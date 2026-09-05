"""Post-training quantization for the physical-tile CJ4ME network."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import (
    FEATURE_COUNT,
    TILE_COUNT,
    TILE_FEATURE_COUNT,
    TILE_FEATURES_COUNT,
    DatasetV1,
)
from .model import MoveEvaluator

INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1


@dataclass(frozen=True)
class QuantizedModel:
    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    weight_scales: tuple[np.ndarray, ...]
    activation_scales: np.ndarray
    requant_multipliers: np.ndarray
    requant_shifts: np.ndarray


def round_ties_away_from_zero(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.copysign(np.floor(np.abs(values) + 0.5), values)


def _positive_scale(maximum: float) -> np.float32:
    if not math.isfinite(maximum) or maximum < 0.0:
        raise ValueError("calibration maximum must be finite and nonnegative")
    scale = np.float32(maximum / 127.0) if maximum else np.float32(1.0)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("quantization scale is not positive and finite")
    return scale


def calibrate_activation_scales(
    model: MoveEvaluator,
    dataset: DatasetV1,
    *,
    batch_size: int = 1024,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Collect scales for tile input/hidden, score input, and score hidden layers."""
    if batch_size < 1:
        raise ValueError("calibration batch size must be at least 1")
    if len(dataset) == 0:
        raise ValueError("calibration dataset must not be empty")

    target_device = torch.device(device)
    model.to(target_device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    maxima = np.zeros(6, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for features, _ in loader:
            features = features.to(target_device)
            tiles = features[..., :TILE_FEATURES_COUNT].reshape(
                -1, TILE_COUNT, TILE_FEATURE_COUNT
            )
            maxima[0] = max(maxima[0], float(tiles.abs().max().item()))
            tile_hidden = torch.relu(model.tile_encoder[0](tiles))
            maxima[1] = max(maxima[1], float(tile_hidden.abs().max().item()))
            tile_output = torch.relu(model.tile_encoder[2](tile_hidden))
            score_input = torch.cat(
                (
                    tile_output.flatten(start_dim=-2),
                    features[..., TILE_FEATURES_COUNT:],
                ),
                dim=-1,
            )
            maxima[2] = max(maxima[2], float(score_input.abs().max().item()))
            value = score_input
            for maximum_index, linear_index in enumerate((0, 2, 4), start=3):
                value = torch.relu(model.score_network[linear_index](value))
                maxima[maximum_index] = max(
                    maxima[maximum_index], float(value.abs().max().item())
                )

    return np.asarray([_positive_scale(value) for value in maxima], dtype="<f4")


def quantize_weight_channels(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weight = np.asarray(weight, dtype=np.float32)
    if weight.ndim != 2 or not np.isfinite(weight).all():
        raise ValueError("weights must be a finite rank-2 float32 array")
    maxima = np.max(np.abs(weight.astype(np.float64)), axis=1)
    scales = np.asarray([_positive_scale(value) for value in maxima], dtype="<f4")
    quantized = round_ties_away_from_zero(
        weight.astype(np.float64) / scales.astype(np.float64)[:, None]
    )
    quantized = np.clip(quantized, -127, 127).astype(np.int8)
    return quantized, scales


def quantize_biases(bias: np.ndarray, accumulator_scales: np.ndarray) -> np.ndarray:
    bias = np.asarray(bias, dtype=np.float32)
    accumulator_scales = np.asarray(accumulator_scales, dtype=np.float32)
    if bias.shape != accumulator_scales.shape:
        raise ValueError("bias and accumulator scale shapes differ")
    if (
        not np.isfinite(bias).all()
        or not np.isfinite(accumulator_scales).all()
        or np.any(accumulator_scales <= 0.0)
    ):
        raise ValueError("biases and accumulator scales must be finite")
    rounded = round_ties_away_from_zero(
        bias.astype(np.float64) / accumulator_scales.astype(np.float64)
    )
    if np.any(rounded < INT32_MIN) or np.any(rounded > INT32_MAX):
        raise ValueError("quantized bias exceeds int32 range")
    return rounded.astype("<i4")


def approximate_requant_ratio(ratio: float) -> tuple[int, int]:
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("requantization ratio must be positive and finite")
    best: tuple[float, int, int] | None = None
    for shift in range(-62, 63):
        scaled = math.ldexp(ratio, shift)
        if not math.isfinite(scaled):
            continue
        multiplier = int(math.floor(scaled + 0.5))
        if multiplier < 1 or multiplier > INT32_MAX:
            continue
        approximation = math.ldexp(float(multiplier), -shift)
        candidate = (abs(approximation - ratio) / ratio, -shift, multiplier)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError(f"requantization ratio {ratio!r} is not representable")
    return best[2], -best[1]


def _check_accumulators(weight: np.ndarray, bias: np.ndarray) -> None:
    weight64 = weight.astype(np.int64)
    bias64 = bias.astype(np.int64)
    minimum = bias64 + np.where(weight64 >= 0, -128 * weight64, 127 * weight64).sum(
        axis=1
    )
    maximum = bias64 + np.where(weight64 >= 0, 127 * weight64, -128 * weight64).sum(
        axis=1
    )
    if np.any(minimum < INT32_MIN) or np.any(maximum > INT32_MAX):
        raise ValueError("worst-case int8 accumulation exceeds int32 range")


def _float_parameters(
    model: MoveEvaluator,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    layers = (
        model.tile_encoder[0],
        model.tile_encoder[2],
        model.score_network[0],
        model.score_network[2],
        model.score_network[4],
        model.score_network[6],
    )
    result = []
    for layer in layers:
        weight = layer.weight.detach().cpu().numpy()
        bias = layer.bias.detach().cpu().numpy()
        if (
            weight.dtype != np.float32
            or bias.dtype != np.float32
            or not np.isfinite(weight).all()
            or not np.isfinite(bias).all()
        ):
            raise ValueError("model parameters must be finite float32 tensors")
        result.append((weight, bias))
    return tuple(result)


def quantize_model(
    model: MoveEvaluator,
    dataset: DatasetV1,
    *,
    batch_size: int = 1024,
    device: torch.device | str = "cpu",
) -> QuantizedModel:
    activation_inputs = calibrate_activation_scales(
        model, dataset, batch_size=batch_size, device=device
    )
    parameters = _float_parameters(model)
    weights = []
    biases = []
    weight_scales = []
    multipliers = []
    shifts = []

    for layer_index, ((weight, bias), input_scale) in enumerate(
        zip(parameters, activation_inputs)
    ):
        quantized_weight, channel_scales = quantize_weight_channels(weight)
        accumulator_scales = np.asarray(
            np.float32(input_scale) * channel_scales, dtype="<f4"
        )
        if (
            not np.isfinite(accumulator_scales).all()
            or np.any(accumulator_scales <= 0.0)
        ):
            raise ValueError("accumulator scale is not positive and finite")
        quantized_bias = quantize_biases(bias, accumulator_scales)
        _check_accumulators(quantized_weight, quantized_bias)
        weights.append(quantized_weight)
        biases.append(quantized_bias)
        weight_scales.append(channel_scales)

        if layer_index < 5:
            output_scale = activation_inputs[layer_index + 1]
            for accumulator_scale in accumulator_scales:
                multiplier, shift = approximate_requant_ratio(
                    float(accumulator_scale) / float(output_scale)
                )
                multipliers.append(multiplier)
                shifts.append(shift)

    output_scale = np.float32(activation_inputs[5] * weight_scales[5][0])
    if not np.isfinite(output_scale) or output_scale <= 0.0:
        raise ValueError("output scale is not positive and finite")
    activation_scales = np.concatenate(
        (activation_inputs, np.asarray([output_scale], dtype="<f4"))
    ).astype("<f4", copy=False)
    return QuantizedModel(
        weights=tuple(weights),
        biases=tuple(biases),
        weight_scales=tuple(weight_scales),
        activation_scales=activation_scales,
        requant_multipliers=np.asarray(multipliers, dtype="<i4"),
        requant_shifts=np.asarray(shifts, dtype="<i4"),
    )


def validate_quantized_model(model: QuantizedModel) -> None:
    weight_shapes = (
        (16, 13),
        (8, 16),
        (128, 1123),
        (64, 128),
        (16, 64),
        (1, 16),
    )
    channel_shapes = ((16,), (8,), (128,), (64,), (16,), (1,))
    if not isinstance(model, QuantizedModel):
        raise ValueError("expected a QuantizedModel")
    if not (
        len(model.weights) == len(model.biases) == len(model.weight_scales) == 6
    ):
        raise ValueError("quantized model must contain six layers")
    for index, (weights, biases, scales, weight_shape, channel_shape) in enumerate(
        zip(
            model.weights,
            model.biases,
            model.weight_scales,
            weight_shapes,
            channel_shapes,
        ),
        start=1,
    ):
        if weights.shape != weight_shape or weights.dtype != np.int8:
            raise ValueError(f"quantized layer {index} weights are invalid")
        if biases.shape != channel_shape or biases.dtype != np.dtype("<i4"):
            raise ValueError(f"quantized layer {index} biases are invalid")
        if (
            scales.shape != channel_shape
            or scales.dtype != np.dtype("<f4")
            or not np.isfinite(scales).all()
            or np.any(scales <= 0.0)
        ):
            raise ValueError(f"quantized layer {index} weight scales are invalid")
        _check_accumulators(weights, biases)

    if (
        model.activation_scales.shape != (7,)
        or model.activation_scales.dtype != np.dtype("<f4")
        or not np.isfinite(model.activation_scales).all()
        or np.any(model.activation_scales <= 0.0)
    ):
        raise ValueError("activation scales are invalid")
    expected_output_scale = np.float32(
        model.activation_scales[5] * model.weight_scales[5][0]
    )
    if model.activation_scales[6] != expected_output_scale:
        raise ValueError("output scale does not match the final accumulator scale")
    if (
        model.requant_multipliers.shape != (232,)
        or model.requant_multipliers.dtype != np.dtype("<i4")
        or np.any(model.requant_multipliers <= 0)
    ):
        raise ValueError("requant multipliers are invalid")
    if (
        model.requant_shifts.shape != (232,)
        or model.requant_shifts.dtype != np.dtype("<i4")
        or np.any(model.requant_shifts < -62)
        or np.any(model.requant_shifts > 62)
    ):
        raise ValueError("requant shifts are invalid")


def _quantize_input_value(value: np.float32, scale: np.float32) -> int:
    if not np.isfinite(value):
        raise ValueError("input features must be finite")
    with np.errstate(over="ignore"):
        scaled = np.float32(value / scale)
    if not np.isfinite(scaled):
        return 127 if scaled > 0 else -128
    if scaled >= np.float32(127):
        return 127
    if scaled <= np.float32(-128):
        return -128
    truncated = math.trunc(float(scaled))
    difference = float(scaled) - truncated
    if difference >= 0.5:
        truncated += 1
    elif difference <= -0.5:
        truncated -= 1
    return truncated


def _requantize(value: int, multiplier: int, shift: int) -> int:
    scaled = value * multiplier
    if shift > 0:
        magnitude = abs(scaled)
        rounded = (magnitude >> shift) + (
            1 if magnitude & (1 << (shift - 1)) else 0
        )
        scaled = rounded if scaled >= 0 else -rounded
    elif shift < 0:
        for _ in range(-shift):
            if scaled > 127:
                return 127
            if scaled < -128:
                return -128
            scaled *= 2
    return min(127, max(-128, scaled))


def _dense_reference(
    inputs: list[int],
    weights: np.ndarray,
    biases: np.ndarray,
    multipliers: np.ndarray | None,
    shifts: np.ndarray | None,
) -> list[int]:
    outputs = []
    for output_index in range(weights.shape[0]):
        accumulator = int(biases[output_index])
        for input_value, weight in zip(inputs, weights[output_index]):
            accumulator += input_value * int(weight)
        if accumulator < INT32_MIN or accumulator > INT32_MAX:
            raise OverflowError("dense accumulator exceeds int32 range")
        if multipliers is None:
            outputs.append(accumulator)
        elif accumulator <= 0:
            outputs.append(0)
        else:
            outputs.append(
                _requantize(
                    accumulator,
                    int(multipliers[output_index]),
                    int(shifts[output_index]),
                )
            )
    return outputs


def infer_int8_reference(model: QuantizedModel, features: np.ndarray) -> np.int32:
    validate_quantized_model(model)
    features = np.asarray(features, dtype=np.float32)
    if features.shape != (FEATURE_COUNT,):
        raise ValueError(f"features must have shape ({FEATURE_COUNT},)")

    tile_scale, _, score_input_scale = model.activation_scales[:3]
    tile_features = features[:TILE_FEATURES_COUNT].reshape(
        TILE_COUNT, TILE_FEATURE_COUNT
    )
    requant_offset = 0
    tile_outputs: list[int] = []
    for tile in tile_features:
        values = [_quantize_input_value(value, tile_scale) for value in tile]
        for layer_index in range(2):
            output_count = model.weights[layer_index].shape[0]
            values = _dense_reference(
                values,
                model.weights[layer_index],
                model.biases[layer_index],
                model.requant_multipliers[
                    requant_offset : requant_offset + output_count
                ],
                model.requant_shifts[requant_offset : requant_offset + output_count],
            )
            requant_offset += output_count
        tile_outputs.extend(values)
        requant_offset = 0

    values = tile_outputs + [
        _quantize_input_value(value, score_input_scale)
        for value in features[TILE_FEATURES_COUNT:]
    ]
    requant_offset = 16 + 8
    for layer_index in range(2, 5):
        output_count = model.weights[layer_index].shape[0]
        values = _dense_reference(
            values,
            model.weights[layer_index],
            model.biases[layer_index],
            model.requant_multipliers[
                requant_offset : requant_offset + output_count
            ],
            model.requant_shifts[requant_offset : requant_offset + output_count],
        )
        requant_offset += output_count
    return np.int32(
        _dense_reference(values, model.weights[5], model.biases[5], None, None)[0]
    )
