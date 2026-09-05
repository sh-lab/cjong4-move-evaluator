"""Export PyTorch checkpoints to the canonical C model formats."""

from __future__ import annotations

import argparse
import pickle
import struct
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from .dataset import (
    ACTION_FEATURE_COUNT,
    FEATURE_SCHEMA_VERSION,
    SCORE_INPUT_COUNT,
    STATE_FEATURE_COUNT,
    TILE_COUNT,
    TILE_EMBEDDING_COUNT,
    TILE_FEATURE_COUNT,
    DatasetV1,
)
from .model import (
    MODEL_HIDDEN1_COUNT,
    MODEL_HIDDEN2_COUNT,
    MODEL_HIDDEN3_COUNT,
    MODEL_TILE_HIDDEN_COUNT,
    MoveEvaluator,
    checkpoint_metadata,
)
from .quantize import QuantizedModel, quantize_model, validate_quantized_model
from .train import select_device

MODEL_MAGIC = b"CJ4MEM02"
MODEL_FORMAT_VERSION = 2
MODEL_HEADER_SIZE = 88
MODEL_LAYER_COUNT = 6
MODEL_KIND_F32 = 1
MODEL_KIND_I8 = 2
TENSOR_F32 = 1
TENSOR_I8 = 2
TENSOR_I32 = 3

_HEADER = struct.Struct("<8s16IQ2I")
_TENSOR_HEADER = struct.Struct("<4I")
_STATE_LAYOUT = (
    ("tile_encoder.0.weight", (MODEL_TILE_HIDDEN_COUNT, TILE_FEATURE_COUNT)),
    ("tile_encoder.0.bias", (MODEL_TILE_HIDDEN_COUNT,)),
    ("tile_encoder.2.weight", (TILE_EMBEDDING_COUNT, MODEL_TILE_HIDDEN_COUNT)),
    ("tile_encoder.2.bias", (TILE_EMBEDDING_COUNT,)),
    ("score_network.0.weight", (MODEL_HIDDEN1_COUNT, SCORE_INPUT_COUNT)),
    ("score_network.0.bias", (MODEL_HIDDEN1_COUNT,)),
    ("score_network.2.weight", (MODEL_HIDDEN2_COUNT, MODEL_HIDDEN1_COUNT)),
    ("score_network.2.bias", (MODEL_HIDDEN2_COUNT,)),
    ("score_network.4.weight", (MODEL_HIDDEN3_COUNT, MODEL_HIDDEN2_COUNT)),
    ("score_network.4.bias", (MODEL_HIDDEN3_COUNT,)),
    ("score_network.6.weight", (1, MODEL_HIDDEN3_COUNT)),
    ("score_network.6.bias", (1,)),
)
_LAYER_INPUTS = (
    TILE_FEATURE_COUNT,
    MODEL_TILE_HIDDEN_COUNT,
    SCORE_INPUT_COUNT,
    MODEL_HIDDEN1_COUNT,
    MODEL_HIDDEN2_COUNT,
    MODEL_HIDDEN3_COUNT,
)
_LAYER_OUTPUTS = (
    MODEL_TILE_HIDDEN_COUNT,
    TILE_EMBEDDING_COUNT,
    MODEL_HIDDEN1_COUNT,
    MODEL_HIDDEN2_COUNT,
    MODEL_HIDDEN3_COUNT,
    1,
)

FLOAT_MODEL_SIZE = (
    MODEL_HEADER_SIZE
    + len(_STATE_LAYOUT) * _TENSOR_HEADER.size
    + 4 * sum(int(np.prod(shape)) for _, shape in _STATE_LAYOUT)
)
_I8_WEIGHT_COUNT = sum(
    input_count * output_count
    for input_count, output_count in zip(_LAYER_INPUTS, _LAYER_OUTPUTS)
)
_I8_CHANNEL_COUNT = sum(_LAYER_OUTPUTS)
_I8_REQUANT_COUNT = sum(_LAYER_OUTPUTS[:-1])
INT8_MODEL_SIZE = (
    MODEL_HEADER_SIZE
    + 21 * _TENSOR_HEADER.size
    + _I8_WEIGHT_COUNT
    + 8 * _I8_CHANNEL_COUNT
    + 7 * 4
    + 8 * _I8_REQUANT_COUNT
)


def validate_checkpoint(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a mapping")
    metadata = checkpoint.get("metadata")
    expected_metadata = checkpoint_metadata()
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint metadata is missing")
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"checkpoint metadata mismatch: {key}")

    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint model_state_dict is missing")
    expected_keys = {name for name, _ in _STATE_LAYOUT}
    if set(state) != expected_keys:
        raise ValueError("checkpoint state_dict keys do not match the fixed model")
    for name, shape in _STATE_LAYOUT:
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"checkpoint tensor {name} is not a tensor")
        if tuple(tensor.shape) != shape:
            raise ValueError(f"checkpoint tensor {name} has invalid shape")
        if tensor.dtype != torch.float32:
            raise ValueError(f"checkpoint tensor {name} must be float32")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"checkpoint tensor {name} contains non-finite values")
    return state


def load_checkpoint(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[dict, MoveEvaluator]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    state = validate_checkpoint(checkpoint)
    model = MoveEvaluator().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return checkpoint, model


def _tensor_record(tensor_id: int, element_type: int, values: np.ndarray) -> bytes:
    if element_type == TENSOR_F32:
        encoded = np.asarray(values, dtype="<f4").tobytes(order="C")
        element_size = 4
    elif element_type == TENSOR_I8:
        encoded = np.asarray(values, dtype=np.int8).tobytes(order="C")
        element_size = 1
    elif element_type == TENSOR_I32:
        encoded = np.asarray(values, dtype="<i4").tobytes(order="C")
        element_size = 4
    else:
        raise ValueError(f"unsupported tensor type: {element_type}")
    count = np.asarray(values).size
    header = _TENSOR_HEADER.pack(tensor_id, element_type, count, count * element_size)
    return header + encoded


def _model_bytes(kind: int, records: Sequence[bytes]) -> bytes:
    payload = b"".join(records)
    header = _HEADER.pack(
        MODEL_MAGIC,
        MODEL_FORMAT_VERSION,
        FEATURE_SCHEMA_VERSION,
        kind,
        MODEL_LAYER_COUNT,
        TILE_COUNT,
        TILE_FEATURE_COUNT,
        MODEL_TILE_HIDDEN_COUNT,
        TILE_EMBEDDING_COUNT,
        STATE_FEATURE_COUNT,
        ACTION_FEATURE_COUNT,
        SCORE_INPUT_COUNT,
        MODEL_HIDDEN1_COUNT,
        MODEL_HIDDEN2_COUNT,
        MODEL_HIDDEN3_COUNT,
        1,
        len(records),
        len(payload),
        MODEL_HEADER_SIZE,
        0,
    )
    if len(header) != MODEL_HEADER_SIZE:
        raise RuntimeError("internal model header size mismatch")
    return header + payload


def serialize_float_model(model: MoveEvaluator) -> bytes:
    records = []
    state = model.state_dict()
    for tensor_id, (name, shape) in enumerate(_STATE_LAYOUT, start=1):
        tensor = state[name]
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != torch.float32
            or not torch.isfinite(tensor).all().item()
        ):
            raise ValueError(f"model tensor {name} is invalid")
        records.append(
            _tensor_record(tensor_id, TENSOR_F32, tensor.detach().cpu().numpy())
        )
    result = _model_bytes(MODEL_KIND_F32, records)
    if len(result) != FLOAT_MODEL_SIZE:
        raise RuntimeError("internal float model size mismatch")
    return result


def serialize_int8_model(model: QuantizedModel) -> bytes:
    validate_quantized_model(model)
    records = []
    tensor_id = 101
    for weights, biases, scales in zip(
        model.weights, model.biases, model.weight_scales
    ):
        records.append(_tensor_record(tensor_id, TENSOR_I8, weights))
        records.append(_tensor_record(tensor_id + 1, TENSOR_I32, biases))
        records.append(_tensor_record(tensor_id + 2, TENSOR_F32, scales))
        tensor_id += 3
    records.extend(
        (
            _tensor_record(119, TENSOR_F32, model.activation_scales),
            _tensor_record(120, TENSOR_I32, model.requant_multipliers),
            _tensor_record(121, TENSOR_I32, model.requant_shifts),
        )
    )
    result = _model_bytes(MODEL_KIND_I8, records)
    if len(result) != INT8_MODEL_SIZE:
        raise RuntimeError("internal INT8 model size mismatch")
    return result


def _write(path: str | Path, data: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a CJ4ME checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--float-output", required=True)
    parser.add_argument("--int8-output", required=True)
    parser.add_argument("--calibration-dataset", required=True)
    parser.add_argument("--calibration-batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.calibration_batch_size < 1:
        parser.error("--calibration-batch-size must be at least 1")
    try:
        device = select_device(args.device)
        _, model = load_checkpoint(args.checkpoint, device)
        dataset = DatasetV1(args.calibration_dataset)
        quantized = quantize_model(
            model,
            dataset,
            batch_size=args.calibration_batch_size,
            device=device,
        )
        float_bytes = serialize_float_model(model)
        int8_bytes = serialize_int8_model(quantized)
        _write(args.float_output, float_bytes)
        _write(args.int8_output, int8_bytes)
    except (
        EOFError,
        OSError,
        pickle.UnpicklingError,
        RuntimeError,
        ValueError,
    ) as error:
        parser.error(str(error))
    print(f"saved float32 model: {args.float_output}")
    print(f"saved INT8 model: {args.int8_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
