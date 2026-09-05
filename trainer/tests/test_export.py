import struct

import numpy as np
import pytest
import torch

from cj4me.dataset import (
    ACTION_FEATURE_COUNT,
    DATASET_MAGIC,
    DATASET_FORMAT_VERSION,
    DATASET_RECORD_SIZE,
    FEATURE_COUNT,
    FEATURE_SCHEMA_VERSION,
    SCORE_INPUT_COUNT,
    STATE_FEATURE_COUNT,
    TILE_COUNT,
    TILE_EMBEDDING_COUNT,
    TILE_FEATURE_COUNT,
    DatasetV1,
)
from cj4me.export import (
    FLOAT_MODEL_SIZE,
    INT8_MODEL_SIZE,
    MODEL_FORMAT_VERSION,
    MODEL_HEADER_SIZE,
    load_checkpoint,
    serialize_float_model,
    serialize_int8_model,
    validate_checkpoint,
)
from cj4me.model import MoveEvaluator, checkpoint_metadata
from cj4me.quantize import (
    QuantizedModel,
    approximate_requant_ratio,
    infer_int8_reference,
    quantize_biases,
    quantize_model,
    quantize_weight_channels,
)


def parse_model(data):
    header = struct.unpack("<8s16IQ2I", data[:MODEL_HEADER_SIZE])
    tensors = []
    offset = MODEL_HEADER_SIZE
    while offset < len(data):
        tensor_id, element_type, count, byte_length = struct.unpack(
            "<4I", data[offset : offset + 16]
        )
        offset += 16
        tensors.append(
            (tensor_id, element_type, count, data[offset : offset + byte_length])
        )
        offset += byte_length
    assert offset == len(data)
    return header, tensors


def write_dataset(path, feature_rows):
    with path.open("wb") as stream:
        stream.write(
            struct.pack(
                "<8s6I",
                DATASET_MAGIC,
                DATASET_FORMAT_VERSION,
                FEATURE_SCHEMA_VERSION,
                FEATURE_COUNT,
                len(feature_rows),
                DATASET_RECORD_SIZE,
                0,
            )
        )
        for features in feature_rows:
            stream.write(np.asarray(features, dtype="<f4").tobytes())
            stream.write(struct.pack("<fBBH", 0.0, 0, 0, 0))


def make_checkpoint(model):
    return {
        "metadata": checkpoint_metadata(),
        "model_state_dict": model.state_dict(),
    }


def test_float_export_header_and_tensors_equal_state_dict():
    torch.manual_seed(7)
    model = MoveEvaluator()
    data = serialize_float_model(model)
    header, tensors = parse_model(data)

    assert len(data) == FLOAT_MODEL_SIZE
    assert header == (
        b"CJ4MEM02",
        MODEL_FORMAT_VERSION,
        FEATURE_SCHEMA_VERSION,
        1,
        6,
        TILE_COUNT,
        TILE_FEATURE_COUNT,
        16,
        TILE_EMBEDDING_COUNT,
        STATE_FEATURE_COUNT,
        ACTION_FEATURE_COUNT,
        SCORE_INPUT_COUNT,
        128,
        64,
        16,
        1,
        12,
        len(data) - MODEL_HEADER_SIZE,
        MODEL_HEADER_SIZE,
        0,
    )
    state_values = list(model.state_dict().values())
    assert [tensor[:3] for tensor in tensors] == [
        (index, 1, value.numel())
        for index, value in enumerate(state_values, start=1)
    ]
    for (_, _, count, encoded), expected in zip(tensors, state_values):
        actual = np.frombuffer(encoded, dtype="<f4", count=count).reshape(
            expected.shape
        )
        np.testing.assert_array_equal(actual, expected.numpy())


def test_checkpoint_validation_rejects_metadata_shape_and_nonfinite(tmp_path):
    model = MoveEvaluator()
    checkpoint = make_checkpoint(model)
    validate_checkpoint(checkpoint)

    bad_metadata = make_checkpoint(model)
    bad_metadata["metadata"] = {**checkpoint_metadata(), "feature_count": 990}
    with pytest.raises(ValueError, match="metadata"):
        validate_checkpoint(bad_metadata)

    bad_shape = make_checkpoint(model)
    bad_shape["model_state_dict"] = dict(model.state_dict())
    bad_shape["model_state_dict"]["score_network.0.bias"] = torch.zeros(127)
    with pytest.raises(ValueError, match="shape"):
        validate_checkpoint(bad_shape)

    bad_finite = make_checkpoint(model)
    bad_finite["model_state_dict"] = dict(model.state_dict())
    bad_finite["model_state_dict"]["score_network.6.bias"] = torch.tensor(
        [float("nan")]
    )
    with pytest.raises(ValueError, match="non-finite"):
        validate_checkpoint(bad_finite)

    path = tmp_path / "model.pt"
    torch.save(checkpoint, path)
    _, loaded = load_checkpoint(path)
    assert torch.equal(
        loaded.state_dict()["score_network.0.weight"], model.score_network[0].weight
    )


def test_quantization_scales_clamp_bias_and_requant():
    weights = np.array([[127.5, -127.5, 0.5], [0.0, 0.0, 0.0]], dtype=np.float32)
    quantized, scales = quantize_weight_channels(weights)
    assert quantized[0].tolist() == [127, -127, 0]
    assert quantized[1].tolist() == [0, 0, 0]
    assert scales[1] == np.float32(1.0)
    assert np.all(np.isfinite(scales)) and np.all(scales > 0)

    biases = quantize_biases(
        np.array([0.5, -0.5], dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32),
    )
    assert biases.tolist() == [1, -1]
    with pytest.raises(ValueError, match="int32"):
        quantize_biases(
            np.array([3.0e9], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
        )

    for ratio in (0.001, 0.5, 1.0, 3.25, 1000.0):
        multiplier, shift = approximate_requant_ratio(ratio)
        actual = multiplier * (2.0 ** (-shift))
        assert multiplier > 0
        assert -62 <= shift <= 62
        assert actual == pytest.approx(ratio, rel=1e-8)


def test_ptq_tensor_order_shapes_and_output_scale(tmp_path):
    model = MoveEvaluator()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.0)
        model.tile_encoder[0].weight[0, 0] = 1.0
        model.tile_encoder[2].weight[0, 0] = 1.0
        model.score_network[0].weight[0, 0] = 1.0
        model.score_network[2].weight[0, 0] = 1.0
        model.score_network[4].weight[0, 0] = 1.0
        model.score_network[6].weight[0, 0] = 1.0

    path = tmp_path / "calibration.cj4medata"
    row = np.zeros(FEATURE_COUNT, dtype=np.float32)
    row[0] = 127.0
    write_dataset(path, [row, -row])
    quantized = quantize_model(model, DatasetV1(path), batch_size=1)
    data = serialize_int8_model(quantized)
    header, tensors = parse_model(data)

    assert len(data) == INT8_MODEL_SIZE
    assert header[3] == 2
    assert header[16] == 21
    assert [tensor[0] for tensor in tensors] == list(range(101, 122))
    assert [tensor[1] for tensor in tensors] == [
        2,
        3,
        1,
        2,
        3,
        1,
        2,
        3,
        1,
        2,
        3,
        1,
        2,
        3,
        1,
        2,
        3,
        1,
        1,
        3,
        3,
    ]
    assert quantized.requant_multipliers.shape == (232,)
    assert quantized.requant_shifts.shape == (232,)
    assert quantized.activation_scales.shape == (7,)
    np.testing.assert_array_equal(
        quantized.activation_scales[:6], np.ones(6, dtype=np.float32)
    )
    assert quantized.activation_scales[6] == np.float32(
        quantized.activation_scales[5] * quantized.weight_scales[5][0]
    )


def test_reference_int8_tracks_float_output_and_argmax(tmp_path):
    model = MoveEvaluator()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.tile_encoder[0].weight[0, 0] = 1.0
        model.tile_encoder[2].weight[0, 0] = 1.0
        model.score_network[0].weight[0, 0] = 1.0
        model.score_network[2].weight[0, 0] = 1.0
        model.score_network[4].weight[0, 0] = 1.0
        model.score_network[6].weight[0, 0] = 1.0

    rows = []
    for value in (0.0, 10.0, 20.0):
        row = np.zeros(FEATURE_COUNT, dtype=np.float32)
        row[0] = value
        rows.append(row)
    path = tmp_path / "reference.cj4medata"
    write_dataset(path, rows)
    quantized = quantize_model(model, DatasetV1(path), batch_size=2)

    float_outputs = model(torch.from_numpy(np.stack(rows))).detach().numpy().ravel()
    integer_outputs = np.asarray(
        [infer_int8_reference(quantized, row) for row in rows], dtype=np.int32
    )
    reference_outputs = (
        integer_outputs.astype(np.float32) * quantized.activation_scales[6]
    )

    np.testing.assert_allclose(reference_outputs, float_outputs, atol=0.1, rtol=0.0)
    assert np.argmax(reference_outputs) == np.argmax(float_outputs)


def test_reference_int8_ties_and_boundary_saturation():
    weights = [
        np.zeros((16, 13), dtype=np.int8),
        np.zeros((8, 16), dtype=np.int8),
        np.zeros((128, SCORE_INPUT_COUNT), dtype=np.int8),
        np.zeros((64, 128), dtype=np.int8),
        np.zeros((16, 64), dtype=np.int8),
        np.zeros((1, 16), dtype=np.int8),
    ]
    for weight in weights:
        weight[0, 0] = 1
    model = QuantizedModel(
        weights=tuple(weights),
        biases=(
            np.zeros(16, dtype="<i4"),
            np.zeros(8, dtype="<i4"),
            np.zeros(128, dtype="<i4"),
            np.zeros(64, dtype="<i4"),
            np.zeros(16, dtype="<i4"),
            np.zeros(1, dtype="<i4"),
        ),
        weight_scales=(
            np.ones(16, dtype="<f4"),
            np.ones(8, dtype="<f4"),
            np.ones(128, dtype="<f4"),
            np.ones(64, dtype="<f4"),
            np.ones(16, dtype="<f4"),
            np.ones(1, dtype="<f4"),
        ),
        activation_scales=np.ones(7, dtype="<f4"),
        requant_multipliers=np.ones(232, dtype="<i4"),
        requant_shifts=np.zeros(232, dtype="<i4"),
    )

    features = np.zeros(FEATURE_COUNT, dtype=np.float32)
    features[0] = 0.5
    assert infer_int8_reference(model, features) == 1
    features[0] = -0.5
    assert infer_int8_reference(model, features) == 0

    for weight in weights:
        weight[0, 0] = 127
    features[0] = 1.0e6
    assert infer_int8_reference(model, features) == 127 * 127
    features[0] = -1.0e6
    assert infer_int8_reference(model, features) == 0
