import struct

import numpy as np
import pytest
import torch

from cj4me.dataset import (
    DATASET_HEADER_SIZE,
    DATASET_MAGIC,
    DATASET_RECORD_SIZE,
    FEATURE_COUNT,
    FEATURE_SCHEMA_VERSION,
    DatasetV1,
)


def write_dataset(path, records=(), **header_overrides):
    values = {
        "magic": DATASET_MAGIC,
        "format_version": 1,
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_count": FEATURE_COUNT,
        "record_count": len(records),
        "record_size": DATASET_RECORD_SIZE,
        "reserved": 0,
    }
    values.update(header_overrides)
    header = struct.pack(
        "<8s6I",
        values["magic"],
        values["format_version"],
        values["schema_version"],
        values["feature_count"],
        values["record_count"],
        values["record_size"],
        values["reserved"],
    )
    with path.open("wb") as stream:
        stream.write(header)
        for features, target, player, action_type, flags in records:
            stream.write(np.asarray(features, dtype="<f4").tobytes())
            stream.write(struct.pack("<fBBH", target, player, action_type, flags))


def test_reads_records_with_numpy_and_torch_access(tmp_path):
    path = tmp_path / "valid.cj4medata"
    features = np.arange(FEATURE_COUNT, dtype=np.float32) / 10
    write_dataset(path, [(features, 1.25, 2, 7, 3)])

    dataset = DatasetV1(path)

    assert len(dataset) == 1
    assert isinstance(dataset.records, np.memmap)
    np.testing.assert_array_equal(dataset.features[0], features)
    assert dataset.targets[0] == pytest.approx(1.25)
    assert (dataset.action_players[0], dataset.action_types[0], dataset.flags[0]) == (
        2,
        7,
        3,
    )
    tensor_features, tensor_target = dataset[0]
    assert tensor_features.shape == (FEATURE_COUNT,)
    assert tensor_features.dtype == torch.float32
    assert tensor_target.shape == ()
    assert tensor_target.item() == pytest.approx(1.25)


def test_empty_dataset_is_valid(tmp_path):
    path = tmp_path / "empty.cj4medata"
    write_dataset(path)
    dataset = DatasetV1(path)
    assert len(dataset) == 0
    assert dataset.features.shape == (0, FEATURE_COUNT)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("magic", b"BADMAGIC", "magic"),
        ("format_version", 2, "format version"),
        ("schema_version", FEATURE_SCHEMA_VERSION + 1, "schema version"),
        ("feature_count", FEATURE_COUNT + 1, "feature count"),
        ("record_size", DATASET_RECORD_SIZE + 1, "record size"),
        ("reserved", 1, "reserved"),
    ],
)
def test_rejects_invalid_headers(tmp_path, field, value, message):
    path = tmp_path / f"{field}.cj4medata"
    write_dataset(path, **{field: value})
    with pytest.raises(ValueError, match=message):
        DatasetV1(path)


def test_rejects_truncated_header_record_and_trailing_bytes(tmp_path):
    header_path = tmp_path / "short-header.cj4medata"
    header_path.write_bytes(b"\0" * (DATASET_HEADER_SIZE - 1))
    with pytest.raises(ValueError, match="truncated"):
        DatasetV1(header_path)

    features = np.zeros(FEATURE_COUNT, dtype=np.float32)
    record_path = tmp_path / "short-record.cj4medata"
    write_dataset(record_path, [(features, 0.0, 0, 0, 0)])
    record_path.write_bytes(record_path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="truncated"):
        DatasetV1(record_path)

    trailing_path = tmp_path / "trailing.cj4medata"
    write_dataset(trailing_path)
    with trailing_path.open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(ValueError, match="trailing"):
        DatasetV1(trailing_path)


@pytest.mark.parametrize(("features_finite", "target"), [(False, 0.0), (True, np.inf)])
def test_rejects_non_finite_values(tmp_path, features_finite, target):
    path = tmp_path / "non-finite.cj4medata"
    features = np.zeros(FEATURE_COUNT, dtype=np.float32)
    if not features_finite:
        features[10] = np.nan
    write_dataset(path, [(features, target, 0, 0, 0)])
    with pytest.raises(ValueError, match="non-finite"):
        DatasetV1(path)


@pytest.mark.parametrize(
    ("player", "action_type", "message"),
    [(4, 0, "action player"), (0, 11, "action type")],
)
def test_rejects_invalid_action_metadata(tmp_path, player, action_type, message):
    path = tmp_path / "invalid-action.cj4medata"
    features = np.zeros(FEATURE_COUNT, dtype=np.float32)
    write_dataset(path, [(features, 0.0, player, action_type, 0)])
    with pytest.raises(ValueError, match=message):
        DatasetV1(path)
