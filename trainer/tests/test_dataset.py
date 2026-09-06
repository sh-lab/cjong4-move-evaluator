import struct

import numpy as np
import pytest
import torch

from cj4me.dataset import (
    DATASET_HEADER_SIZE,
    DATASET_MAGIC,
    DATASET_FORMAT_VERSION,
    DATASET_RECORD_SIZE,
    FEATURE_COUNT,
    FEATURE_SCHEMA_VERSION,
    DatasetV1,
    DatasetV2,
)


def write_facts(stream, facts=None):
    facts = facts or {}
    stream.write(
        struct.pack(
            "<4i6BH",
            facts.get("score_delta", 0),
            facts.get("settlement_delta", 0),
            facts.get("deal_in_points", 0),
            facts.get("win_points", 0),
            facts.get("decision_discard_count", 0),
            facts.get("round_discard_count", 0),
            facts.get("discards_until_end", 0),
            facts.get("round_end_type", 0),
            facts.get("available_call_mask", 0),
            facts.get("tenpai_status", 0),
            facts.get("fact_flags", 0),
        )
    )


def write_dataset(path, records=(), **header_overrides):
    values = {
        "magic": DATASET_MAGIC,
        "format_version": DATASET_FORMAT_VERSION,
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
        for record in records:
            features, target, player, action_type, flags = record[:5]
            facts = record[5] if len(record) == 6 else None
            stream.write(np.asarray(features, dtype="<f4").tobytes())
            stream.write(struct.pack("<fBBH", target, player, action_type, flags))
            write_facts(stream, facts)


def test_reads_records_with_numpy_and_torch_access(tmp_path):
    path = tmp_path / "valid.cj4medata"
    features = np.arange(FEATURE_COUNT, dtype=np.float32) / 10
    facts = {
        "score_delta": 12000,
        "settlement_delta": 13000,
        "win_points": 13000,
        "decision_discard_count": 12,
        "round_discard_count": 20,
        "discards_until_end": 8,
        "round_end_type": 1,
        "tenpai_status": 2,
        "fact_flags": 65,
    }
    write_dataset(path, [(features, 1.25, 2, 7, 3, facts)])

    dataset = DatasetV2(path)

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
    assert dataset.score_deltas[0] == 12000
    assert dataset.settlement_deltas[0] == 13000
    assert dataset.win_points[0] == 13000
    assert dataset.decision_discard_counts[0] == 12
    assert dataset.round_discard_counts[0] == 20
    assert dataset.discards_until_end[0] == 8
    assert dataset.round_end_types[0] == 1
    assert dataset.tenpai_statuses[0] == 2
    assert dataset.fact_flags[0] == 65
    assert DatasetV1 is DatasetV2


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
        ("format_version", DATASET_FORMAT_VERSION + 1, "format version"),
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


@pytest.mark.parametrize(
    ("facts", "message"),
    [
        (
            {"decision_discard_count": 2, "round_discard_count": 1},
            "decision after",
        ),
        (
            {
                "decision_discard_count": 1,
                "round_discard_count": 2,
                "discards_until_end": 0,
            },
            "discard distance",
        ),
        ({"round_end_type": 5}, "round end type"),
        ({"available_call_mask": 8}, "call mask"),
        ({"tenpai_status": 3}, "tenpai status"),
        ({"fact_flags": 1 << 9}, "fact flags"),
    ],
)
def test_rejects_invalid_teacher_facts(tmp_path, facts, message):
    path = tmp_path / "invalid-facts.cj4medata"
    features = np.zeros(FEATURE_COUNT, dtype=np.float32)
    write_dataset(path, [(features, 0.0, 0, 0, 0, facts)])
    with pytest.raises(ValueError, match=message):
        DatasetV2(path)
