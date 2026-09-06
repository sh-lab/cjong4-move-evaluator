"""Reader and PyTorch dataset for the CJ4ME dataset v2 container format."""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

DATASET_MAGIC = b"CJ4MEDA2"
DATASET_FORMAT_VERSION = 2
FEATURE_SCHEMA_VERSION = 3
TILE_COUNT = 136
TILE_FEATURE_COUNT = 13
TILE_EMBEDDING_COUNT = 8
TILE_FEATURES_COUNT = TILE_COUNT * TILE_FEATURE_COUNT
STATE_FEATURE_COUNT = 33
ACTION_FEATURE_COUNT = 2
CONTEXT_FEATURE_COUNT = STATE_FEATURE_COUNT + ACTION_FEATURE_COUNT
FEATURE_COUNT = TILE_FEATURES_COUNT + CONTEXT_FEATURE_COUNT
SCORE_INPUT_COUNT = TILE_COUNT * TILE_EMBEDDING_COUNT + CONTEXT_FEATURE_COUNT
DATASET_HEADER_SIZE = 32
DATASET_RECORD_SIZE = FEATURE_COUNT * 4 + 32
MAX_DISCARDS = 86
CALL_AVAILABLE_CHI = 1 << 0
CALL_AVAILABLE_PON = 1 << 1
CALL_AVAILABLE_MINKAN = 1 << 2
CALL_AVAILABLE_MASK = (
    CALL_AVAILABLE_CHI | CALL_AVAILABLE_PON | CALL_AVAILABLE_MINKAN
)
ROUND_END_NONE = 0
ROUND_END_TSUMO = 1
ROUND_END_RON = 2
ROUND_END_EXHAUSTIVE_DRAW = 3
ROUND_END_ABORTIVE_DRAW = 4
TENPAI_UNKNOWN = 0
TENPAI_NO = 1
TENPAI_YES = 2
FACT_WAS_MENZEN = 1 << 0
FACT_OPENED_HAND = 1 << 1
FACT_CALL_AVAILABLE = 1 << 2
FACT_CHOSE_CALL = 1 << 3
FACT_RIICHI_AVAILABLE = 1 << 4
FACT_CHOSE_RIICHI = 1 << 5
FACT_PLAYER_WON = 1 << 6
FACT_PLAYER_DEALT_IN = 1 << 7
FACT_DEAL_IN_ACTION = 1 << 8
FACT_FLAGS_MASK = (1 << 9) - 1

_HEADER = struct.Struct("<8s6I")
RECORD_DTYPE = np.dtype(
    [
        ("features", "<f4", (FEATURE_COUNT,)),
        ("target", "<f4"),
        ("action_player", "u1"),
        ("action_type", "u1"),
        ("flags", "<u2"),
        ("score_delta", "<i4"),
        ("settlement_delta", "<i4"),
        ("deal_in_points", "<i4"),
        ("win_points", "<i4"),
        ("decision_discard_count", "u1"),
        ("round_discard_count", "u1"),
        ("discards_until_end", "u1"),
        ("round_end_type", "u1"),
        ("available_call_mask", "u1"),
        ("tenpai_status", "u1"),
        ("fact_flags", "<u2"),
    ],
    align=False,
)

if RECORD_DTYPE.itemsize != DATASET_RECORD_SIZE:
    raise RuntimeError("dataset record dtype does not match the v2 format")


@dataclass(frozen=True)
class DatasetHeader:
    format_version: int
    schema_version: int
    feature_count: int
    record_count: int
    record_size: int
    reserved: int


def _read_header(path: Path) -> DatasetHeader:
    try:
        file_size = path.stat().st_size
    except OSError as error:
        raise ValueError(f"cannot stat dataset {path}: {error}") from error

    if file_size < DATASET_HEADER_SIZE:
        raise ValueError(
            f"truncated dataset header: expected {DATASET_HEADER_SIZE} bytes, "
            f"found {file_size}"
        )

    try:
        with path.open("rb") as stream:
            raw_header = stream.read(DATASET_HEADER_SIZE)
    except OSError as error:
        raise ValueError(f"cannot read dataset {path}: {error}") from error

    magic, format_version, schema_version, feature_count, record_count, record_size, reserved = (
        _HEADER.unpack(raw_header)
    )
    if magic != DATASET_MAGIC:
        raise ValueError(f"invalid dataset magic: {magic!r}")
    if format_version != DATASET_FORMAT_VERSION:
        raise ValueError(f"unsupported dataset format version: {format_version}")
    if schema_version != FEATURE_SCHEMA_VERSION:
        raise ValueError(f"unsupported feature schema version: {schema_version}")
    if feature_count != FEATURE_COUNT:
        raise ValueError(f"invalid feature count: {feature_count}")
    if record_size != DATASET_RECORD_SIZE:
        raise ValueError(f"invalid record size: {record_size}")
    if reserved != 0:
        raise ValueError("dataset header reserved field must be zero")

    expected_size = DATASET_HEADER_SIZE + record_count * DATASET_RECORD_SIZE
    if file_size != expected_size:
        detail = "truncated" if file_size < expected_size else "has trailing bytes"
        raise ValueError(
            f"dataset {detail}: expected {expected_size} bytes, found {file_size}"
        )

    return DatasetHeader(
        format_version=format_version,
        schema_version=schema_version,
        feature_count=feature_count,
        record_count=record_count,
        record_size=record_size,
        reserved=reserved,
    )


class DatasetV2(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Memory-mapped dataset that returns ``(features, target)`` tensors."""

    def __init__(self, path: str | os.PathLike[str], *, validate_finite: bool = True):
        self.path = Path(path)
        self.header = _read_header(self.path)
        if self.header.record_count:
            self.records: np.ndarray = np.memmap(
                self.path,
                mode="r",
                dtype=RECORD_DTYPE,
                offset=DATASET_HEADER_SIZE,
                shape=(self.header.record_count,),
            )
        else:
            self.records = np.empty(0, dtype=RECORD_DTYPE)

        if validate_finite:
            if not np.isfinite(self.features).all():
                raise ValueError("dataset contains non-finite features")
            if not np.isfinite(self.targets).all():
                raise ValueError("dataset contains non-finite targets")
        if np.any(self.action_players >= 4):
            raise ValueError("dataset contains an invalid action player")
        if np.any(self.action_types > 10):
            raise ValueError("dataset contains an invalid action type")
        if np.any(self.decision_discard_counts > MAX_DISCARDS):
            raise ValueError("dataset contains an invalid decision discard count")
        if np.any(self.round_discard_counts > MAX_DISCARDS):
            raise ValueError("dataset contains an invalid round discard count")
        if np.any(self.decision_discard_counts > self.round_discard_counts):
            raise ValueError("dataset contains a decision after the round end")
        if np.any(
            self.discards_until_end
            != self.round_discard_counts - self.decision_discard_counts
        ):
            raise ValueError("dataset contains an inconsistent discard distance")
        if np.any(self.round_end_types > ROUND_END_ABORTIVE_DRAW):
            raise ValueError("dataset contains an invalid round end type")
        if np.any(
            self.available_call_masks & np.uint8(0xFF ^ CALL_AVAILABLE_MASK)
        ):
            raise ValueError("dataset contains an invalid available call mask")
        if np.any(self.tenpai_statuses > TENPAI_YES):
            raise ValueError("dataset contains an invalid tenpai status")
        if np.any(self.fact_flags & np.uint16(0xFFFF ^ FACT_FLAGS_MASK)):
            raise ValueError("dataset contains invalid fact flags")

    def __len__(self) -> int:
        return self.header.record_count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        features = torch.from_numpy(np.array(record["features"], dtype=np.float32, copy=True))
        target = torch.tensor(record["target"], dtype=torch.float32)
        return features, target

    @property
    def features(self) -> np.ndarray:
        return self.records["features"]

    @property
    def targets(self) -> np.ndarray:
        return self.records["target"]

    @property
    def action_players(self) -> np.ndarray:
        return self.records["action_player"]

    @property
    def action_types(self) -> np.ndarray:
        return self.records["action_type"]

    @property
    def flags(self) -> np.ndarray:
        return self.records["flags"]

    @property
    def score_deltas(self) -> np.ndarray:
        return self.records["score_delta"]

    @property
    def settlement_deltas(self) -> np.ndarray:
        return self.records["settlement_delta"]

    @property
    def deal_in_points(self) -> np.ndarray:
        return self.records["deal_in_points"]

    @property
    def win_points(self) -> np.ndarray:
        return self.records["win_points"]

    @property
    def decision_discard_counts(self) -> np.ndarray:
        return self.records["decision_discard_count"]

    @property
    def round_discard_counts(self) -> np.ndarray:
        return self.records["round_discard_count"]

    @property
    def discards_until_end(self) -> np.ndarray:
        return self.records["discards_until_end"]

    @property
    def round_end_types(self) -> np.ndarray:
        return self.records["round_end_type"]

    @property
    def available_call_masks(self) -> np.ndarray:
        return self.records["available_call_mask"]

    @property
    def tenpai_statuses(self) -> np.ndarray:
        return self.records["tenpai_status"]

    @property
    def fact_flags(self) -> np.ndarray:
        return self.records["fact_flags"]

    def numpy_records(self) -> np.ndarray:
        """Return the structured memory-mapped record array."""
        return self.records

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for index in range(len(self)):
            yield self[index]


CJ4MEDataset = DatasetV2

# Compatibility name for existing training scripts. It opens only the current
# v2 container and is not a legacy v1 reader.
DatasetV1 = DatasetV2


def read_dataset(
    path: str | os.PathLike[str], *, validate_finite: bool = True
) -> DatasetV2:
    """Open a dataset v2 file using memory-mapped record access."""
    return DatasetV2(path, validate_finite=validate_finite)
