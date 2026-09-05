"""Reader and PyTorch dataset for the CJ4ME dataset v1 container format."""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

DATASET_MAGIC = b"CJ4MEDA1"
DATASET_FORMAT_VERSION = 1
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
DATASET_RECORD_SIZE = FEATURE_COUNT * 4 + 8

_HEADER = struct.Struct("<8s6I")
RECORD_DTYPE = np.dtype(
    [
        ("features", "<f4", (FEATURE_COUNT,)),
        ("target", "<f4"),
        ("action_player", "u1"),
        ("action_type", "u1"),
        ("flags", "<u2"),
    ],
    align=False,
)

if RECORD_DTYPE.itemsize != DATASET_RECORD_SIZE:
    raise RuntimeError("dataset record dtype does not match the v1 format")


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


class DatasetV1(Dataset[tuple[torch.Tensor, torch.Tensor]]):
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

    def numpy_records(self) -> np.ndarray:
        """Return the structured memory-mapped record array."""
        return self.records

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for index in range(len(self)):
            yield self[index]


CJ4MEDataset = DatasetV1


def read_dataset(
    path: str | os.PathLike[str], *, validate_finite: bool = True
) -> DatasetV1:
    """Open a dataset v1 file using memory-mapped record access."""
    return DatasetV1(path, validate_finite=validate_finite)
