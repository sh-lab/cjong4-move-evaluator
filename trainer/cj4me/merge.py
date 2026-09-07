"""Merge compatible CJ4ME dataset shards without loading them into memory."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import BinaryIO, Sequence

from .dataset import (
    DATASET_FORMAT_VERSION,
    DATASET_HEADER_SIZE,
    DATASET_MAGIC,
    DATASET_RECORD_SIZE,
    FEATURE_COUNT,
    FEATURE_SCHEMA_VERSION,
    _HEADER,
    _read_header,
)

COPY_BUFFER_SIZE = 8 * 1024 * 1024
MAX_RECORD_COUNT = (1 << 32) - 1


def _copy_records(source: BinaryIO, destination: BinaryIO, byte_count: int) -> None:
    remaining = byte_count
    while remaining:
        chunk = source.read(min(remaining, COPY_BUFFER_SIZE))
        if not chunk:
            raise ValueError("dataset changed or was truncated while being merged")
        destination.write(chunk)
        remaining -= len(chunk)
    if source.read(1):
        raise ValueError("dataset changed or gained trailing bytes while being merged")


def merge_datasets(
    output: str | os.PathLike[str],
    inputs: Sequence[str | os.PathLike[str]],
    *,
    overwrite: bool = False,
) -> int:
    """Merge dataset shards in the supplied order and return the record count."""
    output_path = Path(output)
    input_paths = [Path(path) for path in inputs]
    if not input_paths:
        raise ValueError("at least one input dataset is required")

    output_resolved = output_path.resolve()
    input_resolved = [path.resolve() for path in input_paths]
    if output_resolved in input_resolved:
        raise ValueError("output dataset must not also be an input dataset")
    if len(set(input_resolved)) != len(input_resolved):
        raise ValueError("input datasets must not contain duplicates")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output dataset already exists: {output_path}")

    headers = [_read_header(path) for path in input_paths]
    record_count = sum(header.record_count for header in headers)
    if record_count > MAX_RECORD_COUNT:
        raise ValueError("merged dataset record count exceeds the format limit")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as destination:
            temporary_path = Path(destination.name)
            destination.write(
                _HEADER.pack(
                    DATASET_MAGIC,
                    DATASET_FORMAT_VERSION,
                    FEATURE_SCHEMA_VERSION,
                    FEATURE_COUNT,
                    record_count,
                    DATASET_RECORD_SIZE,
                    0,
                )
            )
            for path, header in zip(input_paths, headers, strict=True):
                with path.open("rb") as source:
                    source.seek(DATASET_HEADER_SIZE)
                    _copy_records(
                        source,
                        destination,
                        header.record_count * header.record_size,
                    )
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return record_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="input dataset shards in merge order")
    parser.add_argument("--output", required=True, help="merged output dataset")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output dataset",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    record_count = merge_datasets(args.output, args.inputs, overwrite=args.overwrite)
    print(
        f"merged {len(args.inputs)} dataset(s) with {record_count} record(s) "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
