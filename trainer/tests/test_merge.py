import numpy as np
import pytest

from cj4me.dataset import DatasetV3, FEATURE_COUNT
from cj4me.merge import merge_datasets
from test_dataset import write_dataset


def record(value):
    features = np.full(FEATURE_COUNT, value, dtype=np.float32)
    return (features, value / 10, int(value) % 4, 0, 0)


def test_merges_shards_in_input_order(tmp_path):
    first = tmp_path / "first.cj4medata"
    second = tmp_path / "second.cj4medata"
    empty = tmp_path / "empty.cj4medata"
    output = tmp_path / "merged.cj4medata"
    write_dataset(first, [record(1), record(2)])
    write_dataset(second, [record(3)])
    write_dataset(empty)

    count = merge_datasets(output, [first, empty, second])
    dataset = DatasetV3(output)

    assert count == 3
    assert len(dataset) == 3
    np.testing.assert_array_equal(dataset.features[:, 0], [1, 2, 3])
    np.testing.assert_allclose(dataset.targets, [0.1, 0.2, 0.3])


def test_rejects_empty_duplicate_and_self_inputs(tmp_path):
    shard = tmp_path / "shard.cj4medata"
    output = tmp_path / "output.cj4medata"
    write_dataset(shard, [record(1)])

    with pytest.raises(ValueError, match="at least one"):
        merge_datasets(output, [])
    with pytest.raises(ValueError, match="duplicates"):
        merge_datasets(output, [shard, shard])
    with pytest.raises(ValueError, match="must not also"):
        merge_datasets(shard, [shard])


def test_preserves_existing_output_without_overwrite(tmp_path):
    shard = tmp_path / "shard.cj4medata"
    output = tmp_path / "output.cj4medata"
    write_dataset(shard, [record(1)])
    output.write_bytes(b"keep me")

    with pytest.raises(FileExistsError, match="already exists"):
        merge_datasets(output, [shard])

    assert output.read_bytes() == b"keep me"

    assert merge_datasets(output, [shard], overwrite=True) == 1
    assert len(DatasetV3(output)) == 1


def test_rejects_invalid_shard_before_creating_output(tmp_path):
    invalid = tmp_path / "invalid.cj4medata"
    output = tmp_path / "output.cj4medata"
    invalid.write_bytes(b"invalid")

    with pytest.raises(ValueError, match="truncated"):
        merge_datasets(output, [invalid])

    assert not output.exists()
