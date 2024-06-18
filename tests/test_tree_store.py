import tempfile
from typing import Iterator, List, Sequence

import numpy as np
import pytest

from levanter.data import BatchProcessor, ShardedDataset
from levanter.data.utils import batched
from levanter.newstore.jagged_array import JaggedArray
from levanter.newstore.tree_store import TreeStoreBuilder


class SimpleProcessor(BatchProcessor[Sequence[int]]):
    def __init__(self, batch_size: int = 8):
        self._batch_size = batch_size

    def __call__(self, batch: Sequence[Sequence[int]]) -> Sequence[dict[str, Sequence[int]]]:
        return [{"data": x} for x in batch]

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def num_cpus(self) -> int:
        return 1


class SimpleShardSource(ShardedDataset[List[int]]):
    def __init__(self, num_shards: int = 4):
        self._num_shards = num_shards

    @property
    def shard_names(self) -> Sequence[str]:
        return [f"shard_{i}" for i in range(self._num_shards)]

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[List[int]]:
        # parse the shard name to get the shard number
        shard_num = int(shard_name.split("_")[1])
        return ([shard_num * 10 + i] * 10 for i in range(row, 10))


def test_tree_builder_with_processor():
    with tempfile.TemporaryDirectory() as tempdir:
        exemplar = {"data": np.array([0], dtype=np.int32)}

        builder = TreeStoreBuilder.open(exemplar, tempdir, mode="w")
        processor = SimpleProcessor()
        source = SimpleShardSource()

        for batch in batched(source, processor.batch_size):
            processed = processor(batch)
            builder.extend(processed)

        assert len(builder) == 40

        for i, x in enumerate(builder):
            assert len(x) == 1

            np.testing.assert_array_equal(x["data"], np.asarray([i % 10 + i // 10 * 10] * 10))

        assert i == 39

        # now test random access
        for i in range(40):
            x = builder[i]
            assert len(x) == 1
            np.testing.assert_array_equal(x["data"], np.asarray([i % 10 + i // 10 * 10] * 10))

        # double check columnar access
        assert len(builder.tree["data"].data) == 10 * 40


def test_append_batch():
    with tempfile.TemporaryDirectory() as tmpdir:
        exemplar = {"a": np.array([0], dtype=np.float32), "b": np.array([0], dtype=np.float32)}
        builder = TreeStoreBuilder.open(exemplar, tmpdir)

        batch1 = [
            {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])},
            {"a": np.array([5.0, 6.0]), "b": np.array([7.0, 8.0])},
        ]
        builder.extend(batch1)

        assert len(builder) == 2

        result1 = builder[0]
        assert np.all(result1["a"] == np.array([1.0, 2.0]))
        assert np.all(result1["b"] == np.array([3.0, 4.0]))

        result2 = builder[1]
        assert np.all(result2["a"] == np.array([5.0, 6.0]))
        assert np.all(result2["b"] == np.array([7.0, 8.0]))


def test_append_batch_different_shapes():
    with tempfile.TemporaryDirectory() as tmpdir:
        exemplar = {"a": np.array([0], dtype=np.float32), "b": np.array([0], dtype=np.float32)}
        builder = TreeStoreBuilder.open(exemplar, tmpdir)

        batch1 = [
            {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])},
            {"a": np.array([5.0, 6.0]), "b": np.array([7.0, 8.0])},
        ]
        builder.extend(batch1)

        batch2 = [
            {"a": np.array([9.0]), "b": np.array([10.0])},
            {"a": np.array([11.0, 12.0, 13.0]), "b": np.array([14.0, 15.0, 16.0])},
        ]
        builder.extend(batch2)

        assert len(builder) == 4

        result3 = builder[2]
        assert np.all(result3["a"] == np.array([9.0]))
        assert np.all(result3["b"] == np.array([10.0]))

        result4 = builder[3]
        assert np.all(result4["a"] == np.array([11.0, 12.0, 13.0]))
        assert np.all(result4["b"] == np.array([14.0, 15.0, 16.0]))


def test_len():
    with tempfile.TemporaryDirectory() as tmpdir:
        exemplar = {"a": np.array([0], dtype=np.float32), "b": np.array([0], dtype=np.float32)}
        builder = TreeStoreBuilder.open(exemplar, tmpdir)

        assert len(builder) == 0

        batch = [
            {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])},
            {"a": np.array([5.0, 6.0]), "b": np.array([7.0, 8.0])},
        ]
        builder.extend(batch)

        assert len(builder) == 2


def test_getitem():
    with tempfile.TemporaryDirectory() as tmpdir:
        exemplar = {"a": np.array([0], dtype=np.float32), "b": np.array([0], dtype=np.float32)}
        builder = TreeStoreBuilder.open(exemplar, tmpdir)

        batch = [
            {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])},
            {"a": np.array([5.0, 6.0]), "b": np.array([7.0, 8.0])},
        ]
        builder.extend(batch)

        result = builder[0]
        assert np.all(result["a"] == np.array([1.0, 2.0]))
        assert np.all(result["b"] == np.array([3.0, 4.0]))

        result = builder[1]
        assert np.all(result["a"] == np.array([5.0, 6.0]))
        assert np.all(result["b"] == np.array([7.0, 8.0]))

        # test slice
        result = builder[0:2]
        assert isinstance(result["a"], JaggedArray)
        assert isinstance(result["b"], JaggedArray)


def test_getitem_out_of_bounds():
    with tempfile.TemporaryDirectory() as tmpdir:
        exemplar = {"a": np.array([0], dtype=np.float32), "b": np.array([0], dtype=np.float32)}
        builder = TreeStoreBuilder.open(exemplar, tmpdir)

        batch = [
            {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])},
            {"a": np.array([5.0, 6.0]), "b": np.array([7.0, 8.0])},
        ]
        builder.extend(batch)

        with pytest.raises(IndexError):
            builder[2]


def test_iter():
    with tempfile.TemporaryDirectory() as tmpdir:
        exemplar = {"a": np.array([0], dtype=np.float32), "b": np.array([0], dtype=np.float32)}
        builder = TreeStoreBuilder.open(exemplar, tmpdir)

        batch = [
            {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])},
            {"a": np.array([5.0, 6.0]), "b": np.array([7.0, 8.0])},
        ]
        builder.extend(batch)

        for i, result in enumerate(builder):
            if i == 0:
                assert np.all(result["a"] == np.array([1.0, 2.0]))
                assert np.all(result["b"] == np.array([3.0, 4.0]))
            elif i == 1:
                assert np.all(result["a"] == np.array([5.0, 6.0]))
                assert np.all(result["b"] == np.array([7.0, 8.0]))
            else:
                pytest.fail("Unexpected index")


def test_reading_from_written():
    with tempfile.TemporaryDirectory() as tmpdir:
        exemplar = {"a": np.array([0], dtype=np.float32), "b": np.array([0], dtype=np.float32)}
        builder = TreeStoreBuilder.open(exemplar, tmpdir, mode="w")

        batch = [
            {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])},
            {"a": np.array([5.0, 6.0]), "b": np.array([7.0, 8.0])},
        ]
        builder.extend(batch)

        del builder

        builder2 = TreeStoreBuilder.open(exemplar, tmpdir, mode="r")

        for i, result in enumerate(builder2):
            if i == 0:
                assert np.all(result["a"] == np.array([1.0, 2.0]))
                assert np.all(result["b"] == np.array([3.0, 4.0]))
            elif i == 1:
                assert np.all(result["a"] == np.array([5.0, 6.0]))
                assert np.all(result["b"] == np.array([7.0, 8.0]))
            else:
                pytest.fail("Unexpected index")
