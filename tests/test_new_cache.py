import tempfile
from typing import Iterator, Sequence

import numpy as np

from levanter.data import BatchProcessor, ShardedDataset, batched
from levanter.newstore.cache import SerialCacheWriter
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


class SimpleShardSource(ShardedDataset[list[int]]):
    def __init__(self, num_shards: int = 4):
        self._num_shards = num_shards

    @property
    def shard_names(self) -> Sequence[str]:
        return [f"shard_{i}" for i in range(self._num_shards)]

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[list[int]]:
        # parse the shard name to get the shard number
        shard_num = int(shard_name.split("_")[1])
        return ([shard_num * 10 + i] * 10 for i in range(row, 10))


def test_serial_cache_writer():
    with tempfile.TemporaryDirectory() as tmpdir1:
        source = SimpleShardSource(num_shards=4)
        processor = SimpleProcessor()

        with SerialCacheWriter(tmpdir1) as writer:
            for shard_name in source.shard_names:
                for ex in batched(source.open_shard(shard_name), processor.batch_size):
                    writer.write_batch(processor(ex))

        _ = writer.result(batch_size=1)
        data_path = writer._tree_store.path

        exemplar = {"data": np.array([0], dtype=np.int64)}

        builder = TreeStoreBuilder.open(exemplar, data_path, mode="r")

        assert len(builder) == 40

        for i, x in enumerate(builder):
            np.testing.assert_array_equal(x["data"], np.asarray([i % 10 + i // 10 * 10] * 10))


def crappy_du(path):
    import os

    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total
