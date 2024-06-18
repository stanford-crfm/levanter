import tempfile
from typing import Iterator, List, Sequence

import numpy as np

from levanter.data import BatchProcessor, ShardedDataset
from levanter.data.utils import batched
from levanter.newstore.col_tree_store import TreeStoreBuilder


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


def test_tree_builder():
    with tempfile.TemporaryDirectory() as tempdir:
        builder = TreeStoreBuilder.open(tempdir, "w")
        processor = SimpleProcessor()
        source = SimpleShardSource()

        for batch in batched(source, processor.batch_size):
            processed = processor(batch)
            builder.append_batch(processed)

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
