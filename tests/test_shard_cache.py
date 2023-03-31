import tempfile
from typing import Iterator, List, Sequence

import pyarrow as pa

from levanter.data.shard_cache import BatchProcessor, ShardedDataSource, cache_dataset


# tests to write:
# - test that we can recover from a crash in one writer
# - test multiple shards, one writer
# - test multiple shards, multiple writers
# - test idempotency of writes


class TestProcessor(BatchProcessor[List[int]]):
    def __call__(self, batch: List[List[int]]) -> pa.RecordBatch:
        return pa.RecordBatch.from_arrays([pa.array(batch)], ["test"])

    @property
    def batch_size(self) -> int:
        return 8

    @property
    def num_cpus(self) -> int:
        return 1


class TestShardSource(ShardedDataSource[List[int]]):
    @property
    def shard_names(self) -> Sequence[str]:
        return [f"shard_{i}" for i in range(4)]

    def open_shard(self, shard_name: str) -> Iterator[List[int]]:
        # parse the shard name to get the shard number
        shard_num = int(shard_name.split("_")[1])
        return ([shard_num * 10 + i] * 10 for i in range(10))


def test_cache_remembers_its_cached():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dataset(tmpdir, TestProcessor(), TestShardSource())

        class ThrowingProcessor(BatchProcessor[List[int]]):
            def __call__(self, batch: List[List[int]]) -> pa.RecordBatch:
                raise RuntimeError("This should not be called")

            @property
            def batch_size(self) -> int:
                return 8

            @property
            def num_cpus(self) -> int:
                return 1

        # testing this doesn't throw
        cache_dataset(tmpdir, ThrowingProcessor(), TestShardSource())
