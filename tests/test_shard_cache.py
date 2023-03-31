import tempfile
from typing import Iterator, List, Sequence

import pyarrow as pa
import pytest
import ray

from levanter.data.shard_cache import BatchProcessor, ShardedDataSource, cache_dataset


def setup_module(module):
    ray.init(num_cpus=4)


def teardown_module(module):
    ray.shutdown()


# tests to write:
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
    def __init__(self, num_shards: int = 4):
        self.num_shards = num_shards

    @property
    def shard_names(self) -> Sequence[str]:
        return [f"shard_{i}" for i in range(self.num_shards)]

    def open_shard(self, shard_name: str) -> Iterator[List[int]]:
        # parse the shard name to get the shard number
        shard_num = int(shard_name.split("_")[1])
        return ([shard_num * 10 + i] * 10 for i in range(10))


def simple_process(processor, source):
    result = []
    for shard_name in source.shard_names:
        for batch in source.open_shard(shard_name):
            result.append(processor([batch]))


@pytest.mark.parametrize("num_workers", [1, 2, 4])
def test_cache_simple(num_workers):
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dataset(tmpdir, TestProcessor(), TestShardSource(), num_workers=num_workers)

        # simple_processed = simple_process(TestProcessor(), TestShardSource())
        # TODO: ray cache dataset


@pytest.mark.parametrize("num_workers", [1, 2, 4])
def test_cache_remembers_its_cached(num_workers):
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dataset(tmpdir, TestProcessor(), TestShardSource(), num_workers=num_workers)

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


@pytest.mark.parametrize("num_workers", [1, 2, 4])
def test_cache_recover_from_crash(num_workers):
    class CrashingShardSource(ShardedDataSource[List[int]]):
        def __init__(self, crash_point: int):
            self.crash_point = crash_point

        @property
        def shard_names(self) -> Sequence[str]:
            return [f"shard_{i}" for i in range(4)]

        def open_shard(self, shard_name: str) -> Iterator[List[int]]:
            # parse the shard name to get the shard number
            shard_num = int(shard_name.split("_")[1])
            for i in range(10):
                if shard_num * 10 + i == self.crash_point:
                    raise RuntimeError("This is a crash")
                yield [shard_num * 10 + i] * 10

    with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as tmpdir2:
        source = CrashingShardSource(4)
        with pytest.raises(RuntimeError):
            cache_dataset(tmpdir, TestProcessor(), source, num_workers=num_workers)

        source = CrashingShardSource(5)
        with pytest.raises(RuntimeError):
            cache_dataset(tmpdir, TestProcessor(), source, num_workers=num_workers)

        # testing this doesn't throw
        source = CrashingShardSource(1000)
        cache_dataset(tmpdir, TestProcessor(), source, num_workers=num_workers)

        # compare to the original with no crash
        cache_dataset(tmpdir2, TestProcessor(), TestShardSource(), num_workers=num_workers)

        # todo: actual comparison
