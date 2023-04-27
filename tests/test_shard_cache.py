import os
import tempfile
from typing import Iterator, List, Sequence

import pyarrow as pa
import pytest
import ray

from levanter.data.shard_cache import BatchProcessor, ShardedDataSource, _get_broker_actor, cache_dataset


def setup_module(module):
    ray.init("local", num_cpus=min(20, 2 * os.cpu_count()))  # 2x cpu count is faster on my m1


def teardown_module(module):
    ray.shutdown()


# tests to write:
# - test idempotency of writes


class TestProcessor(BatchProcessor[Sequence[int]]):
    def __call__(self, batch: Sequence[Sequence[int]]) -> pa.RecordBatch:
        return pa.RecordBatch.from_arrays([pa.array(batch)], ["test"])

    @property
    def batch_size(self) -> int:
        return 8

    @property
    def num_cpus(self) -> int:
        return 1


class SimpleShardSource(ShardedDataSource[List[int]]):
    def __init__(self, num_shards: int = 4):
        self._num_shards = num_shards

    @property
    def shard_names(self) -> Sequence[str]:
        return [f"shard_{i}" for i in range(self._num_shards)]

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[List[int]]:
        # parse the shard name to get the shard number
        shard_num = int(shard_name.split("_")[1])
        return ([shard_num * 10 + i] * 10 for i in range(row, 10))


def simple_process(processor, source):
    result = []
    for shard_name in source.shard_names:
        for batch in source.open_shard(shard_name):
            result.append(processor([batch]))

    return result


def test_cache_simple():
    td = tempfile.TemporaryDirectory()
    with td as tmpdir:
        ray_ds = cache_dataset(tmpdir, SimpleShardSource(), TestProcessor())

        simple_processed = simple_process(TestProcessor(), SimpleShardSource())

        assert list(ray_ds) == list(simple_processed)


def test_cache_remembers_its_cached():
    directory = tempfile.TemporaryDirectory()
    with directory as tmpdir:
        ds1 = cache_dataset(tmpdir, SimpleShardSource(), TestProcessor())

        class ThrowingProcessor(BatchProcessor[Sequence[int]]):
            def __call__(self, batch: Sequence[Sequence[int]]) -> pa.RecordBatch:
                raise RuntimeError("This should not be called")

            @property
            def batch_size(self) -> int:
                return 8

            @property
            def num_cpus(self) -> int:
                return 1

        # testing this doesn't throw
        ds2 = cache_dataset(tmpdir, SimpleShardSource(), ThrowingProcessor())

        assert list(ds1) == list(ds2)
        # ensure we delete tmpdir, since something is holding onto it


class _CustomException(Exception):
    pass


def test_cache_recover_from_crash():
    class CrashingShardSource(ShardedDataSource[List[int]]):
        def __init__(self, crash_point: int):
            self.crash_point = crash_point

        @property
        def shard_names(self) -> Sequence[str]:
            return [f"shard_{i}" for i in range(4)]

        def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[List[int]]:
            # parse the shard name to get the shard number
            shard_num = int(shard_name.split("_")[1])
            for i in range(10):
                if shard_num * 10 + i == self.crash_point:
                    raise _CustomException(f"Crashing at {shard_num} {i} {self.crash_point}")
                if i >= row:
                    yield [shard_num * 10 + i] * 10

    with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as tmpdir2:
        source = CrashingShardSource(4)
        with pytest.raises(_CustomException):
            cache_dataset(tmpdir, source, TestProcessor())

        # kill the broker actor so that we can test recovery
        ray.kill(_get_broker_actor(tmpdir, source, TestProcessor()), no_restart=True)

        source = CrashingShardSource(5)
        with pytest.raises(_CustomException):
            cache_dataset(tmpdir, source, TestProcessor())

        ray.kill(_get_broker_actor(tmpdir, source, TestProcessor()), no_restart=True)

        # testing this doesn't throw
        source = CrashingShardSource(1000)
        reader1 = cache_dataset(tmpdir, source, TestProcessor(), batch_size=1)

        # compare to the original with no crash
        reader2 = cache_dataset(tmpdir2, SimpleShardSource(), TestProcessor(), batch_size=1)

        assert list(reader1) == list(reader2)
        assert len(list(reader1)) == 40


def test_no_hang_if_empty_shard_source():
    class EmptyShardSource(ShardedDataSource[List[int]]):
        @property
        def shard_names(self) -> Sequence[str]:
            return []

        def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[List[int]]:
            raise RuntimeError("This should not be called")

    with tempfile.TemporaryDirectory() as tmpdir:
        reader = cache_dataset(tmpdir, EmptyShardSource(), TestProcessor(), batch_size=1)
        assert list(reader) == []
