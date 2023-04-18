import os
import tempfile
from typing import Iterator, List, Sequence

import pyarrow as pa
import pytest
import ray

from levanter.data.shard_cache_mk3 import BatchProcessor, ShardedDataSource, cache_dataset


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


class SimpleShardSource(ShardedDataSource[List[int]]):
    def __init__(self, num_shards: int = 4):
        self.num_shards = num_shards

    @property
    def shard_names(self) -> Sequence[str]:
        return [f"shard_{i}" for i in range(self.num_shards)]

    def open_at_row(self, shard_name: str, row: int) -> Iterator[List[int]]:
        # parse the shard name to get the shard number
        shard_num = int(shard_name.split("_")[1])
        return ([shard_num * 10 + i] * 10 for i in range(row, 10))


def simple_process(processor, source):
    result = []
    for shard_name in source.shard_names:
        for batch in source.open_shard(shard_name):
            result.append(processor([batch]))

    return result


@pytest.mark.parametrize("num_workers", [2, 4, 1])
def test_cache_simple(num_workers):
    td = tempfile.TemporaryDirectory()
    with td as tmpdir:
        print(tmpdir)
        ray_ds = cache_dataset(tmpdir, TestProcessor(), SimpleShardSource(), num_workers=num_workers)

        simple_processed = simple_process(TestProcessor(), SimpleShardSource())

        assert list(ray_ds) == list(simple_processed)

    assert not os.path.exists(tmpdir)


@pytest.mark.parametrize("num_workers", [1, 2, 4])
def test_cache_remembers_its_cached(num_workers):
    directory = tempfile.TemporaryDirectory()
    with directory as tmpdir:
        ds1 = cache_dataset(tmpdir, TestProcessor(), SimpleShardSource(), num_workers=num_workers)

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
        ds2 = cache_dataset(tmpdir, ThrowingProcessor(), SimpleShardSource())

        assert list(ds1) == list(ds2)
        # ensure we delete tmpdir, since something is holding onto it


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
                    print(shard_num, i, self.crash_point)
                    raise RuntimeError(f"Crashing at {shard_num} {i} {self.crash_point}")
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
        reader1 = cache_dataset(tmpdir, TestProcessor(), source, num_workers=num_workers, batch_size=1)

        # compare to the original with no crash
        reader2 = cache_dataset(tmpdir2, TestProcessor(), SimpleShardSource(), num_workers=num_workers, batch_size=1)

        assert list(reader1) == list(reader2)
        assert len(list(reader1)) == 40
