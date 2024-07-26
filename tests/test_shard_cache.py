import asyncio
import tempfile
from typing import Iterator, List, Sequence

import pyarrow as pa
import pytest
import ray

from levanter.data._preprocessor import BatchProcessor
from levanter.data.shard_cache import ChunkMetadata, SerialCacheWriter, _get_broker_actor, build_or_load_cache
from levanter.data.sharded_dataset import ShardedDataset, TextUrlDataset
from levanter.utils.py_utils import logical_cpu_core_count
from test_utils import skip_in_ci


def setup_module(module):
    ray.init("local", num_cpus=max(2 * logical_cpu_core_count(), 8))  # 2x cpu count is faster on my m1


def teardown_module(module):
    ray.shutdown()


# tests to write:
# - test idempotency of writes


class TestProcessor(BatchProcessor[Sequence[int]]):
    def __init__(self, batch_size: int = 8):
        self._batch_size = batch_size

    def __call__(self, batch: Sequence[Sequence[int]]) -> pa.RecordBatch:
        return pa.RecordBatch.from_arrays([pa.array(batch)], ["test"])

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


def simple_process(processor, source):
    result = []
    for shard_name in source.shard_names:
        for batch in source.open_shard(shard_name):
            result.append(processor([batch]))

    return result


@pytest.mark.ray
@pytest.mark.parametrize("shards_to_read_at_once", [1, 2, 4])
def test_cache_simple(shards_to_read_at_once):
    td = tempfile.TemporaryDirectory()
    with td as tmpdir:
        ray_ds = build_or_load_cache(
            tmpdir,
            SimpleShardSource(),
            TestProcessor(),
            await_finished=True,
            # shards_to_read_at_once=shards_to_read_at_once,
        )

        simple_processed = simple_process(TestProcessor(), SimpleShardSource())

        assert list(ray_ds) == list(simple_processed)


@pytest.mark.ray
def test_cache_remembers_its_cached():
    directory = tempfile.TemporaryDirectory()
    with directory as tmpdir:
        ds1 = build_or_load_cache(tmpdir, SimpleShardSource(), TestProcessor())

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
        ds2 = build_or_load_cache(tmpdir, SimpleShardSource(), ThrowingProcessor(), await_finished=True)

        assert list(ds1) == list(ds2)
        # ensure we delete tmpdir, since something is holding onto it


class _CustomException(Exception):
    pass


@pytest.mark.ray
@skip_in_ci
def test_cache_recover_from_crash():
    class CrashingShardSource(ShardedDataset[List[int]]):
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
            build_or_load_cache(tmpdir, source, TestProcessor())

        # kill the broker actor so that we can test recovery
        ray.kill(_get_broker_actor(tmpdir, source, TestProcessor()), no_restart=True)

        source = CrashingShardSource(5)
        with pytest.raises(_CustomException):
            build_or_load_cache(tmpdir, source, TestProcessor())

        ray.kill(_get_broker_actor(tmpdir, source, TestProcessor()), no_restart=True)

        # testing this doesn't throw
        source = CrashingShardSource(1000)
        reader1 = build_or_load_cache(tmpdir, source, TestProcessor(), batch_size=1, await_finished=True)

        # compare to the original with no crash
        reader2 = build_or_load_cache(tmpdir2, SimpleShardSource(), TestProcessor(), batch_size=1, await_finished=True)

        assert list(reader1) == list(reader2)
        assert len(list(reader1)) == 40


@pytest.mark.ray
def test_no_hang_if_empty_shard_source():
    class EmptyShardSource(ShardedDataset[List[int]]):
        @property
        def shard_names(self) -> Sequence[str]:
            return []

        def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[List[int]]:
            raise RuntimeError("This should not be called")

    with tempfile.TemporaryDirectory() as tmpdir:
        reader = build_or_load_cache(tmpdir, EmptyShardSource(), TestProcessor(), batch_size=1)
        assert list(reader) == []


@skip_in_ci
@pytest.mark.ray
def test_chunk_ordering_is_correct_with_slow_shards():
    class SlowShardSource(ShardedDataset[List[int]]):
        @property
        def shard_names(self) -> Sequence[str]:
            return ["shard_0", "shard_1"]

        def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[List[int]]:
            max_count = 40 if shard_name == "shard_1" else 20
            for i in range(0, max_count):
                yield [i] * 10

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = build_or_load_cache(
            tmpdir,
            SlowShardSource(),
            TestProcessor(1),
            batch_size=1,
            rows_per_chunk=10,
            await_finished=False,
        )

        # now block until the cache is done
        cache.await_finished(timeout=10)

        # now check that the chunks are in the right order
        # TODO: this is a bit gross
        chunks: List[ChunkMetadata] = ray.get([cache._broker.get_chunk.remote(i) for i in range(6)])
        assert chunks[0].name == "shard_0/chunk-0"
        assert chunks[1].name == "shard_1/chunk-0"
        assert chunks[2].name == "shard_0/chunk-1"
        assert chunks[3].name == "shard_1/chunk-1"
        assert chunks[4].name == "shard_1/chunk-2"
        assert chunks[5].name == "shard_1/chunk-3"

        # make sure there's not a 7th chunk
        chunk = ray.get(cache._broker.get_chunk.remote(6), timeout=0.5)
        assert chunk is None


@skip_in_ci
@pytest.mark.ray
def test_can_get_chunk_before_finished():
    @ray.remote(num_cpus=0)
    class Blocker:
        def __init__(self):
            self.future = asyncio.Future()

        async def block(self):
            await self.future

        def unblock(self):
            self.future.set_result(None)

    blocker_to_wait_on_test = Blocker.remote()

    class SlowShardSource(ShardedDataset[List[int]]):
        @property
        def shard_names(self) -> Sequence[str]:
            return ["shard_0"]

        def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[List[int]]:
            for i in range(10):
                yield [i] * 10
            ray.get(blocker_to_wait_on_test.block.remote())
            for i in range(10, 20):
                yield [i] * 10

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = build_or_load_cache(
            tmpdir, SlowShardSource(), TestProcessor(5), batch_size=1, rows_per_chunk=10, await_finished=False
        )

        def back_to_py(batch: pa.RecordBatch):
            return list(batch["test"].values.to_numpy())

        chunk = [back_to_py(batch) for batch in cache.read_chunk(0)]

        assert [list(x) for x in chunk] == [[i] * 10 for i in range(10)]

        with pytest.raises(TimeoutError):
            cache.get_chunk(1, timeout=0.1)

        ray.get(blocker_to_wait_on_test.unblock.remote())

        chunk = [back_to_py(batch) for batch in cache.read_chunk(1)]

        assert [list(x) for x in chunk] == [[i] * 10 for i in range(10, 20)]

        ray.get(blocker_to_wait_on_test.block.remote())

        # now wait until the cache is finished. mostly so that the tempdir cleanup works
        cache.await_finished(timeout=10)


@skip_in_ci
@pytest.mark.ray
def test_shard_cache_crashes_if_processor_throws():
    class ThrowingProcessor(BatchProcessor[Sequence[int]]):
        def __call__(self, batch: Sequence[Sequence[int]]) -> pa.RecordBatch:
            raise RuntimeError("exc")

        @property
        def batch_size(self) -> int:
            return 8

        @property
        def num_cpus(self) -> int:
            return 1

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(RuntimeError):
            build_or_load_cache(tmpdir, SimpleShardSource(), ThrowingProcessor(), await_finished=True)


@skip_in_ci
@pytest.mark.ray
def test_map_batches_and_map_shard_cache():
    td = tempfile.TemporaryDirectory()
    with td as tmpdir:
        ray_ds = (
            SimpleShardSource()
            .map(lambda list: list * 2)
            .map_batches(TestProcessor(), 8)
            .map(lambda d: {"q": d["test"]})
            .build_or_load_cache(tmpdir, await_finished=True)
        )

        def composite_fn(list):
            assert len(list) == 1
            return {"q": list[0] * 2}

        simple_processed = simple_process(composite_fn, SimpleShardSource())

        # we internally change all the int lists in the ray_ds to np arrays, so we need to convert them back to lists
        ray_entries = []
        for entry in ray_ds:
            assert entry.keys() == {"q"}
            ray_entries.append({"q": entry["q"].tolist()})

        assert ray_entries == list(simple_processed)


@pytest.mark.ray
def test_serial_cache_writer():
    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        source = SimpleShardSource(num_shards=4)
        processor = TestProcessor()

        with SerialCacheWriter(tmpdir1, rows_per_chunk=8) as writer:
            for shard_name in source.shard_names:
                for batch in source.open_shard(shard_name):
                    writer.write_batch(processor([batch]))

        serial = writer.result(batch_size=1)
        ray_ds = build_or_load_cache(tmpdir2, source, processor, await_finished=True)

        def freeze_batch(batch):
            # make it hashable
            return tuple(batch["test"].values.to_numpy())

        assert set(freeze_batch(batch) for batch in serial) == set(freeze_batch(batch) for batch in ray_ds)


@skip_in_ci
@pytest.mark.ray
def test_shard_cache_fails_with_multiple_shards_with_the_same_name():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(f"{tmpdir}/data.txt", "w") as f:
            f.write("")

        with pytest.raises(ValueError):
            TextUrlDataset(
                [f"{tmpdir}/data.txt", f"{tmpdir}/data.txt"],
            )

        with open(f"{tmpdir}/data.txt.1", "w") as f:
            f.write("")

            dataset = TextUrlDataset(
                [f"{tmpdir}/data.txt", f"{tmpdir}/data.txt.1"],
            )

            build_or_load_cache(tmpdir, dataset, TestProcessor(), await_finished=True)


@skip_in_ci
@pytest.mark.ray
def test_shard_cache_fails_gracefully_with_unknown_file_type():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(f"{tmpdir}/data.not_a_real_extension", "w") as f:
            f.write("")

        dataset = TextUrlDataset(
            [f"{tmpdir}/data.not_a_real_extension"],
        )

        with pytest.raises(ValueError):
            build_or_load_cache(tmpdir, dataset, TestProcessor(), await_finished=True)

        # now make sure it works in non-blocking mode

        cache = build_or_load_cache(tmpdir, dataset, TestProcessor(), await_finished=False)

        with pytest.raises(ValueError):
            cache.get_chunk(0, timeout=5)

        with pytest.raises(ValueError):
            cache.await_finished(timeout=10)
