import asyncio
import logging
import tempfile
from typing import Iterator, Sequence
from unittest.mock import MagicMock

import numpy as np
import pytest
import ray
from ray.exceptions import RayTaskError

from levanter.data import BatchProcessor, ShardedDataSource, batched
from levanter.data.sharded_datasource import TextUrlDataSource
from levanter.store.cache import (
    SerialCacheWriter,
    TreeStore,
    _get_builder_actor,
    _OrderedCacheWriter,
    build_or_load_cache,
)
from levanter.utils.py_utils import logical_cpu_core_count
from levanter.utils.ray_utils import ExceptionInfo, SnitchRecipient, ser_exc_info


class TestProcessor(BatchProcessor[Sequence[int], dict[str, np.ndarray]]):
    def __init__(self, batch_size: int = 8):
        self._batch_size = batch_size

    def __call__(self, batch: Sequence[Sequence[int]]) -> Sequence[dict[str, np.ndarray]]:
        # return pa.RecordBatch.from_arrays([pa.array(batch)], ["test"])
        return [{"test": np.asarray(x)} for x in batch]

    @property
    def output_exemplar(self):
        return {"test": np.array([0], dtype=np.int64)}

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def num_cpus(self) -> int:
        return 1


def simple_process(processor, source):
    result = []
    for shard_name in source.shard_names:
        for batch in source.open_shard(shard_name):
            result.append(processor([batch])[0])

    return result


def process_interleave(processor, source):
    batch_size = processor.batch_size
    shard_iterators = {
        shard_name: batched(iter(source.open_shard(shard_name)), batch_size) for shard_name in source.shard_names
    }
    finished = 0

    while finished < len(shard_iterators):
        for shard_name, shard_iter in shard_iterators.items():
            if shard_iter is None:
                continue
            try:
                batch = next(shard_iter)
                yield from processor(batch)
            except StopIteration:
                shard_iterators[shard_name] = None
                finished += 1


def setup_module(module):
    ray.init(
        "local", num_cpus=max(2 * logical_cpu_core_count(), 8), ignore_reinit_error=True
    )  # 2x cpu count is faster on my m1


def teardown_module(module):
    ray.shutdown()


class SimpleProcessor(BatchProcessor[Sequence[int], dict[str, np.ndarray]]):
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

    @property
    def output_exemplar(self) -> dict[str, np.ndarray]:
        return {"data": np.array([0], dtype=np.int64)}


class SimpleShardSource(ShardedDataSource[list[int]]):
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

        exemplar = {"data": np.array([0], dtype=np.int64)}

        with SerialCacheWriter(tmpdir1, exemplar) as writer:
            for shard_name in source.shard_names:
                for ex in batched(source.open_shard(shard_name), processor.batch_size):
                    writer.write_batch(processor(ex))

        _ = writer.result()
        data_path = writer._tree_store.path

        builder = TreeStore.open(exemplar, data_path, mode="r")

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


@ray.remote
class PretendParent(SnitchRecipient):
    def __init__(self):
        self.logger = logging.getLogger("SnitchRecipient")
        self.failure_received = asyncio.Event()
        self.exception_info = None
        self._finished_shards = set()
        self._finished = False
        self._ledger = None
        self._desired_next_item = None

    def _child_failed(self, child: ray.actor.ActorHandle, exception: ExceptionInfo):
        try:
            self.logger.error(f"Child {child} failed with exception {exception}")
            self.exception_info = exception
            self.failure_received.set()
        except Exception as e:
            self.logger.error(f"Error in _child_failed: {e}")

    def shard_failed(self, shard_name, exc_info):
        self.exception_info = exc_info
        self.failure_received.set()

    async def wait_for_failure(self):
        await self.failure_received.wait()
        return self.exception_info

    def shard_finished(self, shard_name):
        self._finished_shards.add(shard_name)

    def get_finished_shards(self):
        return self._finished_shards

    def _updated_ledger(self, ledger):
        if ledger.is_finished:
            self._finished = True

        self._ledger = ledger

    def _finalize(self):
        self._finished = True

    def is_finished(self):
        return self._finished

    def signal_backpressure(self, desired_next_item: float):
        self._desired_next_item = desired_next_item

    def desired_next_item(self):
        return self._desired_next_item


@pytest.mark.asyncio
async def test_batch_finished():
    parent = PretendParent.remote()
    exemplar = np.array([1, 2, 3])
    with tempfile.TemporaryDirectory() as cache_dir:
        shards = ["shard1", "shard2", "shard3"]
        writer = _OrderedCacheWriter.remote(
            parent, "test", exemplar, DEFAULT_BATCH_SIZE, cache_dir, shards, min_items_to_write=1
        )

        try:
            shard_idx = "shard1"
            shard_batch_idx = 0
            batch_result = [np.array([1, 2, 3])]

            await writer.batch_finished.remote(shard_idx, shard_batch_idx, batch_result)
            shard_status = await writer.get_shard_status.remote("shard1")
            assert shard_status.num_rows_committed == 1
        finally:
            ray.kill(parent)
            ray.kill(writer)


@pytest.mark.asyncio
async def test_shard_finished_reading():
    parent = PretendParent.remote()
    exemplar = MagicMock()
    with tempfile.TemporaryDirectory() as cache_dir:
        shards = ["shard1", "shard2", "shard3"]
        writer = _OrderedCacheWriter.remote(parent, "test", exemplar, DEFAULT_BATCH_SIZE, cache_dir, shards)

        try:
            shard_name = "shard1"
            expected_batches = 5

            await writer.shard_finished_reading.remote(shard_name, expected_batches)
            shard_status = await writer.get_shard_status.remote(shard_name)
            assert shard_status.is_finished is False
        finally:
            ray.kill(parent)
            ray.kill(writer)


@pytest.mark.asyncio
async def test_get_shard_status():
    parent = PretendParent.remote()
    exemplar = np.array([1, 2, 3])
    with tempfile.TemporaryDirectory() as cache_dir:
        shards = ["shard1", "shard2", "shard3"]
        writer = _OrderedCacheWriter.remote(parent, "test", exemplar, DEFAULT_BATCH_SIZE, cache_dir, shards)

        try:
            shard_name = "shard1"
            shard_status = await writer.get_shard_status.remote(shard_name)

            assert shard_status.shard_name == shard_name
            assert shard_status.num_rows_committed == 0
            assert not shard_status.is_finished
        finally:
            ray.kill(parent)
            ray.kill(writer)


@pytest.mark.asyncio
async def test_shard_failed():
    parent = PretendParent.remote()
    exemplar = MagicMock()
    with tempfile.TemporaryDirectory() as cache_dir:
        shards = ["shard1", "shard2", "shard3"]
        writer = _OrderedCacheWriter.remote(parent, "test", exemplar, DEFAULT_BATCH_SIZE, cache_dir, shards)

        try:
            shard_name = "shard1"
            batch_id = 0
            try:
                raise Exception("Test Exception")
            except:  # noqa
                exc_info = ser_exc_info()

            await writer.shard_failed.remote(shard_name, batch_id, exc_info)
            exception_received = await parent.wait_for_failure.remote()
            assert str(exception_received.ex) == str(exc_info.ex)
        finally:
            ray.kill(parent)
            ray.kill(writer)


DEFAULT_BATCH_SIZE = 128


@pytest.mark.asyncio
async def test_attempt_to_write_batches():
    parent = PretendParent.remote()
    exemplar = np.array([1, 2, 3])
    with tempfile.TemporaryDirectory() as cache_dir:
        shards = ["shard1", "shard2", "shard3"]
        writer = _OrderedCacheWriter.remote(
            parent, "test", exemplar, DEFAULT_BATCH_SIZE, cache_dir, shards, min_items_to_write=2
        )

        try:
            shard1_batch = [np.asarray([1, 2, 3])]
            shard2_batch = [np.asarray([4, 5, 6, 7])]

            await writer.batch_finished.remote("shard1", 0, shard1_batch)
            await writer.batch_finished.remote("shard2", 0, shard2_batch)

            ledger = await writer.get_ledger.remote()
            assert ledger.is_finished is False
            assert ledger.total_num_rows == 2  # Assuming each batch has 1 row for simplicity

            store = TreeStore.open(exemplar, cache_dir, mode="r")
            assert len(store) == 2
            np.testing.assert_array_equal(store[0], shard1_batch[0])
            np.testing.assert_array_equal(store[1], shard2_batch[0])
        finally:
            ray.kill(parent)
            ray.kill(writer)


@pytest.mark.asyncio
async def test_finalize_cache():
    parent = PretendParent.remote()
    exemplar = np.array([1, 2, 3])
    with tempfile.TemporaryDirectory() as cache_dir:
        shards = ["shard1", "shard2", "shard3"]
        writer = _OrderedCacheWriter.remote(parent, "test", exemplar, DEFAULT_BATCH_SIZE, cache_dir, shards)

        try:
            shard1_batch = [np.array([1, 2, 3])]
            shard2_batch = [np.array([4, 5, 6, 7])]

            await writer.batch_finished.remote("shard1", 0, shard1_batch)
            await writer.shard_finished_reading.remote("shard1", 1)
            await writer.shard_finished_reading.remote("shard2", 1)
            await writer.batch_finished.remote("shard2", 0, shard2_batch)

            ledger = await writer.get_ledger.remote()
            assert ledger.is_finished is False
            assert ledger.total_num_rows == 2  # Assuming each batch has 1 row for simplicity

            await writer.shard_finished_reading.remote("shard3", 0)
            finished_shards = await parent.get_finished_shards.remote()
            assert len(finished_shards) == 3

            ledger = await writer.get_ledger.remote()
            assert ledger.is_finished is True
            assert ledger.total_num_rows == 2
            assert await parent.is_finished.remote() is True
        finally:
            ray.kill(parent)
            ray.kill(writer)


@pytest.mark.asyncio
async def test_error_handling():
    parent = PretendParent.remote()
    exemplar = np.array([1, 2, 3])
    with tempfile.TemporaryDirectory() as cache_dir:
        shards = ["shard1", "shard2", "shard3"]
        writer = _OrderedCacheWriter.remote(parent, "test", exemplar, DEFAULT_BATCH_SIZE, cache_dir, shards)

        try:
            with pytest.raises(TypeError):
                await writer.batch_finished.remote("shard1", 0, None)

            exception_received = await parent.wait_for_failure.remote()
            assert exception_received is not None
        finally:
            ray.kill(parent)
            ray.kill(writer)


@pytest.mark.asyncio
async def test_out_of_order_batches_same_shard():
    parent = PretendParent.remote()
    exemplar = np.array([1, 2, 3])
    with tempfile.TemporaryDirectory() as cache_dir:
        shards = ["shard1"]
        writer = _OrderedCacheWriter.remote(
            parent, "test", exemplar, DEFAULT_BATCH_SIZE, cache_dir, shards, min_items_to_write=2
        )

        try:
            # Sending batch 1 before batch 0 for shard1
            shard1_batch0 = [np.array([1, 2, 3])]
            shard1_batch1 = [np.array([4, 5, 6])]

            await writer.batch_finished.remote("shard1", 1, shard1_batch1)
            await writer.batch_finished.remote("shard1", 0, shard1_batch0)

            store = TreeStore.open(exemplar, cache_dir, mode="r")
            assert len(store) == 2
            np.testing.assert_array_equal(store[0], shard1_batch0[0])
            np.testing.assert_array_equal(store[1], shard1_batch1[0])
        finally:
            ray.kill(parent)
            ray.kill(writer)


@pytest.mark.asyncio
async def test_out_of_order_batches_different_shards():
    parent = PretendParent.remote()
    exemplar = np.array([1, 2, 3])
    with tempfile.TemporaryDirectory() as cache_dir:
        shards = ["shard1", "shard2"]
        writer = _OrderedCacheWriter.remote(
            parent, "test", exemplar, DEFAULT_BATCH_SIZE, cache_dir, shards, min_items_to_write=3
        )

        try:
            # Sending batches out of order across different shards
            shard1_batch0 = [np.array([1, 2, 3])]
            shard2_batch0 = [np.array([4, 5, 6])]
            shard1_batch1 = [np.array([7, 8, 9])]

            await writer.batch_finished.remote("shard1", 1, shard1_batch1)
            await writer.batch_finished.remote("shard2", 0, shard2_batch0)
            await writer.batch_finished.remote("shard1", 0, shard1_batch0)

            store = TreeStore.open(exemplar, cache_dir, mode="r")
            assert len(store) == 3
            np.testing.assert_array_equal(store[0], shard1_batch0[0])
            np.testing.assert_array_equal(store[1], shard2_batch0[0])
            np.testing.assert_array_equal(store[2], shard1_batch1[0])
        finally:
            ray.kill(parent)
            ray.kill(writer)


@pytest.mark.asyncio
async def test_batches_different_orders_all_shards():
    parent = PretendParent.remote()
    exemplar = np.array([1, 2, 3])
    with tempfile.TemporaryDirectory() as cache_dir:
        shards = ["shard1", "shard2", "shard3"]
        writer = _OrderedCacheWriter.remote(
            parent, "test", exemplar, DEFAULT_BATCH_SIZE, cache_dir, shards, min_items_to_write=2
        )

        try:
            # Sending batches in different orders across all shards
            shard1_batch0 = [np.array([1, 2, 3])]
            shard1_batch1 = [np.array([4, 5, 6])]
            shard2_batch0 = [np.array([7, 8, 9])]
            shard3_batch0 = [np.array([10, 11, 12])]

            await writer.batch_finished.remote("shard2", 0, shard2_batch0)
            await writer.batch_finished.remote("shard3", 0, shard3_batch0)
            await writer.batch_finished.remote("shard1", 1, shard1_batch1)
            await writer.batch_finished.remote("shard1", 0, shard1_batch0)

            store = TreeStore.open(exemplar, cache_dir, mode="r")
            assert len(store) == 4
            np.testing.assert_array_equal(store[0], shard1_batch0[0])
            np.testing.assert_array_equal(store[1], shard2_batch0[0])
            np.testing.assert_array_equal(store[2], shard3_batch0[0])
            np.testing.assert_array_equal(store[3], shard1_batch1[0])
        finally:
            ray.kill(parent)
            ray.kill(writer)


@pytest.mark.asyncio
async def test_intermixed_batches_same_and_different_shards():
    parent = PretendParent.remote()
    exemplar = np.array([1, 2, 3])
    with tempfile.TemporaryDirectory() as cache_dir:
        shards = ["shard1", "shard2", "shard3"]
        writer = _OrderedCacheWriter.remote(
            parent, "test", exemplar, DEFAULT_BATCH_SIZE, cache_dir, shards, min_items_to_write=1
        )

        try:
            # Sending intermixed batches from the same and different shards
            shard1_batch0 = [np.array([1, 2, 3])]
            shard2_batch0 = [np.array([4, 5, 6])]
            shard1_batch1 = [np.array([7, 8, 9])]
            shard3_batch0 = [np.array([10, 11, 12])]
            shard2_batch1 = [np.array([13, 14, 15])]

            await writer.batch_finished.remote("shard2", 0, shard2_batch0)
            await writer.batch_finished.remote("shard3", 0, shard3_batch0)
            await writer.batch_finished.remote("shard1", 1, shard1_batch1)
            await writer.batch_finished.remote("shard2", 1, shard2_batch1)
            await writer.batch_finished.remote("shard1", 0, shard1_batch0)

            store = TreeStore.open(exemplar, cache_dir, mode="r")
            assert len(store) == 5
            np.testing.assert_array_equal(store[0], shard1_batch0[0])
            np.testing.assert_array_equal(store[1], shard2_batch0[0])
            np.testing.assert_array_equal(store[2], shard3_batch0[0])
            np.testing.assert_array_equal(store[3], shard1_batch1[0])
            np.testing.assert_array_equal(store[4], shard2_batch1[0])
        finally:
            ray.kill(parent)
            ray.kill(writer)


@pytest.mark.asyncio
async def test_duplicate_batches_same_shard():
    parent = PretendParent.remote()
    exemplar = np.array([1, 2, 3])
    with tempfile.TemporaryDirectory() as cache_dir:
        shards = ["shard1"]
        writer = _OrderedCacheWriter.remote(parent, "test", exemplar, DEFAULT_BATCH_SIZE, cache_dir, shards)

        try:
            # Sending duplicate batches for the same shard
            shard1_batch0 = [np.array([1, 2, 3])]

            await writer.batch_finished.remote("shard1", 0, shard1_batch0)
            with pytest.raises(RayTaskError):
                await writer.batch_finished.remote("shard1", 0, shard1_batch0)  # Duplicate
        finally:
            ray.kill(parent)
            ray.kill(writer)


@pytest.mark.asyncio
async def test_mixed_order_batches_multiple_shards():
    parent = PretendParent.remote()
    exemplar = np.array([1, 2, 3])
    with tempfile.TemporaryDirectory() as cache_dir:
        shards = ["shard1", "shard2", "shard3"]
        writer = _OrderedCacheWriter.remote(
            parent, "test", exemplar, DEFAULT_BATCH_SIZE, cache_dir, shards, min_items_to_write=1
        )

        try:
            # Sending batches in mixed order for multiple shards
            shard1_batch0 = [np.array([1, 2, 3])]
            shard2_batch0 = [np.array([4, 5, 6])]
            shard1_batch1 = [np.array([7, 8, 9])]
            shard2_batch1 = [np.array([10, 11, 12])]
            shard3_batch0 = [np.array([13, 14, 15])]
            shard3_batch1 = [np.array([16, 17, 18])]

            await writer.batch_finished.remote("shard3", 0, shard3_batch0)
            await writer.batch_finished.remote("shard1", 1, shard1_batch1)
            await writer.batch_finished.remote("shard2", 0, shard2_batch0)
            await writer.batch_finished.remote("shard2", 1, shard2_batch1)
            await writer.batch_finished.remote("shard1", 0, shard1_batch0)
            await writer.batch_finished.remote("shard3", 1, shard3_batch1)

            store = TreeStore.open(exemplar, cache_dir, mode="r")
            assert len(store) == 6
            np.testing.assert_array_equal(store[0], shard1_batch0[0])
            np.testing.assert_array_equal(store[1], shard2_batch0[0])
            np.testing.assert_array_equal(store[2], shard3_batch0[0])
            np.testing.assert_array_equal(store[3], shard1_batch1[0])
            np.testing.assert_array_equal(store[4], shard2_batch1[0])
            np.testing.assert_array_equal(store[5], shard3_batch1[0])
        finally:
            ray.kill(parent)
            ray.kill(writer)


@pytest.mark.ray
def test_full_end_to_end_cache_simple():
    td = tempfile.TemporaryDirectory()
    with td as tmpdir:
        ray_ds = build_or_load_cache(
            tmpdir,
            SimpleShardSource(num_shards=1),
            TestProcessor(),
            await_finished=True,
        )

        simple_processed = simple_process(TestProcessor(), SimpleShardSource())

        all_data = ray_ds[:]

        check_datasets_equal(all_data, simple_processed)


@pytest.mark.ray
def test_cache_remembers_its_cached():
    directory = tempfile.TemporaryDirectory()
    with directory as tmpdir:
        ds1 = build_or_load_cache(tmpdir, SimpleShardSource(), TestProcessor())

        class ThrowingProcessor(BatchProcessor[Sequence[int], dict[str, np.ndarray]]):
            def __call__(self, batch: Sequence[Sequence[int]]):
                raise RuntimeError("This should not be called")

            @property
            def output_exemplar(self) -> dict[str, np.ndarray]:
                return {"test": np.array([0], dtype=np.int64)}

            @property
            def batch_size(self) -> int:
                return 8

            @property
            def num_cpus(self) -> int:
                return 1

        # testing this doesn't throw
        ds2 = build_or_load_cache(tmpdir, SimpleShardSource(), ThrowingProcessor(), await_finished=True)

        check_datasets_equal(ds1, ds2)


def check_datasets_equal(ds1, ds2):
    for r1, r2 in zip(ds1, ds2):
        assert r1.keys() == r2.keys()
        for key in r1.keys():
            np.testing.assert_array_equal(r1[key], r2[key])


class _CustomException(Exception):
    pass


@pytest.mark.ray
@pytest.mark.skip("This test segfaults in CI. I think a ray bug")
def test_cache_recover_from_crash():
    class CrashingShardSource(ShardedDataSource[list[int]]):
        def __init__(self, crash_point: int):
            self.crash_point = crash_point

        @property
        def shard_names(self) -> Sequence[str]:
            return [f"shard_{i}" for i in range(4)]

        def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[list[int]]:
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
        ray.kill(
            _get_builder_actor(tmpdir, source, TestProcessor()),
            no_restart=True,
        )

        source = CrashingShardSource(5)
        with pytest.raises(_CustomException):
            build_or_load_cache(tmpdir, source, TestProcessor())

        ray.kill(
            _get_builder_actor(tmpdir, source, TestProcessor()),
            no_restart=True,
        )

        # testing this doesn't throw
        source = CrashingShardSource(1000)
        reader1 = build_or_load_cache(tmpdir, source, TestProcessor(), await_finished=True)

        # compare to the original with no crash
        reader2 = build_or_load_cache(tmpdir2, SimpleShardSource(), TestProcessor(), await_finished=True)

        assert len(list(reader1)) == 40
        check_datasets_equal(reader1, reader2)


@pytest.mark.ray
def test_no_hang_if_empty_shard_source():
    class EmptyShardSource(ShardedDataSource[list[int]]):
        @property
        def shard_names(self) -> Sequence[str]:
            return []

        def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[list[int]]:
            raise RuntimeError("This should not be called")

    with tempfile.TemporaryDirectory() as tmpdir:
        reader = build_or_load_cache(tmpdir, EmptyShardSource(), TestProcessor())
        assert list(reader) == []


@pytest.mark.ray
def test_chunk_ordering_is_correct_with_slow_shards():
    class SlowShardSource(ShardedDataSource[list[int]]):
        @property
        def shard_names(self) -> Sequence[str]:
            return ["shard_0", "shard_1"]

        def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[list[int]]:
            max_count = 40 if shard_name == "shard_1" else 20
            shard_id = int(shard_name.split("_")[1])
            for i in range(0, max_count):
                yield [i * 10 + shard_id] * 10

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = build_or_load_cache(
            tmpdir,
            SlowShardSource(),
            TestProcessor(1),
            await_finished=False,
        )

        # now block until the cache is done
        cache.await_finished(timeout=10)

        expected = process_interleave(TestProcessor(1), SlowShardSource())

        check_datasets_equal(list(cache[:]), expected)


@pytest.mark.asyncio
@pytest.mark.ray
async def test_can_get_elems_before_finished():
    @ray.remote(num_cpus=0)
    class Blocker:
        def __init__(self):
            self.future = asyncio.Future()

        async def block(self):
            await self.future

        def unblock(self):
            self.future.set_result(None)

    blocker_to_wait_on_test = Blocker.remote()

    class SlowShardSource(ShardedDataSource[list[int]]):
        @property
        def shard_names(self) -> Sequence[str]:
            return ["shard_0"]

        def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[list[int]]:
            for i in range(10):
                yield [i] * 10
            ray.get(blocker_to_wait_on_test.block.remote())
            for i in range(10, 20):
                yield [i] * 10

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = build_or_load_cache(
            tmpdir, SlowShardSource(), TestProcessor(5), await_finished=False, items_per_write=5
        )

        # read the first 10 elements
        # ensure the first 10 elements are [{"test": np.array([i] * 10)} for i in range(10)]
        first_10 = list(await cache.get_batch(range(0, 10)))

        for i, x in enumerate(first_10):
            np.testing.assert_array_equal(x["test"], np.array([i] * 10))

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(cache.get_batch(range(10, 20)), timeout=0.1)

        # then unblock:
        ray.get(blocker_to_wait_on_test.unblock.remote())

        # now ensure we can get the next 10 elements, which will be
        # [{"test": np.array([i] * 10)} for i in range(10, 20)]
        batch = await asyncio.wait_for(cache.get_batch(range(10, 20)), timeout=10)

        for i, x in enumerate(batch):
            np.testing.assert_array_equal(x["test"], np.array([i + 10] * 10))

        ray.get(blocker_to_wait_on_test.block.remote())

        # now wait until the cache is finished. mostly so that the tempdir cleanup works
        cache.await_finished(timeout=10)


@pytest.mark.skip("This test segfaults in CI. I think a ray bug")
@pytest.mark.ray
def test_shard_cache_crashes_if_processor_throws():
    class ThrowingProcessor(BatchProcessor[Sequence[int], dict[str, np.ndarray]]):
        def __call__(self, batch: Sequence[Sequence[int]]):
            raise RuntimeError("exc")

        @property
        def output_exemplar(self) -> dict:
            return {"test": np.array([0], dtype=np.int64)}

        @property
        def batch_size(self) -> int:
            return 8

        @property
        def num_cpus(self) -> int:
            return 1

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(RuntimeError):
            build_or_load_cache(tmpdir, SimpleShardSource(), ThrowingProcessor(), await_finished=True)


@pytest.mark.ray
@pytest.mark.skip("This test segfaults in CI. I think a ray bug")
def test_shard_cache_fails_with_multiple_shards_with_the_same_name():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(f"{tmpdir}/data.txt", "w") as f:
            f.write("")

        with pytest.raises(ValueError):
            TextUrlDataSource(
                [f"{tmpdir}/data.txt", f"{tmpdir}/data.txt"],
            )

        with open(f"{tmpdir}/data.txt.1", "w") as f:
            f.write("")

            dataset = TextUrlDataSource(
                [f"{tmpdir}/data.txt", f"{tmpdir}/data.txt.1"],
            )

            build_or_load_cache(tmpdir, dataset, TestProcessor(), await_finished=True)


@pytest.mark.skip("This test segfaults in CI. I think a ray bug")
@pytest.mark.ray
@pytest.mark.asyncio
async def test_shard_cache_fails_gracefully_with_unknown_file_type_async():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(f"{tmpdir}/data.not_a_real_extension", "w") as f:
            f.write("")

        dataset = TextUrlDataSource(
            [f"{tmpdir}/data.not_a_real_extension"],
        )

        with pytest.raises(ValueError):
            build_or_load_cache(tmpdir, dataset, TestProcessor(), await_finished=True)

        # now make sure it works in non-blocking mode

        cache = build_or_load_cache(tmpdir, dataset, TestProcessor(), await_finished=False)

        with pytest.raises(ValueError):
            await cache.get_batch([0])

        with pytest.raises(ValueError):
            cache.await_finished(timeout=10)

        del cache


@pytest.mark.skip("This test segfaults in CI. I think a ray bug")
@pytest.mark.ray
def test_shard_cache_fails_gracefully_with_unknown_file_type():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(f"{tmpdir}/data.not_a_real_extension", "w") as f:
            f.write("")

        dataset = TextUrlDataSource(
            [f"{tmpdir}/data.not_a_real_extension"],
        )

        with pytest.raises(ValueError):
            build_or_load_cache(tmpdir, dataset, TestProcessor(), await_finished=True)

        # now make sure it works in non-blocking mode

        cache = build_or_load_cache(tmpdir, dataset, TestProcessor(), await_finished=False)

        with pytest.raises(ValueError):
            cache.get_batch_sync([0])

        with pytest.raises(ValueError):
            cache.await_finished(timeout=10)

        del cache


@pytest.mark.ray
@pytest.mark.asyncio
async def test_backpressure_mechanism():
    parent = PretendParent.remote()
    exemplar = np.array([1, 2, 3])
    with tempfile.TemporaryDirectory() as cache_dir:
        shards = ["shard1", "shard2", "shard3"]
        writer = _OrderedCacheWriter.remote(
            parent, "test", exemplar, DEFAULT_BATCH_SIZE, cache_dir, shards, min_items_to_write=1
        )

        # Simulate batches being processed
        shard1_batch = [np.array([1, 2, 3])]
        shard2_batch = [np.array([4, 5, 6])]
        shard3_batch = [np.array([7, 8, 9])]

        # await writer.batch_finished.remote("shard1", 0, shard1_batch)
        await writer.batch_finished.remote("shard2", 0, shard2_batch)
        await writer.batch_finished.remote("shard3", 0, shard3_batch)
        await writer.batch_finished.remote("shard1", 1, shard3_batch)
        await writer.batch_finished.remote("shard1", 2, shard3_batch)
        await writer.batch_finished.remote("shard1", 3, shard3_batch)

        # Check if backpressure is signaled
        is_overwhelmed = await writer.is_overwhelmed.remote()
        assert is_overwhelmed is True

        for i in range(4):
            if (await parent.desired_next_item.remote()) == 0:
                break

            await asyncio.sleep(0.1 * (i + 1) * (i + 1))
        else:
            assert False, "Backpressure wasn't sent"

        await writer.batch_finished.remote("shard1", 0, shard1_batch)

        # Reduce the queue size to relieve backpressure
        # Check if backpressure is relieved
        is_overwhelmed = await writer.is_overwhelmed.remote()
        assert is_overwhelmed is False

        for i in range(4):
            if (await parent.desired_next_item.remote()) is None:
                break

            await asyncio.sleep(0.1 * (i + 1) * (i + 1))
        else:
            assert False, "Backpressure wasn't relieved"
