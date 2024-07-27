import asyncio
import logging
import tempfile
from typing import Iterator, Sequence
from unittest.mock import MagicMock

import numpy as np
import pytest
import ray

from levanter.data import BatchProcessor, ShardedDataset, batched
from levanter.newstore.cache import SerialCacheWriter, TreeStoreBuilder, _OrderedCacheWriter
from levanter.utils.py_utils import logical_cpu_core_count
from levanter.utils.ray_utils import ExceptionInfo, SnitchRecipient, ser_exc_info


def setup_module(module):
    ray.init("local", num_cpus=max(2 * logical_cpu_core_count(), 8))  # 2x cpu count is faster on my m1


def teardown_module(module):
    ray.shutdown()


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

        exemplar = {"data": np.array([0], dtype=np.int64)}

        with SerialCacheWriter(tmpdir1, exemplar) as writer:
            for shard_name in source.shard_names:
                for ex in batched(source.open_shard(shard_name), processor.batch_size):
                    writer.write_batch(processor(ex))

        _ = writer.result()
        data_path = writer._tree_store.path

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


@ray.remote
class PretendParent(SnitchRecipient):
    def __init__(self):
        self.logger = logging.getLogger("SnitchRecipient")
        self.failure_received = asyncio.Event()
        self.exception_info = None
        self._finished_shards = set()
        self._finished = False

    def _child_failed(self, child: ray.actor.ActorHandle, exception: ExceptionInfo):
        self.logger.error(f"Child {child} failed with exception {exception}")
        self.exception_info = exception
        self.failure_received.set()

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

    def _set_finished(self):
        self._finished = True

    def is_finished(self):
        return self._finished


@pytest.mark.asyncio
async def test_batch_finished():
    parent = PretendParent.remote()
    exemplar = np.array([1, 2, 3])
    with tempfile.TemporaryDirectory() as cache_dir:
        shards = ["shard1", "shard2", "shard3"]
        writer = _OrderedCacheWriter.remote(parent, exemplar, cache_dir, shards)

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
        writer = _OrderedCacheWriter.remote(parent, exemplar, cache_dir, shards)

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
        writer = _OrderedCacheWriter.remote(parent, exemplar, cache_dir, shards)

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
        writer = _OrderedCacheWriter.remote(parent, exemplar, cache_dir, shards)

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


@pytest.mark.asyncio
async def test_attempt_to_write_batches():
    parent = PretendParent.remote()
    exemplar = np.array([1, 2, 3])
    with tempfile.TemporaryDirectory() as cache_dir:
        shards = ["shard1", "shard2", "shard3"]
        writer = _OrderedCacheWriter.remote(parent, exemplar, cache_dir, shards)

        try:
            shard1_batch = [np.asarray([1, 2, 3])]
            shard2_batch = [np.asarray([4, 5, 6, 7])]

            await writer.batch_finished.remote("shard1", 0, shard1_batch)
            await writer.batch_finished.remote("shard2", 0, shard2_batch)

            ledger = await writer.get_ledger.remote()
            assert ledger.is_finished is False
            assert ledger.total_num_rows == 2  # Assuming each batch has 1 row for simplicity

            store = TreeStoreBuilder.open(exemplar, cache_dir, mode="r")
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
        writer = _OrderedCacheWriter.remote(parent, exemplar, cache_dir, shards)

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
        writer = _OrderedCacheWriter.remote(parent, exemplar, cache_dir, shards)

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
        writer = _OrderedCacheWriter.remote(parent, exemplar, cache_dir, shards)

        try:
            # Sending batch 1 before batch 0 for shard1
            shard1_batch0 = [np.array([1, 2, 3])]
            shard1_batch1 = [np.array([4, 5, 6])]

            await writer.batch_finished.remote("shard1", 1, shard1_batch1)
            await writer.batch_finished.remote("shard1", 0, shard1_batch0)

            store = TreeStoreBuilder.open(exemplar, cache_dir, mode="r")
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
        writer = _OrderedCacheWriter.remote(parent, exemplar, cache_dir, shards)

        try:
            # Sending batches out of order across different shards
            shard1_batch0 = [np.array([1, 2, 3])]
            shard2_batch0 = [np.array([4, 5, 6])]
            shard1_batch1 = [np.array([7, 8, 9])]

            await writer.batch_finished.remote("shard1", 1, shard1_batch1)
            await writer.batch_finished.remote("shard2", 0, shard2_batch0)
            await writer.batch_finished.remote("shard1", 0, shard1_batch0)

            store = TreeStoreBuilder.open(exemplar, cache_dir, mode="r")
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
        writer = _OrderedCacheWriter.remote(parent, exemplar, cache_dir, shards)

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

            store = TreeStoreBuilder.open(exemplar, cache_dir, mode="r")
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
        writer = _OrderedCacheWriter.remote(parent, exemplar, cache_dir, shards)

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

            store = TreeStoreBuilder.open(exemplar, cache_dir, mode="r")
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
        shards = ["shard1", "shard2", "shard3"]
        writer = _OrderedCacheWriter.remote(parent, exemplar, cache_dir, shards)

        try:
            # Sending duplicate batches for the same shard
            shard1_batch0 = [np.array([1, 2, 3])]
            shard1_batch1 = [np.array([4, 5, 6])]

            await writer.batch_finished.remote("shard1", 0, shard1_batch0)
            with pytest.raises(ValueError):
                await writer.batch_finished.remote("shard1", 0, shard1_batch0)  # Duplicate
            await writer.batch_finished.remote("shard1", 1, shard1_batch1)

            store = TreeStoreBuilder.open(exemplar, cache_dir, mode="r")
            assert len(store) == 2
            np.testing.assert_array_equal(store[0], shard1_batch0[0])
            np.testing.assert_array_equal(store[1], shard1_batch1[0])
        finally:
            ray.kill(parent)
            ray.kill(writer)


@pytest.mark.asyncio
async def test_mixed_order_batches_multiple_shards():
    parent = PretendParent.remote()
    exemplar = np.array([1, 2, 3])
    with tempfile.TemporaryDirectory() as cache_dir:
        shards = ["shard1", "shard2", "shard3"]
        writer = _OrderedCacheWriter.remote(parent, exemplar, cache_dir, shards)

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

            store = TreeStoreBuilder.open(exemplar, cache_dir, mode="r")
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
