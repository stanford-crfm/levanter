import asyncio
import concurrent
import dataclasses
import logging as pylogging
import os
import threading
import time
from asyncio import InvalidStateError
from concurrent.futures import Future as threading_Future
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, TypeVar, Union

import fsspec.core
import pyarrow as pa
import ray
from dataclasses_json import dataclass_json
from fsspec import AbstractFileSystem
from ray.actor import ActorHandle
from ray.remote_function import RemoteFunction

from levanter.data.dataset import AsyncDataset
from levanter.store._prefetch_actor import RayPrefetchQueue
from levanter.utils.py_utils import Stopwatch

from ..data._preprocessor import BatchProcessor, BatchResult, dict_from_record_batch
from ..data.metrics_monitor import InProgressCacheMetrics, LoggerMetricsMonitor, MetricsMonitor
from ..data.sharded_datasource import ShardedDataSource
from ..utils.ray_utils import ExceptionInfo, RefBox, SnitchRecipient, current_actor_handle, log_failures_to
from .tree_store import TreeStore


T = TypeVar("T")
U = TypeVar("U")
T_co = TypeVar("T_co", covariant=True)

logger = pylogging.getLogger(__name__)

LEDGER_FILE_NAME = "shard_ledger.json"

DEFAULT_LOG_LEVEL = pylogging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# TODO: should probably do this in terms of bytes
# this is kinda silly, but the bigger the better.
MIN_ITEMS_TO_WRITE = 32 * 1024
MAX_TIME_BETWEEN_WRITES = 100.0


def build_or_load_cache(
    cache_dir: str,
    input_shards: ShardedDataSource[T],
    processor: BatchProcessor[T, U],
    await_finished: bool = True,
    monitors: Optional[Sequence["MetricsMonitor"]] = None,
    cache_config: Optional[Dict[str, Any]] = None,
    force_flush: bool = False,
) -> "TreeCache[U]":
    """
    Produces a sharded cache of the dataset using Ray for distributed processing. The cache can be any path
    on any file system understood by fsspec.

    This system is designed with tokenization and similar processes in mind, but it can potentially be used for any kind
    of preprocessing that converts input batches to output batches. The main design goal is to make it easy to
    parallelize preprocessing across multiple machines while maintaining reproducibility and fault tolerance.
    Usually the machines in question are the ones doing the training, but they could be separate machines as well.

    See the [Dataloader Design Doc](https://github.com/stanford-crfm/levanter/blob/main/docs/design/Data-Loader-Design.md)
    for a somewhat out of date overview of the design.

    Args:
        cache_dir: The directory to write the cache to. This can be any path understood by fsspec.
        input_shards: A ShardedDataset that will be used to read the input data. Conceptually, it's just a mapping
                    from shard names to iterators over the data in that shard.
        processor: A BatchProcessor that will be used to process batches of data. This is the main place where
                    you can customize the preprocessing pipeline.
        await_finished: If True, this function will block until the cache is finished. If False, it will return
                    immediately.
        monitors: a list of MetricsMonitors to attach to the cache. These will be called periodically with
            metrics about the cache build process. If None, will add a LoggerMetricsMonitor.

        cache_config: A dictionary of configuration options for the cache. This is passed to the cache writer.

        force_flush: for testing, forces the cache to flush after every batch. This is useful for testing.

    Returns:
       (TreeCache) A TreeCache object that can be used to read the cache.

    """
    # first see if we need to do anything
    cache = TreeCache.build_or_load(
        cache_dir=cache_dir,
        shard_source=input_shards,
        processor=processor,
        cache_config=cache_config,
        force_flush=force_flush,
    )

    if cache.is_finished:
        logger.info("Cache already finished. Skipping.")
        return cache

    if monitors is None:
        monitors = [LoggerMetricsMonitor()]

    for monitor in monitors:
        cache.attach_metrics_monitor(monitor)

    while await_finished:
        try:
            cache.await_finished(4.0)
            break
        except TimeoutError:
            pass

    return cache


class TreeCache(AsyncDataset[T_co]):
    ledger: Optional["CacheLedger"]
    _builder: Optional[ActorHandle]  # handle of _TreeStoreCacheBuilder
    # monitor_thread waits for new metrics and also periodically reloads the cache
    _monitor_thread: Optional[threading.Thread]
    _metrics_monitors: List[MetricsMonitor]

    def __init__(
        self,
        cache_dir: str,
        exemplar: T_co,
        ledger: Optional["CacheLedger"],
        _broker,  # handle of _TreeStoreCacheBuilder
    ):
        self.cache_dir = cache_dir
        self.ledger = ledger
        self._was_already_finished = ledger is not None and ledger.is_finished
        self._builder = _broker
        self._exemplar = exemplar

        self._metrics_monitors = []
        name = os.path.join(*cache_dir.split("/")[-2:])
        self.logger = pylogging.getLogger(f"TreeCache.{name}")
        self._store_future: threading_Future[TreeStore] = threading_Future()
        self._stop = False
        # assert _broker is None

        if self._builder is not None:
            self._monitor_thread = threading.Thread(target=self._monitor_metrics, daemon=True)
            self._monitor_thread.start()
        else:
            self._attempt_to_load_store()
            assert self._store_future.done()

    @property
    def store(self) -> TreeStore[T_co]:
        return self._store_future.result()

    async def store_async(self) -> TreeStore[T_co]:
        if self._builder is not None:
            return await asyncio.wrap_future(self._store_future)
        else:
            return self.store

    async def async_len(self) -> int:
        if self._builder is not None:
            self.await_finished()

        return len(await self.store_async())

    def __len__(self):
        self.await_finished()

        return len(self.store)

    async def final_length_is_known(self) -> bool:
        if self._builder is not None:
            return await self._builder.is_finished.remote()

        return True

    def is_finite(self) -> bool:
        return True

    async def current_len(self) -> int:
        if not self._store_future.done():
            return 0

        return len(await self.store_async())

    async def get_batch(self, indices: Sequence[int] | slice):
        # this is tricky: we want to wait until either the cache is finished or we have the max index
        if isinstance(indices, slice):
            start, step, stop = await self._get_start_stops_async(indices)
            await self._wait_for_len(max(stop, start))
            indices = range(start, stop, step)

        max_index = max(indices)
        await self._wait_for_len(max_index + 1)

        return await self.store.get_batch(indices)

    async def _wait_for_len(self, needed_len: int):
        if self._builder is not None:
            while needed_len > await self.current_len():
                new_ledger: CacheLedger = await self._builder.updated_ledger.remote()

                if needed_len <= new_ledger.total_num_rows:
                    break

                if new_ledger.is_finished:
                    if needed_len >= new_ledger.total_num_rows:
                        raise IndexError(
                            f"Index {needed_len} out of bounds for cache of size {new_ledger.total_num_rows}"
                        )
                    break
        else:
            if needed_len > len(self.store):
                raise IndexError(f"Index {needed_len} out of bounds for cache of size {len(self.store)}")

    def _wait_for_len_sync(self, needed_len, timeout: Optional[float] = None):
        time_in = time.time()
        t_max = time_in + (timeout or 1e6)
        if self._builder is not None:
            while needed_len > len(self.store):
                cur_time = time.time()
                if cur_time > t_max:
                    raise TimeoutError(f"Timed out waiting for cache to reach {needed_len}")
                try:
                    new_ledger: CacheLedger = ray.get(
                        self._builder.updated_ledger.remote(), timeout=max(t_max - cur_time, 10)
                    )
                except TimeoutError:
                    continue

                if needed_len <= new_ledger.total_num_rows:
                    break

                if new_ledger.is_finished:
                    if needed_len >= new_ledger.total_num_rows:
                        raise IndexError(
                            f"Index {needed_len} out of bounds for cache of size {new_ledger.total_num_rows}"
                        )
                    break
        else:
            if needed_len > len(self.store):
                raise IndexError(f"Index {needed_len} out of bounds for cache of size {len(self.store)}")

    @staticmethod
    def load(cache_dir: str, exemplar: T, cache_config: dict[str, Any] | None) -> "TreeCache":
        """Loads a cache from disk or an object store. Raises FileNotFoundError if the cache doesn't exist"""
        logger.info(f"Loading cache from {cache_dir}")
        ledger = CacheLedger.load(cache_dir, cache_config)
        if not ledger.is_finished:
            raise FileNotFoundError(f"Cache at {cache_dir} is not finished. Use build_or_load to build it.")
        return TreeCache(cache_dir, exemplar, ledger, None)

    @staticmethod
    def build_or_load(
        cache_dir: str,
        shard_source: ShardedDataSource[T],
        processor: BatchProcessor[T, U],
        cache_config: dict[str, Any] | None = None,
        force_flush: bool = False,
    ) -> "TreeCache[U]":
        try:
            return TreeCache.load(cache_dir, processor.output_exemplar, cache_config)
        except FileNotFoundError:
            broker = _get_builder_actor(
                cache_dir=cache_dir,
                input_shards=shard_source,
                processor=processor,
                cache_config=cache_config,
                force_flush=force_flush,
            )
            return TreeCache(cache_dir=cache_dir, exemplar=processor.output_exemplar, ledger=None, _broker=broker)

    def finished_sentinel(self):
        """Returns a Ray-awaitable object that will be set when the cache is finished"""
        if self._builder is None:
            return ray.remote(num_cpus=0)(lambda: None).remote()
        else:
            return self._builder.finished_sentinel.remote()

    @property
    def is_finished(self):
        if self._builder is None:
            return True
        else:
            return ray.get(self._builder.is_finished.remote())

    def __getitem__(self, item):
        if isinstance(item, slice):
            start, step, stop = self._get_start_stops(item)
            # TODO: wait for store to be set
            return self.store[start:stop:step]
        else:
            if item < 0:
                item += len(self)
            if item < 0 or item >= len(self):
                raise IndexError(f"Index {item} out of bounds for cache of size {len(self)}")
            return self.store[item]

    def get_batch_sync(self, indices_or_slice, *, timeout: Optional[float] = None):
        store = self.store
        if isinstance(indices_or_slice, slice):
            start, step, stop = self._get_start_stops(indices_or_slice)
            indices_or_slice = range(start, stop, step)

        max_index = max(indices_or_slice)

        self._wait_for_len_sync(max_index + 1, timeout=timeout)

        return store.get_batch_sync(indices_or_slice)

    def _get_start_stops(self, slice):
        start = slice.start or 0
        if slice.stop is None:
            stop = len(self)
        elif slice.stop < 0:
            stop = len(self) + slice.stop
        else:
            stop = slice.stop
        if start < 0:
            start = len(self) + slice.start
        step = slice.step or 1
        return start, step, stop

    async def _get_start_stops_async(self, slice):
        start = slice.start or 0
        if slice.stop is None:
            stop = await self.async_len()
        elif slice.stop < 0:
            stop = (await self.async_len()) + slice.stop
        else:
            stop = slice.stop
        if start < 0:
            start = (await self.async_len()) + slice.start

        step = slice.step or 1
        return start, step, stop

    def await_finished(self, timeout: Optional[float] = None):
        if self._builder is None:
            return
        x = ray.get(self.finished_sentinel(), timeout=timeout)
        self._attempt_to_load_store()
        return x

    async def finished(self):
        if self._builder is None:
            return
        x = await self.finished_sentinel()
        # TODO: make an async version of this
        self._attempt_to_load_store()
        return x

    def _attempt_to_load_store(self):
        if self._store_future.done():
            return

        logger.info(f"Attempting to load store from {self.cache_dir}")

        try:
            store = TreeStore.open(self._exemplar, self.cache_dir, mode="r")
        except FileNotFoundError:
            assert self._builder is not None
            ledger = ray.get(self._builder.current_ledger.remote())
            metrics = _ledger_to_metrics(ledger)
            if metrics.rows_finished == 0 and metrics.is_finished:
                # this means we built an empty cache. go with it
                store = TreeStore.open(self._exemplar, f"memory://{self.cache_dir}", mode="a")
            else:
                raise
        try:
            self._store_future.set_result(store)
        except concurrent.futures.InvalidStateError:
            pass

    def attach_metrics_monitor(self, monitor: MetricsMonitor):
        if self._builder is None:
            logger.warning("Cannot attach metrics monitor to finished cache.")
            # TODO: decide what to do about attaching if the cache is already finished
            # maybe get the final metrics?
            return

        self._metrics_monitors.append(monitor)

    def _monitor_metrics(self):
        while not self._stop:
            try:
                try:
                    ledger = ray.get(self._builder.updated_ledger.remote(), timeout=10.0)
                    metrics = _ledger_to_metrics(ledger)
                    for monitor in self._metrics_monitors:
                        monitor(metrics)
                    if metrics.is_finished:
                        break
                except TimeoutError:
                    pass
                except Exception as e:
                    if str(e).startswith("Failed to submit task to actor"):
                        logger.warning("Cache builder actor is gone. Stopping monitoring.")
                        break
                try:
                    self._attempt_to_load_store()
                except FileNotFoundError:
                    pass
            except Exception as e:
                if str(e).startswith("Failed to submit task to actor"):
                    logger.warning("Cache builder actor is gone. Stopping monitoring.")
                    break
                else:
                    self.logger.exception("Error while reading metrics from shard cache.")
                    raise e


@dataclass_json
@dataclass
class CacheLedger:
    # NB: unlike the old cache, the mere existence of a ledger doesn't mean the cache is finished
    total_num_rows: int
    shard_rows: Dict[str, int]
    is_finished: bool = False
    finished_shards: List[str] = dataclasses.field(default_factory=list)
    field_counts: Dict[str, int] = dataclasses.field(default_factory=dict)
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @staticmethod
    def load_or_initialize(source: ShardedDataSource, cache_dir: str, metadata: dict | None = None) -> "CacheLedger":
        try:
            return CacheLedger.load(cache_dir, metadata)
        except FileNotFoundError:
            return CacheLedger(
                total_num_rows=0,
                shard_rows={shard: 0 for shard in source.shard_names},
                is_finished=False,
                metadata=metadata or {},
            )

    @staticmethod
    def load(cache_dir: str, metadata: dict | None) -> "CacheLedger":
        ledger_path = os.path.join(cache_dir, LEDGER_FILE_NAME)
        try:
            logger.debug(f"Attempting to load cache ledger from {ledger_path}")
            with fsspec.open(ledger_path) as file:
                cache_ledger = CacheLedger.from_json(file.read())  # type: ignore
            if metadata:
                cache_ledger.compare_metadata(metadata)
            return cache_ledger
        except FileNotFoundError:
            raise FileNotFoundError(f"Cache ledger not found at {ledger_path}")

    def _serialize_and_commit(self, cache_dir):
        path = os.path.join(cache_dir, LEDGER_FILE_NAME)
        return _serialize_json_and_commit(path, self)  # type: ignore

    def _compare_metadata(self, cache_dir, metadata):
        differences: dict[str, tuple[Any, Any]] = {}
        for key, value in metadata.items():
            if self.metadata.get(key) != value:
                differences[key] = (self.metadata.get(key), value)
        # check for keys in our metadata that aren't in the new metadata
        for key in self.metadata.keys():
            if key not in metadata:
                differences[key] = (self.metadata[key], None)

        if differences:

            def format_difference(k, v):
                return f"{k}: {v[0]} -> {v[1]}"

            formatted_diffs = "\n    ".join(format_difference(k, v) for k, v in differences.items())
            logger.warning(
                f"Metadata differences between loaded ledger at {cache_dir} and expectation:\n    {formatted_diffs}"
            )


@dataclass
class _ShardStatus:
    shard_name: str
    num_rows_committed: int
    is_finished: bool


class SerialCacheWriter(AbstractContextManager):
    """
    Writes TreeCache-compatible caches to disk. This is a serial version of TreeCacheWriter that doesn't use Ray.
    Mostly for scripts and debugging.

    Examples:
        >>> with SerialCacheWriter(cache_dir, exemplar) as writer:
        ...     for batch in process_batches():
        ...         writer.write_batch(batch)
    """

    def __init__(
        self,
        cache_dir: str,
        exemplar: T,
        cache_config: Optional[Dict[str, Any]] = None,
    ):
        self.cache_dir = cache_dir
        self.cache_config = cache_config
        self._exemplar = exemplar
        self._tree_store = TreeStore.open(exemplar, self.cache_dir, mode="w")  # type: ignore
        self._is_closed = False

    def __enter__(self) -> "SerialCacheWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # if successful, write the ledger
        # TODO: store field counts in the ledger
        ledger = CacheLedger(
            total_num_rows=len(self._tree_store),
            is_finished=True,
            shard_rows={"": len(self._tree_store)},
            finished_shards=[""],
            field_counts={},
            metadata=self.cache_config,
        )

        if exc_type is None:
            ledger._serialize_and_commit(self.cache_dir)
            logger.info(f"Cache ledger written to {self.cache_dir}")
            self._is_closed = True

    def result(self) -> "TreeCache":
        if not self._is_closed:
            raise RuntimeError("Cannot get result until TreeCacheWriter is closed")
        return TreeCache.load(self.cache_dir, self._exemplar, self.cache_config)

    def write_batch(self, batch: BatchResult):
        if isinstance(batch, pa.RecordBatch):
            raise NotImplementedError("Only non-RecordBatch batches are supported for now")

        batch = _canonicalize_batch(batch)  # type: ignore

        self._tree_store.extend(batch)


class ShardedCacheWriter:
    """
    Similar to SerialCacheWriter, but tracks shard metadata.

    Similar to _OrderedCacheWriter, it also supports resuming, and it
    groups together batches before writing (at some interval) in order to improve performance.
    It does its actual writes in a background thread.
    """

    def __init__(
        self,
        source: ShardedDataSource[T],
        cache_dir: str,
        exemplar: T,
        cache_config: Optional[Dict[str, Any]] = None,
        on_write: Optional[Callable[[CacheLedger], None]] = None,
    ):
        self.cache_dir = cache_dir
        self.cache_config = cache_config
        self._exemplar = exemplar
        self._on_write = on_write

        self._ledger = CacheLedger.load_or_initialize(source, os.path.join(cache_dir, LEDGER_FILE_NAME), cache_config)

        self._tree_store = TreeStore.open(exemplar, self.cache_dir, mode="a")  # type: ignore
        self._tree_store.trim_to_size(self._ledger.total_num_rows)

        # set up the writer thread
        if not self._ledger.is_finished:
            self._stop_loop = threading.Event()
            self._done = False
            self._actual_writer_thread = threading.Thread(target=self._write_loop, daemon=True)

            self._last_write_time = time.time()
            self._items_ready_to_write: list = []

            self._actual_writer_thread.start()
        else:
            logger.info("Cache is already finished. Not starting writer thread")

    @property
    def ledger(self):
        return self._ledger

    @property
    def is_finished(self):
        return self._ledger.is_finished

    @property
    def last_write_time(self):
        return self._last_write_time

    def write_batch(self, shard_name: str, batch: BatchResult):
        if self.is_finished:
            raise RuntimeError("Cannot write to a finished cache")

        if isinstance(batch, pa.RecordBatch):
            raise NotImplementedError("Only non-RecordBatch batches are supported for now")

        batch = _canonicalize_batch(batch)  # type: ignore

        self._items_ready_to_write.append((shard_name, batch))

    def _write_loop(self):
        done = False
        while not done:
            self._attempt_to_write_batches()
            try:
                if self._stop_loop.wait(1):
                    logger.info("Stopping writer thread")
                    done = True
            except TimeoutError:
                pass
            if self._ledger.is_finished:
                logger.info("Ledger is finished. Stopping writer thread")
                return

    def flush(self):
        self._attempt_to_write_batches()

    def finish(self):
        self._done = True
        self._stop_loop.set()
        self._actual_writer_thread.join()
        self.flush()

        # if successful, write the ledger
        logger.info("Finished writing cache")
        self._ledger.is_finished = True
        self._ledger._serialize_and_commit(self.cache_dir)
        if self._on_write:
            self._on_write(self._ledger)

        return self._tree_store

    def _attempt_to_write_batches(self):
        if self._ledger.is_finished:
            return

        if len(self._items_ready_to_write) == 0:
            return

        logger.warning(f"Writing {len(self._items_ready_to_write)} batches")

        updated_shards = self._write_available_batches()

        logger.debug(f"Updated shards: {updated_shards}")

        did_write = len(updated_shards) > 0

        if did_write:
            for shard, num_rows in updated_shards.items():
                self._ledger.shard_rows[shard] = self._ledger.shard_rows.get(shard, 0) + num_rows

            self._last_write_time = time.time()
            total_rows = self._ledger.total_num_rows + sum(updated_shards.values())
            self._ledger.total_num_rows = total_rows
            self._ledger._serialize_and_commit(self.cache_dir)

            if self._on_write:
                self._on_write(self._ledger)

    def _write_available_batches(self):
        if len(self._items_ready_to_write) == 0:
            return {}
        ready = self._items_ready_to_write
        self._items_ready_to_write = []

        to_write = []
        written_by_shard = {}
        for shard, batch in ready:
            to_write.extend(batch)
            written_by_shard[shard] = written_by_shard.get(shard, 0) + len(batch)

        self._tree_store.extend(to_write)
        return written_by_shard


def _serialize_json_and_commit(path, obj):
    # just to be paranoid, we write to a temp file and then rename it
    # TODO: probably we could do better here
    with fsspec.open(f"{path}.tmp", "w") as file:
        file.write(obj.to_json())
    # now copy the old file to a backup
    fs: AbstractFileSystem = fsspec.core.url_to_fs(path)[0]
    fs.mkdirs(os.path.dirname(path), exist_ok=True)
    if fs.exists(path):
        fs.copy(path, f"{path}.bak")
    fs.rename(f"{path}.tmp", path)


@ray.remote(num_cpus=0.1)  # keep this small b/c it doesn't do a lot
class _TreeStoreCacheBuilder(SnitchRecipient):
    """
    Actor that coordinates the building of a cache. It spins up a bunch of workers to read from each shard
    and write to the cache.

    """

    def __init__(
        self,
        cache_dir: str,
        name: str,
        source: ShardedDataSource[T],
        processor: BatchProcessor[T, U],
        cache_config: Dict[str, Any],
        force_flush: bool,
    ):
        pylogging.basicConfig(level=DEFAULT_LOG_LEVEL, format=LOG_FORMAT)
        self.logger = pylogging.getLogger(f"{__name__}.{name}")
        self.source = source
        self._cache_dir = cache_dir
        self._cache_config = cache_config
        self._updated_ledger_condition = asyncio.Condition()
        # used to subscribe to metrics updates
        self._finished_promise: asyncio.Future[None] = asyncio.Future()

        self._ledger = CacheLedger.load_or_initialize(source, cache_dir, cache_config)

        if self._ledger.is_finished:
            self._finished_promise.set_result(None)

        path_for_name = os.path.join(*self._cache_dir.split("/")[-2:])
        name = f"broker::{path_for_name}"
        self.logger = pylogging.getLogger(f"{name}")

        if self._ledger.is_finished:
            self.logger.info("Cache already finished. Nothing to do.")
            return
        self._cache_writer = _preprocess_task_chain(
            current_actor_handle(), cache_dir, name, source, processor, cache_config, force_flush
        )

    def current_ledger(self):
        return self._ledger

    def other_failed(self, error: ExceptionInfo):
        """Callback method for when a shard worker has failed."""
        self._writer_exception(None, error)

    def _child_failed(self, child: ray.actor.ActorHandle, exception: ExceptionInfo):
        self.logger.error(f"Child {child} failed with exception", exc_info=exception.restore())
        self._writer_exception(None, exception)

    def is_finished(self):
        return self._ledger.is_finished

    async def finished_sentinel(self):
        await self._finished_promise

    async def updated_ledger(self) -> CacheLedger:
        if self._finished_promise.done():
            if self._finished_promise.exception() is not None:
                raise self._finished_promise.exception()  # type: ignore
            else:
                return self._ledger

        async with self._updated_ledger_condition:
            await self._updated_ledger_condition.wait()
            return self._ledger

    def _writer_exception(self, shard_name, exc_info: ExceptionInfo):
        info = exc_info.restore()

        logger.exception(f"Writer task {shard_name} failed with exception", exc_info=info)

        try:
            self._finished_promise.set_exception(info[1])
        except InvalidStateError:
            pass
        except concurrent.futures.InvalidStateError:
            pass
        self._do_notify()

    def _notify_updated_ledger(self, ledger: CacheLedger):
        """
        Called by the cache writer when it has updated the ledger.
        """
        was_finished = self._ledger.is_finished
        self._ledger = ledger

        if was_finished:
            raise RuntimeError("Ledger was already finished")

        if self._ledger.is_finished:
            logger.info(f"Finalizing cache {self._cache_dir}...")
            # guard against invalid state errors
            if not self._finished_promise.done():
                self._finished_promise.set_result(None)

            self._cache_writer = None

        self._do_notify()

    def _do_notify(self):
        async def _do_notify_async():
            async with self._updated_ledger_condition:
                self._updated_ledger_condition.notify_all()

        asyncio.create_task(_do_notify_async())


def _get_builder_actor(cache_dir, input_shards, processor, cache_config=None, force_flush=False):
    name = f"lev_cache_manager::{cache_dir}"
    path_for_name = os.path.join(*os.path.split(cache_dir)[-2:])
    name_for_display = f"builder::{path_for_name}"

    return _TreeStoreCacheBuilder.options(name=name, get_if_exists=True).remote(  # type: ignore
        name=name_for_display,
        cache_dir=cache_dir,
        source=input_shards,
        processor=processor,
        cache_config=cache_config,
        force_flush=force_flush,
    )


def _preprocess_task_chain(
    parent,
    cache_dir: str,
    name: str,
    source: ShardedDataSource,
    processor: BatchProcessor,
    cache_config: Optional[Dict[str, Any]],
    force_flush: bool = False,
):
    """
    This is the main task that processes the data and writes it to the cache.

    It chains together:
    * 1 generator per shard
    * interleaving of the generators
    * processing of the batches
    * writing of the batches to the cache

    It handles resumes on the reading side
    """
    with log_failures_to(parent):
        initial_ledger = CacheLedger.load_or_initialize(source, cache_dir, cache_config)
        statuses = [
            _ShardStatus(shard_name, initial_ledger.shard_rows.get(shard_name, 0), False)
            for shard_name in source.shard_names
        ]

        writer_task = _core_writer_task.remote(
            parent, cache_dir, source, statuses, processor, cache_config, force_flush
        )

    return writer_task


@ray.remote(num_cpus=1)
def _core_writer_task(
    parent,
    cache_dir,
    source: ShardedDataSource,
    initial_statuses: Sequence[_ShardStatus],
    processor,
    cache_config: Optional[Dict[str, Any]],
    force_flush: bool,
):
    logger.setLevel(DEFAULT_LOG_LEVEL)
    logger.info("Starting writer task")

    def on_write(ledger):
        # todo: send serial numbers to ensure we don't process old data
        ray.get(parent._notify_updated_ledger.remote(ledger))

    with log_failures_to(parent):
        sharded_cache_writer = ShardedCacheWriter(
            source, cache_dir, processor.output_exemplar, cache_config=cache_config, on_write=on_write
        )

        interleave = RayPrefetchQueue(lambda: _make_interleave_for_shards(source, initial_statuses, processor), 1024)
        del initial_statuses

        total_time = Stopwatch()
        loading_time = Stopwatch()
        get_time = Stopwatch()
        append_time = Stopwatch()

        # iterator = iter(enumerate(interleave))
        i = 0

        # for i, batch_box in enumerate(interleave):
        while True:
            with total_time:
                try:
                    with loading_time:
                        # i, batch_box = next(iterator)
                        shard, batch = interleave.get_next()
                        i += 1

                    # with get_time:
                    #     shard, batch = batch_box.get()
                    with append_time:
                        sharded_cache_writer.write_batch(shard, batch)
                    if force_flush:
                        sharded_cache_writer.flush()
                    if i % 1000 == 0:
                        print(
                            f"Processed {i} batches: {loading_time.average()}s load, {get_time.average()}s get,"
                            f" {append_time.average()}s append, {total_time.average()}s total"
                        )
                except StopIteration:
                    print("stop!")
                    break
                except Exception as e:
                    logger.exception("Error while processing batch")
                    raise e

        logger.warning(f"Got {i} batches")
        sharded_cache_writer.finish()

        out = sharded_cache_writer.ledger
        return out


def _interleave_shards(
    shard_names: Sequence[str], readers: Sequence[RayPrefetchQueue[RefBox]], first_shard_index: int
) -> Iterator[T]:
    """
    Interleaves the results of multiple iterators. To support resume,
    we need to be able to start from not the "first" iterator.
    """
    if len(shard_names) != len(readers):
        raise ValueError("shard_names and readers must have the same length")

    if len(shard_names) == 0:
        logger.info("No shards to interleave")
        return

    finished: set[str] = set()
    total = 0
    while len(finished) < len(shard_names):
        for i in range(first_shard_index, len(shard_names)):
            shard_name = shard_names[i]
            reader = readers[i]
            if shard_name not in finished:
                try:
                    out = reader.get_next()
                    total += 1
                    yield out.ref
                except StopIteration:
                    logger.info(f"Finished shard {shard_name}")
                    finished.add(shard_name)
                except Exception as e:
                    logger.exception(f"Error while processing shard {shard_name}")
                    raise e

        first_shard_index = 0

    logger.info(f"Finished all shards, got {total} batches")


def _make_interleave_for_shards(
    source: ShardedDataSource, statuses: Sequence[_ShardStatus], processor: BatchProcessor
):
    """
    Given a list of ShardStatus objects and sources, creates an interleaving generator
    that reads from shards and tokenizes them in parallel.

    We use ShardStatus objects to track the progress of each shard. If we're preempted, we can resume
    from the last shard we were working on. This function starts each shard at the last committed row
    and starts interleaving from the next shard (i.e. the one with the fewest rows that isn't finished).
    """
    logger.setLevel(DEFAULT_LOG_LEVEL)

    unfinished_shards: list[_ShardStatus] = []
    shards_with_progress: dict[str, int] = {}

    for status in statuses:
        if not status.is_finished:
            unfinished_shards.append(status)
        if status.num_rows_committed > 0:
            shards_with_progress[status.shard_name] = status.num_rows_committed

    if not unfinished_shards:
        logger.info("All shards finished. Nothing to do.")
        return
    elif shards_with_progress:
        formatted = ", ".join(f"{k}: {v}" for k, v in shards_with_progress.items())
        logger.info(f"Resuming from shards with progress: {formatted}")
    logger.warning(f"Starting cache build with {len(unfinished_shards)} shards")

    process_task = _mk_process_task(processor)

    def _make_generator_fn(status: _ShardStatus) -> Callable[[], Iterator[RefBox]]:
        def generator():
            for shard_name, batch in _shard_reader_generator(
                source, status.shard_name, status.num_rows_committed, processor.batch_size
            ):
                processed = process_task.remote(status.shard_name, batch)
                # We don't want Ray to dereference the ObjectRef, so we wrap it in a RefBox
                yield RefBox(processed)

        return generator

    generator_fns = [_make_generator_fn(status) for status in unfinished_shards]

    readers = [
        RayPrefetchQueue(generator_fn, max_queue_size=4)
        for shard, generator_fn in zip(unfinished_shards, generator_fns)
    ]

    # then figure out the first shard to start from. This is the first unfinished shard with the minimum number of rows
    min_unfinished = min(range(len(unfinished_shards)), key=lambda x: unfinished_shards[x].num_rows_committed)

    yield from _interleave_shards([status.shard_name for status in unfinished_shards], readers, min_unfinished)


def _shard_reader_generator(shard_source: ShardedDataSource[T], shard_name: str, start_row: int, batch_size: int):
    # TODO: track more metadata about batches
    logger.info(f"Opening shard {shard_name} at row {start_row}")
    shard_iter = shard_source.open_shard_at_row(shard_name, start_row)
    batch = []
    i = 0
    for row in shard_iter:
        batch.append(row)

        if len(batch) == batch_size:
            i += 1
            yield (shard_name, batch)
            batch = []

    if len(batch) > 0:
        yield (shard_name, batch)
        i += 1

    logger.info(f"Finished generating shard {shard_name} with {i} batches")


def _mk_process_task(processor: BatchProcessor[T, U]) -> RemoteFunction:
    """
    Returns a Ray remote function that processes a batch of data. Basically it takes the resources from
    the processor and wraps its call
    Args:
        processor:

    Returns:

    """

    @ray.remote(num_cpus=processor.num_cpus, num_gpus=processor.num_gpus, resources=processor.resources)
    def process_task(desc, batch: List[T]):
        # pylogging.basicConfig(level=DEFAULT_LOG_LEVEL, format=LOG_FORMAT)
        logger.debug(f"Processing batch {desc}")
        try:
            result = processor(batch)
            result = _canonicalize_batch(result)  # type: ignore
            logger.debug(f"Finished processing batch {desc}")
            return desc, result
        except Exception as e:
            logger.exception(f"Error while processing batch {desc}")
            raise e
        finally:
            pass

    return process_task


def _canonicalize_batch(batch: Union[dict, List[dict]]) -> List[dict]:
    if isinstance(batch, pa.RecordBatch):
        batch = dict_from_record_batch(batch)

    if isinstance(batch, dict):
        return _to_list_of_dicts(batch)
    else:
        return batch


def _to_list_of_dicts(batch: dict) -> List[dict]:
    """
    Convert a batch of dictionaries to a list of dictionaries, suitable for writing to a cache.
    """
    keys = list(batch.keys())
    values = list(batch.values())
    num_rows = len(values[0])
    return [{key: values[i][j] for i, key in enumerate(keys)} for j in range(num_rows)]


def _ledger_to_metrics(ledger: CacheLedger) -> InProgressCacheMetrics:
    # TODO: remove this
    return InProgressCacheMetrics(
        rows_finished=ledger.total_num_rows,
        is_finished=ledger.is_finished,
        # shard_rows=ledger.shard_rows,
        shards_finished=len(ledger.finished_shards),
        field_counts=ledger.field_counts,
    )
