import asyncio
import concurrent
import copy
import dataclasses
import logging as pylogging
import os
import pprint
import random
import threading
import time
from asyncio import InvalidStateError
from concurrent.futures import Future as threading_Future
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, TypeVar, Union

import deepdiff
import fsspec.core
import humanfriendly
import jax
import pyarrow as pa
import ray
from dataclasses_json import dataclass_json
from fsspec import AbstractFileSystem
from jaxtyping import PyTree
from ray.actor import ActorHandle

from levanter.data.dataset import AsyncDataset
from levanter.store._prefetch_actor import QueueEmpty, RayPrefetchQueue
from levanter.utils.py_utils import Stopwatch

from ..data._preprocessor import BatchProcessor, BatchProcessorPool, BatchResult, dict_from_record_batch
from ..data.metrics_monitor import InProgressCacheMetrics, LoggerMetricsMonitor, MetricsMonitor
from ..data.sharded_datasource import ShardedDataSource
from ..utils.ray_utils import (
    ExceptionInfo,
    RefBox,
    SnitchRecipient,
    current_actor_handle,
    log_failures_to,
    ser_exc_info,
)
from ..utils.thread_utils import ExceptionTrackingThread
from .jagged_array import PreparedBatch
from .tree_store import TreeStore


T = TypeVar("T")
U = TypeVar("U")
T_co = TypeVar("T_co", covariant=True)

logger = pylogging.getLogger(__name__)

LEDGER_FILE_NAME = "shard_ledger.json"

DEFAULT_LOG_LEVEL = pylogging.INFO
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


@dataclass_json
@dataclass(frozen=True)
class CacheOptions:
    """
    Configuration for a cache. This is used to configure a few parts of the cache creation process and to
    store metadata that can be checked to ensure that the cache being loaded was created with the expected
    configuration. It combined with the [[BatchProcessor]] metadata to form the [[CacheMetadata]].

    It is intended that caching it deterministic conditional on the input data, processor, and these options.
    """

    num_shard_groups: Optional[int] = 128
    """Number of groups to divide the shards into. This is used to parallelize the cache building process without
    overloading Ray. If None, all shards will be in their own group."""
    shard_order_randomization_key: Optional[int] = 0
    """A key used to randomize the order of the shards before building and grouping."""
    batch_size: int = 128
    """The batch size to use when processing the data. This is used to control the memory usage of the cache building
    process. Lower values will use less memory but take somewhat longer to build the cache."""

    # the below options don't actually impact the cache's result, but do impact construction
    target_size_per_flush: int | str = "512MB"
    """The number of bytes to buffer before flushing to disk. This is used to control the memory usage of the cache
    building process. Lower values will use less memory but could take somewhat longer to build the cache."""
    prefetch_per_group: int = 4
    """The number of batches to prefetch per group. This is used to keep the processors busy and to reduce the time"""

    @property
    def target_bytes_per_flush(self):
        return humanfriendly.parse_size(self.target_size_per_flush)

    @staticmethod
    def default():
        return CacheOptions()

    @staticmethod
    def no_fanciness(batch_size: Optional[int] = None):
        """
        For testing, disables all the fancy features of the cache. This makes it easier to predict the behavior
        """
        if batch_size is None:
            batch_size = 128
        return CacheOptions(num_shard_groups=None, shard_order_randomization_key=None, batch_size=batch_size)

    @staticmethod
    def one_group():
        """
        For testing, disables all the fancy features of the cache. This makes it easier to predict the behavior
        """
        return CacheOptions(num_shard_groups=1, shard_order_randomization_key=None, batch_size=128)


def build_or_load_cache(
    cache_dir: str,
    input_shards: ShardedDataSource[T],
    processor: BatchProcessor[T, U],
    await_finished: bool = True,
    monitors: Optional[Sequence["MetricsMonitor"]] = None,
    options: CacheOptions = CacheOptions.default(),
    force_flush: bool = False,
    split: str = "test",
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

        options: Configuration for the cache. This is used to configure a few parts of the cache creation process

        force_flush: for testing, forces the cache to flush after every batch. This is useful for testing.

    Returns:
       (TreeCache) A TreeCache object that can be used to read the cache.

    """
    # first see if we need to do anything
    cache = TreeCache.build_or_load(
        cache_dir=cache_dir,
        shard_source=input_shards,
        processor=processor,
        options=options,
        force_flush=force_flush,
        split=split,
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
        return self.ledger is not None and self.ledger.is_finished

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
    def load(cache_dir: str, exemplar: T, options: Optional["CacheMetadata"] = None) -> "TreeCache":
        """Loads a cache from disk or an object store. Raises FileNotFoundError if the cache doesn't exist"""
        logger.info(f"Loading cache from {cache_dir}")
        ledger = CacheLedger.load(cache_dir, options)
        if not ledger.is_finished:
            raise FileNotFoundError(f"Cache at {cache_dir} is not finished. Use build_or_load to build it.")
        return TreeCache(cache_dir, exemplar, ledger, None)

    @staticmethod
    def build_or_load(
        cache_dir: str,
        shard_source: ShardedDataSource[T],
        processor: BatchProcessor[T, U],
        options: Optional["CacheOptions"] = None,
        force_flush: bool = False,
        split: str = "test",
    ) -> "TreeCache[U]":
        if options is None:
            options = CacheOptions.default()
        metadata = CacheMetadata(options=options, preprocessor_metadata=processor.metadata)
        try:
            return TreeCache.load(cache_dir, processor.output_exemplar, metadata)
        except FileNotFoundError:
            broker = _get_builder_actor(
                cache_dir=cache_dir,
                shard_source=shard_source,
                processor=processor,
                options=options,
                force_flush=force_flush,
                split=split,
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
        return self.ledger is not None and self.ledger.is_finished

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
                    # it's better to let the Ray actor handle the timeout
                    ledger_or_timeout = ray.get(self._builder.updated_ledger.remote(timeout=4.0), timeout=10.0)
                    if isinstance(ledger_or_timeout, Exception):
                        raise ledger_or_timeout
                    self.ledger = ledger_or_timeout
                    metrics = _ledger_to_metrics(self.ledger)
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
                    else:
                        raise
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
    metadata: "CacheMetadata" = dataclasses.field(default_factory=lambda: CacheMetadata(CacheOptions(), {}))

    @staticmethod
    def load_or_initialize(
        cache_dir: str, source: ShardedDataSource, processor: BatchProcessor, config: "CacheOptions"
    ):
        metadata = CacheMetadata(options=config, preprocessor_metadata=processor.metadata)
        try:
            return CacheLedger.load(cache_dir, metadata)
        except FileNotFoundError:
            return CacheLedger(
                total_num_rows=0,
                shard_rows={shard: 0 for shard in source.shard_names},
                is_finished=False,
                metadata=metadata,
            )

    @staticmethod
    def load(cache_dir: str, metadata: Optional["CacheMetadata"] = None) -> "CacheLedger":
        ledger_path = os.path.join(cache_dir, LEDGER_FILE_NAME)
        try:
            logger.debug(f"Attempting to load cache ledger from {ledger_path}")
            with fsspec.open(ledger_path) as file:
                cache_ledger = CacheLedger.from_json(file.read())  # type: ignore
            if metadata:
                diff = cache_ledger.metadata.compare_to(metadata)
                if not diff:
                    logger.debug("Metadata matches")
                else:
                    logger.warning(f"Metadata mismatch: {pprint.pformat(diff, indent=2)}")
            return cache_ledger
        except FileNotFoundError:
            raise FileNotFoundError(f"Cache ledger not found at {ledger_path}")

    def _serialize_and_commit(self, cache_dir):
        path = os.path.join(cache_dir, LEDGER_FILE_NAME)
        return _serialize_json_and_commit(path, self)  # type: ignore


@dataclass_json
@dataclass(frozen=True)
class CacheMetadata:
    options: CacheOptions = CacheOptions.default()
    preprocessor_metadata: Optional[dict[str, Any]] = None

    def compare_to(self, other: "CacheMetadata") -> deepdiff.DeepDiff:
        """
        Compare this metadata to another set of metadata. This is used to check if the cache being loaded
        was created with the expected configuration.

        if other.preprocessor_metadata is None, we ignore it for the purposes of comparison.
        """
        if other.preprocessor_metadata is None:
            sorta_self = dataclasses.replace(self, preprocessor_metadata=None)
        else:
            sorta_self = self
        return deepdiff.DeepDiff(sorta_self, other)

    @staticmethod
    def empty():
        return CacheMetadata()


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
        >>> with SerialCacheWriter(cache_dir,exemplar) as writer:
        ...     for batch in process_batches():
        ...         writer.write_batch(batch)
    """

    def __init__(
        self,
        cache_dir: str,
        exemplar: T,
        metadata: Optional["CacheMetadata"] = None,
    ):
        self.cache_dir = cache_dir
        self.metadata = metadata
        self._exemplar = exemplar
        self._tree_store = TreeStore.open(exemplar, self.cache_dir, mode="w", cache_metadata=True)
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
            metadata=self.metadata or CacheMetadata.empty(),
        )

        if exc_type is None:
            ledger._serialize_and_commit(self.cache_dir)
            logger.info(f"Cache ledger written to {self.cache_dir}")
            self._is_closed = True

    def result(self) -> "TreeCache":
        if not self._is_closed:
            raise RuntimeError("Cannot get result until TreeCacheWriter is closed")
        return TreeCache.load(self.cache_dir, self._exemplar, self.metadata)

    def write_batch(self, batch: BatchResult):
        if isinstance(batch, pa.RecordBatch):
            raise NotImplementedError("Only non-RecordBatch batches are supported for now")

        cbatch = _canonicalize_batch(batch)  # type: ignore

        self._tree_store.extend(cbatch)


class ShardedCacheWriter:
    """
    Similar to SerialCacheWriter, but tracks shard metadata.

    Similar to _OrderedCacheWriter, it also supports resuming, and it
    groups together batches before writing (at some interval) in order to improve performance.
    """

    def __init__(
        self,
        cache_dir: str,
        initial_ledger: CacheLedger,
        exemplar: T,
        on_write: Optional[Callable[[CacheLedger], None]] = None,
    ):
        self.cache_dir = cache_dir
        self._on_write = on_write

        self._ledger = copy.deepcopy(initial_ledger)

        self._tree_store = TreeStore.open(exemplar, self.cache_dir, mode="a")  # type: ignore
        self._tree_store.trim_to_size(self._ledger.total_num_rows)

    @property
    def ledger(self):
        return self._ledger

    # we have both versions b/c we need this one for actors
    def get_ledger(self):
        return self._ledger

    @property
    def is_finished(self):
        return self._ledger.is_finished

    def finish_shard(self, shard_name: str, num_rows: int):
        current_rows = self._ledger.shard_rows.get(shard_name, 0)
        if current_rows != num_rows:
            raise ValueError(f"Expected {num_rows} rows in finished shard {shard_name}, but found {current_rows}")

        self._ledger.finished_shards.append(shard_name)
        self._ledger._serialize_and_commit(self.cache_dir)

    def write_prepared_batch(self, shard_counts: Mapping[str, int], batch: PyTree[PreparedBatch]):
        if self.is_finished:
            raise RuntimeError("Cannot write to a finished cache")
        self._tree_store.extend_with_batch(batch)

        for shard, num_rows in shard_counts.items():
            self._ledger.shard_rows[shard] = self._ledger.shard_rows.get(shard, 0) + num_rows

        total_rows = self._ledger.total_num_rows + sum(shard_counts.values())
        self._ledger.total_num_rows = total_rows
        self._ledger._serialize_and_commit(self.cache_dir)

        if self._on_write:
            self._on_write(self._ledger)

    def write_batch(self, shard_name: str, batch: BatchResult):
        if self.is_finished:
            raise RuntimeError("Cannot write to a finished cache")

        if isinstance(batch, pa.RecordBatch):
            raise NotImplementedError("Only non-RecordBatch batches are supported for now")

        batch = _canonicalize_batch(batch)  # type: ignore
        prepared = self._tree_store.batch_preparer(batch)

        return self.write_prepared_batch({shard_name: len(batch)}, prepared)

    def finish(self):
        # if successful, write the ledger
        logger.info("Finished writing cache")
        # check that all shards are finished
        if set(self._ledger.shard_rows.keys()) != set(self._ledger.finished_shards):
            raise ValueError("Not all shards are finished")

        self._ledger.is_finished = True
        self._ledger._serialize_and_commit(self.cache_dir)
        if self._on_write:
            self._on_write(self._ledger)

        return self._tree_store


def _serialize_json_and_commit(path, obj):
    # just to be paranoid, we write to a temp file and then rename it
    # TODO: probably we could do better here
    fs: AbstractFileSystem = fsspec.core.url_to_fs(path)[0]
    fs.mkdirs(os.path.dirname(path), exist_ok=True)
    if fs.exists(path):
        # copy the old file to a backup
        fs.copy(path, f"{path}.bak")

    for i in range(10):
        with fsspec.open(f"{path}.tmp", "w") as file:
            file.write(obj.to_json())

        try:
            fs.rename(f"{path}.tmp", path)
            break
        except FileNotFoundError:
            # this happens for some reason sometimes. It makes no sense.
            # FileNotFoundError: b/levanter-data/o/scratch%2Fdlwh%2Fpile-YYY%2Fpubmed_abs%2Ftrain%2Fshard_ledger.json.tmp/rewriteTo/b/levanter-data/o/scratch%2Fdlwh%2Fpile-YYY%2Fpubmed_abs%2Ftrain%2Fshard_ledger.json
            logger.exception(f"Failed to rename {path}.tmp to {path}")
            pass


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
        split: str,  # to workaround https://github.com/ray-project/ray/issues/44083
        source: ShardedDataSource[T],
        processor: BatchProcessor[T, U],
        options: CacheOptions,
        force_flush: bool,
    ):
        pylogging.basicConfig(level=DEFAULT_LOG_LEVEL, format=LOG_FORMAT)
        self.logger = pylogging.getLogger(f"{__name__}.{name}")
        self._finished_promise: asyncio.Future[None] = asyncio.Future()
        try:
            self.source = source
            self._cache_dir = cache_dir
            self._options = options
            self._updated_ledger_condition = asyncio.Condition()  # used to subscribe to metrics updates

            self._ledger = CacheLedger.load_or_initialize(cache_dir, source, processor, options)

            if self._ledger.is_finished:
                self._finished_promise.set_result(None)

            path_for_name = os.path.join(*self._cache_dir.split("/")[-2:])
            name = f"broker::{path_for_name}"
            self.logger = pylogging.getLogger(f"{name}")

            if self._ledger.is_finished:
                self.logger.info("Cache already finished. Nothing to do.")
                return
            self._cache_writer = _core_writer_task.options(
                name=f"writer::{path_for_name}",
                scheduling_strategy="SPREAD",
                # memory needed for the writer is twice the options' target size per flush
                # (we get twice from we need to concatenate prepared batches into the accumulator)
                # TODO: measure.
                memory=2 * self._options.target_bytes_per_flush,
            ).remote(current_actor_handle(), cache_dir, self._ledger, source, processor, force_flush)
        except Exception:
            # Ray behaves poorly if the constructor of an actor fails, so we catch and log here
            # this also propagates to the finished promise, so we can handle it there
            self._writer_exception(None, ser_exc_info())

    def current_ledger(self):
        if self._finished_promise.done() and self._finished_promise.exception() is not None:
            raise self._finished_promise.exception()
        return self._ledger

    def is_finished(self):
        if self.failed():
            return False
        return self._ledger.is_finished

    def failed(self):
        return self._finished_promise.done() and self._finished_promise.exception() is not None

    async def finished_sentinel(self):
        await self._finished_promise

    async def updated_ledger(self, timeout: float | None = None) -> CacheLedger | TimeoutError:
        """
        NB: we **return** a timeout error, we don't raise it. This is because we want to find real failures
        in the ray dashboard, and it's a real pain to find exceptions in the logs.
        """
        if self._finished_promise.done():
            if self._finished_promise.exception() is not None:
                raise self._finished_promise.exception()  # type: ignore
            else:
                return self._ledger

        try:
            async with self._updated_ledger_condition:
                cond = self._updated_ledger_condition.wait()
                if timeout is not None:
                    await asyncio.wait_for(cond, timeout=timeout)
                else:
                    await cond
            return self._ledger
        except asyncio.TimeoutError:
            return TimeoutError("Timed out waiting for cache to update")

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


def _get_builder_actor(split, cache_dir, shard_source, processor, options=CacheOptions.default(), force_flush=False):
    name = f"lev_cache_manager::{split}::{cache_dir}"
    path_for_name = os.path.join(*os.path.split(cache_dir)[-2:])
    name_for_display = f"builder::{path_for_name}"

    return _TreeStoreCacheBuilder.options(name=name, get_if_exists=True).remote(  # type: ignore
        name=name_for_display,
        split=split,
        cache_dir=cache_dir,
        source=shard_source,
        processor=processor,
        options=options,
        force_flush=force_flush,
    )


#####
# Core implementation starts below.
#####
# The main idea is to have a bunch of reader tasks that read batches, dispatch tokenization tasks, producing
# a stream of tokenized batches. We then interleave these tokenized batches and write them to the cache.
# The reader tasks are given a group of shards, which are implicitly concatenated together.


@dataclass
class _Batch:
    """
    A batch of data that has either been read or tokenized.
    """

    shard_name: str
    row_indices: List[int]
    payload: ray.ObjectRef


@dataclass
class _ShardFinished:
    """
    A message indicating that a shard has finished.
    """

    shard_name: str
    total_rows: int


_Message = _Batch | _ShardFinished
"""
A message that can be sent from a reader task to the writer task.
"""

_TIME_BETWEEN_WRITES = 20.0  # seconds


@ray.remote(num_cpus=1)
def _core_writer_task(
    parent,
    cache_dir,
    initial_ledger: CacheLedger,
    source: ShardedDataSource,
    processor,
    force_flush: bool,
):
    """
    This is the main task that processes the data and writes it to the cache.

    It chains together:
        * 1 generator per shard group
        * interleaving of the generators
        * processing of the batches
        * writing of the batches to the cache
    """
    pylogging.basicConfig(level=DEFAULT_LOG_LEVEL, format=LOG_FORMAT)
    logger.info("Starting writer task")

    name = str(os.path.join(*cache_dir.split("/")[-2:]))
    # append a small random number to the name to avoid collisions
    name += f"::{random.randint(0, 1000)}"

    with log_failures_to(parent):

        def on_write(ledger):
            ray.get(parent._notify_updated_ledger.remote(ledger))

        sharded_cache_writer = ShardedCacheWriter(
            cache_dir, initial_ledger, processor.output_exemplar, on_write=on_write
        )

        options = initial_ledger.metadata.options
        num_groups = min(options.num_shard_groups or 1000000, len(source.shard_names))

        processor_pool = _mk_processor_pool(processor, 0, num_groups * 4)

        interleave: RayPrefetchQueue = RayPrefetchQueue(
            lambda: _make_interleave(name, source, initial_ledger, processor_pool),
            64,
            producer_options={"num_cpus": 1, "name": f"{name}::interleave"},
        )

        total_time = Stopwatch()
        loading_time = Stopwatch()
        append_time = Stopwatch()
        flush_time = Stopwatch()
        flush_amortized_time = Stopwatch()

        current_prepared_batch: Optional[PyTree[PreparedBatch]] = None
        current_shard_rows: dict[str, int] = {}
        time_of_last_write = time.time()
        batches_total = 0.0
        flush_thread = None
        finished_shards_last_flush: list = []

        while True:
            with total_time:  # 0.0051
                try:
                    cur_time = time.time()
                    time_since_last_write = cur_time - time_of_last_write
                    remaining_time = _TIME_BETWEEN_WRITES - time_since_last_write

                    if current_prepared_batch is not None:
                        with flush_amortized_time:  # 6e-4
                            current_byte_size = sum(
                                b.byte_size for b in jax.tree_util.tree_flatten(current_prepared_batch)[0]
                            )
                            should_flush = (
                                force_flush
                                or remaining_time <= 0
                                or (current_byte_size >= options.target_bytes_per_flush)
                            )
                            if should_flush:
                                with flush_time:  # 0.613s
                                    if flush_thread is not None:
                                        flush_thread.join()

                                    flush_thread = ExceptionTrackingThread(
                                        target=_write_batches,
                                        args=(
                                            sharded_cache_writer,
                                            current_shard_rows,
                                            current_prepared_batch,
                                            finished_shards_last_flush,
                                        ),
                                    )
                                    flush_thread.start()

                                    current_prepared_batch = None
                                    current_shard_rows = {}
                                    finished_shards_last_flush = []

                                    time_of_last_write = time.time()
                                    continue
                    else:
                        remaining_time = _TIME_BETWEEN_WRITES

                    with loading_time:
                        try:
                            message = interleave.get_next(timeout=max(remaining_time, 0.1))
                        except QueueEmpty:
                            logger.info("Writer running ahead of reader.")
                            continue

                    with append_time:
                        match message:
                            case _Batch(shard, row_indices, payload):
                                batches_total += 1
                                this_prepared_batch = ray.get(payload)
                                if current_prepared_batch is None:
                                    # TODO: actually check row indices
                                    current_shard_rows = {shard: len(row_indices)}
                                    current_prepared_batch = this_prepared_batch
                                else:
                                    current_shard_rows[shard] = current_shard_rows.get(shard, 0) + len(row_indices)
                                    current_prepared_batch = _concat_prepared_batches(
                                        current_prepared_batch, this_prepared_batch
                                    )
                                    del this_prepared_batch

                                if force_flush:
                                    _write_batches(
                                        sharded_cache_writer,
                                        current_shard_rows,
                                        current_prepared_batch,
                                        finished_shards_last_flush,
                                    )
                                    finished_shards_last_flush = []
                                    current_prepared_batch = None
                                    current_shard_rows = {}

                            case _ShardFinished(shard, total_rows):
                                finished_shards_last_flush.append((shard, total_rows))
                            case _:
                                raise AssertionError(f"Unexpected message type {type(message)}")

                    # if batches_total % 1000 == 0:
                    #     print(
                    #         f"Processed {batches_total} batches: {loading_time.average()}s load,"
                    #         f" {append_time.average()}s append, {flush_time.average()}s flush blocked, "
                    #         f"{flush_amortized_time.average()}s amortized flush, "
                    #         f"{total_time.average()}s total"
                    #     )
                except StopIteration:
                    logger.info("Finished all shards")
                    break
                except Exception as e:
                    logger.exception("Error while processing batch")
                    raise e

        # force a flush
        if current_prepared_batch is not None or finished_shards_last_flush:
            if flush_thread is not None:
                flush_thread.join()
            _write_batches(
                sharded_cache_writer, current_shard_rows, current_prepared_batch, finished_shards_last_flush
            )

        sharded_cache_writer.finish()

        out = sharded_cache_writer.get_ledger()
        return out


def _concat_prepared_batches(
    current_prepared_batch: PyTree[PreparedBatch], this_prepared_batch: PyTree[PreparedBatch]
):
    return jax.tree.map(lambda *bs: PreparedBatch.concat(bs), current_prepared_batch, this_prepared_batch)


def _write_batches(writer: ShardedCacheWriter, shard_totals, batch: Optional[PyTree[PreparedBatch]], finished_shards):
    # concatenate the payloads
    if batch is not None:
        writer.write_prepared_batch(shard_totals, batch)

    for shard, total_rows in finished_shards:
        writer.finish_shard(shard, total_rows)


def _fetch_batches(batches) -> tuple[dict[str, int], list[PreparedBatch]]:
    shards_for_batches, payloads_for_batches = zip(*batches)
    payloads_for_batches = ray.get(list(payloads_for_batches))

    shard_row_totals: dict[str, int] = {}
    for shard, payload in zip(shards_for_batches, payloads_for_batches):
        shard_row_totals[shard] = shard_row_totals.get(shard, 0) + jax.tree.leaves(payload)[0].num_rows

    return shard_row_totals, payloads_for_batches


def _interleave_shards(readers: Sequence[RayPrefetchQueue], first_index: int) -> Iterator[T]:  # _Message
    """
    Interleaves the results of multiple iterators. To support resume,
    we need to be able to start from not the "first" iterator.

    Args:
        readers: A list of iterators
        first_index: The index of the first iterator to start from. We use this to support resuming.
    """

    finished: set[int] = set()
    total = 0
    while len(finished) < len(readers):
        for i in range(first_index, len(readers)):
            reader = readers[i]
            if i not in finished:
                try:
                    message = reader.get_next()
                    total += 1
                    yield message
                except StopIteration:
                    finished.add(i)
                except Exception as e:
                    logger.exception(f"Error while processing group {i}")
                    raise e

        first_index = 0

    logger.info(f"Finished all shards, got {total} batches")


def _assign_shards_to_groups(shards: Sequence[_ShardStatus], num_groups: int) -> list["_ShardGroup"]:
    """
    Assigns shards to groups in a round-robin fashion.
    """
    groups: list[list] = [[] for _ in range(num_groups)]
    for i, shard in enumerate(shards):
        groups[i % num_groups].append(shard)
    return [_ShardGroup(group) for group in groups]


def _randomize_shards(shards: Sequence[T], seed: int) -> list[T]:
    prng = random.Random(seed)
    shuffled = list(shards)
    prng.shuffle(shuffled)
    return shuffled


class _ShardGroup:
    """
    Given a group of shards and a list of statuses, implicitly concatenates the shards and reads from them.

    This class mostly exists for resuming: we want to be able to start from the last shard we were working on.
    """

    def __init__(self, group: list[_ShardStatus]):
        self.shards = group
        self.total_rows_committed, _all_finished = self._impute_total_rows_committed_and_check_invariants()

    def _impute_total_rows_committed_and_check_invariants(self):
        # we also want to ensure that we haven't started any shards until we've finished the previous ones
        total_committed = 0
        last_shard_name = None
        last_was_finished = True
        all_finished = True

        for status in self.shards:
            shard_name = status.shard_name
            if not last_was_finished and status.num_rows_committed > 0:
                raise ValueError(
                    f"Shard {shard_name} has rows committed but previous shard in group {last_shard_name} "
                    "is not finished. Something about the cache configuration has changed: either the "
                    "number/order of shards, the shard shuffle random seed, or the number of groups."
                )
            total_committed += status.num_rows_committed
            if not status.is_finished:
                all_finished = False
            last_was_finished = status.is_finished
            last_shard_name = shard_name

        return total_committed, all_finished


def _make_interleave(name: str, source: ShardedDataSource, initial_ledger: CacheLedger, processor_pool: ActorHandle):
    """
    Given a list of ShardStatus objects and sources, creates an interleaving generator
    that reads from shards and tokenizes them in parallel.

    We use ShardStatus objects to track the progress of each shard. If we're preempted, we can resume
    from the last shard we were working on. This function starts each shard at the last committed row
    and starts interleaving from the next shard (i.e. the one with the fewest rows that isn't finished).
    """
    logger.setLevel(DEFAULT_LOG_LEVEL)
    statuses = _get_shard_statuses(initial_ledger, source)

    options = initial_ledger.metadata.options

    unfinished_shards = _check_current_shard_progress(statuses)

    if not unfinished_shards:
        logger.info("All shards finished. Nothing to do.")
        return

    group_names, groups = _randomize_and_group_shards(name, options, statuses)

    logger.warning(f"Starting cache build with {len(statuses)} shards, in {len(groups)} groups")

    def _make_generator_fn(group: _ShardGroup):
        def generator():
            pylogging.basicConfig(level=DEFAULT_LOG_LEVEL, format=LOG_FORMAT)
            for message in _shard_reader_generator(source, group, options.batch_size):
                match message:
                    case _Batch():
                        # processed = ray.put(process_task(ray.get(message.payload)))
                        # processed = process_task.remote(processor_ref, message.payload)
                        processed = processor_pool.process_batch.remote(RefBox(message.payload))
                        yield dataclasses.replace(message, payload=processed)
                    case _ShardFinished():
                        yield message
                    case _:
                        raise AssertionError(f"Unexpected message type {type(message)}")

        return generator

    generator_fns = [_make_generator_fn(group) for group in groups]

    readers = [
        RayPrefetchQueue(
            fn,
            options.prefetch_per_group,
            producer_options=dict(num_cpus=0.1, name=name, scheduling_strategy="SPREAD"),
        )
        for name, fn in zip(group_names, generator_fns)
    ]

    # then figure out the first shard to start from. This is the first unfinished shard with the minimum number of rows
    first_group_to_start = min(
        range(len(groups)),
        key=lambda i: groups[i].total_rows_committed,
    )

    yield from _interleave_shards(readers, first_group_to_start)


def _mk_processor_pool(processor, min_size, max_size):
    import hashlib

    metadata_hash = hashlib.md5(str(processor.metadata).encode()).hexdigest()
    processor_pool_name = f"processor_pool::{metadata_hash}"
    processor_pool = BatchProcessorPool.options(  # type: ignore
        name=processor_pool_name, get_if_exists=True, lifetime="detached"
    ).remote(  # type: ignore
        processor, min_size, max_size
    )

    ray.get(processor_pool.ensure_max_at_least.remote(max_size))

    return processor_pool


def _check_current_shard_progress(statuses):
    unfinished_shards: list[_ShardStatus] = []
    shards_with_progress: dict[str, int] = {}
    for status in statuses:
        if not status.is_finished:
            unfinished_shards.append(status)
        if status.num_rows_committed > 0:
            shards_with_progress[status.shard_name] = status.num_rows_committed
    if unfinished_shards and shards_with_progress:
        formatted = ", ".join(f"{k}: {v}" for k, v in shards_with_progress.items())
        logger.info(f"Resuming from shards with progress: {formatted}")
    return unfinished_shards


def _randomize_and_group_shards(name, options, statuses):
    if options.shard_order_randomization_key is not None:
        seed = options.shard_order_randomization_key
        logger.info(f"Randomizing shard order with seed {seed}")
        statuses = _randomize_shards(statuses, seed)

    num_groups = min(
        options.num_shard_groups if options.num_shard_groups is not None else len(statuses), len(statuses)
    )
    if num_groups == 1:
        group_names = [f"generator::{name}::all_shards"]
    elif len(statuses) == num_groups:
        group_names = [f"generator::{name}::{status.shard_name}" for status in statuses]
    else:
        group_names = [f"generator::{name}::group_{i}" for i in range(num_groups)]

    groups = _assign_shards_to_groups(statuses, num_groups)
    return group_names, groups


def _shard_reader_generator(
    shard_source: ShardedDataSource[T], group: _ShardGroup, batch_size: int
) -> Iterator[_Message]:
    """
    Given a group of shards, implicitly concatenates the shards and reads from them.
    """
    for status in group.shards:
        if status.is_finished:
            logger.info(f"Skipping finished shard {status.shard_name}")
            continue
        start_row = status.num_rows_committed
        logger.info(f"Opening shard {status.shard_name} at row {start_row}")
        shard_iter = shard_source.open_shard_at_row(status.shard_name, start_row)

        batch = []
        batch_idxes = []
        row_idx = start_row
        for row in shard_iter:
            batch.append(row)
            batch_idxes.append(row_idx)
            row_idx += 1

            if len(batch) == batch_size:
                yield _Batch(status.shard_name, batch_idxes, ray.put(batch))
                batch = []
                batch_idxes = []

        if len(batch) > 0:
            yield _Batch(status.shard_name, batch_idxes, ray.put(batch))

        logger.info(f"Finished generating shard {status.shard_name} with {row_idx} rows")
        yield _ShardFinished(status.shard_name, row_idx)


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


def _get_shard_statuses(ledger: CacheLedger, source: ShardedDataSource):
    return [
        _ShardStatus(name, ledger.shard_rows.get(name, 0), name in ledger.finished_shards)
        for name in source.shard_names
    ]
