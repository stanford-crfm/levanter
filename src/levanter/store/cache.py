import asyncio
import concurrent
import dataclasses
import functools
import heapq
import logging as pylogging
import os
import threading
import time
from asyncio import InvalidStateError
from concurrent.futures import Future as threading_Future
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Sequence, TypeVar, Union

import fsspec.core
from ray.remote_function import RemoteFunction

from levanter.store._prefetch_actor import PrefetchIteratorActor
import pyarrow as pa
import ray
from dataclasses_json import dataclass_json
from fsspec import AbstractFileSystem
from ray.actor import ActorHandle

from levanter.data.dataset import AsyncDataset

from ..data._preprocessor import BatchProcessor, BatchResult, dict_from_record_batch
from ..data._queue import (
    PriorityWorkItem,
    PriorityWorkTaskGroup,
    PriorityWorkTaskGroupSpec,
    WorkQueueDispatcherActor,
    _BatchProcessorQueue,
)
from ..data.metrics_monitor import InProgressCacheMetrics, LoggerMetricsMonitor, MetricsMonitor
from ..data.sharded_datasource import ShardedDataSource
from ..utils.ray_utils import (
    ExceptionInfo,
    RefBox,
    SnitchRecipient,
    current_actor_handle,
    log_failures_to,
    map_ref,
    ser_exc_info,
)
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


@dataclass
class ShardStatus:
    shard_name: str
    num_rows_committed: int
    is_finished: bool  # todo, need to add this to the ledger



def _interleave_shards(shard_names: Sequence[str],
    readers: Sequence[ActorHandle],  # PrefetchIteratorActor
    first_shard_index: int):
    """
    Interleaves the results of multiple iterators. To support resume,
    we need to be able to start from not the "first" iterator.
    """
    raise NotImplementedError("This is not implemented yetXXX")
    if len(shard_names) != len(readers):
        raise ValueError("shard_names and readers must have the same length")

    if len(shard_names) == 0:
        raise ValueError("No shards to interleave")
        logger.info("No shards to interleave")
        return

    iterators = list(readers)
    finished = set()
    i = first_shard_index
    while len(finished) < len(shard_names):
        raise NotImplementedError("This is not implemented yetXXX")
        while i < len(iterators):
            shard_name = shard_names[i]
            it = iterators[i]
            if shard_name not in finished:
                try:
                    # TODO: push shard_name into the actual iterator
                    yield map_ref(lambda x: (shard_name, x), it.get_next())
                except StopIteration:
                    logger.info(f"Finished shard {shard_name}")
                    finished.add(shard_name)

            i += 1
            if i == len(iterators):
                i = 0

    logger.info("Finished all shards")


def _make_interleave_for_shards(
    source: ShardedDataSource,
    statuses: Sequence[ShardStatus],
    batch_size: int):
    """
    Given a list of ShardStatus objects and sources, creates an interleave
    """
    unfinished_shards: list[ShardStatus] = []
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
    print("Unfinished shards", unfinished_shards)

    raise NotImplementedError("This is not implemented yet")

    generator_fns = [lambda: _shard_reader_generator(source, status.shard_name, status.num_rows_committed, batch_size)
                        for status in unfinished_shards]

    readers = [PrefetchIteratorActor.remote(generator_fn, max_queue_size=32) for generator_fn in generator_fns]

    # then figure out the first shard to start from. This is the first unfinished shard with the minimum number of rows
    min_unfinished = min(range(len(unfinished_shards)), key=lambda x: unfinished_shards[x].num_rows_committed)

    yield from _interleave_shards([status.shard_name for status in unfinished_shards], readers, min_unfinished)


def _mk_process_task(processor: BatchProcessor[T, U]) -> RemoteFunction:
    @ray.remote(num_cpus=processor.num_cpus, num_gpus=processor.num_gpus, resources=processor.resources)
    def process_task(desc, batch: List[T]):
        # pylogging.basicConfig(level=DEFAULT_LOG_LEVEL, format=LOG_FORMAT)
        logger.debug(f"Processing batch {desc}")
        try:
            result = processor(batch)
            logger.debug(f"Finished processing batch {desc}")
            return result
        except Exception as e:
            logger.exception(f"Error while processing batch {desc}")
            raise e
        finally:
            pass

    return process_task


def _process_batches(
    processor: BatchProcessor,
    batch_iter: ActorHandle  # PrefetchIteratorActor[(str, batch)]
    ):
    """
    Processes batches of data from a PrefetchIteratorActor.
    This is a generator that yields refs representing the processed batches.
    """

    process_fn = _mk_process_task(processor)


    while True:
        try:
            shard_name, batch = ray.get(batch_iter.get_next.remote())
            yield map_ref(lambda x: (shard_name, x), process_fn.remote(shard_name, batch))
            raise NotImplementedError("This is not implemented yet")
        except StopIteration:
            break


@ray.remote(num_cpus=1)
def _core_writer_task(
    parent,
    cache_dir,
    processor,
    batch_iter: ActorHandle,  # PrefetchIteratorActor[(str, batch)]
    cache_config: Optional[Dict[str, Any]],
    force_flush: bool,
):
    logger.setLevel(DEFAULT_LOG_LEVEL)
    logger.info("Starting writer task")
    with log_failures_to(parent):
        sharded_cache_writer = ShardedCacheWriter(cache_dir, processor.output_exemplar, cache_config=cache_config)
        last_update_time = time.time()

        i = 0

        while True:
            try:
                shard, batch = ray.get(batch_iter.get_next.remote())
                logger.info(f"Writing batch {i} for {shard}")
                sharded_cache_writer.write_batch(shard, batch)
                if force_flush:
                    sharded_cache_writer.flush()
                i += 1
            except StopIteration:
                break

            if sharded_cache_writer.last_write_time - last_update_time > 10.0:
                logger.info(f"Updating ledger")
                parent._notify_updated_ledger.remote(sharded_cache_writer.ledger)
                last_update_time = sharded_cache_writer.last_write_time

        logger.warning(f"Finished writing {i} batches")
        sharded_cache_writer.finish()
        parent._notify_updated_ledger.remote(sharded_cache_writer.ledger)
        return sharded_cache_writer.ledger


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
        initial_ledger = _load_or_initialize_ledger(os.path.join(cache_dir, LEDGER_FILE_NAME))
        statuses = [ShardStatus(shard_name, num_rows, False) for shard_name, num_rows in initial_ledger.shard_rows.items()]
        batch_size = processor.batch_size

        interleave = PrefetchIteratorActor.remote(lambda: _make_interleave_for_shards(source, statuses, batch_size), max_queue_size=64)
        process = PrefetchIteratorActor.remote(lambda: _process_batches(processor, interleave), max_queue_size=64)
        writer_task = _core_writer_task.remote(parent, cache_dir, processor, process, cache_config, force_flush=force_flush)

    return writer_task



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
        )

        if exc_type is None:
            _serialize_json_and_commit(os.path.join(self.cache_dir, LEDGER_FILE_NAME), ledger)
            logger.info(f"Cache ledger written to {self.cache_dir}")
            self._is_closed = True

    def result(self) -> "TreeCache":
        if not self._is_closed:
            raise RuntimeError("Cannot get result until TreeCacheWriter is closed")
        return TreeCache.load(self.cache_dir, self._exemplar)

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
        cache_dir: str,
        exemplar: T,
        cache_config: Optional[Dict[str, Any]] = None,
    ):
        self.cache_dir = cache_dir
        self.cache_config = cache_config
        self._exemplar = exemplar

        self._ledger = _load_or_initialize_ledger(os.path.join(cache_dir, LEDGER_FILE_NAME))

        self._tree_store = TreeStore.open(exemplar, self.cache_dir, mode="a")  # type: ignore
        self._tree_store.trim_to_size(self._ledger.total_num_rows)

        # set up the writer thread
        if not self._ledger.is_finished:
            self._stop_loop = threading.Event()
            self._actual_writer_thread = threading.Thread(target=self._write_loop, daemon=True)

            self._last_write_time = time.time()
            self._items_ready_to_write: list = []

            self._actual_writer_thread.start()

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
        while True:
            try:
                self._stop_loop.wait(1)
                if self._stop_loop.is_set():
                    break
            except TimeoutError:
                pass
            self._attempt_to_write_batches()
            if self._ledger.is_finished:
                break

    def flush(self):
        self._attempt_to_write_batches()

    def finish(self):
        self._stop_loop.set()
        self._actual_writer_thread.join()
        self.flush()

        # if successful, write the ledger
        self._ledger.is_finished = True
        _serialize_json_and_commit(os.path.join(self.cache_dir, LEDGER_FILE_NAME), self._ledger)

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
            _serialize_json_and_commit(os.path.join(self.cache_dir, LEDGER_FILE_NAME), self._ledger)

    def _write_available_batches(self):
        if len(self._items_ready_to_write) == 0:
            return {}

        to_write = []
        written_by_shard = {}
        for shard, batch in self._items_ready_to_write:
            to_write.extend(batch)
            written_by_shard[shard] = written_by_shard.get(shard, 0) + len(batch)

        self._tree_store.extend(to_write)
        self._items_ready_to_write = []
        return written_by_shard



def _load_or_initialize_ledger(path):
    try:
        with fsspec.open(path, "r") as file:
            return CacheLedger.from_json(file.read())
    except FileNotFoundError:
        return CacheLedger(0, {})


def _to_list_of_dicts(batch: dict) -> List[dict]:
    """
    Convert a batch of dictionaries to a list of dictionaries, suitable for writing to a cache.
    """
    keys = list(batch.keys())
    values = list(batch.values())
    num_rows = len(values[0])
    return [{key: values[i][j] for i, key in enumerate(keys)} for j in range(num_rows)]


def _canonicalize_batch(batch: Union[dict, List[dict]]) -> List[dict]:
    if isinstance(batch, pa.RecordBatch):
        batch = dict_from_record_batch(batch)

    if isinstance(batch, dict):
        return _to_list_of_dicts(batch)
    else:
        return batch


# thinking through the design of the cache system

# we decided to use Ray, which was maybe a mistake, but here we are.
# Ray doesn't like it when the number of actors gets too large, so we can't have one actor per shard.
# we have N nodes and K shards.

# at a high level, we have 3 steps:
# 1. read batches from the shard source
# 2. process batches
# 3. write batches to the cache for that shard

# The difficulty is that we want parallelism, and we want to control the order of the written data.
# Reading batches requires CPU and network.
# ==> This means we should limit the number of shard groups to roughly the number of nodes, maybe times 2.
# We ideally want to read from shards roughly evenly (at least within a group of shards)


def _shard_reader_generator(shard_source: ShardedDataSource[T], shard_name: str, start_row: int, batch_size: int):
    shard_iter = shard_source.open_shard_at_row(shard_name, start_row)
    batch = []
    raise NotImplementedError("This is not implemented yet")
    for row in shard_iter:
        batch.append(row)

        if len(batch) == batch_size:
            yield batch
            batch = []

    if len(batch) > 0:
        yield batch


@dataclass
class ShardGroupToBeProcessed(PriorityWorkTaskGroupSpec):
    name: str
    builder_ref: ray.actor.ActorHandle  # _TreeStoreCacheBuilder
    writer: ray.actor.ActorHandle  # _GroupedShardWriter
    shard_source: ShardedDataSource
    shard_names: Sequence[str]
    priority_fn: Callable[[int, int], float]
    processor_actor: ray.actor.ActorHandle  # BatchProcessorQueue
    batch_size: int
    group_id: int

    def build(self) -> "PriorityWorkTaskGroup":
        return ShardGroupTaskGroup(self)


class ShardGroupTaskGroup(PriorityWorkTaskGroup):
    def __init__(self, spec: ShardGroupToBeProcessed):
        self.spec: ShardGroupToBeProcessed = spec
        self.logger = pylogging.getLogger(f"shard_reader.{spec.group_id}.{spec.name}")

        current_shard_status: dict[str, ShardStatus] = {}
        for shard_name in self.spec.shard_names:
            try:
                current_shard_status[shard_name] = ray.get(self.spec.writer.get_shard_status.remote(shard_name))
            except Exception as e:
                self.spec.builder_ref.shard_failed.remote(shard_name, ser_exc_info())
                raise e

        batch_size = self.spec.batch_size

        self._items: list[PriorityWorkItem] = []

        for shard_name in self.spec.shard_names:
            try:
                status = current_shard_status[shard_name]
                if status.is_finished:
                    self.logger.info(f"Shard {shard_name} already finished. Skipping.")
                    continue

                reader = _shard_reader_generator(
                    self.spec.shard_source, shard_name, status.num_rows_committed, batch_size
                )

                task_name = f"shard_reader.{self.spec.name}.{shard_name}"

                batch_idx = status.num_rows_committed // batch_size

                shard_idx = self.spec.shard_source.shard_names.index(shard_name)
                item = ShardReaderItem(
                    self,
                    task_name,
                    shard_name,
                    shard_idx,
                    batch_idx=batch_idx,
                    reader=reader,
                    current_row=status.num_rows_committed,
                )

                heapq.heappush(self._items, item)
            except Exception as e:
                self.logger.exception(f"Error while initializing shard {shard_name}")
                self.spec.writer[shard_name].shard_failed.remote(ser_exc_info())
                raise e

    @property
    def name(self):
        return self.spec.name

    def items(self) -> Sequence["PriorityWorkItem"]:
        return self._items


# NB This class is stateful
@dataclass
class ShardReaderItem(PriorityWorkItem):
    """
    Each time execute is called, this class reads a batch of examples from the shard
    and dispatches them to the processor.
    """

    group: ShardGroupTaskGroup
    name: str
    shard_name: str
    shard_idx: int
    batch_idx: int
    reader: Iterator[list]
    current_row: int = 0

    @property
    def priority(self):
        return self.group.spec.priority_fn(self.shard_idx, self.batch_idx)

    @property
    def spec(self):
        return self.group.spec

    def execute(self) -> tuple[bool, Optional[ray.ObjectRef]]:
        writer = self.spec.writer
        write_finished_ref = None

        self.group.logger.debug(f"Reading one batch of shard {self.shard_name}: {self.batch_idx}")

        try:
            batch = next(self.reader, None)
            exhausted_shard = batch is None or (len(batch) < self.spec.batch_size)

            if batch:
                priority = self.spec.priority_fn(self.shard_idx, self.batch_idx)
                try:
                    batch_result_ref = ray.get(
                        self.spec.processor_actor.submit.remote(
                            priority=priority,
                            desc=f"{self.shard_name}.{self.batch_idx}",
                            batch=RefBox(ray.put(batch)),
                        )
                    )
                    logger.debug(f"Got batch result: {batch_result_ref}")
                    write_finished_ref = writer.batch_finished.remote(
                        self.shard_name, self.batch_idx, RefBox(batch_result_ref)
                    )
                    self.batch_idx += 1
                    self.current_row += len(batch)
                except Exception as e:
                    self.group.logger.exception(f"Error while processing batch {self.batch_idx}")
                    # fire and forget
                    writer.shard_failed.remote(self.shard_name, self.batch_idx, ser_exc_info())
                    raise e

            if exhausted_shard:
                logger.info(f"Shard {self.shard_name} exhausted. Expecting {self.current_row} rows.")
                writer.shard_finished_reading.remote(self.shard_name, self.current_row)

            self.group.logger.debug(f"Finished reading one batch of shard {self.shard_name}: {self.batch_idx}")

            return exhausted_shard, write_finished_ref
        except Exception as e:  # noqa
            self.group.logger.exception(f"Error while processing shard {self.shard_name}")
            # fire and forget
            writer.shard_failed.remote(self.shard_name, self.batch_idx, ser_exc_info())
            raise e


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


def _load_cache_ledger(cache_dir) -> CacheLedger:
    try:
        ledger_path = os.path.join(cache_dir, LEDGER_FILE_NAME)
        logger.debug(f"Attempting to load cache ledger from {ledger_path}")
        with fsspec.open(ledger_path) as file:
            cache_ledger = CacheLedger.from_json(file.read())  # type: ignore
        return cache_ledger
    except FileNotFoundError:
        raise FileNotFoundError(f"Cache ledger not found at {ledger_path}")


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

        self._ledger = _load_or_initialize_ledger(os.path.join(cache_dir, LEDGER_FILE_NAME))

        if self._ledger.is_finished:
            self._finished_promise.set_result(None)

        path_for_name = os.path.join(*self._cache_dir.split("/")[-2:])
        name = f"broker::{path_for_name}"
        self.logger = pylogging.getLogger(f"{name}")

        if self._ledger.is_finished:
            self.logger.info("Cache already finished. Nothing to do.")
            return
        self._cache_writer = _preprocess_task_chain(current_actor_handle(), cache_dir, name, source, processor, cache_config, force_flush)

    def _notify_updated_ledger(self, ledger: CacheLedger):
        self._ledger = ledger
        # assert not self._ledger.is_finished
        if self._ledger.is_finished:
            self._finalize()

        self._do_notify()

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

    def _do_notify(self):
        async def _do_notify_async():
            async with self._updated_ledger_condition:
                self._updated_ledger_condition.notify_all()

        asyncio.create_task(_do_notify_async())

    def current_ledger(self):
        return self._ledger

    def _finalize(self):
        logger.info(f"Finalizing cache {self._cache_dir}...")

        self._ledger.is_finished = True
        self._finished_promise.set_result(None)

        self._cache_writer = None


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


class TreeCache(AsyncDataset[T_co]):
    ledger: Optional[CacheLedger]
    _broker: Optional[ActorHandle]
    # monitor_thread waits for new metrics and also periodically reloads the cache
    _monitor_thread: Optional[threading.Thread]
    _metrics_monitors: List[MetricsMonitor]

    def __init__(
        self,
        cache_dir: str,
        exemplar: T_co,
        ledger: Optional[CacheLedger],
        _broker,  # handle of _TreeStoreCacheBuilder
    ):
        self.cache_dir = cache_dir
        self.ledger = ledger
        self._was_already_finished = ledger is not None and ledger.is_finished
        self._broker = _broker
        self._exemplar = exemplar

        self._metrics_monitors = []
        name = os.path.join(*cache_dir.split("/")[-2:])
        self.logger = pylogging.getLogger(f"TreeCache.{name}")
        self._store_future: threading_Future[TreeStore] = threading_Future()
        self._stop = False
        # assert _broker is None

        if self._broker is not None:
            self._monitor_thread = threading.Thread(target=self._monitor_metrics, daemon=True)
            self._monitor_thread.start()
        else:
            self._attempt_to_load_store()
            assert self._store_future.done()

    @property
    def store(self) -> TreeStore[T_co]:
        return self._store_future.result()

    async def store_async(self) -> TreeStore[T_co]:
        if self._broker is not None:
            return await asyncio.wrap_future(self._store_future)
        else:
            return self.store

    async def async_len(self) -> int:
        if self._broker is not None:
            self.await_finished()

        return len(await self.store_async())

    def __len__(self):
        self.await_finished()

        return len(self.store)

    async def final_length_is_known(self) -> bool:
        if self._broker is not None:
            return await self._broker.is_finished.remote()

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
        if self._broker is not None:
            while needed_len > await self.current_len():
                new_ledger: CacheLedger = await self._broker.updated_ledger.remote()

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
        if self._broker is not None:
            while needed_len > len(self.store):
                cur_time = time.time()
                if cur_time > t_max:
                    raise TimeoutError(f"Timed out waiting for cache to reach {needed_len}")
                try:
                    new_ledger: CacheLedger = ray.get(
                        self._broker.updated_ledger.remote(), timeout=max(t_max - cur_time, 10)
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
    def load(cache_dir: str, exemplar: T) -> "TreeCache":
        """Loads a cache from disk or an object store. Raises FileNotFoundError if the cache doesn't exist"""
        logger.info(f"Loading cache from {cache_dir}")
        ledger = _load_cache_ledger(cache_dir)
        if not ledger.is_finished:
            raise FileNotFoundError(f"Cache at {cache_dir} is not finished. Use build_or_load to build it.")
        return TreeCache(cache_dir, exemplar, ledger, None)

    @staticmethod
    def build_or_load(
        cache_dir: str,
        shard_source: ShardedDataSource[T],
        processor: BatchProcessor[T, U],
        cache_config: Optional[Dict[str, Any]] = None,
        force_flush: bool = False,
    ) -> "TreeCache[U]":
        try:
            return TreeCache.load(cache_dir, processor.output_exemplar)
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
        if self._broker is None:
            return ray.remote(num_cpus=0)(lambda: None).remote()
        else:
            return self._broker.finished_sentinel.remote()

    @property
    def is_finished(self):
        if self._broker is None:
            return True
        else:
            return ray.get(self._broker.is_finished.remote())

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
        if self._broker is None:
            return
        x = ray.get(self.finished_sentinel(), timeout=timeout)
        self._attempt_to_load_store()
        return x

    async def finished(self):
        if self._broker is None:
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
            logger.error(f"Cache at {self.cache_dir} not found.")
            assert self._broker is not None
            ledger = ray.get(self._broker.current_ledger.remote())
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
        if self._broker is None:
            logger.warning("Cannot attach metrics monitor to finished cache.")
            # TODO: decide what to do about attaching if the cache is already finished
            # maybe get the final metrics?
            return

        self._metrics_monitors.append(monitor)

    def _monitor_metrics(self):
        while not self._stop:
            try:
                try:
                    ledger = ray.get(self._broker.updated_ledger.remote(), timeout=10.0)
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


def _ledger_to_metrics(ledger: CacheLedger) -> InProgressCacheMetrics:
    return InProgressCacheMetrics(
        rows_finished=ledger.total_num_rows,
        is_finished=ledger.is_finished,
        # shard_rows=ledger.shard_rows,
        # finished_shards=ledger.finished_shards,
        field_counts=ledger.field_counts,
    )


class GroupRoundRobinBuffer(Generic[T]):
    """
    A buffer that holds items from multiple groups and returns them in a round-robin fashion.
    The groups need not have the same number of items. If a group is exhausted, it is removed from the rotation.
    """

    def __init__(self, groups: Sequence[str]):
        self.groups = groups
        self._current_group = 0
        self.buffers: dict[str, list[tuple[int, T]]] = {group: [] for group in groups}
        self._remaining_groups = set(groups)
        self._totals_written: dict[str, int] = {group: 0 for group in groups}
        self._totals_expected: dict[str, Optional[int]] = {group: None for group in groups}

    def __len__(self):
        return sum(len(buffer) for buffer in self.buffers.values())

    def append_to_group(self, group: str, item_serial: int, item: T):
        if group not in self.groups:
            raise ValueError(f"Group {group} not in {self.groups}")

        if group not in self._remaining_groups:
            raise ValueError(f"Group {group} already finished")

        logger.debug(f"Appending item {item_serial} to {group}")

        heapq.heappush(self.buffers[group], (item_serial, item))

    def group_total_known(self, group: str, total: int):
        if group not in self.groups:
            raise ValueError(f"Group {group} not in {self.groups}")

        if group not in self._remaining_groups:
            raise ValueError(f"Group {group} already finished: {total} vs {self._totals_expected[group]}")

        self._totals_expected[group] = total

        if self._totals_written[group] == total:
            assert len(self.buffers[group]) == 0
            self._remaining_groups.remove(group)
        elif self._totals_written[group] > total:
            raise ValueError(f"Group {group} has written more than expected: {self._totals_written[group]} > {total}")

    def is_finished(self):
        return len(self._remaining_groups) == 0

    def pop(self) -> Optional[tuple[str, T]]:
        group = self._next_group_to_read_from()
        if group is None:
            return None

        if len(self.buffers[group]) == 0:
            return None

        cur_serial, item = self.buffers[group][0]

        # logger.debug(
        #     f"group: {group}, cur_serial: {cur_serial}, totals_written: {self._totals_written[group]},"
        #     f" totals_expected: {self._totals_expected.get(group)}"
        # )

        if cur_serial > self._totals_written[group]:
            return None
        elif cur_serial < self._totals_written[group]:
            raise ValueError(f"Duplicate serial {cur_serial} for group {group}")

        heapq.heappop(self.buffers[group])
        logger.debug(f"Read item {cur_serial} from {group}")

        self._totals_written[group] += 1

        if self._totals_written[group] == self._totals_expected[group]:
            assert len(self.buffers[group]) == 0
            assert group in self._remaining_groups
            self._remaining_groups.remove(group)

        self._current_group = (self._current_group + 1) % len(self.groups)

        return group, item

    def drain(self) -> Iterator[tuple[str, T]]:
        while True:
            item = self.pop()
            if item is None:
                break
            yield item

    def _next_group_to_read_from(self):
        """
        Returns the next group to read from. This is always the group with the least that is not finished.
        """
        if len(self._remaining_groups) == 0:
            return None

        # careful: this is only correct if self._current_group is correct. whenever we fast forward, we have to
        # recompute it
        while True:
            group = self.groups[self._current_group]
            if group not in self._remaining_groups:
                assert self._totals_written[group] == self._totals_expected[group]
                assert len(self.buffers[group]) == 0
                self._current_group = (self._current_group + 1) % len(self.groups)
            else:
                break
        return group

    def fast_forward(self, group, num_rows):
        """
        Fast forwards the buffer for a group to a certain number of rows. This sets the "next" item to be the
        num_rows-th item.
        """
        if group not in self.groups:
            raise ValueError(f"Group {group} not in {self.groups}")

        if self._totals_written[group] != 0:
            raise ValueError(f"Group {group} already written to: {self._totals_written[group]}")

        self._totals_written[group] = num_rows

        self._fix_current_group()

    def _fix_current_group(self):
        # This is always the minimum total written group that is not finished
        self._current_group = 0
        min_total = None

        for i, group in enumerate(self.groups):
            if group not in self._remaining_groups:
                continue
            total = self._totals_written[group]
            if min_total is None or total < min_total:
                min_total = total
                self._current_group = i

    def next_missing_item_index(self):
        """
        Returns the index of the next item that is not in the buffer
        (i.e. what's stopping us from yielding the next item).
        """
        if len(self._remaining_groups) == 0:
            return None

        group = self.groups[self._current_group]
        if group not in self._remaining_groups:
            self._fix_current_group()
            return self.next_missing_item_index()

        if len(self.buffers[group]) == 0:
            return self._totals_written[group]

        cur_serial, _ = self.buffers[group][0]

        if cur_serial > self._totals_written[group]:
            return self._totals_written[group]
        elif cur_serial < self._totals_written[group]:
            raise ValueError(f"Duplicate serial {cur_serial} for group {group}")

        return None


def div_round_up(x, y):
    return (x + y - 1) // y
