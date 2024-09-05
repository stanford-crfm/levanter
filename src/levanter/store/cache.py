import asyncio
import concurrent
import dataclasses
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
MIN_ITEMS_TO_WRITE = 8192
MAX_TIME_BETWEEN_WRITES = 100.0


def build_or_load_cache(
    cache_dir: str,
    input_shards: ShardedDataSource[T],
    processor: BatchProcessor[T, U],
    await_finished: bool = True,
    monitors: Optional[Sequence["MetricsMonitor"]] = None,
    cache_config: Optional[Dict[str, Any]] = None,
    items_per_write: int = MIN_ITEMS_TO_WRITE,
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

        items_per_write: The number of items to write to the cache at a time. This is a performance tuning parameter,
            and you probably don't need to change it. We mostly use it for testing.

    Returns:
       (TreeCache) A TreeCache object that can be used to read the cache.

    """
    # first see if we need to do anything
    cache = TreeCache.build_or_load(
        cache_dir=cache_dir,
        shard_source=input_shards,
        processor=processor,
        cache_config=cache_config,
        items_per_write=items_per_write,
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


def _load_or_initialize_ledger(path):
    try:
        with fsspec.open(path, "r") as file:
            return CacheLedger.from_json(file.read())
    except FileNotFoundError:
        return CacheLedger(0, {})


@ray.remote(num_cpus=0.5)  # type: ignore
class _OrderedCacheWriter:
    """
    This cache writer receives examples from some number of shards (generally out of order) and writes them to the store
    in a defined round-robin order. It also keeps track of the metadata for each shard.

    Once a shard finishes sending batches, it notifies this writer, which then updates the metadata and writes it to disk.
    """

    def __init__(
        self,
        parent,
        name,
        exemplar,
        batch_size,
        cache_dir: str,
        shards: Sequence[str],
        min_items_to_write=MIN_ITEMS_TO_WRITE,
    ):
        pylogging.basicConfig(level=DEFAULT_LOG_LEVEL, format=LOG_FORMAT)
        with log_failures_to(parent):
            self._parent = parent
            self.cache_dir = cache_dir
            self.shards = shards
            self.batch_size = batch_size
            self._min_items_to_write = min_items_to_write
            self._failed = False
            self._logger = pylogging.getLogger(name)

            # these are batches that we've received but haven't ordered them for writing yet
            self._batch_queue = GroupRoundRobinBuffer(shards)  # type: ignore
            self._total_queue_length = 0
            self._was_overwhelmed = False  # whether the queue has gotten too big
            # writes are very slow (~2s) so we want to batch them up
            self._ordered_but_unwritten_items: list = []
            self._batches_in_next_write_by_shard: dict[str, int] = {shard: 0 for shard in shards}
            # we also want to write every so often
            self._last_write_time = time.time()

            self._ledger = _load_or_initialize_ledger(os.path.join(cache_dir, LEDGER_FILE_NAME))
            self._expected_num_rows: dict[str, Optional[int]] = {shard: None for shard in shards}

            self._tree_store = TreeStore.open(exemplar, self.cache_dir, mode="a")
            # careful: trim the store to the total number of rows in the cache that we've committed to
            self._tree_store.trim_to_size(self._ledger.total_num_rows)
            # we also have to tell the queue how many rows for each shard we've already written
            for shard, num_rows in self._ledger.shard_rows.items():
                if num_rows > 0:
                    self._logger.info(f"Already written {num_rows} rows for shard {shard}")

                # careful: this is in terms of batch size
                # Have to round up to the nearest batch size
                self._batch_queue.fast_forward(shard, div_round_up(num_rows, self.batch_size))
                if shard in self._ledger.finished_shards:
                    self._expected_num_rows[shard] = num_rows
                    self._batch_queue.group_total_known(shard, div_round_up(num_rows, self.batch_size))

            # double check that we're not finished by committing the ledger
            self._attempt_to_write_batches()

    def batch_finished(self, shard_name: str, shard_batch_idx: int, batch_result_box):
        with log_failures_to(self._parent):
            if self._failed:
                self._logger.warning("Received batch after failure. Ignoring.")
                return

            if isinstance(batch_result_box, RefBox):
                batch_result = ray.get(batch_result_box.ref)
            else:
                batch_result = batch_result_box

            # we need to keep track of the order of the batches so that we can write them out in order
            self._total_queue_length += len(batch_result)
            self._batch_queue.append_to_group(shard_name, shard_batch_idx, batch_result)
            self._attempt_to_write_batches()
            next_missing_item = self._batch_queue.next_missing_item_index()

            overwhelmed = self.is_overwhelmed()
            if overwhelmed:
                if not self._was_overwhelmed:
                    self._logger.warning(f"Writer queue is getting long ({self._total_queue_length}).")
                self._parent.signal_backpressure.remote(next_missing_item)
            elif self._was_overwhelmed:
                self._logger.info(f"Writer queue is no longer overwhelmed ({self._total_queue_length}).")
                self._parent.signal_backpressure.remote(None)

            self._was_overwhelmed = overwhelmed

    def shard_failed(self, shard_name: str, batch_id: int, exc_info: ExceptionInfo):
        with log_failures_to(self._parent):
            self._failed = True
            logger.error(f"Shard {shard_name} failed at batch {batch_id}", exc_info=exc_info.restore())
            self._parent.shard_failed.remote(shard_name, exc_info)

    def shard_finished_reading(self, shard_name: str, expected_num_rows: int):
        with log_failures_to(self._parent):
            # careful: this is in terms of batch size
            self._batch_queue.group_total_known(shard_name, div_round_up(expected_num_rows, self.batch_size))
            self._expected_num_rows[shard_name] = expected_num_rows
            logger.debug(
                f"Attempting to write batches because {shard_name} finished reading with {expected_num_rows} batches."
            )
            self._attempt_to_write_batches()

    def get_shard_status(self, shard_name: str):
        with log_failures_to(self._parent):
            rows = self._ledger.shard_rows.get(shard_name, 0)
            is_finished = shard_name in self._ledger.finished_shards
            return ShardStatus(shard_name, rows, is_finished)

    def get_ledger(self):
        return self._ledger

    def _attempt_to_write_batches(self):
        if self._ledger.is_finished:
            raise RuntimeError("Trying to write batches after cache is finished")

        if self._failed:
            logger.warning("Not writing batches because of failure.")
            return

        self._dequeue_ready_batches()
        updated_shards = self._write_available_batches()

        logger.debug(f"Updated shards: {updated_shards}")

        need_to_commit = len(updated_shards) > 0
        total_rows = self._ledger.total_num_rows + sum(updated_shards.values())

        for shard, num_rows in updated_shards.items():
            self._ledger.shard_rows[shard] = self._ledger.shard_rows.get(shard, 0) + num_rows

        futures_to_await_shards, need_to_commit_for_shards = self._check_for_finished_shards()

        need_to_commit = need_to_commit or need_to_commit_for_shards

        futures_to_await = []
        if need_to_commit:
            self._ledger.total_num_rows = total_rows
            _serialize_json_and_commit(os.path.join(self.cache_dir, LEDGER_FILE_NAME), self._ledger)

            futures_to_await.append(self._parent._updated_ledger.remote(self._ledger))

            if self._ledger.is_finished:
                f = self._parent._finalize.remote()
                futures_to_await.append(f)

        ray.wait(futures_to_await + futures_to_await_shards)

    def _dequeue_ready_batches(self):
        for shard, batch in self._batch_queue.drain():
            logger.debug(f"Writing batch for {shard}")
            batch = _canonicalize_batch(batch)
            self._total_queue_length -= len(batch)
            self._ordered_but_unwritten_items.extend(batch)
            self._batches_in_next_write_by_shard[shard] = self._batches_in_next_write_by_shard.get(shard, 0) + len(
                batch
            )

    def _write_available_batches(self):
        if len(self._ordered_but_unwritten_items) == 0:
            return {}

        any_shard_finished_reading = any(num_rows is not None for num_rows in self._expected_num_rows.values())

        if (
            len(self._ordered_but_unwritten_items) >= self._min_items_to_write
            or (time.time() - self._last_write_time > MAX_TIME_BETWEEN_WRITES)
            or any_shard_finished_reading
        ):
            time_in = time.time()
            self._tree_store.extend(self._ordered_but_unwritten_items)
            time_out = time.time()
            logger.debug(f"Wrote {len(self._ordered_but_unwritten_items)} rows in {time_out - time_in:.2f} seconds")
            self._ordered_but_unwritten_items = []

            written_by_shard = self._batches_in_next_write_by_shard
            self._batches_in_next_write_by_shard = {}
            self._last_write_time = time.time()
            return written_by_shard
        else:
            return {}

    def _check_for_finished_shards(self):
        futures_to_await_shards = []
        need_to_commit_for_shards = False
        for shard, expected_rows in self._expected_num_rows.items():
            if expected_rows is None:
                continue

            current_rows = self._ledger.shard_rows.get(shard, 0)
            if current_rows == expected_rows:
                if shard not in self._ledger.finished_shards:
                    logger.info(f"Shard {shard} finished.")
                    self._ledger.finished_shards.append(shard)
                    futures_to_await_shards.append(self._parent.shard_finished.remote(shard))
                    need_to_commit_for_shards = True
            elif current_rows > expected_rows:
                raise ValueError(f"Shard {shard} has more rows than expected: {current_rows} > {expected_rows}")

        if len(self._ledger.finished_shards) == len(self.shards) and set(self._ledger.finished_shards) == set(
            self.shards
        ):
            self._ledger.is_finished = True
            need_to_commit_for_shards = True
        return futures_to_await_shards, need_to_commit_for_shards

    def is_overwhelmed(self) -> bool:
        max_queue_size = self._min_items_to_write * 3
        return self._total_queue_length > max_queue_size


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
        min_items_to_write: int,
    ):
        pylogging.basicConfig(level=DEFAULT_LOG_LEVEL, format=LOG_FORMAT)
        self.logger = pylogging.getLogger(f"{__name__}.{name}")
        self.source = source
        self._cache_dir = cache_dir
        # self._metrics = InProgressCacheMetrics()
        self._updated_ledger_condition = asyncio.Condition()
        self._ledger = CacheLedger(0, {})
        self.shards_in_progress: set[str] = set()
        exemplar = processor.output_exemplar

        self._finished_promise: asyncio.Future[None] = asyncio.Future()
        # used to subscribe to metrics updates
        self._cache_config = cache_config
        path_for_name = os.path.join(*self._cache_dir.split("/")[-2:])
        name = f"broker::{path_for_name}"
        self.logger = pylogging.getLogger(f"{name}")
        self._cache_writer: Optional[ActorHandle] = _OrderedCacheWriter.remote(  # type: ignore
            current_actor_handle(),
            f"writer::{path_for_name}",
            exemplar,
            processor.batch_size,
            cache_dir,
            source.shard_names,
            min_items_to_write,
        )

        try:
            cache_ledger = _load_cache_ledger(self._cache_dir)
            self._ledger = cache_ledger
        except FileNotFoundError:
            pass

        if self._ledger.is_finished:
            self._finished_promise.set_result(None)
        self._start_workers(cache_dir, name, processor, source)

    def _start_workers(self, cache_dir, name, processor, source):
        if len(source.shard_names) == 0:
            self.logger.warning("No shards to index?!?")
            self._finalize()
        else:
            self.logger.debug(f"Starting cache build for {source.shard_names}")
            self.logger.info(f"Starting cache build for {len(source.shard_names)} shards")

            self_ref = current_actor_handle()

            self._shard_writers = []
            self._shard_readers = []
            self._processor_actors = []

            for shard_name in source.shard_names:
                self.shards_in_progress.add(shard_name)

            num_shards = len(source.shard_names)
            num_worker_groups = len(ray.nodes())
            num_shard_groups = max(min(num_worker_groups, num_shards), 1)

            # if we have a bunch of caches to build with one shard, we don't want them all
            # assigned to the same node, so we use an offset based on the hash of the name (for stability)
            # in an attempt to spread them out
            group_offset = int(hash(name) % num_worker_groups)

            shard_groups: list[list[str]] = [[] for _ in range(num_shard_groups)]
            for i, shard_name in enumerate(source.shard_names):
                shard_groups[i % num_shard_groups].append(shard_name)

            def priority_fn(shard_idx, batch_idx):
                return batch_idx * num_shards + shard_idx

            for group_id, shard_group in enumerate(shard_groups):
                # TODO: would probably be better if we didn't create one of these per shard group
                processor_actor = _BatchProcessorQueue.remote(processor)  # type: ignore
                self._processor_actors.append(processor_actor)

                assert self._cache_writer is not None

                work_item = ShardGroupToBeProcessed(
                    name=name,
                    builder_ref=self_ref,
                    writer=self._cache_writer,
                    shard_source=source,
                    shard_names=shard_group,
                    priority_fn=priority_fn,
                    processor_actor=processor_actor,
                    batch_size=processor.batch_size,
                    group_id=group_id,
                )

                # we want global names so that different tasks can coordinate priorities
                worker_to_assign = (group_id + group_offset) % num_worker_groups
                priority_actor_name = f"priority_processor.{worker_to_assign}"

                reader_actor = WorkQueueDispatcherActor.options(  # type: ignore
                    name=priority_actor_name, get_if_exists=True
                ).remote()

                reader_actor.assign_work.remote(work_item)
                self._shard_readers.append(reader_actor)

    def shard_finished(self, shard_name: str):
        """Callback method for when a shard worker has finished."""
        self.shards_in_progress.remove(shard_name)

    def shard_failed(self, shard_name: str, error: ExceptionInfo):
        """Callback method for when a shard worker has failed."""
        self._writer_exception(shard_name, error)

    def _updated_ledger(self, ledger: CacheLedger):
        self._ledger = ledger
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

        # notify metrics subscribers
        self._do_notify()
        self._cache_writer = None

    def signal_backpressure(self, next_item_desired: Optional[int]):
        # get the priority of the item we want
        if next_item_desired is not None:
            self.logger.debug(f"Signaling backpressure for {next_item_desired}")
            # our priority function above is basically (batch_index, shard_index). We just ask we don't get more
            # than one round of batches ahead
            max_priority = (next_item_desired + 1) * len(self.source.shard_names)

            for reader in self._shard_readers:
                reader.set_max_dispatch_priority.remote(max_priority)
        else:
            self.logger.debug("Signaling no backpressure")
            for reader in self._shard_readers:
                reader.set_max_dispatch_priority.remote(None)


def _get_builder_actor(cache_dir, input_shards, processor, cache_config=None, items_per_write=MIN_ITEMS_TO_WRITE):
    name = f"lev_cache_manager::{cache_dir}"
    path_for_name = os.path.join(*os.path.split(cache_dir)[-2:])
    name_for_display = f"builder::{path_for_name}"

    return _TreeStoreCacheBuilder.options(name=name, get_if_exists=True).remote(  # type: ignore
        name=name_for_display,
        cache_dir=cache_dir,
        source=input_shards,
        processor=processor,
        cache_config=cache_config,
        min_items_to_write=items_per_write,
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

    async def _wait_for_len(self, needed_len):
        if self._broker is not None:
            while needed_len > await self.current_len():
                new_ledger = await self._broker.updated_ledger.remote()

                if needed_len <= new_ledger.total_num_rows:
                    break

                if new_ledger.is_finished:
                    if needed_len >= new_ledger.rows_finished:
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
                    new_ledger = ray.get(self._broker.updated_ledger.remote(), timeout=max(t_max - cur_time, 10))
                except TimeoutError:
                    continue

                if needed_len <= new_ledger.total_num_rows:
                    break

                if new_ledger.is_finished:
                    if needed_len >= new_ledger.rows_finished:
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
        items_per_write: int = MIN_ITEMS_TO_WRITE,
    ) -> "TreeCache[U]":
        try:
            return TreeCache.load(cache_dir, processor.output_exemplar)
        except FileNotFoundError:
            broker = _get_builder_actor(
                cache_dir=cache_dir,
                input_shards=shard_source,
                processor=processor,
                cache_config=cache_config,
                items_per_write=items_per_write,
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
        x = ray.get(self.finished_sentinel(), timeout=timeout)
        self._attempt_to_load_store()
        return x

    async def finished(self):
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
