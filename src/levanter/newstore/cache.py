import asyncio
import dataclasses
import heapq
import logging as pylogging
import os
import threading
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Sequence, TypeVar, Union

import fsspec.core
import pyarrow as pa
import ray
from dataclasses_json import dataclass_json
from fsspec import AbstractFileSystem
from ray.actor import ActorHandle

from .tree_store import TreeStoreBuilder
from ..data.metrics_monitor import InProgressCacheMetrics, LoggerMetricsMonitor, MetricsMonitor
from ..data._preprocessor import BatchProcessor, BatchResult, dict_from_record_batch
from ..data._queue import PriorityWorkItem, PriorityWorkTaskGroup, PriorityWorkTaskGroupSpec
from ..data.sharded_dataset import ShardedDataset
from ..utils.ray_utils import ExceptionInfo, RefBox, current_actor_handle, ser_exc_info

T = TypeVar("T")


logger = pylogging.getLogger(__name__)

LEDGER_FILE_NAME = "shard_ledger.json"

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def build_or_load_cache(
    cache_dir: str,
    exemplar: T,
    input_shards: ShardedDataset[T],
    processor: BatchProcessor[T],
    await_finished: bool = True,
    monitors: Optional[Sequence["MetricsMonitor"]] = None,
    cache_config: Optional[Dict[str, Any]] = None,
) -> "TreeCache":
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

    Returns:
       (TreeCache) A TreeCache object that can be used to read the cache.

    """
    # first see if we need to do anything
    cache = TreeCache.build_or_load(
        cache_dir=cache_dir,
        exemplar=exemplar,
        shard_source=input_shards,
        processor=processor,
        cache_config=cache_config,
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
class ShardMetadata:
    path: str
    num_rows: int
    is_finished: bool = False
    field_counts: Dict[str, int] = dataclasses.field(default_factory=dict)


@dataclass_json
@dataclass
class CacheLedger:
    """Written at the end of the cache build process."""
    shards: List[ShardMetadata] = dataclasses.field(default_factory=list)
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


class SerialCacheWriter(AbstractContextManager):
    """
    Writes TreeCache-compatible caches to disk. This is a serial version of TreeCacheWriter that doesn't use Ray.
    Mostly for scripts and debugging.

    Examples:
        >>> with SerialCacheWriter(cache_dir) as writer:
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
        self._tree_store: Optional[TreeStoreBuilder] = None

    def __enter__(self) -> "SerialCacheWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # if successful, write the ledger
        # TODO: store field counts in the ledger
        shard_metadata = ShardMetadata(self.cache_dir, is_finished=True, field_counts={})

        if exc_type is None:
            _serialize_json_and_commit(
                os.path.join(self.cache_dir, LEDGER_FILE_NAME), CacheLedger([shard_metadata], self.cache_config)
            )
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

        if self._tree_store is None:
            self._tree_store = TreeStoreBuilder.open(batch[0], self.cache_dir, mode="w")  # type: ignore

        self._tree_store.extend(batch)


def _load_or_initialize_shard_metadata(path):
    try:
        with fsspec.open(path, "r") as file:
            return ShardMetadata.from_json(file.read())  # type: ignore
    except FileNotFoundError:
        return ShardMetadata(path=path, num_rows=0, is_finished=False, field_counts={})



class _OutOfOrderCacheWriter:
    """
    This cache writer receives batches from some number of shards and writes them to the store in a defined round-robin
    order. It is responsible for managing the order of the batches and writing them to the store in the correct order.

    Once a shard finishes sending batches, it notifies this writer, which then updates the metadata and writes it to disk.
    """
    def __init__(self, cache_dir: str, shards: Sequence[str]):
        self.cache_dir = cache_dir
        self.shards = shards

        self._metadata_path = os.path.join(self.cache_base_dir, f"{shard_name}.json")
        self.metadata = _load_or_initialize_shard_metadata(self._metadata_path)

        self._tree_store: Optional[TreeStoreBuilder] = None

        self._batches = heapq.heapify([])
        self._next_batch = (0, 0)
        self._expected_num_batches: Optional[int] = None

    def batch_finished(self, shard_idx: int, shard_batch_idx: int, batch_result):
        # we need to keep track of the order of the batches
        # so that we can write them out in order
        heapq.heappush(self._batches, (batch_idx, shard_idx, batch_result))
        return self._attempt_to_write_batches()

    def shard_failed(self, batch_id: int, exc_info: ExceptionInfo):
        raise NotImplementedError() from exc_info.restore()

    def _attempt_to_write_batches(self):
        num_new_rows = 0
        while self._batches and self._batches[0][0] == self._next_batch_idx:
            batch_id, batch = self._batches[0]
            if batch_id != self._next_batch_idx:
                break

            heapq.heappop(self._batches)

            batch = _canonicalize_batch(batch)

            if self._tree_store is None:
                self._tree_store = TreeStoreBuilder.open(batch[0], self.cache_dir, mode="a")
                self._tree_store = self._tree_store.trim_to_size(self.metadata.num_rows)

            self._tree_store.extend(batch)
            self._next_batch_idx += 1
            num_new_rows += len(batch)

            # TODO: bring back field counts
            # for i in range(batch.num_columns):
            #     name = batch.field(i).name
            #     value = batch.column(i)
            #     if isinstance(value, pa.ListArray):
            #         value = value.flatten()
            #         self.field_counts[name] = self.field_counts.get(name, 0) + len(value)
            #     elif isinstance(value, pa.ChunkedArray):
            #         self.field_counts[name] = self.field_counts.get(name, 0) + value.length()

        if self._expected_num_batches is not None and self._next_batch_idx >= self._expected_num_batches:
            self.metadata.is_finished = True
            self._tree_store = None

        self.metadata.num_rows += num_new_rows

    def shard_finished_reading(self, expected_batches: int):
        if self._tree_store is None:
            raise RuntimeError("ShardWriter not initialized")

        self._expected_num_batches = expected_batches

    def _commit(self):
        _serialize_json_and_commit(self._metadata_path, self.metadata)


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
        return dict_from_record_batch(batch)
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

def _shard_reader_generator(shard_source: ShardedDataset[T], shard_idx: int, start_row: int, batch_size: int):
    shard_name = shard_source.shard_names[shard_idx]
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
    shard_source: ShardedDataset
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

        try:
            metadata: dict[str, ShardMetadata] = _initial_shard_metadatas(
                self.spec.shard_source, self.spec.shard_names, self.spec.writer
            )
        except Exception as e:
            self.spec.builder_ref.other_failed.remote(ser_exc_info())
            raise e

        batch_size = self.spec.batch_size

        self._items: list[PriorityWorkItem] = []

        for shard_name in self.spec.shard_names:
            shard_idx = self.spec.shard_source.shard_names.index(shard_name)
            try:
                shard_metadata = metadata[shard_name]
                reader = _shard_reader_generator(
                    self.spec.shard_source, shard_idx, shard_metadata.num_rows, batch_size
                )

                if shard_metadata.is_finished:
                    self.logger.info(f"Shard {shard_name} already finished. Skipping.")

                task_name = f"shard_reader.{self.spec.name}.{shard_name}"

                batch_idx = shard_metadata.num_rows // batch_size

                item = ShardReaderItem(self, task_name, shard_name, shard_idx, batch_idx=batch_idx, reader=reader)

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

    @property
    def priority(self):
        return self.group.spec.priority_fn(self.shard_idx, self.batch_idx)

    @property
    def spec(self):
        return self.group.spec

    def execute(self) -> tuple[bool, Optional[ray.ObjectRef]]:
        writer = self.spec.writer
        batch_result_ref = None

        self.group.logger.debug(f"Reading one batch of shard {self.shard_name}: {self.batch_idx}")

        try:
            batch = next(self.reader, None)
            exhausted_shard = batch is None or (len(batch) < self.spec.batch_size)

            if batch:
                priority = self.spec.priority_fn(self.shard_idx, self.batch_idx)
                # these times aren't exact because the times might be from different machines
                # but they're just for logging
                time_in = time.time()
                batch_result_ref = ray.get(
                    self.spec.processor_actor.submit.remote(
                        priority=priority,
                        desc=f"{self.shard_name}.{self.batch_idx}",
                        batch=RefBox(ray.put(batch)),
                    )
                )
                writer.batch_finished.remote(
                    self.shard_name, self.batch_idx, RefBox(batch_result_ref), time_in
                )
                self.batch_idx += 1

            if exhausted_shard:
                writer.shard_finished_reading.remote(self.shard_name, self.batch_idx)

            self.group.logger.debug(
                f"Finished reading one batch of shard {self.shard_name}: {self.batch_idx}"
            )

            return exhausted_shard, batch_result_ref
        except Exception as e:  # noqa
            self.group.logger.exception(f"Error while processing shard {self.shard_name}")
            # fire and forget
            writer.shard_failed.remote(self.shard_name, ser_exc_info())
            raise e


def _initial_shard_metadatas(shard_source, shard_names, shard_group_writer):
    shard_metadatas: dict[str, ShardMetadata] = {}
    _metadata_futures = [shard_group_writer.current_metadata.remote(name) for name in shard_names]
    shard_metadatas_rs = ray.get(_metadata_futures)
    for shard_name, shard_metadata in zip(shard_names, shard_metadatas_rs):
        shard_metadatas[shard_name] = shard_metadata
    return shard_metadatas


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


def _mk_queue_aware_process_task(processor: BatchProcessor[T], queue: ActorHandle):
    @ray.remote(num_cpus=processor.num_cpus, num_gpus=processor.num_gpus, resources=processor.resources)
    def process_task(desc, batch: List[T]):
        pylogging.basicConfig(level=pylogging.INFO, format=LOG_FORMAT)
        logger.debug(f"Processing batch {desc}")
        queue.task_running.remote()
        # timer_thread = WaitTimeReportingThread(
        #     lambda t: logger.info(f"Waiting for {desc} to be processed for {t} seconds"), interval=30
        # )
        # timer_thread.start()
        try:
            result = processor(batch)
            del batch
            logger.debug(f"Finished processing batch {desc}")
            return result
        except Exception as e:
            logger.exception(f"Error while processing batch {desc}")
            raise e
        finally:
            # timer_thread.shutdown()
            # timer_thread.join()
            pass

    return process_task


# Ray does poorly with large numbers of actors (grumble grumble), so we can't have one actor per shard.
# This class wraps a map of shard names to _ShardWriterWorkers, and manages the lifecycle of the workers.
@ray.remote(num_cpus=0.0, scheduling_strategy="SPREAD")  # type: ignore
class _GroupShardWriterWorker:
    def __init__(self, parent_ref, cache_dir: str, shard_names: Sequence[str]):
        pylogging.basicConfig(level=pylogging.INFO, format=LOG_FORMAT)
        self.cache_dir = cache_dir
        self.shard_names = shard_names
        self.shard_writers: dict[str, _ShardWriterWorker] = {
            shard_name: _ShardWriterWorker(parent_ref, cache_dir, shard_name) for shard_name in shard_names
        }

    def current_metadata(self, shard_name: str):
        return self.shard_writers[shard_name].current_metadata()

    async def batch_finished(self, shard_name: str, batch_idx: int, batch: RefBox, time_in):
        try:
            time_mid = time.time()
            logger.debug(
                f"Received in progress batch {batch_idx} of shard {shard_name} in"
                f" {time_mid - time_in}"
            )
            # do a backoff loop until the batch is actually processed. log if it's been a while
            timeout_interval = 20
            total_time_waited = 0

            while True:
                try:
                    # batch = await asyncio.wait_for(asyncio.shield(batch.ref), timeout_interval)
                    batch = await batch.ref
                    break
                except asyncio.TimeoutError:
                    # to keep to round numbers, we log how much we asked for rather than how much we got
                    total_time_waited += timeout_interval
                    timeout_interval = min(2 * timeout_interval, 100)
                    logger.info(
                        f"Waiting for {shard_name}.{batch_idx} to be processed. "
                        f"Waited {total_time_waited} seconds."
                    )

            if logger.isEnabledFor(pylogging.DEBUG):
                logger.debug(
                    f"Received finished {shard_name}.{batch_idx} in {(time.time() - time_in):.2f} seconds."
                )
            elif total_time_waited > 40:
                logger.info(
                    f"Waited {total_time_waited} seconds for {shard_name}.{batch_idx} to be processed."
                )
            return self.shard_writers[shard_name].batch_finished(batch_idx, batch)
        except Exception as e:
            print(f"Error while processing batch {batch_idx} of  shard {shard_name}", flush=True)
            self.shard_writers[shard_name].shard_failed(ser_exc_info())
            raise e

    def shard_finished_reading(self, shard_name: str, expected_num_chunks: int):
        return self.shard_writers[shard_name].shard_finished_reading(expected_num_chunks)

    def shard_failed(self, shard_name: str, error: ExceptionInfo):
        return self.shard_writers[shard_name].shard_failed(error)


class _ShardWriterWorker:  # type: ignore
    """
    Actor that writes chunks to disk and updates the ShardMetadata. It reports to the ChunkCacheBroker
    """

    def __init__(
        self,
        parent_ref: ActorHandle,  # ChunkCacheBuilder
        cache_base_dir: str,
        shard_name: str,
    ):
        pylogging.basicConfig(level=pylogging.INFO, format=LOG_FORMAT)
        self.parent_ref = parent_ref
        self.cache_base_dir = cache_base_dir
        self.shard_name = shard_name

        self._writer = _OutOfOrderShardWriter(cache_base_dir=cache_base_dir, shard_name=shard_name)

    def current_metadata(self):
        return self._writer.metadata

    # forward some methods to the collator, handle any metadata that comes back
    def batch_finished(self, batch_idx: int, batch):
        metadata = self._writer.batch_finished(batch_idx, batch)
        if metadata is not None and metadata.is_finished:
            self._finished_shard(metadata)

        return metadata

    def batch_failed(self, batch_id: int, error: ExceptionInfo):
        self._writer.shard_failed(batch_id, error)
        print(f"Error while processing shard {self.shard_name} at {batch_id}", flush=True)
        self.parent_ref.shard_failed.remote(self.shard_name, error)

    def _finished_shard(self, shard_metadata: ShardMetadata):
        self.finished = True
        _serialize_json_and_commit(os.path.join(self.cache_base_dir, self.shard_name, "metadata.json"), shard_metadata)
        self.parent_ref.shard_finished.remote(self.shard_name, shard_metadata)

    def shard_finished_reading(self, expected_num_batches: int):
        # TODO: add that we're done reading to metrics
        self._writer.shard_finished_reading(expected_num_batches)

    def shard_failed(self, error: ExceptionInfo):
        self.parent_ref.shard_failed.remote(self.shard_name, error)


@ray.remote(num_cpus=0.5)  # keep this small b/c it doesn't do a lot
class _TreeStoreCacheBuilder:
    """
    Actor that coordinates the building of a cache. It spins up a bunch of workers to read from each shard
    and write to the cache.

    NB: the predecessor of this class used to support reading a prefix of the cache while it was being built,
    but that feature is currently disabled.
    """

    def __init__(
        self,
        cache_dir: str,
        name: str,
        source: ShardedDataset[T],
        processor: BatchProcessor[T],
        cache_config: Dict[str, Any],
    ):
        pylogging.basicConfig(level=pylogging.INFO, format=LOG_FORMAT)
        self.logger = pylogging.getLogger(f"{__name__}.{name}")
        self.source = source
        self._cache_dir = cache_dir
        self._metrics = InProgressCacheMetrics()
        self.shards_in_progress = set()

        self._finished_promise = asyncio.Future()
        # used to subscribe to metrics updates
        self._metrics_condition = asyncio.Condition()
        self._cache_config = cache_config
        path_for_name = os.path.join(*self._cache_dir.split("/")[-2:])
        name = f"broker::{path_for_name}"
        self.logger = pylogging.getLogger(f"{name}")

        try:
            cache_ledger = _load_cache_ledger(self._cache_dir)
            self._ledger = cache_ledger
            self._is_finished = True
            self._finished_promise.set_result(None)
        except FileNotFoundError:
            self._start_workers(cache_dir, name, processor, source)

    def _start_workers(self, cache_dir, name, processor, source):
        if len(source.shard_names) == 0:
            self.logger.warning("No shards to index?!?")
            self._finalize()
        else:
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
                writer = _GroupShardWriterWorker.remote(self_ref, cache_dir, shard_group)  # type: ignore
                self._shard_writers.append(writer)

                # TODO: would probably be better if we didn't create one of these per shard group
                processor_actor = _BatchProcessorQueue.remote(processor)  # type: ignore
                self._processor_actors.append(processor_actor)

                work_item = ShardGroupToBeProcessed(
                    name=name,
                    builder_ref=self_ref,
                    writer=writer,
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

                reader_actor = PriorityProcessorActor.options(  # type: ignore
                    name=priority_actor_name, get_if_exists=True
                ).remote()

                reader_actor.add_work_group.remote(work_item)

                self._shard_readers.append(reader_actor)

    def shard_finished(self, shard_name: str, shard_metadata: ShardMetadata):
        """Callback method for when a shard worker has finished."""
        self._metrics.shards_finished += 1
        self._do_notify()
        self.shards_in_progress.remove(shard_name)

        # if there are no more active shards, we're done
        if self._all_shards_done():
            self._finalize()

    def _all_shards_done(self):
        return len(self.shards_in_progress) == 0

    def shard_failed(self, shard_name: str, error: ExceptionInfo):
        """Callback method for when a shard worker has failed."""
        self._writer_exception(shard_name, error)

    def other_failed(self, error: ExceptionInfo):
        """Callback method for when a shard worker has failed."""
        self._writer_exception(None, error)

    def is_finished(self):
        return self._metrics.is_finished

    async def finished_sentinel(self):
        await self._finished_promise

    async def updated_metrics(self) -> InProgressCacheMetrics:
        if self._finished_promise.done():
            if self._finished_promise.exception() is not None:
                raise self._finished_promise.exception()  # type: ignore
            else:
                return self._latest_metrics

        async with self._metrics_condition:
            await self._metrics_condition.wait()
            return self._latest_metrics

    def _writer_exception(self, shard_name, exc_info: ExceptionInfo):
        info = exc_info.restore()

        logger.exception(f"Writer task {shard_name} failed with exception", exc_info=info)
        for future in self._reader_promises.values():
            future.set_exception(info[1])

        self._reader_promises = {}

        self._finished_promise.set_exception(info[1])
        self._do_notify()

    def _do_notify(self):
        async def _do_notify_async():
            async with self._metrics_condition:
                self._metrics_condition.notify_all()

        asyncio.create_task(_do_notify_async())

    def _finalize(self):
        logger.info(f"Finalizing cache {self._cache_dir}...")

        self._metrics.is_finished = True
        self._is_finished = True
        for k, future in self._reader_promises.items():
            future.set_result(None)

        # write ledger
        _serialize_json_and_commit(
            os.path.join(self._cache_dir, LEDGER_FILE_NAME), CacheLedger(self.chunks, self._cache_config)
        )

        self._reader_promises = {}
        # TODO: For some reason this crashes other actors with weird reference counting assertion errors.
        # pretty sure it's a ray bug
        # self._builder_actor = None
        self._finished_promise.set_result(None)

        # notify metrics subscribers
        self._do_notify()

def _get_builder_actor(cache_dir, input_shards, processor, cache_config=None):
    name = f"lev_cache_manager::{cache_dir}"
    path_for_name = os.path.join(*os.path.split(cache_dir)[-2:])
    name_for_display = f"builder::{path_for_name}"

    return _TreeStoreCacheBuilder.options(name=name, get_if_exists=True).remote(  # type: ignore
        name=name_for_display,
        cache_dir=cache_dir,
        source=input_shards,
        processor=processor,
        cache_config=cache_config,
    )


class TreeCache:
    ledger: Optional[CacheLedger]
    _broker: Optional[ActorHandle]
    _stores: dict[str, TreeStoreBuilder]
    # We use a thread here instead of an actor because we want to ensure it's in the same process as the TreeCache
    # object.
    _monitor_thread: Optional[threading.Thread]
    _metrics_monitors: List[MetricsMonitor]

    def __init__(
        self,
        cache_dir: str,
        exemplar: T,
        ledger: Optional[CacheLedger],
        _broker: Optional[ActorHandle],
    ):
        self.cache_dir = cache_dir
        self.ledger = ledger
        self._broker = _broker
        self._stores = {}

        self._metrics_monitors = []
        self._monitor_thread = None

        name = os.path.join(*cache_dir.split("/")[-2:])
        self.logger = pylogging.getLogger(f"TreeCache.{name}")

    @staticmethod
    def load(cache_dir: str, exemplar: T) -> "TreeCache":
        """Loads a cache from disk or an object store. Raises FileNotFoundError if the cache doesn't exist"""
        logger.info(f"Loading cache from {cache_dir}")
        ledger = _load_cache_ledger(cache_dir)
        return TreeCache(cache_dir, exemplar, ledger, None)

    @staticmethod
    def build_or_load(
        cache_dir: str,
        exemplar: T,
        shard_source: ShardedDataset[T],
        processor: BatchProcessor[T],
        cache_config: Optional[Dict[str, Any]] = None,
    ):
        try:
            return TreeCache.load(cache_dir, exemplar)
        except FileNotFoundError:
            broker = _get_builder_actor(
                cache_dir=cache_dir,
                input_shards=shard_source,
                processor=processor,
                cache_config=cache_config,
            )
            return TreeCache(cache_dir=cache_dir, exemplar=exemplar, ledger=None, _broker=broker)

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
        ...

    def await_finished(self, timeout: Optional[float] = None):
        return ray.get(self.finished_sentinel(), timeout=timeout)

    def attach_metrics_monitor(self, monitor: MetricsMonitor):
        if self._broker is None:
            logger.warning("Cannot attach metrics monitor to finished cache.")
            # TODO: decide what to do about attaching if the cache is already finished
            # maybe get the final metrics?
            return

        self._metrics_monitors.append(monitor)
        if self._monitor_thread is None:
            self._monitor_thread = threading.Thread(target=self._monitor_metrics)
            self._monitor_thread.start()

    def _monitor_metrics(self):
        while True:
            try:
                metrics = ray.get(self._broker.updated_metrics.remote())
                for monitor in self._metrics_monitors:
                    monitor(metrics)
                if metrics.is_finished:
                    break
            except Exception as e:
                self.logger.exception("Error while reading metrics from shard cache.")
                raise e



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

    def append_to_group(self, group: str, item_serial: int, item: T):
        if group not in self.groups:
            raise ValueError(f"Group {group} not in {self.groups}")

        if group not in self._remaining_groups:
            raise ValueError(f"Group {group} already finished")

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

    def is_finished(self):
        return len(self._remaining_groups) == 0

    def pop(self) -> Optional[T]:
        group = self._next_group_to_read_from()
        if group is None:
            return None

        if len(self.buffers[group]) == 0:
            return None

        cur_serial, item = self.buffers[group][0]

        if cur_serial != self._totals_written[group]:
            return None

        heapq.heappop(self.buffers[group])

        self._totals_written[group] += 1

        if self._totals_written[group] == self._totals_expected[group]:
            assert len(self.buffers[group]) == 0
            assert group in self._remaining_groups
            self._remaining_groups.remove(group)

        self._current_group = (self._current_group + 1) % len(self.groups)

        return item

    def drain(self) -> list[T]:
        items = []
        while True:
            item = self.pop()
            if item is None:
                break
            items.append(item)

        return items

    def _next_group_to_read_from(self):
        if len(self._remaining_groups) == 0:
            return None

        while True:
            group = self.groups[self._current_group]
            if group not in self._remaining_groups:
                assert self._totals_written[group] == self._totals_expected[group]
                assert len(self.buffers[group]) == 0
                self._current_group = (self._current_group + 1) % len(self.groups)
            else:
                break
        return group