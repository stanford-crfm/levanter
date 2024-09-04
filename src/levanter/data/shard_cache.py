# Dataset for preprocessing data, tokenizing, and caching to disk.
import asyncio
import dataclasses
import heapq
import logging as pylogging
import os
import threading
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import IO, Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, TypeVar

import fsspec.core
import pyarrow as pa
import pyarrow.parquet as pq
import ray
from dataclasses_json import dataclass_json
from fsspec import AbstractFileSystem
from ray.actor import ActorHandle
from ray.exceptions import GetTimeoutError

from ..utils.ray_utils import (
    ExceptionInfo,
    RefBox,
    SnitchRecipient,
    current_actor_handle,
    log_failures_to,
    ser_exc_info,
)
from ._preprocessor import BatchProcessor, BatchResult, as_record_batch, dict_from_record_batch
from ._queue import (
    PriorityProcessorActor,
    PriorityWorkItem,
    PriorityWorkTaskGroup,
    PriorityWorkTaskGroupSpec,
    _BatchProcessorQueue,
)
from .dataset import ShardableDataset
from .metrics_monitor import InProgressCacheMetrics, LoggerMetricsMonitor, MetricsMonitor
from .sharded_dataset import ShardedDataset


G = TypeVar("G")
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


logger = pylogging.getLogger(__name__)

DEFAULT_ROWS_PER_CHUNK = 8192
DEFAULT_MAX_BYTES_PER_BATCH = 256 * 1024 * 1024  # 256 MB, this is pre-preprocessing python object size
DEFAULT_MAX_SHARDS_TO_READ_AT_ONCE = 32
LEDGER_FILE_NAME = "cache_ledger.json"

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LEVEL_TO_LOG = pylogging.INFO


def build_or_load_cache(
    cache_dir: str,
    input_shards: ShardedDataset[T],
    processor: BatchProcessor[T],
    batch_size: int = 1,
    rows_per_chunk: int = DEFAULT_ROWS_PER_CHUNK,
    await_finished: bool = True,
    monitors: Optional[Sequence["MetricsMonitor"]] = None,
    cache_config: Optional[Dict[str, Any]] = None,
) -> "ShardCache":
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
        batch_size: When reading from the cache, how many examples to read at a time.
        rows_per_chunk: The number of rows to write to each chunk. May be smaller at the end of a shard.
        await_finished: If True, this function will block until the cache is finished. If False, it will return
                    immediately.
        monitors: a list of MetricsMonitors to attach to the cache. These will be called periodically with
            metrics about the cache build process. If None, will add a LoggerMetricsMonitor.

    Returns:
       (ShardCache) A ShardCache object that can be used to read the cache.

    """
    # first see if we need to do anything
    cache = ShardCache.build_or_load(
        cache_dir=cache_dir,
        shard_source=input_shards,
        processor=processor,
        batch_size=batch_size,
        rows_per_chunk=rows_per_chunk,
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
class ChunkMetadata:
    name: str
    num_rows: int
    field_counts: Dict[str, int]


@dataclass_json
@dataclass
class ShardMetadata:
    chunks: List[ChunkMetadata] = dataclasses.field(default_factory=list)
    is_finished: bool = False

    @property
    def total_rows(self):
        return sum(chunk.num_rows for chunk in self.chunks)

    @property
    def total_chunks_produced(self):
        return len(self.chunks)


@dataclass_json
@dataclass
class CacheLedger:
    """Written at the end of the cache build process. Contains the global chunk order."""

    chunks: List[ChunkMetadata] = dataclasses.field(default_factory=list)
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


class SerialCacheWriter(AbstractContextManager):
    """
    Writes ShardCache-compatible caches to disk. This is a serial version of ShardCacheWriter that doesn't use Ray.
    Mostly for scripts and debugging.

    Examples:
        >>> with SerialCacheWriter(cache_dir, rows_per_chunk=1024) as writer:
        ...     for batch in process_batches():
        ...         writer.write_batch(batch)
    """

    def __init__(
        self,
        cache_dir: str,
        rows_per_chunk: int = DEFAULT_ROWS_PER_CHUNK,
        cache_config: Optional[Dict[str, Any]] = None,
    ):
        if rows_per_chunk <= 0:
            raise ValueError("rows_per_chunk must be positive")
        self.cache_dir = cache_dir
        self.cache_config = cache_config
        self._rows_per_chunk = rows_per_chunk
        self._chunks: List[ChunkMetadata] = []
        self._current_chunk_writer: Optional[_ChunkWriter] = None
        self._is_closed = False

    def __enter__(self) -> "SerialCacheWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # if successful, write the ledger
        if self._current_chunk_writer is not None:
            self._current_chunk_writer.__exit__(exc_type, exc_val, exc_tb)
            self._chunks.append(self._current_chunk_writer.get_metadata())
            self._current_chunk_writer = None

        if exc_type is None:
            _serialize_json_and_commit(
                os.path.join(self.cache_dir, LEDGER_FILE_NAME), CacheLedger(self._chunks, self.cache_config)
            )
            logger.info(f"Cache ledger written to {self.cache_dir}")
            self._is_closed = True

    def result(self, batch_size: int = 1) -> "ShardCache":
        if not self._is_closed:
            raise RuntimeError("Cannot get result until ShardCacheWriter is closed")
        return ShardCache.load(self.cache_dir, batch_size=batch_size)

    def write_batch(self, batch: BatchResult):
        rb = as_record_batch(batch)

        while rb.num_rows > 0:
            if self._current_chunk_writer is None:
                self._current_chunk_writer = _ChunkWriter(
                    self.cache_dir, f"chunk-{len(self._chunks)}", rb.schema
                ).__enter__()

            slice = rb.slice(0, min(rb.num_rows, self._rows_per_chunk - self._current_chunk_writer.num_rows))
            self._current_chunk_writer.write_batch(slice)
            rb = rb.slice(slice.num_rows)

            if self._current_chunk_writer.num_rows >= self._rows_per_chunk:
                self._current_chunk_writer.__exit__(None, None, None)
                self._chunks.append(self._current_chunk_writer.get_metadata())
                self._current_chunk_writer = None


class _ChunkWriter:
    def __init__(self, cache_dir: str, chunk_name: str, schema: pa.Schema):
        self.cache_dir = cache_dir
        self.chunk_name = chunk_name
        self.schema = schema
        self.file: Optional[IO] = None
        self.writer: Optional[pq.ParquetWriter] = None
        self.num_rows = 0
        self.field_counts: Dict[str, int] = {}

        self.is_finished = False

    def __enter__(self):
        self.file = fsspec.open(os.path.join(self.cache_dir, f"{self.chunk_name}.parquet"), "wb").__enter__()
        self.writer = pq.ParquetWriter(self.file, self.schema, version="2.6", compression="ZSTD").__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer is not None:
            self.writer.__exit__(exc_type, exc_val, exc_tb)
        if self.file is not None:
            self.file.__exit__(exc_type, exc_val, exc_tb)

        self.is_finished = True

    def write_batch(self, batch: pa.RecordBatch):
        assert not self.is_finished
        assert self.writer is not None
        self.writer.write_batch(batch)
        self.num_rows += batch.num_rows

        for i in range(batch.num_columns):
            name = batch.field(i).name
            value = batch.column(i)
            if isinstance(value, pa.ListArray):
                value = value.flatten()
                self.field_counts[name] = self.field_counts.get(name, 0) + len(value)
            elif isinstance(value, pa.ChunkedArray):
                self.field_counts[name] = self.field_counts.get(name, 0) + value.length()

    def get_metadata(self) -> ChunkMetadata:
        if not self.is_finished:
            raise RuntimeError("Cannot get metadata for unfinished chunk")
        return ChunkMetadata(self.chunk_name, self.num_rows, self.field_counts)


class _ShardMetadataWriter:
    def __init__(self, metadata_path):
        self.metadata_path = metadata_path
        try:
            with fsspec.open(self.metadata_path, "r") as file:
                self.metadata = ShardMetadata.from_json(file.read())  # type: ignore
        except FileNotFoundError:
            self.metadata = ShardMetadata()

    @property
    def is_finished(self):
        return self.metadata.is_finished

    @property
    def chunks(self):
        return self.metadata.chunks

    @property
    def num_chunks(self):
        return len(self.metadata.chunks)

    def commit_chunk(self, chunk: ChunkMetadata):
        assert not self.metadata.is_finished
        self.metadata.chunks.append(chunk)
        self._commit()

    def finish(self):
        self.metadata.is_finished = True
        self._commit()

    def _commit(self):
        _serialize_json_and_commit(self.metadata_path, self.metadata)


# thinking through the design of the cache system

# we decided to use Ray, which was maybe a mistake, but here we are.
# Ray doesn't like it when the number of actors gets too large, so we can't have one actor per shard.
# we have N nodes and K shards. We want to produce chunks of size C examples, from each shards.
# We define a global order over chunks [shard[0].chunk[0], shard[1].chunk[0], ... shard[K].chunk[0], shard[0].chunk[1], ...]
# with the obvious extension for if one shard has more chunks than another.
# We want to produce chunks in roughly this order, but we want to do it in parallel.
# We also want to be able to recover from failures, and we want to be able to resume a cache build.

# at a high level, we have 3 steps:
# 1. read batches from the source
# 2. process batches, concatenating them into chunks
# 3. write chunks to disk

# The difficulty is that we want parallelism and we want to control the order of chunks.
# reading batches requires CPU and network. This means we should limit the number to roughly the number of nodes, maybe times 2.
# We want to prioritize so that we read 1 chunks worth of batches from each shard before reading more from another shard.
# We also want to prioritize reading earlier shards before later shards (within a chunk generation round).
# Ray also seems to get upset about having too many processes, and we can't serialize the iterators over shards.


def _shard_reader_generator(shard_source: ShardedDataset[T], shard_name: str, start_row: int, batch_size: int):
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
    builder_ref: ray.actor.ActorHandle  # _ChunkCacheBuilder
    writer: ray.actor.ActorHandle  # _GroupedShardWriter
    shard_source: ShardedDataset
    shard_names: Sequence[str]
    priority_fn: Callable[[int, int], float]
    processor_actor: ray.actor.ActorHandle  # BatchProcessorQueue
    batch_size: int
    num_rows_per_chunk: int
    group_id: int

    def build(self) -> "PriorityWorkTaskGroup":
        return ShardGroupTaskGroup(self)


class ShardGroupTaskGroup(PriorityWorkTaskGroup):
    def __init__(self, spec: ShardGroupToBeProcessed):
        self.spec = spec
        self.logger = pylogging.getLogger(f"shard_reader.{spec.group_id}.{spec.name}")

        try:
            metadata: dict[str, ShardMetadata] = _initial_shard_metadatas(
                self.spec.shard_source, self.spec.shard_names, self.spec.writer
            )
        except Exception as e:
            self.spec.builder_ref.other_failed.remote(ser_exc_info())
            raise e

        batch_size = min(self.spec.batch_size, self.spec.num_rows_per_chunk)

        self._items: list[PriorityWorkItem] = []

        for shard_name in self.spec.shard_names:
            shard_idx = self.spec.shard_source.shard_names.index(shard_name)
            try:
                shard_metadata = metadata[shard_name]
                reader = _shard_reader_generator(
                    self.spec.shard_source, shard_name, shard_metadata.total_rows, batch_size
                )

                if shard_metadata.is_finished:
                    self.logger.info(f"Shard {shard_name} already finished. Skipping.")

                task_name = f"shard_reader.{self.spec.name}.{shard_name}"

                chunk_idx = len(shard_metadata.chunks)
                item = ShardReaderItem(self, task_name, shard_name, shard_idx, chunk_idx, reader)

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
    Each time execute is called, this class reads one chunk's worth of batches from the shard
    and dispatches them to the processor.
    """

    group: ShardGroupTaskGroup
    name: str
    shard_name: str
    shard_idx: int
    chunk_idx: int
    reader: Iterator[list]

    @property
    def priority(self):
        return self.group.spec.priority_fn(self.shard_idx, self.chunk_idx)

    @property
    def spec(self):
        return self.group.spec

    def execute(self) -> tuple[bool, Optional[ray.ObjectRef]]:
        exhausted_shard = False
        writer = self.spec.writer

        chunk_batch_idx = 0  # the index of the batch within the chunk
        chunk_filled = False  # whether or not we've filled the chunk to max size
        total_chunk_rows = 0  # the total number of rows in the chunk
        batch_result_ref = None

        self.group.logger.debug(f"Reading one chunk of shard {self.shard_name}: {self.chunk_idx}")

        try:
            while not chunk_filled:
                batch = next(self.reader, None)
                if batch is None:
                    exhausted_shard = True
                    break

                exhausted_shard = len(batch) < self.spec.batch_size
                total_chunk_rows += len(batch)

                if batch:
                    priority = self.spec.priority_fn(self.shard_idx, self.chunk_idx)
                    # these times aren't exact because the times might be from different machines
                    # but they're just for logging
                    time_in = time.time()
                    batch_result_ref = ray.get(
                        self.spec.processor_actor.submit.remote(
                            priority=priority,
                            desc=f"{self.shard_name}.{self.chunk_idx}.{chunk_batch_idx}",
                            batch=RefBox(ray.put(batch)),
                        )
                    )
                    writer.chunk_batch_finished.remote(
                        self.shard_name, self.chunk_idx, chunk_batch_idx, RefBox(batch_result_ref), time_in
                    )
                    chunk_batch_idx += 1
                    del batch

                if total_chunk_rows >= self.spec.num_rows_per_chunk or exhausted_shard:
                    chunk_filled = True

            if chunk_batch_idx > 0:
                writer.chunk_finished_reading.remote(self.shard_name, self.chunk_idx, chunk_batch_idx)
                old_prio = self.priority
                self.chunk_idx += 1
                assert self.priority > old_prio

            if exhausted_shard:
                writer.shard_finished_reading.remote(self.shard_name, self.chunk_idx)

            self.group.logger.debug(
                f"Finished reading one chunk of shard {self.shard_name}: {self.chunk_idx} {exhausted_shard}"
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


@dataclass
class _ShardStatus:
    num_chunks_sent: int = 0
    current_buffer: list[ChunkMetadata] = dataclasses.field(default_factory=list)
    expected_num_chunks: Optional[int] = None

    def pop_chunk_to_send(self) -> Optional[ChunkMetadata]:
        if len(self.current_buffer) == 0:
            return None
        else:
            self.num_chunks_sent += 1
            return self.current_buffer.pop(0)

    @property
    def is_finished_and_buffer_empty(self):
        return self.expected_num_chunks is not None and self.num_chunks_sent >= self.expected_num_chunks


# Ray does poorly with large numbers of actors (grumble grumble), so we can't have one actor per shard.
# This class wraps a map of shard names to _ShardWriterWorkers, and manages the lifecycle of the workers.
@ray.remote(num_cpus=0.0, scheduling_strategy="SPREAD")  # type: ignore
class _GroupShardWriterWorker:
    def __init__(self, parent_ref, cache_dir: str, shard_names: Sequence[str]):
        with log_failures_to(parent_ref):
            pylogging.basicConfig(level=LEVEL_TO_LOG, format=LOG_FORMAT)
            self.cache_dir = cache_dir
            self.shard_names = shard_names
            self.shard_writers: dict[str, _ShardWriterWorker] = {
                shard_name: _ShardWriterWorker(parent_ref, cache_dir, shard_name) for shard_name in shard_names
            }

    def current_metadata(self, shard_name: str):
        return self.shard_writers[shard_name].current_metadata()

    async def chunk_batch_finished(self, shard_name: str, chunk_id: int, batch_idx: int, batch: RefBox, time_in):
        # batch is a pa.RecordBatch ref box
        try:
            time_mid = time.time()
            logger.debug(
                f"Received in progress batch {batch_idx} of chunk {chunk_id} of shard {shard_name} in"
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
                        f"Waiting for {shard_name}.{chunk_id}.{batch_idx} to be processed. "
                        f"Waited {total_time_waited} seconds."
                    )

            if logger.isEnabledFor(pylogging.DEBUG):
                logger.debug(
                    f"Received finished {shard_name}.{chunk_id}.{batch_idx} in {(time.time() - time_in):.2f} seconds."
                )
            elif total_time_waited > 40:
                logger.info(
                    f"Waited {total_time_waited} seconds for {shard_name}.{chunk_id}.{batch_idx} to be processed."
                )
            return self.shard_writers[shard_name].chunk_batch_finished(chunk_id, batch_idx, batch)
        except Exception as e:
            print(f"Error while processing batch {batch_idx} of chunk {chunk_id} of shard {shard_name}", flush=True)
            self.shard_writers[shard_name].chunk_failed(chunk_id, ser_exc_info())
            raise e

    def chunk_finished_reading(self, shard_name: str, chunk_id: int, expected_num_batches: int):
        return self.shard_writers[shard_name].chunk_finished_reading(chunk_id, expected_num_batches)

    def chunk_failed(self, shard_name: str, chunk_id: int, error: ExceptionInfo):
        return self.shard_writers[shard_name].chunk_failed(chunk_id, error)

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
        cache_dir: str,
        shard_name: str,
    ):
        pylogging.basicConfig(level=LEVEL_TO_LOG, format=LOG_FORMAT)
        self.parent_ref = parent_ref
        self.cache_dir = cache_dir
        self.shard_name = shard_name
        self.uncommited_chunks: list[tuple[int, ChunkMetadata]] = []  # heapq of (chunk index, chunk)

        self.metadata_writer = _ShardMetadataWriter(os.path.join(cache_dir, f"{shard_name}.json"))
        self._expected_num_chunks: Optional[int] = None

        if self.metadata_writer.num_chunks > 0:
            self.parent_ref.new_chunk.remote(shard_name, *self.metadata_writer.chunks)

        if self.metadata_writer.is_finished:
            self._expected_num_chunks = self.metadata_writer.num_chunks
            self.parent_ref.shard_finished.remote(self.shard_name, self._expected_num_chunks)
            self.finished = True
        else:
            self.finished = False

        self.collator = _ChunkCollator(cache_dir, shard_name)

    def current_metadata(self):
        return self.metadata_writer.metadata

    # forward some methods to the collator, handle any metadata that comes back
    def chunk_batch_finished(self, chunk_id: int, batch_idx: int, batch: pa.RecordBatch):
        metadata = self.collator.new_batch(chunk_id, batch_idx, batch)
        if metadata is not None:
            self._finished_chunk(chunk_id, metadata)

        return metadata

    def chunk_finished_reading(self, chunk_id: int, expected_num_batches: int):
        metadata = self.collator.chunk_finished_reading(chunk_id, expected_num_batches)
        if metadata is not None:
            self._finished_chunk(chunk_id, metadata)

        return metadata

    def chunk_failed(self, chunk_id: int, error: ExceptionInfo):
        self.collator.chunk_failed(chunk_id, error)
        print(f"Error while processing chunk {chunk_id} of shard {self.shard_name}", flush=True)
        self.parent_ref.shard_failed.remote(self.shard_name, error)

    def _finished_chunk(self, idx: int, chunk: ChunkMetadata):
        if (idx < self.metadata_writer.num_chunks) or (
            self._expected_num_chunks is not None and idx >= self._expected_num_chunks
        ):
            logger.error(f"Received chunk {idx} for {self.shard_name} but it's already finished")
            error = RuntimeError(f"Received chunk {idx} for {self.shard_name} but it's already finished")
            self.parent_ref.shard_failed.remote(self.shard_name, ser_exc_info(error))
            raise error

        heapq.heappush(self.uncommited_chunks, (idx, chunk))
        self._attempt_to_commit_chunks()

    def shard_finished_reading(self, expected_num_chunks: int):
        # TODO: add metadata that we're done reading to metrics
        self._expected_num_chunks = expected_num_chunks
        self._attempt_to_commit_chunks()

    def shard_failed(self, error: ExceptionInfo):
        self.parent_ref.shard_failed.remote(self.shard_name, error)

    def _attempt_to_commit_chunks(self):
        chunks_committed = []
        while len(self.uncommited_chunks) > 0 and self.uncommited_chunks[0][0] == self.metadata_writer.num_chunks:
            _, chunk = heapq.heappop(self.uncommited_chunks)
            chunk_number = self.metadata_writer.num_chunks
            logger.debug(f"Committing chunk {chunk.name} of shard {self.shard_name}. It is chunk {chunk_number}")
            self.metadata_writer.commit_chunk(chunk)
            chunks_committed.append(chunk)

        if len(chunks_committed) > 0:
            if self.finished:
                raise RuntimeError("Tried to commit chunks after shard finished")
            # TODO: this is called inside an async call so we need to not block, but we do need to sequence
            # this to come before the shard_finished
            self.parent_ref.new_chunk.remote(self.shard_name, *chunks_committed)

        if not self.finished and self.metadata_writer.num_chunks == self._expected_num_chunks:
            self.metadata_writer.finish()
            self.finished = True
            self.parent_ref.shard_finished.remote(self.shard_name, self._expected_num_chunks)


class _ChunkCollator:
    """
    This class is responsible for taking batches from the processor and writing them to disk in order.
    It also handles the logic of when to commit chunks to disk.

    For each chunk (that is has data for and hasn't finished), it keeps a heapq of batches that have been
    processed but not yet written to disk. When a new batch comes in, it checks if it's the next batch in the
    chunk. If so, it writes it to disk and flushes any other batches that are ready to be written.

    A chunk isn't finished until it's received all the batches it's expecting and it knows how many batches
    to expect.

    """

    def __init__(self, cache_dir: str, shard_name: str):
        self.cache_dir = cache_dir
        self.shard_name = shard_name
        self.chunk_writers: dict[int, _ChunkWriter] = {}  # chunk index -> writer
        self.batch_counts: dict[int, int] = {}  # chunk index -> number of batches written
        self.expected_totals: dict[int, int] = {}  # chunk index -> expected num batches.
        self.failed_chunks: dict[int, ExceptionInfo] = {}  # chunk index -> error
        self.chunk_partial_batches: dict[
            int, list[tuple[int, pa.RecordBatch]]
        ] = {}  # chunk index -> heapq of (batch index, batch)

    def new_batch(self, chunk_id, batch_idx, batch) -> Optional[ChunkMetadata]:
        if chunk_id not in self.chunk_partial_batches:
            self.chunk_partial_batches[chunk_id] = []
            self.batch_counts[chunk_id] = 0

        heapq.heappush(self.chunk_partial_batches[chunk_id], (batch_idx, batch))

        return self._attempt_to_write_chunk_fragments(chunk_id)

    def chunk_finished_reading(self, chunk_id, expected_num_batches) -> Optional[ChunkMetadata]:
        self.expected_totals[chunk_id] = expected_num_batches
        return self._attempt_to_write_chunk_fragments(chunk_id)

    def chunk_failed(self, chunk_id, error: ExceptionInfo):
        self.failed_chunks[chunk_id] = error
        if chunk_id in self.chunk_writers:
            self.chunk_writers[chunk_id].__exit__(*error.restore())
            del self.chunk_writers[chunk_id]

    def _attempt_to_write_chunk_fragments(self, chunk_id) -> Optional[ChunkMetadata]:
        if chunk_id in self.failed_chunks:
            logger.error(f"Chunk {chunk_id} of shard {self.shard_name} already failed, not writing more")
            raise self.failed_chunks[chunk_id].restore()

        if chunk_id in self.chunk_partial_batches:
            chunk_batches = self.chunk_partial_batches[chunk_id]

            while len(chunk_batches) > 0:
                batch_id, batch = chunk_batches[0]
                if batch_id != self.batch_counts[chunk_id]:
                    break

                # we can write this batch
                batch_id, batch = heapq.heappop(chunk_batches)

                if chunk_id not in self.chunk_writers:
                    assert batch_id == 0, f"Expected batch 0 but got {batch_id}"
                    chunk_name = os.path.join(self.shard_name, f"chunk-{chunk_id}")
                    writer = _ChunkWriter(self.cache_dir, chunk_name, batch.schema)
                    writer.__enter__()
                    self.chunk_writers[chunk_id] = writer

                self.chunk_writers[chunk_id].write_batch(batch)
                self.batch_counts[chunk_id] += 1

        if chunk_id not in self.batch_counts:
            return None

        if chunk_id in self.expected_totals and self.batch_counts[chunk_id] == self.expected_totals[chunk_id]:
            assert len(chunk_batches) == 0
            # we're done with this chunk
            writer = self.chunk_writers[chunk_id]
            writer.__exit__(None, None, None)
            del self.chunk_writers[chunk_id]
            del self.batch_counts[chunk_id]
            del self.chunk_partial_batches[chunk_id]
            return writer.get_metadata()
        else:
            return None


@ray.remote(num_cpus=0.5)  # keep this small b/c it doesn't do a lot
class ChunkCacheBuilder(SnitchRecipient):
    """
    Actor that manages the in-progress global ordering on chunks. ChunkCacheWriter's job is to hold the list of all
    chunks as well as chunks from each shard while caching is running.

    This is a separate actor from the ChunkCacheBroker because
    we need something that gets messages from shards in-order, and async methods make actors
    lose that property.
    """

    def __init__(
        self,
        broker_ref,
        cache_dir: str,
        name: str,
        source: ShardedDataset[T],
        processor: BatchProcessor[T],
        rows_per_chunk: int,
    ):
        with log_failures_to(broker_ref):
            pylogging.basicConfig(level=pylogging.INFO, format=LOG_FORMAT)
            self.logger = pylogging.getLogger(f"{__name__}.{name}")
            self.broker_ref = broker_ref
            self.shard_status: Dict[str, _ShardStatus] = dict()
            self._current_round_robin = []
            self.source = source
            self._metrics = InProgressCacheMetrics()

            self_ref = current_actor_handle()

            if len(source.shard_names) == 0:
                self.logger.warning("No shards to index?!?")
                self._finish()
            else:
                self.logger.info(f"Starting cache build for {len(source.shard_names)} shards")

                self._shard_writers = []
                self._shard_readers = []
                self._processor_actors = []

                for shard_name in source.shard_names:
                    self._current_round_robin.append(shard_name)
                    self.shard_status[shard_name] = _ShardStatus()

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

                def priority_fn(shard_idx, chunk_idx):
                    return chunk_idx * num_shards + shard_idx

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
                        num_rows_per_chunk=rows_per_chunk,
                        group_id=group_id,
                    )

                    # we want global names so that different tasks can coordinate priorities
                    worker_to_assign = (group_id + group_offset) % num_worker_groups
                    priority_actor_name = f"priority_processor.{worker_to_assign}"

                    reader_actor = PriorityProcessorActor.options(  # type: ignore
                        name=priority_actor_name, get_if_exists=True
                    ).remote()

                    reader_actor.assign_work.remote(work_item)

                    self._shard_readers.append(reader_actor)

    def new_chunk(self, shard_name: str, *chunks: ChunkMetadata):
        """Callback method for when a shard worker has produced a new chunk."""
        self.shard_status[shard_name].current_buffer.extend(chunks)

        # if we have buffered chunks, we need to check if we can send them to the broker
        self._attempt_to_flush_buffers()

        self._metrics.chunks_finished += len(chunks)
        # update metrics
        for chunk in chunks:
            self._metrics.rows_finished += chunk.num_rows
            for field, count in chunk.field_counts.items():
                self._metrics.field_counts[field] = self._metrics.field_counts.get(field, 0) + count

        if len(chunks) > 0:
            ray.get(self.broker_ref._new_metrics.remote(self._metrics))

    def shard_finished(self, shard_name: str, expected_num_chunks: int):
        """Callback method for when a shard worker has finished."""
        shard_status = self.shard_status[shard_name]
        assert (
            shard_status.expected_num_chunks is None
        ), f"Shard {shard_name} already finished: {shard_status.expected_num_chunks} {expected_num_chunks}"
        shard_status.expected_num_chunks = expected_num_chunks

        # we might still have buffered chunks, so we need to check if we can append them
        self._attempt_to_flush_buffers()
        self._metrics.shards_finished += 1
        ray.get(self.broker_ref._new_metrics.remote(self._metrics))

        # if there are no more active shards, we're done
        if self._all_shards_done():
            assert len(self._current_round_robin) == 0
            self._finish()

    def _all_shards_done(self):
        return all(status.is_finished_and_buffer_empty for status in self.shard_status.values())

    def shard_failed(self, shard_name: str, error: ExceptionInfo):
        """Callback method for when a shard worker has failed."""
        ray.get(self.broker_ref._writer_exception.remote(shard_name, error))

    def other_failed(self, error: ExceptionInfo):
        """Callback method for when a shard worker has failed."""
        ray.get(self.broker_ref._writer_exception.remote(None, error))

    def _attempt_to_flush_buffers(self):
        # this is the most complex logic in this class.
        # The global order on chunks is defined as a roundrobin over shards, until one shard is done.
        # After that, that shard is removed from the roundrobin and the process continues.
        # Roundrobin order is determined by self.source.shard_names

        # We are happy to release chunks that form a prefix of the global order so that they can be read.
        # To do that, we maintain the roundrobin order in self._current_round_robin
        # and we maintain the current buffer for each shard in self.shard_status.
        # When we get a new chunk, we append it to the buffer for that shard.
        # When we get a finished message, we mark that shard as finished.
        # In either case, we check if we can send any chunks from the front of the roundrobin.
        # If we can, we send them to the broker

        # here "finished" means that the shard has sent all of its chunks and has told us that it's done.

        chunks_to_send = []

        while len(self._current_round_robin) > 0:
            name = self._current_round_robin[0]
            status = self.shard_status[name]
            if status.is_finished_and_buffer_empty:
                # we're done with this shard, so we can remove it from the roundrobin
                self._current_round_robin.pop(0)
                logger.debug(f"Shard {name} is finished, removing from round robin")
                continue

            # now let's see if we can send a chunk from this shard
            next_chunk = status.pop_chunk_to_send()
            if next_chunk is not None:
                # we can send a chunk from this shard
                self._current_round_robin.pop(0)
                self._current_round_robin.append(name)
                chunks_to_send.append(next_chunk)
                continue
            else:
                # we can't send a chunk from this shard, so we can't send any additional chunks
                if self.logger.level <= pylogging.DEBUG:
                    chunks_waiting = [
                        f"{n2} ({len(s2.current_buffer)})"
                        for n2, s2 in self.shard_status.items()
                        if len(s2.current_buffer) > 0
                    ]
                    msg = (
                        f"Shard {name} has no chunks to send and is not known to be finished. We have this many queued"
                        f" chunks: {chunks_waiting}"
                    )
                    self.logger.debug(msg)
                break

        if len(chunks_to_send) > 0:
            logger.debug(f"Sending {len(chunks_to_send)} chunks to broker")
            ray.get(self.broker_ref._append_chunks.remote(*chunks_to_send))

    def _finish(self):
        self._metrics.is_finished = True
        ray.get(self.broker_ref._new_metrics.remote(self._metrics))
        ray.get(self.broker_ref._finalize.remote())
        # self._shard_writers = []
        # self._shard_readers = []


@ray.remote(num_cpus=0)
class ChunkCacheBroker(SnitchRecipient):
    """Actor that manages the global order on chunks and vends chunk metadata to readers."""

    chunks: List[ChunkMetadata]
    _reader_promises: Dict[int, asyncio.Future[ChunkMetadata]]
    _finished_promise: asyncio.Future[None]

    def __init__(
        self,
        cache_dir: str,
        source: ShardedDataset[T],
        processor: BatchProcessor[T],
        rows_per_chunk: int,
        cache_config: Optional[Dict[str, Any]],
    ):
        pylogging.basicConfig(level=pylogging.INFO, format=LOG_FORMAT)
        self.chunks = []
        self._reader_promises = {}
        self._is_finished = False
        self._source = source
        self._processor = processor
        self._cache_dir = cache_dir
        self._rows_per_chunk = rows_per_chunk
        self._finished_promise = asyncio.Future()
        # used to subscribe to metrics updates
        self._latest_metrics = InProgressCacheMetrics()
        self._metrics_condition = asyncio.Condition()
        self._cache_config = cache_config
        path_for_name = os.path.join(*self._cache_dir.split("/")[-2:])
        name = f"broker::{path_for_name}"
        self.logger = pylogging.getLogger(f"{name}")

        # initialize writer task
        # first see if we need to do anything: check the ledger for is_finished
        try:
            cache_ledger = _load_cache_ledger(self._cache_dir)
            self._append_chunks(*cache_ledger.chunks)
            self._is_finished = True
            self._finished_promise.set_result(None)
        except FileNotFoundError:
            self_ref = ray.runtime_context.get_runtime_context().current_actor
            # only use the last two components of the name since it gets kind of long
            name = f"builder::{path_for_name}"
            self._builder_actor = ChunkCacheBuilder.remote(  # type: ignore
                self_ref,
                self._cache_dir,
                name,
                self._source,
                self._processor,
                rows_per_chunk,
            )  # type: ignore

    def is_finished(self):
        return self._is_finished

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

    async def get_chunk(self, chunk_idx: int) -> Optional[ChunkMetadata]:
        assert isinstance(self.chunks, list), self.chunks
        if chunk_idx < len(self.chunks):
            return self.chunks[chunk_idx]
        elif self._is_finished:
            return None
        elif self._finished_promise.exception() is not None:
            raise self._finished_promise.exception()  # type: ignore
        else:
            if chunk_idx not in self._reader_promises:
                self._reader_promises[chunk_idx] = asyncio.Future()
            return await self._reader_promises[chunk_idx]

    async def final_chunk_count(self) -> Optional[int]:
        if self._is_finished:
            return len(self.chunks)
        else:
            return None

    def _append_chunks(self, *chunks: ChunkMetadata):
        for chunk in chunks:
            self.chunks.append(chunk)
            chunk_idx = len(self.chunks) - 1
            self.logger.debug(f"Received chunk {chunk_idx}")
            if chunk_idx in self._reader_promises:
                self.logger.debug(f"Resolving promise for chunk {chunk_idx}")
                self._reader_promises[chunk_idx].set_result(chunk)
                del self._reader_promises[chunk_idx]

    def _new_metrics(self, metrics):
        self._latest_metrics = metrics
        self._do_notify()

    def _do_notify(self):
        async def _do_notify_async():
            async with self._metrics_condition:
                self._metrics_condition.notify_all()

        asyncio.create_task(_do_notify_async())

    def _child_failed(self, child: ray.actor.ActorHandle, exception: ExceptionInfo):
        try:
            super()._child_failed(child, exception)
        except Exception as e:
            logger.exception("Error in child_failed")
            self._writer_exception(None, ser_exc_info(e))

    def _writer_exception(self, shard_name, exc_info: ExceptionInfo):
        info = exc_info.restore()

        logger.exception(f"Writer task {shard_name} failed with exception", exc_info=info)
        for future in self._reader_promises.values():
            future.set_exception(info[1])

        self._reader_promises = {}

        self._finished_promise.set_exception(info[1])
        self._do_notify()

    def _finalize(self):
        logger.info(f"Finalizing cache {self._cache_dir}...")
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


def _get_broker_actor(
    cache_dir,
    input_shards,
    processor,
    cache_config=None,
    rows_per_chunk=DEFAULT_ROWS_PER_CHUNK,
):
    return ChunkCacheBroker.options(
        name="lev_cache_manager::" + cache_dir.replace("/", "--"), get_if_exists=True, lifetime="detached"
    ).remote(
        # type: ignore
        cache_dir=cache_dir,
        source=input_shards,
        processor=processor,
        cache_config=cache_config,
        rows_per_chunk=rows_per_chunk,
    )


class DictCacheDataset(ShardableDataset[dict]):
    """
    A Dataset that yields HF BatchEncodings from a ShardCache.
    This basically yields a dict-of-arrays, just the HF BatchEncoding class version of dict.
    """

    def __init__(self, cache: "ShardCache", return_batches: bool = False):
        self.cache = cache
        self.return_batches = return_batches

    def __iter__(self) -> Iterator[dict]:
        for batch in self.cache:
            encoding = dict_from_record_batch(batch)

            if self.return_batches:
                yield encoding
            else:
                batch_size = 0
                for v in encoding.values():
                    batch_size = len(v)
                    break

                for i in range(batch_size):
                    yield {k: v[i] for k, v in encoding.items()}

    def shard(self, shard_id: int, num_shards: int) -> "DictCacheDataset":
        return DictCacheDataset(self.cache.shard(shard_id, num_shards))

    @staticmethod
    def load(cache_dir: str, return_batches: bool = False, batch_size: Optional[int] = None) -> "DictCacheDataset":
        if batch_size is None:
            batch_size = 1
        cache = ShardCache.load(cache_dir, batch_size=batch_size)
        return DictCacheDataset(cache, return_batches=return_batches)


class ShardCache(Iterable[pa.RecordBatch]):
    """A cache which is backed by a collection of chunks of preprocessed documents. These chunks
    are produced by tokenizing/preprocessing a ShardedDataset.

        This is the main interface for building and reading from a shard cache.

        ShardCache has the following objectives:

        1) Deterministic ordering over the data
        2) Sharded reading
        3) Sharded writing
        4) Simultaneous reading and writing of shards
        5) Fast resumption of writing
        6) Fast resumption of reading

        ShardCache achieves (1), (2), and (3) maintaining a reproducible global ordering over "chunks" created from shards.
        The global ordering is defined by taking chunks round-robin from each shard. This allows us to read shards
        in parallel and deterministically.

        ShardCache achieves (4) also via the chunking mechanism. As soon as all shards have written a chunk, the next
        chunk can be read. This allows us to read and write in parallel.

        ShardCache achieves (5) by writing chunks to disk as soon as they are completed and serializing a state
        of the chunks that have been written for each shard. This allows us to resume from the last chunk that was written.

        # TODO (6) isn't implemented just yet

        ShardCache achieves (6) by storing metadata about the chunks that have been written in a state. In addition
        to the global ordering, the state also stores the number of documents in each chunk as well as the number
        of tokens.
    """

    ledger: Optional[CacheLedger]
    _broker: Optional[ActorHandle]
    # We use a thread here instead of an actor because we want to ensure it's on the same process as the ShardCache
    # object.
    _monitor_thread: Optional[threading.Thread]
    _metrics_monitors: List[MetricsMonitor]

    def __init__(
        self,
        cache_dir: str,
        batch_size: int,
        ledger: Optional[CacheLedger],
        _broker: Optional[ActorHandle],
        reader_offset: int = 0,
        num_readers: int = 1,
    ):
        self.cache_dir = cache_dir
        self.ledger = ledger
        self._broker = _broker
        self._batch_size = batch_size

        self._metrics_monitors = []
        self._monitor_thread = None

        self._num_readers = num_readers
        self._reader_offset = reader_offset
        name = os.path.join(*cache_dir.split("/")[-2:])
        self.logger = pylogging.getLogger(f"ShardCache.{name}")

    @staticmethod
    def load(cache_dir: str, batch_size: int) -> "ShardCache":
        """Loads a cache from disk. Raises FileNotFoundError if the cache doesn't exist"""
        logger.info(f"Loading cache from {cache_dir}")
        ledger = _load_cache_ledger(cache_dir)
        return ShardCache(cache_dir, batch_size, ledger, None)

    @staticmethod
    def build_or_load(
        cache_dir: str,
        shard_source: ShardedDataset[T],
        processor: BatchProcessor[T],
        batch_size: int,
        rows_per_chunk: int,
        cache_config: Optional[Dict[str, Any]] = None,
    ):
        try:
            return ShardCache.load(cache_dir, batch_size)
        except FileNotFoundError:
            broker = _get_broker_actor(
                cache_dir=cache_dir,
                input_shards=shard_source,
                processor=processor,
                cache_config=cache_config,
                rows_per_chunk=rows_per_chunk,
            )
            return ShardCache(cache_dir=cache_dir, batch_size=batch_size, ledger=None, _broker=broker)

    def finished_sentinel(self):
        """Returns a Ray-awaitable object that will be set when the cache is finished"""
        if self._broker is None:
            return ray.remote(num_cpus=0)(lambda: None).remote()
        else:
            return self._broker.finished_sentinel.remote()

    @property
    def is_finished(self):
        """Returns whether the cache is finished"""
        if self._broker is None:
            return True
        else:
            return ray.get(self._broker.is_finished.remote())

    def read_chunk(self, chunk_idx: int) -> Iterator[pa.RecordBatch]:
        """Reads a chunk from the cache"""
        chunk = self.get_chunk(chunk_idx)
        yield from self._read_chunk(chunk)

    def _map_index(self, index):
        return index * self._num_readers + self._reader_offset

    def get_chunk(self, index: int, *, timeout: Optional[float] = None) -> ChunkMetadata:
        """Returns the metadata for a given chunk index"""
        mapped_index = self._map_index(index)
        return self._get_chunk_unmapped(mapped_index, timeout=timeout)

    def _get_chunk_unmapped(self, mapped_index: int, *, timeout: Optional[float] = None) -> ChunkMetadata:
        if self.ledger is not None:
            return self.ledger.chunks[mapped_index]
        else:
            assert self._broker is not None
            time_in = time.time()
            next_time = time_in
            # we want to also log if we're waiting for a long time, so we do this in a loop
            while timeout is None or next_time - time_in < timeout:
                current_timeout = 20.0
                if timeout is not None:
                    current_timeout = min(current_timeout, timeout - (next_time - time_in))
                try:
                    chunk = ray.get(self._broker.get_chunk.remote(mapped_index), timeout=current_timeout)
                except GetTimeoutError:
                    self.logger.warning(f"Waiting for chunk {mapped_index} for {int(next_time - time_in)} seconds")
                    next_time = time.time()
                    current_timeout *= 2
                    current_timeout = min(current_timeout, 100)
                    continue
                except asyncio.exceptions.InvalidStateError:
                    self.logger.warning(
                        f"Invalid state waiting for chunk {mapped_index} for {int(next_time - time_in)} seconds"
                    )
                    next_time = time.time()
                    current_timeout *= 2
                    current_timeout = min(current_timeout, 100)
                    time.sleep(current_timeout)
                    continue

                if chunk is None:
                    raise IndexError(f"Chunk index out of bounds. (Mapped index {mapped_index})")

                return chunk

            if timeout is not None:
                raise TimeoutError(f"Timeout while waiting for chunk {mapped_index}")

    async def get_chunk_async(self, index: int) -> ChunkMetadata:
        """Returns the metadata for a given chunk index"""
        mapped_index = self._map_index(index)
        if self.ledger is not None:
            return self.ledger.chunks[mapped_index]
        else:
            assert self._broker is not None
            chunk = await self._broker.get_chunk.remote(mapped_index)
            if chunk is None:
                raise IndexError(f"Chunk index {index} out of bounds. (Mapped index {mapped_index})")
            return chunk

    def final_chunk_count(self) -> Optional[int]:
        """Returns the number of chunks in the cache, if known"""
        if self.ledger is not None:
            return len(self.ledger.chunks)
        else:
            assert self._broker is not None
            return ray.get(self._broker.final_chunk_count.remote())

    def iter_batches_from_chunks(self, loop: bool = False):
        shard_offset = self._reader_offset

        if self.ledger is not None:
            num_chunks = len(self.ledger.chunks)

            if num_chunks == 0:
                return

            while True:
                i = 0
                for i in range(shard_offset, num_chunks, self._num_readers):
                    chunk = self.ledger.chunks[i]
                    yield from self._read_chunk(chunk)

                if not loop:
                    break

                shard_offset = i % len(self.ledger.chunks)
        else:
            assert self._broker is not None
            i = shard_offset
            while True:
                try:
                    self.logger.debug(f"Reading chunk {i}")
                    chunk = self._get_chunk_unmapped(i)
                    i += self._num_readers
                    yield from self._read_chunk(chunk)
                except IndexError:
                    if loop:
                        num_chunks = ray.get(self._broker.final_chunk_count.remote())
                        assert num_chunks is not None

                        i = i % num_chunks
                    else:
                        break
                except Exception as e:
                    self.logger.exception("Error while reading from shard cache.")
                    raise e

    def __iter__(self):
        return self.iter_batches_from_chunks()

    def shard(self, offset, num_readers):
        """
        Returns a shard of this shard cache. This method shards w.r.t the current shard cache, not the base shard cache.

        Args:
            offset:
            num_readers:

        Returns:
            (ShardCache): A shard of this shard cache.
        """
        if offset >= num_readers:
            raise ValueError(f"Shard index {offset} is out of range")

        if num_readers == 1:
            return self

        new_offset = self._reader_offset * num_readers + offset
        new_num_readers = self._num_readers * num_readers
        return ShardCache(self.cache_dir, self._batch_size, self.ledger, self._broker, new_offset, new_num_readers)

    def unshard(self):
        """
        Gets the "base" shard cache that this shard cache is a shard of.
        """
        return ShardCache(self.cache_dir, self._batch_size, self.ledger, self._broker, 0, 1)

    def with_batch_size(self, batch_size):
        return ShardCache(
            self.cache_dir, batch_size, self.ledger, self._broker, self._reader_offset, self._num_readers
        )

    def _read_chunk(self, chunk):
        reader = _ChunkReader.from_metadata(self.cache_dir, chunk, self._batch_size)
        for batch in reader:
            yield batch

    def await_finished(self, timeout: Optional[float] = None):
        return ray.get(self.finished_sentinel(), timeout=timeout)

    def attach_metrics_monitor(self, monitor: MetricsMonitor):
        if self._broker is None:
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


class _ChunkReader:
    """Reads batches of documents from a chunk"""

    metadata: ChunkMetadata
    file: pq.ParquetFile
    batch_size: int

    # TODO: seek by doc
    # TODO: seek by token etc

    def __init__(self, metadata: ChunkMetadata, file: pq.ParquetFile, batch_size: int):
        self.metadata = metadata
        self.file = file
        self.batch_size = batch_size

    def with_batch_size(self, batch_size):
        return _ChunkReader(self.metadata, self.file, batch_size)

    @property
    def num_docs(self):
        return self.metadata.num_rows

    def field_count(self, field, default=None):
        return self.metadata.field_counts.get(field, default)

    @property
    def __len__(self):
        return (self.num_docs + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[pa.RecordBatch]:
        for record_batch in self.file.iter_batches(batch_size=self.batch_size):
            yield record_batch

    @staticmethod
    def from_metadata(cache_dir, metadata: ChunkMetadata, batch_size: int) -> "_ChunkReader":
        file = pq.ParquetFile(fsspec.open(os.path.join(cache_dir, f"{metadata.name}.parquet"), "rb").open())
        return _ChunkReader(metadata, file, batch_size)
