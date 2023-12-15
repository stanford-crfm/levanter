# Dataset for preprocessing data, tokenizing, and caching to disk.
import asyncio
import dataclasses
import heapq
import logging
import os
import threading
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from queue import PriorityQueue
from typing import IO, Any, Callable, Dict, Iterable, Iterator, List, Optional, Protocol, Sequence, TypeVar, Union

import fsspec.core
import jax
import pyarrow as pa
import pyarrow.parquet as pq
import ray
import wandb
from dataclasses_json import dataclass_json
from fsspec import AbstractFileSystem
from ray.actor import ActorHandle
from ray.exceptions import GetTimeoutError
from ray.util.queue import Queue
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from levanter.utils.ray_utils import ExceptionInfo, RefBox, ser_exc_info

from . import ShardableDataset
from ._preprocessor import BatchProcessor, BatchResult, as_record_batch, dict_from_record_batch
from ._priority_work_queue import _QueueItem
from .sharded_dataset import ShardedDataset


T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


logger = logging.getLogger(__name__)

DEFAULT_ROWS_PER_CHUNK = 1024 * 32
LEDGER_FILE_NAME = "cache_ledger.json"


def build_cache(
    cache_dir: str,
    input_shards: ShardedDataset[T],
    processor: BatchProcessor[T],
    batch_size: int = 1,
    rows_per_chunk: int = DEFAULT_ROWS_PER_CHUNK,
    await_finished: bool = True,
    monitors: Optional[Sequence["MetricsMonitor"]] = None,
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
        batch_size: The number of input examples to process at once.
        rows_per_chunk: The number of rows to write to each chunk. May be smaller at the end of a shard.
        await_finished: If True, this function will block until the cache is finished. If False, it will return
                    immediately.
        monitors: a list of MetricsMonitors to attach to the cache. These will be called periodically with
            metrics about the cache build process. If None, will add a LoggerMetricsMonitor.

    Returns:
       (ShardCache) A ShardCache object that can be used to read the cache.

    """
    # first see if we need to do anything
    cache = ShardCache.build_or_load(cache_dir, input_shards, processor, batch_size, rows_per_chunk)

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


# this class is used as the state for _produce_chunks_for_shard
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


class SerialCacheWriter(AbstractContextManager):
    """
    Writes ShardCache-compatible caches to disk. This is a serial version of ShardCacheWriter that doesn't use Ray.
    Mostly for scripts and debugging.

    Examples:
        >>> with SerialCacheWriter(cache_dir, rows_per_chunk=1024) as writer:
        ...     for batch in process_batches():
        ...         writer.write_batch(batch)
    """

    def __init__(self, cache_dir: str, rows_per_chunk: int = DEFAULT_ROWS_PER_CHUNK):
        if rows_per_chunk <= 0:
            raise ValueError("rows_per_chunk must be positive")
        self.cache_dir = cache_dir
        self._rows_per_chunk = rows_per_chunk
        self._chunks: List[ChunkMetadata] = []
        self._current_chunk_writer: Optional[_ChunkWriter] = None
        self._is_closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # if successful, write the ledger
        if self._current_chunk_writer is not None:
            self._current_chunk_writer.__exit__(exc_type, exc_val, exc_tb)
            self._chunks.append(self._current_chunk_writer.get_metadata())
            self._current_chunk_writer = None

        if exc_type is None:
            _serialize_json_and_commit(os.path.join(self.cache_dir, LEDGER_FILE_NAME), CacheLedger(self._chunks))
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


@ray.remote(num_cpus=0.0, scheduling_strategy="SPREAD")  # type: ignore
def _produce_cache_for_shard(
    sink: ActorHandle,  # _ChunkCacheBuilder
    source: ShardedDataset[T],
    priority_fn: Callable[[int, int], float],
    shard_idx: int,
    processor: BatchProcessor,
    process_queue: ActorHandle,  # _BatchProcessorQueue
    cache_dir: str,
    rows_per_chunk: int,
):
    """Produces chunks of preprocessed data from a single shard and writes them to disk. Chunks are written to sink,
    which is an actor of ChunkCacheBuilder."""
    # TODO: thread logging level through calls
    logging.basicConfig(level=logging.INFO)
    # load or create shard metadata (for recovery)
    try:
        shard_name = source.shard_names[shard_idx]
        metadata_path = os.path.join(cache_dir, f"{shard_name}.json")
        shard_writer = _ShardMetadataWriter(metadata_path)

        # yield from existing chunks
        if len(shard_writer.chunks) > 0:
            logger.debug(f"Yielding {len(shard_writer.chunks)} already finished chunks from {shard_name}")
            sink.new_chunk.remote(shard_name, *shard_writer.chunks)

        if not shard_writer.is_finished:
            _produce_chunks_for_shard(
                sink, cache_dir, shard_writer, source, priority_fn, shard_idx, processor, process_queue, rows_per_chunk
            )
            shard_writer.finish()

        sink.shard_finished.remote(shard_name)

    except Exception as e:
        logger.exception(f"Error while processing shard {shard_name}")
        ray.get(sink.shard_failed.remote(shard_name, ser_exc_info()))
        raise e


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
# We want to prioritize so that we read 1 chunks worth of batches from each shard before reading more from any shard.
# We also want to prioritize reading earlier shards before later shards (within a chunk generation round).
# Ray also seems to get upset about having too many processes, and we can't serialize the iterators over shards.


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


# This class is responsible for reading batches from a set of shards, prioritizing earlier
# chunks and earlier shards. (So that we approximately generate following the global order.)
@ray.remote(num_cpus=1, scheduling_strategy="SPREAD")
def _alternating_shard_reader(
    builder_ref: ActorHandle,  # _ChunkCacheBuilder
    shard_writers: dict[str, ActorHandle],  # _ShardWriterWorker
    shard_source: ShardedDataset[T],
    shard_idxes: Sequence[int],
    priority_fn: Callable[[int, int], float],
    processor_actor: ActorHandle,  # _BatchProcessorQueue
    batch_size,
    num_rows_per_chunk,
):
    shard_pqueue: list[tuple[int, int]] = []  # heapq of (num_chunks, shard_idx)
    shard_readers: dict[int, Iterator[list[T]]] = {}
    try:
        shard_metadatas = _initial_shard_metadatas(shard_source, shard_idxes, shard_writers)
    except Exception as e:
        builder_ref.other_failed.remote(ser_exc_info())
        raise e

    batch_size = min(batch_size, num_rows_per_chunk)

    for shard_idx in shard_idxes:
        shard_name = shard_source.shard_names[shard_idx]
        try:
            shard_metadata = shard_metadatas[shard_idx]
            heapq.heappush(shard_pqueue, (len(shard_metadata.chunks), shard_idx))
            shard_readers[shard_idx] = _shard_reader_generator(
                shard_source, shard_idx, shard_metadata.total_rows, batch_size
            )
        except Exception as e:  # noqa
            logger.exception(f"Error while initializing shard {shard_name}")
            ray.get(shard_writers[shard_name].shard_failed.remote(ser_exc_info()))
            raise e

    while len(shard_pqueue) > 0:
        chunk_id, shard_idx = heapq.heappop(shard_pqueue)
        shard_name = shard_source.shard_names[shard_idx]
        try:
            shard_iter = shard_readers[shard_idx]
            writer = shard_writers[shard_source.shard_names[shard_idx]]

            exhausted_shard = False

            chunk_batch_idx = 0
            chunk_filled = False
            total_chunk_rows = 0

            while not chunk_filled:
                batch = next(shard_iter, None)
                if batch is None:
                    exhausted_shard = True
                    break

                exhausted_shard = len(batch) < batch_size
                total_chunk_rows += len(batch)

                if batch:
                    priority = priority_fn(shard_idx, chunk_id)
                    batch = ray.put(batch)
                    batch_result_ref = ray.get(processor_actor.submit.remote(priority=priority, batch=RefBox(batch)))
                    writer.chunk_batch_finished.remote(chunk_id, chunk_batch_idx, batch_result_ref)
                    chunk_batch_idx += 1

                if total_chunk_rows >= num_rows_per_chunk or exhausted_shard:
                    chunk_filled = True

            if chunk_batch_idx > 0:
                writer.chunk_finished_reading.remote(chunk_id, chunk_batch_idx)
                chunk_id += 1

            if exhausted_shard:
                writer.shard_finished_reading.remote(chunk_id)
                del shard_readers[shard_idx]
                del shard_metadatas[shard_idx]
            else:
                # we're not done with this shard, so put it back in the queue
                heapq.heappush(shard_pqueue, (chunk_id, shard_idx))

        except Exception as e:  # noqa
            logger.exception(f"Error while processing shard {shard_name}")
            ray.get(shard_writers[shard_name].shard_failed.remote(ser_exc_info()))
            raise e


def _initial_shard_metadatas(shard_source, shard_idxes, shard_writers):
    shard_metadatas: dict[int, ShardMetadata] = {}
    _metadata_futures = [
        shard_writers[shard_source.shard_names[shard_idx]].current_metadata.remote() for shard_idx in shard_idxes
    ]
    shard_metadatas_rs = ray.get(_metadata_futures)
    for shard_idx, shard_metadata in zip(shard_idxes, shard_metadatas_rs):
        shard_metadatas[shard_idx] = shard_metadata
    return shard_metadatas


def _produce_chunks_for_shard(
    sink,
    cache_dir,
    shard_writer,
    source,
    priority_fn,
    shard_idx,
    processor: BatchProcessor,
    process_queue,
    rows_per_chunk,
):
    shard_name = source.shard_names[shard_idx]
    total_rows_written = sum(chunk.num_rows for chunk in shard_writer.chunks)
    if total_rows_written > 0:
        logger.info(f"Resuming shard {shard_name} at row {total_rows_written}")
    else:
        logger.info(f"Starting shard {shard_name}")

    target_batch_size = min(processor.batch_size, rows_per_chunk)

    def yield_chunk(chunk: ChunkMetadata):
        logger.debug(f"Yielding new chunk {chunk.name} from {shard_name} with {chunk.num_rows} rows")
        shard_writer.commit_chunk(chunk)
        sink.new_chunk.remote(shard_name, chunk)
        return chunk.num_rows

    def submit_task_for_processing(batch):
        priority = priority_fn(shard_idx, shard_writer.num_chunks)
        output_batch_future = ray.get(process_queue.submit.remote(priority=priority, batch=RefBox(batch)))
        return output_batch_future

    def open_iterator():
        logger.info(f"Starting to get rows for shard {shard_name}")
        return source.open_shard_at_row(shard_name, total_rows_written)

    result_iterator = _map_batches_in_parallel(
        open_iterator, submit_task_for_processing, batch_size=target_batch_size, max_queue_size=10
    )

    writer: Optional[_ChunkWriter] = None
    for processed_batch in result_iterator:
        if writer is None:
            chunk_name = os.path.join(shard_name, f"chunk-{shard_writer.num_chunks}")
            writer = _ChunkWriter(cache_dir, chunk_name, processed_batch.schema)
            writer.__enter__()

        writer.write_batch(processed_batch)

        if writer.num_rows >= rows_per_chunk:
            writer.__exit__(None, None, None)
            chunk = writer.get_metadata()
            writer = None
            yield_chunk(chunk)

    if writer is not None:
        writer.__exit__(None, None, None)
        chunk = writer.get_metadata()
        writer = None
        yield_chunk(chunk)


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


# TODO: should we just make the ledger have all this?
@dataclass_json
@dataclass
class InProgressCacheMetrics:
    rows_finished: int = 0
    chunks_finished: int = 0
    shards_finished: int = 0
    field_counts: Dict[str, int] = dataclasses.field(default_factory=dict)
    is_finished: bool = False


class MetricsMonitor(Protocol):
    def __call__(self, metrics: InProgressCacheMetrics):
        ...


class RichMetricsMonitor(MetricsMonitor):

    progress: Optional[Progress]  # type: ignore
    task: Optional[TaskID]

    def __init__(self, num_shards, **kwargs):
        """kwargs are passed to rich.progress.Progress"""
        self.kwargs = kwargs
        self.progress: Optional[Progress] = None
        self.task = None
        self.num_shards = num_shards

    def __call__(self, metrics: InProgressCacheMetrics):
        if self.progress is None:
            self._init_progress(metrics)

        self.progress.update(self.task, completed=metrics.shards_finished, **dataclasses.asdict(metrics))  # type: ignore

        self.progress.refresh()  # type: ignore

        if metrics.is_finished:
            self.progress.stop()  # type: ignore

    def _init_progress(self, metrics):
        columns = [
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("| {task.fields[chunks_finished]} chunks", justify="center"),
            TextColumn("| {task.fields[rows_finished]} docs", justify="center"),
        ]

        for field in metrics.field_counts:
            columns.append(TextColumn(f"| {{task.fields[field_counts][{field}]}} {field}", justify="center"))

        columns.append(TimeElapsedColumn())
        columns.append(TimeRemainingColumn())

        self.progress = Progress(
            *columns,
            **self.kwargs,
        )

        self.task = self.progress.add_task(
            "Shards", total=self.num_shards, completed=metrics.shards_finished, **dataclasses.asdict(metrics)
        )
        self.progress.start()


class WandbMetricsMonitor(MetricsMonitor):
    last_metrics: Optional[InProgressCacheMetrics]
    last_time: Optional[float]

    def __init__(self, prefix: str = "preproc", commit=False):
        """
        :param prefix:
        :param commit: Forwarded to wandb.log. Use False (default) if it's part of a simultaneous training run,
        and True if you're running standalone.
        """
        self.prefix = prefix
        self.commit = commit
        self.last_metrics = None
        self.last_time = None

    def __call__(self, metrics: InProgressCacheMetrics):
        to_log: Dict[str, Any] = {}

        to_log[f"{self.prefix}/shards"] = metrics.shards_finished
        to_log[f"{self.prefix}/chunks"] = metrics.chunks_finished
        to_log[f"{self.prefix}/rows"] = metrics.rows_finished

        for field, count in metrics.field_counts.items():
            to_log[f"{self.prefix}/{field}"] = count

        if metrics.is_finished:
            to_log[f"{self.prefix}/finished"] = 1

        # estimate the rate of progress
        # if self.last_metrics is not None:
        #     assert self.last_time is not None
        #     elapsed = time.time() - self.last_time
        #     to_log[f"{self.prefix}/shards_per_s"] = (metrics.shards_finished - self.last_metrics.shards_finished) / elapsed
        #     to_log[f"{self.prefix}/chunks_per_s"] = (metrics.chunks_finished - self.last_metrics.chunks_finished) / elapsed
        #     to_log[f"{self.prefix}/rows_per_s"] = (metrics.rows_finished - self.last_metrics.rows_finished) / elapsed
        #
        #     for field, count in metrics.field_counts.items():
        #         to_log[f"{self.prefix}/{field}_per_s"] = (count - self.last_metrics.field_counts[field]) / elapsed

        self.last_metrics = metrics
        self.last_time = time.time()

        wandb.log(to_log, commit=self.commit)


class LoggerMetricsMonitor(MetricsMonitor):
    # TODO: I'd like to get the trainer pbar migrated to rich and just use rich everywhere, but until then,
    # we have separate logging
    def __init__(self, logger: Optional[Union[logging.Logger, str]] = None, level=logging.INFO):
        if isinstance(logger, str):
            logger = logging.getLogger(logger)
        self.logger = logger or logging.getLogger(__name__)
        self.level = level

    def __call__(self, metrics: InProgressCacheMetrics):
        if jax.process_index() == 0:
            self.logger.log(
                self.level,
                f" done: Shards: {metrics.shards_finished} | Chunks: {metrics.chunks_finished} | Docs:"
                f" {metrics.rows_finished}",
            )

        if metrics.is_finished:
            self.logger.info("Cache creation finished")


@dataclass
class _ShardStatus:
    writer_task: Optional[ActorHandle]
    num_chunks_sent: int = 0
    current_buffer: list[ChunkMetadata] = dataclasses.field(default_factory=list)

    def pop_chunk_to_send(self) -> Optional[ChunkMetadata]:
        if len(self.current_buffer) == 0:
            return None
        else:
            self.num_chunks_sent += 1
            return self.current_buffer.pop(0)

    @property
    def is_finished_and_buffer_empty(self):
        return not self.is_producing and len(self.current_buffer) == 0

    @property
    def is_producing(self):
        return self.writer_task is not None


def _mk_queue_aware_process_task(processor: BatchProcessor[T], queue: ActorHandle):
    @ray.remote(num_cpus=processor.num_cpus, num_gpus=processor.num_gpus, resources=processor.resources)
    def process_task(batch: List[T]) -> pa.RecordBatch:
        logging.basicConfig(level=logging.INFO)
        ray.get(queue.task_running.remote())
        result = processor(batch)
        del batch
        return as_record_batch(result)

    return process_task


@ray.remote
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
        logging.basicConfig(level=logging.INFO)
        self.parent_ref = parent_ref
        self.cache_dir = cache_dir
        self.shard_name = shard_name
        self.uncommited_chunks: list[tuple[int, ChunkMetadata]] = []  # heapq of (chunk index, chunk)

        self.metadata_writer = _ShardMetadataWriter(os.path.join(cache_dir, f"{shard_name}.json"))
        self._expected_num_chunks: Optional[int] = None

        if self.metadata_writer.num_chunks > 0:
            self.parent_ref.new_chunk.remote(shard_name, *self.metadata_writer.chunks)

        if self.metadata_writer.is_finished:
            logger.info(f"Shard {shard_name} already finished. Skipping.")
            self._expected_num_chunks = self.metadata_writer.num_chunks
            self.parent_ref.shard_finished.remote(self.shard_name)

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
        self.parent_ref.shard_failed.remote(self.shard_name, error)

    def _finished_chunk(self, idx: int, chunk: ChunkMetadata):
        if idx < self.metadata_writer.num_chunks:
            logger.error(f"Received chunk {idx} for {self.shard_name} but it's already finished")
            error = RuntimeError(f"Received chunk {idx} for {self.shard_name} but it's already finished")
            self.parent_ref.shard_failed.remote(self.shard_name, ser_exc_info(error))
            raise error

        if self._expected_num_chunks is not None and idx >= self._expected_num_chunks:
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
        ray.get(self.parent_ref.shard_failed.remote(self.shard_name, error))

    def _attempt_to_commit_chunks(self):
        chunks_committed = []
        while len(self.uncommited_chunks) > 0 and self.uncommited_chunks[0][0] == self.metadata_writer.num_chunks:
            _, chunk = heapq.heappop(self.uncommited_chunks)
            self.metadata_writer.commit_chunk(chunk)
            chunks_committed.append(chunk)

        if len(chunks_committed) > 0:
            ray.get(self.parent_ref.new_chunk.remote(self.shard_name, *chunks_committed))

        if self._expected_num_chunks is not None and self.metadata_writer.num_chunks == self._expected_num_chunks:
            self.metadata_writer.finish()
            ray.get(self.parent_ref.shard_finished.remote(self.shard_name))


class _ChunkCollator:
    def __init__(self, cache_dir: str, shard_name: str):
        self.cache_dir = cache_dir
        self.shard_name = shard_name
        self.chunk_writers: dict[int, _ChunkWriter] = {}  # chunk index -> writer
        self.batch_counts: dict[int, int] = {}  # chunk index -> number of batches written
        self.expected_totals: dict[
            int, int
        ] = {}  # chunk index -> expected num batches. set when we finish reading the shard
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
        if chunk_id in self.chunk_writers:
            self.chunk_writers[chunk_id].__exit__(*error.restore())
            del self.chunk_writers[chunk_id]

        error.reraise()

    def _attempt_to_write_chunk_fragments(self, chunk_id) -> Optional[ChunkMetadata]:
        assert chunk_id in self.chunk_partial_batches

        chunk_batches = self.chunk_partial_batches[chunk_id]

        while len(chunk_batches) > 0 and chunk_batches[0][0] == self.batch_counts[chunk_id]:
            # we can write this batch
            batch_id, batch = heapq.heappop(chunk_batches)

            if chunk_id not in self.chunk_writers:
                assert batch_id == 0
                chunk_name = os.path.join(self.shard_name, f"chunk-{chunk_id}")
                writer = _ChunkWriter(self.cache_dir, chunk_name, batch.schema)
                writer.__enter__()
                self.chunk_writers[chunk_id] = writer

            self.chunk_writers[chunk_id].write_batch(batch)
            self.batch_counts[chunk_id] += 1

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


@ray.remote
class ChunkCacheBuilder:
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
        source: ShardedDataset[T],
        processor: BatchProcessor[T],
        rows_per_chunk: int,
    ):
        logging.basicConfig(level=logging.INFO)
        self.broker_ref = broker_ref
        self.shard_status: Dict[str, _ShardStatus] = dict()
        self._current_round_robin = []
        self.source = source
        self._metrics = InProgressCacheMetrics()
        self._processor_actor = BatchProcessorQueue.remote(processor)  # type: ignore

        self_ref = ray.runtime_context.get_runtime_context().current_actor

        if len(source.shard_names) == 0:
            logger.warning("No shards to index?!?")
            self._finish()
        else:
            logger.info(f"Starting cache build for {len(source.shard_names)} shards")

            self._shard_writers = {}
            self._shard_readers = []
            for shard_name in source.shard_names:
                writer = _ShardWriterWorker.remote(self_ref, cache_dir, shard_name)  # type: ignore
                self._shard_writers[shard_name] = writer
                self.shard_status[shard_name] = _ShardStatus(writer)

            num_shards = len(source.shard_names)

            def priority_fn(shard_idx, chunk_idx):
                return chunk_idx * num_shards + shard_idx

            # we make 1 task for min(2 * nodes, num_shards) groups of shards
            # num_shard_groups = min(2 * ray.cluster_resources()["CPU"], num_shards)
            num_shard_groups = num_shards

            shard_groups: list[list[int]] = [[] for _ in range(num_shard_groups)]
            for i, shard_name in enumerate(source.shard_names):
                self._current_round_robin.append(shard_name)
                shard_groups[i % num_shard_groups].append(i)

            for shard_group in shard_groups:
                reader = _alternating_shard_reader.remote(
                    self_ref,
                    self._shard_writers,
                    source,
                    shard_group,
                    priority_fn,
                    self._processor_actor,
                    processor.batch_size,
                    rows_per_chunk,
                )
                self._shard_readers.append(reader)

            # for shard_idx, shard_name in enumerate(source.shard_names):
            #     self._current_round_robin.append(shard_name)
            #
            #     task = _produce_cache_for_shard.remote(
            #         self_ref,
            #         source,
            #         priority_fn,
            #         shard_idx,
            #         processor,
            #         self._processor_actor,
            #         cache_dir,
            #         rows_per_chunk,
            #     )
            #
            #     self.shard_status[shard_name] = _ShardStatus(task)

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

    def shard_finished(self, shard_name: str):
        """Callback method for when a shard worker has finished."""
        shard_status = self.shard_status[shard_name]
        shard_status.writer_task = None

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
        # The global order on chunks is defined as "roundrobin" over shards, until one shard is done.
        # after that, that shard is removed from the roundrobin and the process continues.
        # roundrobin order is determined by self.source.shard_names

        # we are happy to release chunks that form a prefix of the global order so that they can be read
        # to do that, we maintain the roundrobin order in self._current_round_robin
        # and we maintain the current buffer for each shard in self.shard_status
        # when we get a new chunk, we append it to the buffer for that shard
        # when we get a finished message, we mark that shard as finished
        # in either case, we check if we can send any chunks from the front of the roundrobin
        # if we can, we send them to the broker

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
                logger.debug(f"Sending chunk from {name}")
                self._current_round_robin.pop(0)
                self._current_round_robin.append(name)
                chunks_to_send.append(next_chunk)
                continue
            else:
                logger.debug(f"Shard {name} has no chunks to send and is not known to be finished")
                # we can't send a chunk from this shard, so we can't send any additional chunks
                break

        if len(chunks_to_send) > 0:
            logger.debug(f"Sending {len(chunks_to_send)} chunks to broker")
            ray.get(self.broker_ref._append_chunks.remote(*chunks_to_send))

    def _finish(self):
        self._metrics.is_finished = True
        ray.get(self.broker_ref._new_metrics.remote(self._metrics))
        ray.get(self.broker_ref._finalize.remote())


@ray.remote(num_cpus=0)
class ChunkCacheBroker:
    """Actor that manages the global order on chunks and vends chunk metadata to readers."""

    chunks: List[ChunkMetadata]
    _reader_promises: Dict[int, asyncio.Future[ChunkMetadata]]
    _finished_promise: asyncio.Future[None]

    def __init__(self, cache_dir: str, source: ShardedDataset[T], processor: BatchProcessor[T], rows_per_chunk: int):
        logging.basicConfig(level=logging.INFO)
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

        # initialize writer task
        # first see if we need to do anything: check the ledger for is_finished
        try:
            cache_ledger = _load_cache_ledger(self._cache_dir)
            self._append_chunks(*cache_ledger.chunks)
            self._is_finished = True
            self._finished_promise.set_result(None)
        except FileNotFoundError:
            self_ref = ray.runtime_context.get_runtime_context().current_actor
            self._builder_actor = ChunkCacheBuilder.remote(self_ref, self._cache_dir, self._source, self._processor, self._rows_per_chunk)  # type: ignore

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
        if chunk_idx < len(self.chunks):
            return self.chunks[chunk_idx]
        elif self._is_finished:
            return None
        else:
            # we don't have this chunk yet, so we need to wait
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
            if chunk_idx in self._reader_promises:
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
        _serialize_json_and_commit(os.path.join(self._cache_dir, LEDGER_FILE_NAME), CacheLedger(self.chunks))

        self._reader_promises = {}
        self._builder_actor = None
        self._finished_promise.set_result(None)

        # notify metrics subscribers
        self._do_notify()


def _get_broker_actor(cache_dir, input_shards, processor, rows_per_chunk=DEFAULT_ROWS_PER_CHUNK):
    return ChunkCacheBroker.options(name="lev_cache_manager::" + cache_dir, get_if_exists=True).remote(
        # type: ignore
        cache_dir,
        input_shards,
        processor,
        rows_per_chunk,
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

    _ledger: Optional[CacheLedger]
    _broker: Optional[ActorHandle]
    # We use a thread here instead of an actor because we want to ensure it's on the same process as the ShardCache
    # object.
    _monitor_thread: Optional[threading.Thread]
    _metrics_monitors: List[MetricsMonitor]

    def __init__(
        self,
        cache_dir: str,
        batch_size: int,
        _ledger: Optional[CacheLedger],
        _broker: Optional[ActorHandle],
        reader_offset: int = 0,
        num_readers: int = 1,
    ):
        self.cache_dir = cache_dir
        self._ledger = _ledger
        self._broker = _broker
        self._batch_size = batch_size

        self._metrics_monitors = []
        self._monitor_thread = None

        self._num_readers = num_readers
        self._reader_offset = reader_offset

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
    ):
        try:
            return ShardCache.load(cache_dir, batch_size)
        except FileNotFoundError:
            broker = _get_broker_actor(cache_dir, shard_source, processor, rows_per_chunk)
            return ShardCache(cache_dir, batch_size, None, broker)

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
        if self._ledger is not None:
            return self._ledger.chunks[mapped_index]
        else:
            assert self._broker is not None
            time_in = time.time()
            # we want to also log if we're waiting for a long time, so we do this in a loop
            while timeout is None or time.time() - time_in < timeout:
                current_timeout = 20.0  # be generous
                if timeout is not None:
                    current_timeout = min(current_timeout, timeout - (time.time() - time_in))
                try:
                    chunk = ray.get(self._broker.get_chunk.remote(mapped_index), timeout=current_timeout)
                except GetTimeoutError:
                    logger.warning(f"Waiting for chunk {mapped_index} after {int(time.time() - time_in)} seconds")
                    continue

                if chunk is None:
                    raise IndexError(f"Chunk index out of bounds. (Mapped index {mapped_index})")

                return chunk

            if timeout is not None:
                raise TimeoutError(f"Timeout while waiting for chunk {mapped_index}")

    async def get_chunk_async(self, index: int) -> ChunkMetadata:
        """Returns the metadata for a given chunk index"""
        mapped_index = self._map_index(index)
        if self._ledger is not None:
            return self._ledger.chunks[mapped_index]
        else:
            assert self._broker is not None
            chunk = await self._broker.get_chunk.remote(mapped_index)
            if chunk is None:
                raise IndexError(f"Chunk index {index} out of bounds. (Mapped index {mapped_index})")
            return chunk

    def final_chunk_count(self) -> Optional[int]:
        """Returns the number of chunks in the cache, if known"""
        if self._ledger is not None:
            return len(self._ledger.chunks)
        else:
            assert self._broker is not None
            return ray.get(self._broker.final_chunk_count.remote())

    def iter_batches_from_chunks(self, loop: bool = False):
        shard_offset = self._reader_offset

        if self._ledger is not None:
            num_chunks = len(self._ledger.chunks)

            if num_chunks == 0:
                return

            while True:
                i = 0
                for i in range(shard_offset, num_chunks, self._num_readers):
                    chunk = self._ledger.chunks[i]
                    yield from self._read_chunk(chunk)

                if not loop:
                    break

                shard_offset = i % len(self._ledger.chunks)
        else:
            assert self._broker is not None
            i = shard_offset
            while True:
                try:
                    logger.debug(f"Reading chunk {i}")
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
                    logger.exception("Error while reading from shard cache.")
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
        return ShardCache(self.cache_dir, self._batch_size, self._ledger, self._broker, new_offset, new_num_readers)

    def unshard(self):
        """
        Gets the "base" shard cache that this shard cache is a shard of.
        """
        return ShardCache(self.cache_dir, self._batch_size, self._ledger, self._broker, 0, 1)

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
                logger.exception("Error while reading metrics from shard cache.")
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


def _map_batches_in_parallel(
    it: Callable[[], Iterator[T]], fn: Callable[[list[T]], ray.ObjectRef], batch_size, max_queue_size
):
    # Uses Ray to map a function over batches in parallel. It's assumed that the function runs things
    # in the background. We yield the results in order.

    # `it` is a Callable[[], Iterator[T]] b/c we can't pass in certain stateful iterators to Ray tasks.
    queue = Queue(maxsize=max_queue_size, actor_options={"num_cpus": 0})

    @ray.remote(num_cpus=0)
    def producer_task(it, fn):
        try:
            batch = []
            for item in it():
                batch.append(item)
                if len(batch) == batch_size:
                    queue.put(RefBox(fn(batch)))
                    batch = []

            if len(batch) > 0:
                queue.put(RefBox(fn(batch)))
        except Exception:  # noqa
            # always push an exception to the queue so that the consumer can raise it
            queue.put(ray.put(ser_exc_info()))

        queue.put(None)

    producer_task.remote(it, fn)

    while True:
        result = queue.get()
        if result is None:
            break
        elif isinstance(result, RefBox):
            yield ray.get(result.ref)
        else:
            assert isinstance(result, ExceptionInfo)
            result.reraise()


@ray.remote(num_cpus=0)
class BatchProcessorQueue:  # (Generic[T]): ray doesn't like generics
    """
    A queue of tasks to be processed by a BatchProcessor.

    BatchProcessorQueue spins up tasks to process batches of data.
    It spins up tasks until it reaches the maximum number of tasks that can be run in parallel.
    It then waits for a task to finish before spinning up another one.
    """

    pqueue: PriorityQueue[_QueueItem]
    processor: BatchProcessor
    _next_task_id: int
    ready: bool  # whether or not we can spin up a new task

    @property
    def batch_size(self):
        return self.processor.batch_size

    def __init__(self, batch_processor: BatchProcessor[T]):
        self.pqueue = PriorityQueue()
        self.processor = batch_processor
        self._next_task_id = 0
        self.ready = True  # whether we're ready to ask ray to start a new task
        self_ref = ray.runtime_context.get_runtime_context().current_actor
        self._task_processor = _mk_queue_aware_process_task(batch_processor, self_ref)

    # we don't need/want to dereference the batch, so we wrap it in a RefBox
    # one virtue of doing things this way is that we can let Ray try to schedule the compute near the data.
    async def submit(self, priority: float, batch: RefBox):
        """Returns a future that is set to the *ObjectRef* of the processed batch. The future is "complete" when the task
        starts, not when it finishes. You then call ray.get on the future's result to get the actual batch."""
        task_id = self._next_task_id
        self._next_task_id += 1
        f: asyncio.Future = asyncio.Future()
        self.pqueue.put(_QueueItem(priority, batch.ref, task_id, f))
        self._maybe_start_task()
        return await f

    def _maybe_start_task(self):
        if self.ready and not self.pqueue.empty():
            self.ready = False
            item = self.pqueue.get()
            batch = item.batch
            item.task_future.set_result(self._task_processor.remote(batch))

    def task_running(self):
        self.ready = True
        self._maybe_start_task()
