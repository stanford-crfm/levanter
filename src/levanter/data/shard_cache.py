# Dataset for preprocessing data, tokenizing, and caching to disk.
import asyncio
import dataclasses
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from queue import PriorityQueue
from typing import (
    IO,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import fsspec.core
import jax
import pyarrow as pa
import pyarrow.parquet as pq
import ray
import tblib
import wandb
from dataclasses_json import dataclass_json
from fsspec import AbstractFileSystem
from ray.actor import ActorHandle
from ray.exceptions import GetTimeoutError
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from . import ShardableDataset
from ._preprocessor import BatchProcessor, as_record_batch, dict_from_record_batch
from .sharded_dataset import ShardedDataset


T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
_ExcInfo = Tuple[Optional[BaseException], tblib.Traceback]

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


@dataclass_json
@dataclass
class CacheLedger:
    """Written at the end of the cache build process. Contains the global chunk order."""

    chunks: List[ChunkMetadata] = dataclasses.field(default_factory=list)


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
        shard_writer = _ShardWriter(metadata_path)

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
        ray.get(sink.shard_failed.remote(shard_name, _exc_info()))
        raise e


class _ShardWriter:
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


def _produce_chunks_for_shard(
    sink, cache_dir, shard_writer, source, priority_fn, shard_idx, processor, process_queue, rows_per_chunk
):
    shard_name = source.shard_names[shard_idx]
    total_rows_written = sum(chunk.num_rows for chunk in shard_writer.chunks)
    if total_rows_written > 0:
        logger.info(f"Resuming shard {shard_name} at row {total_rows_written}")
    else:
        logger.info(f"Starting shard {shard_name}")
    shard_iter = source.open_shard_at_row(shard_name, total_rows_written)
    target_batch_size = min(processor.batch_size, rows_per_chunk)
    writer: Optional[_ChunkWriter] = None
    batch = []

    def yield_chunk(chunk: ChunkMetadata):
        nonlocal total_rows_written
        total_rows_written += chunk.num_rows
        logger.debug(f"Yielding new chunk {chunk.name} from {shard_name} with {chunk.num_rows} rows")
        sink.new_chunk.remote(shard_name, chunk)
        shard_writer.commit_chunk(chunk)
        return chunk.num_rows

    def do_preprocess(batch):
        nonlocal writer

        # TODO: don't do a .get here, but spawn a whole bunch of tasks as soon as we can
        # the issue is we need to implement some kind of backpressure or latch-type thing so we don't starve
        # other shards since we want to stream them round-robin
        priority = priority_fn(shard_idx, shard_writer.num_chunks)
        output_batch_future = ray.get(process_queue.submit.remote(priority=priority, batch=_RefBox(batch)))
        record_batch = ray.get(output_batch_future)

        if writer is None:
            chunk_name = os.path.join(shard_name, f"chunk-{shard_writer.num_chunks}")
            writer = _ChunkWriter(cache_dir, chunk_name, record_batch.schema)
            writer.__enter__()

        writer.write_batch(record_batch)

        if writer.num_rows >= rows_per_chunk:
            writer.__exit__(None, None, None)
            chunk = writer.get_metadata()
            writer = None
            yield_chunk(chunk)

    logger.info(f"Starting to get rows for shard {shard_name}")
    for row in shard_iter:
        batch.append(row)

        if len(batch) == target_batch_size:
            do_preprocess(batch)
            batch = []

    if batch:
        do_preprocess(batch)
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
    producer_task: Optional[ray.ObjectRef]
    num_chunks_sent: int = 0
    current_buffer: List[ChunkMetadata] = dataclasses.field(default_factory=list)

    def pop_chunk_to_send(self) -> Optional[ChunkMetadata]:
        if len(self.current_buffer) == 0:
            return None
        else:
            self.num_chunks_sent += 1
            return self.current_buffer.pop(0)

    @property
    def total_chunks_produced(self) -> int:
        return self.num_chunks_sent + len(self.current_buffer)

    @property
    def is_finished_and_buffer_empty(self):
        return not self.is_producing and len(self.current_buffer) == 0

    @property
    def is_producing(self):
        return self.producer_task is not None


def _mk_process_task(processor: BatchProcessor[T]):
    @ray.remote(num_cpus=processor.num_cpus, num_gpus=processor.num_gpus, resources=processor.resources)
    def process_task(batch: List[T]) -> pa.RecordBatch:
        logging.basicConfig(level=logging.INFO)
        return processor(batch)

    return process_task


def _mk_queue_aware_process_task(processor: BatchProcessor[T], queue: ActorHandle):
    @ray.remote(num_cpus=processor.num_cpus, num_gpus=processor.num_gpus, resources=processor.resources)
    def process_task(batch: List[T]) -> pa.RecordBatch:
        logging.basicConfig(level=logging.INFO)
        ray.get(queue.task_running.remote())
        result = processor(batch)
        del batch
        return as_record_batch(result)

    return process_task


@dataclass(order=True, frozen=True)
class _QueueItem:
    priority: float
    batch: ray.ObjectRef = dataclasses.field(compare=False)
    task_id: int
    task_future: asyncio.Future


@dataclass
class _RefBox:
    """Ray doesn't dereference ObjectRefs if they're nested in another object. So we use this to take advantage of that.
    https://docs.ray.io/en/latest/ray-core/objects.html#passing-object-arguments"""

    ref: ray.ObjectRef


@ray.remote(num_cpus=0)
class _BatchProcessorQueue:  # (Generic[T]): ray doesn't like generics
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
    async def submit(self, priority: float, batch: _RefBox):
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


@ray.remote(num_cpus=0)
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
        self._processor_actor = _BatchProcessorQueue.remote(processor)  # type: ignore

        self_ref = ray.runtime_context.get_runtime_context().current_actor

        if len(source.shard_names) == 0:
            logger.warning("No shards to index?!?")
            self._finish()
        else:
            logger.info(f"Starting cache build for {len(source.shard_names)} shards")

            num_shards = len(source.shard_names)

            def priority_fn(shard_idx, chunk_idx):
                return chunk_idx * num_shards + shard_idx

            for shard_idx, shard_name in enumerate(source.shard_names):
                self._current_round_robin.append(shard_name)

                task = _produce_cache_for_shard.remote(
                    self_ref,
                    source,
                    priority_fn,
                    shard_idx,
                    processor,
                    self._processor_actor,
                    cache_dir,
                    rows_per_chunk,
                )

                self.shard_status[shard_name] = _ShardStatus(task)

    def new_chunk(self, shard_name: str, *chunks: ChunkMetadata):
        """Callback method for when a shard worker has produced a new chunk."""
        assert self.shard_status[shard_name].is_producing
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
        # we should only get this message if we're still producing
        assert shard_status.is_producing
        shard_status.producer_task = None

        # we might still have buffered chunks, so we need to check if we can append them
        self._attempt_to_flush_buffers()
        self._metrics.shards_finished += 1
        ray.get(self.broker_ref._new_metrics.remote(self._metrics))

        # if there are no more active shards, we're done
        if all(status.is_finished_and_buffer_empty for status in self.shard_status.values()):
            assert len(self._current_round_robin) == 0
            assert all(len(status.current_buffer) == 0 for status in self.shard_status.values())
            self._finish()

    def shard_failed(self, shard_name: str, error: _ExcInfo):
        """Callback method for when a shard worker has failed."""
        assert self.shard_status[shard_name].is_producing
        ray.get(self.broker_ref._writer_exception.remote(shard_name, error))

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
            # send the chunk to the broker
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

    def _writer_exception(self, shard_name, exc_info: _ExcInfo):
        info = _restore_exc_info(exc_info)

        logger.exception(f"Writer task {shard_name} failed with exception", exc_info=info)
        for future in self._reader_promises.values():
            future.set_exception(info[1])

        self._reader_promises = {}

        self._finished_promise.set_exception(info[1])
        self._do_notify()

    def _finalize(self):
        logger.info(f"Finalizing cache {self._cache_dir}...")
        self._is_finished = True
        for future in self._reader_promises.values():
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
            return ray.remote(lambda: None).remote()
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
    def from_name(cache_dir, name: str, batch_size: int) -> "_ChunkReader":
        fs, path = fsspec.core.url_to_fs(cache_dir)
        with fs.open(os.path.join(path, f"{name}.json"), "r") as f:
            metadata = ChunkMetadata.from_json(f.read())  # type: ignore
        return _ChunkReader.from_metadata(cache_dir, metadata, batch_size)

    @staticmethod
    def from_metadata(cache_dir, metadata: ChunkMetadata, batch_size: int) -> "_ChunkReader":
        file = pq.ParquetFile(fsspec.open(os.path.join(cache_dir, f"{metadata.name}.parquet"), "rb").open())
        return _ChunkReader(metadata, file, batch_size)


def _exc_info() -> _ExcInfo:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    tb = tblib.Traceback(exc_traceback)
    return (exc_value, tb)


def _restore_exc_info(exc_info):
    exc_value, tb = exc_info
    if exc_value is not None:
        exc_value = exc_value.with_traceback(tb.as_traceback())
        return (exc_value.__class__, exc_value, tb.as_traceback())
    else:
        return (Exception, Exception("Process failed with no exception"), tb.as_traceback())
