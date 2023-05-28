# Dataset for preprocessing data, tokenizing, and caching to disk.
import asyncio
import dataclasses
import logging
import os
import sys
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Generic, Iterable, Iterator, List, Optional, Protocol, Sequence, Tuple, TypeVar

import fsspec.core
import pyarrow as pa
import pyarrow.parquet as pq
import ray
import tblib
from dataclasses_json import dataclass_json
from fsspec import AbstractFileSystem
from ray.actor import ActorHandle
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

import wandb


T = TypeVar("T")
T_contra = TypeVar("T_contra", contravariant=True)
T_co = TypeVar("T_co", covariant=True)
_ExcInfo = Tuple[Optional[BaseException], tblib.Traceback]

logger = logging.getLogger(__name__)

DEFAULT_ROWS_PER_CHUNK = 1024 * 32
LEDGER_FILE_NAME = "cache_ledger.json"


class BatchProcessor(Generic[T_contra], ABC):
    @abstractmethod
    def __call__(self, batch: Sequence[T_contra]) -> pa.RecordBatch:
        raise NotImplementedError

    @property
    def resources(self) -> Dict[str, float]:
        return {}

    @property
    @abstractmethod
    def num_cpus(self) -> int:
        raise NotImplementedError

    @property
    def num_gpus(self) -> int:
        return 0


class ShardedDataSource(Generic[T_co]):
    @property
    def shard_names(self) -> Sequence[str]:
        raise NotImplementedError

    @property
    def num_shards(self) -> int:
        return len(self.shard_names)

    def open_shard(self, shard_name: str) -> Iterator[T_co]:
        return self.open_shard_at_row(shard_name, 0)

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[T_co]:
        raise NotImplementedError


def cache_dataset(
    cache_dir: str,
    input_shards: ShardedDataSource[T],
    processor: BatchProcessor[T],
    batch_size: int = 1,
    rows_per_chunk: int = DEFAULT_ROWS_PER_CHUNK,
    await_finished: bool = True,
) -> "ShardCache":
    # first see if we need to do anything
    cache = ShardCache(cache_dir, input_shards, processor, batch_size, rows_per_chunk)

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


def _mk_process_task(processor: BatchProcessor[T]):
    @ray.remote(num_cpus=processor.num_cpus, num_gpus=processor.num_gpus, resources=processor.resources)
    def process_task(batch: List[T]) -> pa.RecordBatch:
        return processor(batch)

    return process_task


@ray.remote(num_cpus=0)
def _produce_chunk(batch: List[T], processor: BatchProcessor[T], cache_dir: str, chunk_name: str) -> ChunkMetadata:
    process_task = _mk_process_task(processor)
    record_batch = ray.get(process_task.remote(batch))
    logger.debug(f"Produced chunk {chunk_name} with {record_batch.num_rows} rows. Writing to {cache_dir}/{chunk_name}")
    with fsspec.open(os.path.join(cache_dir, f"{chunk_name}.parquet"), "wb") as file:
        with pq.ParquetWriter(file, record_batch.schema, version="2.6", compression="ZSTD") as writer:
            writer.write_batch(record_batch)

        field_counts = {}

        for i in range(record_batch.num_columns):
            name = record_batch.field(i).name
            value = record_batch.column(i)
            if isinstance(value, pa.ListArray):
                value = value.flatten()
                field_counts[name] = len(value)
            elif isinstance(value, pa.ChunkedArray):
                field_counts[name] = value.length()

        return ChunkMetadata(chunk_name, record_batch.num_rows, field_counts)


@ray.remote(num_cpus=0, scheduling_strategy="SPREAD")  # type: ignore
def _produce_cache_for_shard(
    sink: ActorHandle,
    source: ShardedDataSource[T],
    shard_name: str,
    processor: BatchProcessor[T],
    cache_dir: str,
    rows_per_chunk: int,
):
    """Produces chunks of preprocessed data from a single shard and writes them to disk. Chunks are written to sink,
    which is an actor of ChunkCacheBuilder."""
    # load or create shard metadata (for recovery)
    try:
        shard_metadata_path = os.path.join(cache_dir, f"{shard_name}.json")
        try:
            with fsspec.open(shard_metadata_path, "r") as file:
                shard_metadata = ShardMetadata.from_json(file.read())  # type: ignore
        except FileNotFoundError:
            shard_metadata = ShardMetadata()

        was_finished = shard_metadata.is_finished

        total_rows_written = sum(chunk.num_rows for chunk in shard_metadata.chunks)
        if not was_finished:
            shard_iter = source.open_shard_at_row(shard_name, total_rows_written)

        # yield from existing chunks
        logger.info(f"Yielding {len(shard_metadata.chunks)} chunks from {shard_name}")
        sink.new_chunk.remote(shard_name, *shard_metadata.chunks)

        def yield_chunk(chunk: ChunkMetadata):
            nonlocal total_rows_written
            total_rows_written += chunk.num_rows
            logger.info(f"Yielding new chunk {chunk.name} from {shard_name}")
            sink.new_chunk.remote(shard_name, chunk)
            shard_metadata.chunks.append(chunk)
            _serialize_json_and_commit(os.path.join(cache_dir, f"{shard_name}.json"), shard_metadata)

        if not was_finished:
            count = len(shard_metadata.chunks)
            batch = []
            for row in shard_iter:
                batch.append(row)
                if len(batch) == rows_per_chunk:
                    # TODO: don't do a .get here, but spawn a whole bunch of tasks as soon as we can
                    # the issue is we need to implement some kind of backpressure or latch-type thing so we don't starve
                    # other shards since we want to stream them round-robin
                    chunk_name = os.path.join(shard_name, f"chunk-{count}")
                    count += 1
                    chunk = ray.get(_produce_chunk.remote(batch, processor, cache_dir, chunk_name))
                    yield_chunk(chunk)

                    batch = []

            if batch:
                chunk_name = os.path.join(shard_name, f"chunk-{count}")
                chunk = ray.get(_produce_chunk.remote(batch, processor, cache_dir, chunk_name))
                yield_chunk(chunk)

            shard_metadata.is_finished = True
            _serialize_json_and_commit(os.path.join(cache_dir, f"{shard_name}.json"), shard_metadata)

        sink.shard_finished.remote(shard_name)
    except Exception as e:
        logger.exception(f"Error while processing shard {shard_name}")
        ray.get(sink.shard_failed.remote(shard_name, _exc_info()))
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
        logger.info(f"Attempting to load cache ledger from {cache_dir}/{LEDGER_FILE_NAME}")
        ledger_path = os.path.join(cache_dir, LEDGER_FILE_NAME)
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
    def __init__(self, prefix: str = "preprocessing", commit=False):
        """
        :param prefix:
        :param commit: Forwarded to wandb.log. Use False (default) if it's part of a simultaneous training run,
        and True if you're running standalone.
        """
        self.prefix = prefix
        self.commit = commit

    def __call__(self, metrics: InProgressCacheMetrics):
        to_log = {}

        to_log[f"{self.prefix}/shards_finished"] = metrics.shards_finished
        to_log[f"{self.prefix}/chunks_finished"] = metrics.chunks_finished
        to_log[f"{self.prefix}/rows_finished"] = metrics.rows_finished

        for field, count in metrics.field_counts.items():
            to_log[f"{self.prefix}/{field}"] = count

        wandb.log(to_log, commit=self.commit)


def _ledger_or_broker(
    cache_dir: str, input_shards: ShardedDataSource[T], processor: BatchProcessor[T], rows_per_chunk: int
):
    try:
        return _load_cache_ledger(cache_dir)
    except FileNotFoundError:
        return _get_broker_actor(cache_dir, input_shards, processor, rows_per_chunk)


@ray.remote(num_cpus=0)
class ChunkCacheBuilder:
    """
    Actor that manages the in-progress global ordering on chunks.
    ChunkCacheWriter's job is to hold the list of all chunks as well as chunks from each shard while
    caching is running.

    This is a separate actor from the ChunkCacheBroker because
    we need something that gets messages from shards in-order, and async methods make actors
    lose that property.
    """

    def __init__(
        self,
        broker_ref,
        cache_dir: str,
        source: ShardedDataSource[T],
        processor: BatchProcessor[T],
        rows_per_chunk: int,
    ):
        self.broker_ref = broker_ref
        self.buffered_shard_chunks: Dict[str, List[ChunkMetadata]] = {}
        self.current_shard_tasks: Dict[str, ray.ObjectRef] = dict()
        self.source = source
        self._metrics = InProgressCacheMetrics()

        self_ref = ray.runtime_context.get_runtime_context().current_actor

        if len(source.shard_names) == 0:
            logger.warning("No shards to index?!?")
            self._finish()
        else:
            for shard_name in source.shard_names:
                self.buffered_shard_chunks[shard_name] = []

                self.current_shard_tasks[shard_name] = _produce_cache_for_shard.remote(
                    self_ref, source, shard_name, processor, cache_dir, rows_per_chunk
                )

    def new_chunk(self, shard_name: str, *chunks: ChunkMetadata):
        """Callback method for when a shard worker has produced a new chunk."""
        assert shard_name in self.current_shard_tasks
        assert shard_name in self.buffered_shard_chunks
        self.buffered_shard_chunks[shard_name] += chunks

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
        assert shard_name in self.current_shard_tasks
        # we're done with this shard, so remove it from the list of active shards
        del self.current_shard_tasks[shard_name]
        # we might still have buffered chunks, so we need to check if we can append them

        if len(self.buffered_shard_chunks[shard_name]) == 0:
            # we don't have to worry about this shard anymore
            del self.buffered_shard_chunks[shard_name]

        self._attempt_to_flush_buffers()
        self._metrics.shards_finished += 1
        ray.get(self.broker_ref._new_metrics.remote(self._metrics))

        # if there are no more active shards, we're done
        if len(self.current_shard_tasks) == 0:
            assert len(self.buffered_shard_chunks) == 0, f"Buffered chunks: {self.buffered_shard_chunks}"
            # we're done, so tell the broker to finalize
            self._finish()

    def shard_failed(self, shard_name: str, error: _ExcInfo):
        """Callback method for when a shard worker has failed."""
        ray.get(self.broker_ref._writer_exception.remote(shard_name, error))

    def _attempt_to_flush_buffers(self):
        # this is the most complex logic in this class.
        # The global order on chunks is defined as "round robin" over shards, until one shard is done.
        # after that, that shard is removed from the round robin and the process continues.
        # round robin order is determined by self.source.shard_names
        # So we loop over the shards in order. If we find a shard that has buffered chunks, we send
        # the first chunk to the broker. If we find a shard that has no buffered chunks, we check
        # if it's done. If it's done, we remove it from the list of buffered shards. If it's not done,
        chunks_to_send = []
        done = False
        while not done:
            found_one = False
            for name in self.source.shard_names:
                assert not done
                if name not in self.buffered_shard_chunks:
                    continue

                shard_is_finished = name not in self.current_shard_tasks

                if len(self.buffered_shard_chunks[name]) == 0:
                    if shard_is_finished:
                        # we're done with this shard, so we can remove it
                        del self.buffered_shard_chunks[name]
                    else:
                        # we can't send any chunks yet because we're waiting for a chunk from this shard
                        done = True
                        break

                chunk = self.buffered_shard_chunks[name].pop(0)
                chunks_to_send.append(chunk)
                found_one = True

                # check again
                if len(self.buffered_shard_chunks[name]) == 0:
                    if shard_is_finished:
                        # we're done with this shard, so we can remove it
                        del self.buffered_shard_chunks[name]

            done = done or not found_one

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

    def __init__(
        self, cache_dir: str, source: ShardedDataSource[T], processor: BatchProcessor[T], rows_per_chunk: int
    ):
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


class ShardCache(Iterable[pa.RecordBatch]):
    """A cache which is backed by a collection of chunks of preprocessed documents. These chunks
    are produced by tokenizing/preprocessing a ShardedDataSource.

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

        # TODO: (2) isn't implemented just yet

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
        shard_source: ShardedDataSource[T],
        processor: BatchProcessor[T],
        batch_size: int,
        rows_per_chunk: int,
    ):
        self.cache_dir = cache_dir
        self.shard_source = shard_source
        self.processor = processor
        self.rows_per_chunk = rows_per_chunk

        ledger_or_broker = _ledger_or_broker(cache_dir, shard_source, processor, rows_per_chunk)
        if isinstance(ledger_or_broker, CacheLedger):
            self._ledger = ledger_or_broker
            self._broker = None
        else:
            self._ledger = None
            self._broker = ledger_or_broker

        self._batch_size = batch_size

        self._metrics_monitors = []
        self._monitor_thread = None

    def finished_sentinel(self):
        """Returns a Ray-awaitable object that will be set when the cache is finished"""
        if self._broker is None:
            return ray.remote(lambda: None).remote()
        else:
            return self._broker.finished_sentinel.remote()

    def __iter__(self):
        if self._ledger is not None:
            for i in range(len(self._ledger.chunks)):
                chunk = self._ledger.chunks[i]
                reader = _ChunkReader.from_metadata(self.cache_dir, chunk, self._batch_size)
                for batch in reader:
                    yield batch
        else:
            i = 0
            while True:
                try:
                    chunk = ray.get(self._broker.get_chunk.remote(i))
                    if chunk is None:
                        break
                    i += 1
                    reader = _ChunkReader.from_metadata(self.cache_dir, chunk, self._batch_size)
                    for batch in reader:
                        yield batch
                except Exception as e:
                    logger.exception("Error while reading from shard cache.")
                    raise e

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
