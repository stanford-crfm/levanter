# Dataset for preprocessing data, tokenizing, and caching to disk.
import asyncio
import dataclasses
import logging
import os
import sys
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
from ray.exceptions import GetTimeoutError


T = TypeVar("T")
T_contra = TypeVar("T_contra", contravariant=True)
T_co = TypeVar("T_co", covariant=True)
_ExcInfo = Tuple[Optional[BaseException], tblib.Traceback]

logger = logging.getLogger(__name__)

ROWS_PER_CHUNK = 32 * 1024  # if a doc produces ~1200 tokens, this is ~150MB chunks
LEDGER_FILE_NAME = "cache_ledger.json"


class BatchProcessor(Generic[T_contra], ABC):  # type: ignore
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


class ShardedDataSource(Protocol[T_co]):
    @property
    def shard_names(self) -> Sequence[str]:
        raise NotImplementedError

    def open_shard(self, shard_name: str) -> Iterator[T_co]:
        return self.open_shard_at_row(shard_name, 0)

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[T_co]:
        raise NotImplementedError


def cache_dataset(
    cache_dir: str,
    processor: BatchProcessor[T],
    input_shards: ShardedDataSource[T],
    batch_size: Optional[int] = None,
) -> "ShardCache":
    broker = _get_broker_actor(cache_dir, input_shards, processor)

    logger.debug(f"Waiting for cache {cache_dir} to be built")
    sentinel_remote = broker.finished_sentinel.remote()
    while True:
        try:
            _ = ray.get(sentinel_remote, timeout=4)
            break
        except GetTimeoutError:
            pass

    return ShardCache(cache_dir, input_shards, processor, batch_size or 1)


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
    with fsspec.open(f"{cache_dir}/{chunk_name}.parquet", "wb") as file:
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
    sink, source: ShardedDataSource[T], shard_name: str, processor: BatchProcessor[T], cache_dir: str
):
    """Produces chunks of preprocessed data from a single shard and writes them to disk. Chunks are written to sink,
    which is an actor of ChunkCacheBuilder."""
    # load or create shard metadata (for recovery)
    try:
        shard_metadata_path = f"{cache_dir}/{shard_name}.json"
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
            _serialize_json_and_commit(f"{cache_dir}/{shard_name}.json", shard_metadata)

        if not was_finished:
            count = len(shard_metadata.chunks)
            batch = []
            for row in shard_iter:
                batch.append(row)
                if len(batch) == ROWS_PER_CHUNK:
                    # TODO: don't do a .get here, but spawn a whole bunch of tasks as soon as we can
                    # the issue is we need to implement some kind of backpressure or latch-type thing so we don't starve
                    # other shards since we want to stream them round-robin
                    chunk = ray.get(_produce_chunk.remote(batch, processor, cache_dir, f"{shard_name}/chunk-{count}"))
                    yield_chunk(chunk)

                    batch = []
                    count += 1

            if batch:
                chunk = ray.get(_produce_chunk.remote(batch, processor, cache_dir, f"{shard_name}/chunk-{count}"))
                yield_chunk(chunk)

            shard_metadata.is_finished = True
            _serialize_json_and_commit(f"{cache_dir}/{shard_name}.json", shard_metadata)

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
        print(f"Loading cache state from {cache_dir}/{LEDGER_FILE_NAME}")
        with fsspec.open(f"{cache_dir}/{LEDGER_FILE_NAME}") as file:
            cache_ledger = CacheLedger.from_json(file.read())  # type: ignore
        return cache_ledger
    except FileNotFoundError:
        raise FileNotFoundError(f"Cache state not found at {cache_dir}/{LEDGER_FILE_NAME}")


@ray.remote(num_cpus=0)
class ChunkCacheBuilder:
    """
    Actor that manages the in-progress global ordering on chunks.
    ChunkCacheWriter's job is to hold the list of all chunks as well as active shards and update
    the cache state as new chunks are produced.

    This is a separate actor from the ChunkCacheBroker because
    we need something that gets messages from shards in-order, and async methods make actors
    lose that property.
    """

    def __init__(self, broker_ref, cache_dir: str, source: ShardedDataSource[T], processor: BatchProcessor[T]):

        self.broker_ref = broker_ref
        self.buffered_shard_chunks: Dict[str, List[ChunkMetadata]] = {}
        self.current_shard_tasks: Dict[str, ray.ObjectRef] = dict()
        self.source = source

        self_ref = ray.runtime_context.get_runtime_context().current_actor

        for shard_name in source.shard_names:
            self.buffered_shard_chunks[shard_name] = []

            self.current_shard_tasks[shard_name] = _produce_cache_for_shard.remote(
                self_ref, source, shard_name, processor, cache_dir
            )

    def new_chunk(self, shard_name: str, *chunks: ChunkMetadata):
        assert shard_name in self.current_shard_tasks
        assert shard_name in self.buffered_shard_chunks
        self.buffered_shard_chunks[shard_name] += chunks

        # if we have buffered chunks, we need to check if we can send them to the broker
        self._attempt_to_flush_buffers()

    def shard_finished(self, shard_name: str):
        assert shard_name in self.current_shard_tasks
        # we're done with this shard, so remove it from the list of active shards
        del self.current_shard_tasks[shard_name]
        # we might still have buffered chunks, so we need to check if we can append them

        if len(self.buffered_shard_chunks[shard_name]) == 0:
            # we don't have to worry about this shard anymore
            del self.buffered_shard_chunks[shard_name]

        self._attempt_to_flush_buffers()

        # if there are no more active shards, we're done
        if len(self.current_shard_tasks) == 0:
            assert len(self.buffered_shard_chunks) == 0, f"Buffered chunks: {self.buffered_shard_chunks}"
            # we're done, so tell the broker to finalize
            self._finish()

    def shard_failed(self, shard_name: str, error: _ExcInfo):
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
            ray.get(self.broker_ref._append_chunk.remote(*chunks_to_send))

    def _finish(self):
        ray.get(self.broker_ref._finalize.remote())


@ray.remote(num_cpus=0)
class ChunkCacheBroker:
    """Actor that manages the global order on chunks and vends chunk metadata to readers."""

    _reader_promises: Dict[int, asyncio.Future[ChunkMetadata]]
    _finished_promise: asyncio.Future[None]

    def __init__(self, cache_dir: str, source: ShardedDataSource[T], processor: BatchProcessor[T]):
        self._reader_promises = {}
        self._is_finished = False
        self._source = source
        self._processor = processor
        self._cache_dir = cache_dir

        # initialize writer task
        # first see if we need to do anything: check the state for is_finished
        try:
            cache_ledger = _load_cache_ledger(self._cache_dir)
        except FileNotFoundError:
            cache_ledger = None

        self._finished_promise = asyncio.Future()
        if cache_ledger is not None:
            self.chunks = cache_ledger.chunks
            self._is_finished = True
            self._finished_promise.set_result(None)
        else:
            self.chunks = []
            self_ref = ray.runtime_context.get_runtime_context().current_actor
            self._builder_actor = ChunkCacheBuilder.remote(self_ref, self._cache_dir, self._source, self._processor)  # type: ignore

    def is_finished(self):
        return self._is_finished

    async def finished_sentinel(self):
        await self._finished_promise

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

    def _append_chunk(self, *chunks: ChunkMetadata):
        for chunk in chunks:
            self.chunks.append(chunk)
            chunk_idx = len(self.chunks) - 1
            if chunk_idx in self._reader_promises:
                self._reader_promises[chunk_idx].set_result(chunk)
                del self._reader_promises[chunk_idx]

    def _writer_exception(self, shard_name, exc_info: _ExcInfo):
        info = _restore_exc_info(exc_info)

        logger.exception(f"Writer task {shard_name} failed with exception", exc_info=info)
        for future in self._reader_promises.values():
            future.set_exception(info[1])

        self._reader_promises = {}
        self._finished_promise.set_exception(info[1])

    def _finalize(self):
        self._is_finished = True
        for future in self._reader_promises.values():
            future.set_result(None)

        # write ledger
        _serialize_json_and_commit(f"{self._cache_dir}/{LEDGER_FILE_NAME}", CacheLedger(self.chunks))

        self._reader_promises = {}
        self._builder_actor = None
        self._finished_promise.set_result(None)


def _get_broker_actor(cache_dir, input_shards, processor):
    return ChunkCacheBroker.options(name="lev_cache_manager::" + cache_dir, get_if_exists=True).remote(
        # type: ignore
        cache_dir,
        input_shards,
        processor,
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

    def __init__(
        self, cache_dir: str, shard_source: ShardedDataSource[T], processor: BatchProcessor[T], batch_size: int
    ):
        self.cache_dir = cache_dir
        self.shard_source = shard_source
        self.processor = processor

        self._broker = _get_broker_actor(cache_dir, shard_source, processor)
        self._batch_size = batch_size

    def __iter__(self):
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
            # except IndexError:
            #     break
            except Exception as e:
                logger.exception("Error while reading from shard cache.")
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
    def from_name(cache_dir, name: str, batch_size: int):
        fs, path = fsspec.core.url_to_fs(cache_dir)
        with fs.open(f"{path}/{name}.json", "r") as f:
            metadata = ChunkMetadata.from_json(f.read())  # type: ignore
        return _ChunkReader.from_metadata(cache_dir, metadata, batch_size)

    @staticmethod
    def from_metadata(cache_dir, metadata: ChunkMetadata, batch_size: int):
        file = pq.ParquetFile(fsspec.open(f"{cache_dir}/{metadata.name}.parquet", "rb").open())
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
