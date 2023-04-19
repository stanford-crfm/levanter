# Dataset for preprocessing data, tokenizing, and caching to disk.
import asyncio
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import IO, Dict, Generic, Iterator, List, Optional, Protocol, Sequence, Tuple, TypeVar

import fsspec.core
import pyarrow as pa
import pyarrow.parquet as pq
import ray
import tblib
from dataclasses_json import dataclass_json
from tqdm import tqdm

from levanter.data import Dataset


logger = logging.getLogger(__name__)

TARGET_CACHE_SHARD_SIZE = 512 * 1024 * 1024  # 512MB

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_con = TypeVar("T_con", contravariant=True)
_ExcInfo = Tuple[Optional[BaseException], tblib.Traceback]


class BatchProcessor(Protocol[T_con]):  # type: ignore
    def __call__(self, batch: List[T]) -> pa.RecordBatch:
        raise NotImplementedError

    def resources(self) -> Dict[str, float]:
        raise NotImplementedError


class ShardedDataSource(Protocol[T_co]):
    @property
    def shard_names(self) -> Sequence[str]:
        raise NotImplementedError

    def open_shard(self, shard_name: str) -> Iterator[T]:
        ...

    # TODO: seek to row?


@dataclass_json
@dataclass
class ChunkMetadata:
    name: str
    num_rows: int
    field_counts: Dict[str, int]

    @staticmethod
    def new(name):
        return ChunkMetadata(name, 0, {})


@dataclass_json
@dataclass
class ShardMetadata:
    chunks: List[ChunkMetadata]
    is_finished: bool


def _initialize_cache_ledger(cache_dir, input_shards):
    fs, path = fsspec.core.url_to_fs(cache_dir)
    if fs.exists(path):
        raise ValueError(f"Cache directory {cache_dir} already exists")

    fs.makedirs(path)

    for shard_id, shard in enumerate(input_shards):
        shard_path = f"{path}/shard_{shard_id}"
        fs.makedirs(shard_path)
        shard_ledger = ShardMetadata([], False)
        shard_ledger_path = f"{shard_path}/ledger.json"

        with fs.open(shard_ledger_path, "w") as f:
            f.write(shard_ledger.to_json())


@dataclass_json
@dataclass(frozen=True)
class CacheConfig:
    ideal_readers: int  # defines the global ordering of data for sharded reads
    target_cache_shard_size: int
    doc_batch_size: int  # number of docs to process at a time


class ChunkReader:
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
        return ChunkReader(self.metadata, self.file, batch_size)

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
    def open(cache_dir, name: str, batch_size: int):
        fs, path = fsspec.core.url_to_fs(cache_dir)
        file = pq.ParquetFile(fsspec.open(f"{path}/{name}.parquet", "rb").open())
        with fs.open(f"{path}/{name}.json", "r") as f:
            metadata = ChunkMetadata.from_json(f.read())  # type: ignore
        return ChunkReader(metadata, file, batch_size)


class _ChunkWriter:
    metadata: ChunkMetadata
    writer: Optional[pq.ParquetWriter]
    file_stream: IO
    cache_dir: str
    bytes_written: int

    def __init__(self, cache_dir, name: str):
        fs, path = fsspec.core.url_to_fs(cache_dir)
        self.file_stream = fsspec.open(f"{path}/{name}.parquet", "wb")
        self.metadata = ChunkMetadata.new(name)
        self.cache_dir = cache_dir
        self.bytes_written = 0

    def write(self, record_batch: pa.RecordBatch):
        # record_batch = _batch_encoding_to_arrow(batch)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.file_stream, record_batch.schema, version="2.6", compression="ZSTD")
        self.writer.write_batch(record_batch)

        # update metadata
        self.metadata.num_rows += record_batch.num_rows

        self.bytes_written += record_batch.nbytes

        for i in range(record_batch.num_columns):
            name = record_batch.field(i).name
            value = record_batch.column(i)
            if isinstance(value, pa.ListArray):
                value = value.flatten()
                self.metadata.field_counts[name] = self.metadata.field_counts.get(name, 0) + len(value)

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        self.file_stream.close()

        # write metadata
        fs, path = fsspec.core.url_to_fs(self.cache_dir)
        with fs.open(f"{path}/{self.metadata.name}.json", "w") as f:
            f.write(self.metadata.to_json())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @staticmethod
    def open(cache_dir, name: str):
        return _ChunkWriter(cache_dir, name)


class _ShardWriter:
    cache_dir: str
    written_chunk_metadata: List[ChunkMetadata]
    current_chunk_writer: Optional[_ChunkWriter]
    cache_config: CacheConfig
    shard_name: str

    is_finished: bool

    def __init__(self, cache_dir, shard_name, cache_config: CacheConfig):
        self.cache_dir = cache_dir
        self.shard_name = shard_name
        self.cache_config = cache_config

        self._load_metadata()

    @property
    def _in_progress_metadata(self):
        current_writer_list = [self.current_chunk_writer.metadata] if self.current_chunk_writer is not None else []
        return ShardMetadata(
            chunks=self.written_chunk_metadata + current_writer_list,
            is_finished=self.is_finished,
        )

    def rows_written(self):
        committed = sum(c.num_rows for c in self.written_chunk_metadata)
        in_progress = self.current_chunk_writer.metadata.num_rows if self.current_chunk_writer is not None else 0
        return committed + in_progress

    @property
    def num_chunks_written(self):
        return len(self.written_chunk_metadata) + (1 if self.current_chunk_writer is not None else 0)

    def write(self, batch: pa.RecordBatch) -> Optional[ChunkMetadata]:
        """Returns the metadata of the chunk that was just written, if it was closed."""
        if self.is_finished:
            raise ValueError("Shard is finalized")

        if self.current_chunk_writer is None:
            self.current_chunk_writer = self._open_chunk_writer()

        self.current_chunk_writer.write(batch)

        if self.current_chunk_writer.bytes_written >= self.cache_config.target_cache_shard_size:
            self.current_chunk_writer.close()
            current_metadata = self.current_chunk_writer.metadata
            self.written_chunk_metadata.append(current_metadata)
            self.current_chunk_writer = None

            self._write_in_progress_metadata()
            return current_metadata

        return None

    def _write_in_progress_metadata(self):
        fs, path = fsspec.core.url_to_fs(self.cache_dir)
        with fs.open(f"{path}/{self.shard_name}/metadata.json", "w") as f:
            f.write(self._in_progress_metadata.to_json())

    def finalize(self):
        if self.current_chunk_writer is not None:
            self.current_chunk_writer.close()
            self.written_chunk_metadata.append(self.current_chunk_writer.metadata)
            self.current_chunk_writer = None

        try:
            self.is_finished = True
            self._write_in_progress_metadata()
        except Exception:
            logger.exception("Failed to finalize shard")
            self.is_finished = False
            raise

    def _open_chunk_writer(self):
        fs, path = fsspec.core.url_to_fs(self.cache_dir)
        fs.makedirs(f"{path}/{self.shard_name}", exist_ok=True)

        return _ChunkWriter(self.cache_dir, f"{self.shard_name}/chunk-{len(self.written_chunk_metadata)}")

    def _load_metadata(self):
        fs, path = fsspec.core.url_to_fs(self.cache_dir)
        try:
            with fs.open(f"{path}/{self.shard_name}/metadata.json", "r") as f:
                obj = ShardMetadata.from_json(f.read())
                self.written_chunk_metadata = obj.chunks
                self.is_finished = obj.is_finished
        except FileNotFoundError:
            self.written_chunk_metadata = []
            self.is_finished = False

        self.current_chunk_writer = None


def _seek_to_row(name, data_iterator, row_count, pos):
    count = 0
    for _ in tqdm(data_iterator, desc=f"[Shard {name}] seeking raw data iterator", total=row_count, position=pos + 1):
        count += 1
        if count >= row_count:
            break


def _take(source: Iterator[T], n: int) -> List[T]:
    """Take the first n elements from an iterator, or until the iterator is exhausted."""
    result = []
    for _ in range(n):
        try:
            result.append(next(source))
        except StopIteration:
            break

    return result


def _exc_info():
    exc_type, exc_value, exc_traceback = sys.exc_info()
    tb = tblib.Traceback(exc_traceback)
    return (exc_value, tb)


def _restore_exc_info(exc_info):
    exc_value, tb = exc_info
    if exc_value is not None:
        exc_value = exc_value.with_traceback(tb.as_traceback())
    else:
        return (None, None, tb.as_traceback())


class _ShardState:
    buffered_chunks: List[ChunkMetadata]
    finished: bool

    def __init__(self):
        self.buffered_chunks = []
        self.finished = False

    def ready(self) -> bool:
        return len(self.buffered_chunks) > 0 or self.finished


@ray.remote
class _ShardCacheManager(Generic[T]):
    """ShardCacheManager is responsible for coordinating the writing and reading of a shard cache. Its main
    job is to create the global ordering of chunks."""

    shard_source: ShardedDataSource[T]
    cache_dir: str
    config: CacheConfig
    processor: BatchProcessor[T]

    global_chunks: List[ChunkMetadata]
    shard_states: Dict[str, _ShardState]
    reader_promises: Dict[int, asyncio.Future[Optional[ChunkMetadata]]]
    writer_tasks: List[ray.ObjectRef]

    _exception: Optional[_ExcInfo]
    _exception_source: Optional[str]

    def __init__(
        self,
        cache_dir: str,
        config: CacheConfig,
        shard_source: ShardedDataSource[T],
        processor: BatchProcessor[T],
        num_writers: int,
    ):

        self.shard_source = shard_source
        self.config = config
        self.cache_dir = cache_dir
        self.processor = processor

        # kick off the writer tasks
        self.writer_tasks = []
        shard_assignments = [self.shard_source.shard_names[i::num_writers] for i in range(num_writers)]
        worker_fn = ray.remote(resources=processor.resources())(_shard_writer_task)
        for shard_names in shard_assignments:
            self.writer_tasks.append(worker_fn.remote(self, shard_names))

        self.shard_states = {name: _ShardState() for name in self.shard_source.shard_names}
        self.reader_promises = {}
        self.global_chunks = []

        self._exception = None
        self._exception_source = None

    def finished_chunk(self, shard_name: str, chunk: ChunkMetadata):
        """Called by a shard writer when it has finished writing a chunk. This will update the global ordering
        of chunks and notify any readers that are waiting for this chunk."""
        logger.debug(f"Finished writing chunk {chunk} for shard {shard_name}")

        assert not self.shard_states[shard_name].finished
        self.shard_states[shard_name].buffered_chunks.append(chunk)

        # if we have a chunk for every incomplete shard, we can add them to the global ordering
        self._check_chunks_ready()

    def finished_shard(self, shard_name: str):
        """Called by a shard writer when it has finished writing a shard. This will update the global ordering
        of chunks and notify any readers that are waiting for this chunk."""
        logger.debug(f"Finished writing shard {shard_name}")
        self.shard_states[shard_name].finished = True
        self._check_chunks_ready()

    def shard_exception(self, shard_name: str, exc: _ExcInfo):
        """Called by a shard writer when it has encountered an exception. This will notify any readers that
        are waiting for this chunk."""
        logger.error(f"Shard {shard_name} encountered an exception: {exc}", exc_info=_restore_exc_info(exc))
        # only log the first exception for now
        if self._exception is not None:
            self._exception = exc
            self._exception_source = shard_name

            self._propagate_exceptions()

    def other_exception(self, exc: _ExcInfo):
        """Called by a shard writer when it has encountered an exception. This will notify any readers that
        are waiting for this chunk."""
        logger.error(f"Writing encountered an exception: {exc}", exc_info=_restore_exc_info(exc))
        if self._exception is not None:
            self._exception = exc
            self._propagate_exceptions()

    def _propagate_exceptions(self):
        # send exceptions to all blocked readers, terminate all writers
        for future in self.reader_promises.values():
            future.set_exception(self._exception[0].with_traceback(self._exception[1].as_traceback()))
        for task in self.writer_tasks:
            ray.cancel(task)

    async def get_chunk(self, global_index: int) -> Optional[ChunkMetadata]:
        """Called by a shard reader when it is ready for the next chunk. This will return the next chunk
        in the global ordering, or wait until it is ready."""
        logger.debug(f"Reader ready for chunk {global_index}")
        if self._exception is not None:
            restored = _restore_exc_info(self._exception)
            if isinstance(restored, Exception):
                raise restored
            else:
                raise RuntimeError().with_traceback(restored[2])

        if global_index < len(self.global_chunks):
            return self.global_chunks[global_index]
        elif self._all_done():
            return None
        else:
            # we don't have this chunk yet, so we need to wait
            if global_index not in self.reader_promises:
                self.reader_promises[global_index] = asyncio.get_event_loop().create_future()
            return await self.reader_promises[global_index]

    def _all_done(self):
        return all(state.finished for state in self.shard_states.values())

    def _check_chunks_ready(self):
        while all(state.ready() for state in self.shard_states.values()):
            # get the next chunk for each shard (if it has one)
            # ordering determined by shard_names
            old_size = len(self.global_chunks)
            next_chunks = []
            # use shard_names for ordering
            for shard_name in self.shard_source.shard_names:
                state = self.shard_states[shard_name]
                if len(state.buffered_chunks) > 0:
                    next_chunks.append(state.buffered_chunks.pop(0))
                else:
                    assert state.finished

            if len(next_chunks) == 0:
                # all shards are finished, need to notify readers that we are done
                for k, v in self.reader_promises.items():
                    assert k >= len(self.global_chunks)
                    v.set_result(None)
                self.reader_promises = {}

            # add the next chunk for each shard to the global ordering
            self.global_chunks += next_chunks

            # Notify any readers that are waiting for this chunk
            for i in range(old_size, len(self.global_chunks)):
                if i in self.reader_promises:
                    self.reader_promises[i].set_result(self.global_chunks[i])
                    del self.reader_promises[i]


def _shard_writer_task(manager: "_ShardCacheManager[T]", shard_names):
    """This should generally be wrapped in a ray.remote"""
    try:
        sources: List[Iterator[T]] = [manager.shard_source.open_shard(shard_name) for shard_name in shard_names]
        writers = [_ShardWriter(manager.cache_dir, name, manager.config) for name in shard_names]
        # send any finished chunks to the manager
        # TODO: probably better to read from a global list of finished chunks
        i = 0
        while i < len(writers):
            writer = writers[i]
            try:
                for chunk in writer.written_chunk_metadata:
                    manager.finished_chunk(writer.shard_name, chunk)

                if writer.is_finished:
                    manager.finished_shard(writer.shard_name)
                    del writers[i]
                    del sources[i]
                else:
                    i += 1
            except Exception as e:
                manager.shard_exception(writer.shard_name, _exc_info())
                raise e

        # now, skip to the first unfinished chunk for each shard
        # do this in threads b/c it's likely io bound for each shard
        # TODO: maybe support seek
        executor = ThreadPoolExecutor(max_workers=len(writers))
        seek_futures = [
            executor.submit(_seek_to_row, writers[i].shard_name, sources[i], writers[i].rows_written(), i)
            for i in range(len(writers))
        ]

        # now, start writing chunks
        # we write one chunk at a time, always the chunk for the furthest behind shard (ties to the first)
        while len(writers) > 0:
            min_chunk_idx = min(range(len(writers)), key=lambda i: writers[i].num_chunks_written)
            min_writer = writers[min_chunk_idx]
            min_source = sources[min_chunk_idx]
            shard_name = min_writer.shard_name
            logger.debug(f"Writing chunk {min_writer.num_chunks_written} for shard {min_writer.shard_name}")
            # Our contract is that we always want to make sure we've
            # written chunks evenly from each shard, so we need to wait for the seek to finish before we
            # start writing chunks
            try:
                seek_futures[min_chunk_idx].result()
                chunk_written = False
                while not chunk_written:
                    # write until we have a full chunk or the source is exhausted
                    batch = _take(min_source, manager.config.doc_batch_size)
                    if len(batch) != 0:
                        processed = manager.processor(batch)
                        opt_chunk = min_writer.write(processed)
                        if opt_chunk is not None:
                            # we have a full chunk
                            manager.finished_chunk(shard_name, opt_chunk)
                            chunk_written = True

                    if len(batch) < manager.config.doc_batch_size:
                        # the source is exhausted
                        min_writer.finalize()
                        sources.pop(min_chunk_idx)
                        writers.pop(min_chunk_idx)
                        manager.finished_shard(shard_name)
                        break
            except Exception:
                logger.exception(f"Writer process raised an exception while processing shard {min_writer.shard_name}.")
                manager.shard_exception(shard_name, _exc_info())
                raise
    except Exception:
        logger.exception("Writer process raised an exception.")
        manager.other_exception(_exc_info())
        raise


class ShardCache:
    """A cache which is backed by a collection of cached shards, or shards that are being cached.
    This is the main interface for building and reading from a shard cache.

    ShardCache has the following objectives:

    1) Deterministic ordering over the data
    2) Sharded reading and writing
    3) Simultaneous reading and writing of shards
    4) Fast resumption without losing too much progress

    ShardCache achieves (1) maintaining a global ordering over "chunks" written by shards. Shards produce "chunks"
    of data in order, and we define a global ordering over the chunks. The chunks are then read in order.
    So long as all shards have more data, the ordering of chunks is shard_chunks[0][0], shard_chunks[1][0], ...
    shard_chunks[K][0], shard_chunks[0][1], shard_chunks[1][1], ... Once a shard has no more data, it will be removed
    from the shard_chunks list, and the next chunk will be read from the next shard that has data.

    ShardCache achieves (2) by allowing shards to be read and written in parallel. Shards are written
    round-robin by each worker. *Chunks* are read round-robin from their global ordering. This introduces nontrivial
    communication, but allows us to have uneven shards, or shards that don't divide over the workers.

    ShardCache achieves (3) also via the chunking mechanism. As soon as all shards have written a chunk, the next
    chunk can be read. This allows us to read and write in parallel.

    ShardCache achieves (4) by writing chunks to disk as soon as they are completed and serializing a ledger
    of the chunks that have been written. This allows us to resume from the last chunk that was written.
    """

    # TODO: we may one day want to support writing from workers that aren't doing the reading,
    # but for now we'll just assume that the writing workers are also the reading workers.
    # (We can still easily support the "pre-caching" case where we do no reading)
    def __init__(
        self,
        cache_dir: str,
        cache_config: CacheConfig,
        shard_source: ShardedDataSource[T],
        process_fn: BatchProcessor[T],
        num_writer_workers: Optional[int] = None,
    ):
        self.cache_dir = cache_dir
        self.cache_config = cache_config
        self.shard_source = shard_source
        self.process_fn = process_fn

        writer_workers = num_writer_workers or ray.cluster_resources().get("worker", 1)

        # start the writer actor if it doesn't exist
        self._writer_actor = _ShardCacheManager.options(  # type: ignore
            name="levanter_shard_cache_manager", get_if_exists=True
        ).remote(cache_dir, cache_config, shard_source, process_fn, writer_workers)

    def get_chunk(self, chunk_idx: int) -> ray.ObjectRef[ChunkReader]:
        """Get a chunk from the cache, blocking until it is available."""
        return self._writer_actor.get_chunk.remote(chunk_idx)


class ShardedChunkDataset(Dataset[pa.RecordBatch]):
    """Iterates through the chunks of a dataset, reading them from a shard cache."""

    # we have an idealized number of readers and an actual number of readers
    # our determinism is b

    def __init__(self, cache: ShardCache, reader_id: int, num_readers: int):
        self.cache = cache
        self.reader_id = reader_id
        self.num_readers = num_readers

    def __iter__(self):
        chunk_idx = self.reader_id
        while True:
            chunk = self.cache.get_chunk(chunk_idx)
            chunk = ray.get(chunk)
            if chunk is None:
                break
            yield chunk
            chunk_idx += self.num_readers

    # TODO: remove __len__ from dataset interface


# In general, we have M input shards, N output shards, and K reader processes.
