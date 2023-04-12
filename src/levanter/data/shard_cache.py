# Dataset for preprocessing data, tokenizing, and caching to disk.
import asyncio
import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import IO, Dict, Generic, Iterable, Iterator, List, Optional, Protocol, Sequence, Tuple, TypeVar

import fsspec.core
import pyarrow as pa
import pyarrow.parquet as pq
import ray
import tblib
from dataclasses_json import dataclass_json
from ray.actor import ActorHandle


logger = logging.getLogger(__name__)

ROWS_PER_CHUNK = 32 * 1024  # if a doc produces ~1200 tokens, this is ~150MB chunks
LEDGER_FILE_NAME = "ledger.json"

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
_ExcInfo = Tuple[Optional[BaseException], tblib.Traceback]


class BatchProcessor(Generic[T], ABC):  # type: ignore
    @abstractmethod
    def __call__(self, batch: List[T]) -> pa.RecordBatch:
        raise NotImplementedError

    def resources(self) -> Dict[str, float]:
        return {}

    @property
    @abstractmethod
    def num_cpus(self) -> int:
        raise NotImplementedError

    @property
    def num_gpus(self) -> int:
        return 0

    @property
    def batch_size(self) -> int:
        """Number of items to process in a batch"""
        raise NotImplementedError


class ShardedDataSource(Protocol[T_co]):
    @property
    def shard_names(self) -> Sequence[str]:
        raise NotImplementedError

    def open_shard(self, shard_name: str) -> Iterator[T_co]:
        raise NotImplementedError

    # TODO: seek to row?


def cache_dataset(
    cache_dir: str,
    processor: BatchProcessor[T],
    input_shards: ShardedDataSource[T],
    num_workers: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> "ShardCacheIterable":
    manager = _get_manager_actor(cache_dir, input_shards, processor, num_workers)

    logger.info("Waiting for cache to be built")
    while not ray.get(manager.is_finished.remote()):
        pass

    logger.info("Finished caching")

    return ShardCacheIterable(cache_dir, input_shards, processor, batch_size or 1)


def _get_manager_actor(cache_dir, input_shards, processor, num_workers):
    return _ShardCacheManager.options(name="lev_cache_manager::" + cache_dir, get_if_exists=True).remote(
        # type: ignore
        cache_dir,
        input_shards,
        processor,
        num_workers,
    )


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


class _ChunkWriter:
    metadata: ChunkMetadata
    writer: Optional[pq.ParquetWriter]
    file: fsspec.core.OpenFile
    file_stream: Optional[IO[bytes]]
    cache_dir: str

    finalized: bool

    def __init__(self, cache_dir, name: str):
        fs, path = fsspec.core.url_to_fs(cache_dir)
        self.file = fsspec.open(f"{path}/{name}.parquet", "wb")
        self.file_stream = None
        self.metadata = ChunkMetadata.new(name)
        self.cache_dir = cache_dir
        self.writer = None
        self.finalized = False

    def write(self, record_batch: pa.RecordBatch):
        if self.finalized:
            raise ValueError("Cannot write to a finalized chunk")

        if self.writer is None:
            self.file_stream = self.file.open()
            self.writer = pq.ParquetWriter(self.file_stream, record_batch.schema, version="2.6", compression="ZSTD")
        self.writer.write_batch(record_batch)

        # update metadata
        self.metadata.num_rows += record_batch.num_rows

        for i in range(record_batch.num_columns):
            name = record_batch.field(i).name
            value = record_batch.column(i)
            if isinstance(value, pa.ListArray):
                value = value.flatten()
                self.metadata.field_counts[name] = self.metadata.field_counts.get(name, 0) + len(value)

    def close(self):
        self.finalized = True

        if self.writer is not None:
            self.writer.close()
            self.writer = None

        if self.file_stream is not None:
            self.file_stream.close()
            self.file_stream = None

        self.file.close()

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
    def from_name(cache_dir, name: str, batch_size: int):
        fs, path = fsspec.core.url_to_fs(cache_dir)
        with fs.open(f"{path}/{name}.json", "r") as f:
            metadata = ChunkMetadata.from_json(f.read())  # type: ignore
        return ChunkReader.from_metadata(cache_dir, metadata, batch_size)

    @staticmethod
    def from_metadata(cache_dir, metadata: ChunkMetadata, batch_size: int):
        file = pq.ParquetFile(fsspec.open(f"{cache_dir}/{metadata.name}.parquet", "rb").open())
        return ChunkReader(metadata, file, batch_size)


class _ShardWriter:
    cache_dir: str
    written_chunk_metadata: List[ChunkMetadata]
    current_chunk_writer: Optional[_ChunkWriter]
    shard_name: str

    is_finished: bool

    def __init__(self, cache_dir, shard_name):
        self.cache_dir = cache_dir
        self.shard_name = shard_name

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

        if self.current_chunk_writer.metadata.num_rows >= ROWS_PER_CHUNK:
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

    def finalize(self) -> Optional[ChunkMetadata]:
        wrote_last_chunk = False
        if self.current_chunk_writer is not None:
            self.current_chunk_writer.close()
            self.written_chunk_metadata.append(self.current_chunk_writer.metadata)
            self.current_chunk_writer = None
            wrote_last_chunk = True

        try:
            self.is_finished = True
            self._write_in_progress_metadata()
        except Exception:
            logger.exception("Failed to finalize shard")
            self.is_finished = False
            raise

        # return the last chunk if we wrote one
        return self.written_chunk_metadata[-1] if wrote_last_chunk else None

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


def _seek_to_row(data_iterator, row_count):
    count = 0
    for _ in data_iterator:
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
        return (exc_value.__class__, exc_value, tb.as_traceback())
    else:
        return (None, None, tb.as_traceback())


class _ShardState:
    # reflects the state of the shard in the cache manager
    buffered_chunks: List[ChunkMetadata]  # chunks that have been written but are still awaiting commitment
    finished: bool

    def __init__(self):
        self.buffered_chunks = []
        self.finished = False

    def ready(self) -> bool:
        return len(self.buffered_chunks) > 0 or self.finished


# constraints:
# can't pass an actor ref to self from within a method
# so we can't pass the ShardCacheManager to the ShardWriter
# so we need an intermediate actor to coordinate the writing
# The ShardCacheManager needs to be able to tell writers to stop
# Ray doesn't guarantee message ordering, so we need to use a sequence number to ensure that writers send their
# messages in the correct order
# Structure:
# ShardCacheManager <-> CacheMessageHandler <-> ShardWriter


@ray.remote
class _CacheMessageHandler:
    global_chunks: List[ChunkMetadata]
    shard_states: Dict[str, _ShardState]

    exception: Optional[_ExcInfo] = None
    exception_source: Optional[str] = None

    writer_tasks: List[ray.ObjectRef]
    reader_promises: Dict[int, asyncio.Future[Optional[ChunkMetadata]]]

    def __init__(self, shard_names: List[str]):
        self.shard_states = {name: _ShardState() for name in shard_names}
        self.global_chunks = []

        self.writer_tasks = []
        self.reader_promises = {}

    # has to be separate from __init__ because we can't pass an actor ref to self from within a method
    def set_tasks(self, tasks):
        self.writer_tasks = tasks

    def finished_chunk(self, shard_name: str, chunk: ChunkMetadata):
        """Called by a shard writer when it has finished writing a chunk. This will update the global ordering
        of chunks and notify any readers that are waiting for this chunk."""
        print("chunk" + shard_name)
        logger.debug(f"Finished writing chunk {chunk} for shard {shard_name}")

        assert not self.shard_states[shard_name].finished
        self.shard_states[shard_name].buffered_chunks.append(chunk)

        # if we have a chunk for every incomplete shard, we can add them to the global ordering
        self._check_chunks_ready()

    def finished_shard(self, shard_name: str):
        """Called by a shard writer when it has finished writing a shard. This will update the global ordering
        of chunks and notify any readers that are waiting for this chunk."""
        logger.debug(f"Finished writing shard {shard_name}")
        print("shard" + shard_name)
        self.shard_states[shard_name].finished = True
        self._check_chunks_ready()

    def shard_exception(self, shard_name: str, exc: _ExcInfo):
        """Called by a shard writer when it has encountered an exception. This will notify any readers that
        are waiting for this chunk."""
        logger.error(f"Shard {shard_name} encountered an exception: {exc}", exc_info=_restore_exc_info(exc))
        # only log the first exception for now
        if self.exception is None:
            self.exception = exc
            self.exception_source = shard_name

            self._propagate_exceptions()

    def other_exception(self, exc: _ExcInfo):
        """Called by a shard writer when it has encountered an exception. This will notify any readers that
        are waiting for this chunk."""
        logger.error(f"Writing encountered an exception: {exc}", exc_info=_restore_exc_info(exc))
        if self.exception is None:
            self.exception = exc
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
        if self.exception is not None:
            restored = _restore_exc_info(self.exception)
            if isinstance(restored, Exception):
                raise restored
            else:
                raise RuntimeError().with_traceback(restored[2])

        print(f"global index: {global_index}, global chunks: {self.global_chunks}")

        if global_index < len(self.global_chunks):
            return self.global_chunks[global_index]
        elif self._all_done():
            return None
        else:
            # we don't have this chunk yet, so we need to wait
            if global_index not in self.reader_promises:
                self.reader_promises[global_index] = asyncio.get_event_loop().create_future()
            print("await return")
            return await self.reader_promises[global_index]

    def _all_done(self):
        return all(state.finished for state in self.shard_states.values()) or self.exception is not None

    def is_finished(self):
        if self.exception is not None:
            raise self.exception[0].with_traceback(self.exception[1].as_traceback())
        res = self._all_done()
        return res

    def _check_chunks_ready(self):
        while all(state.ready() for state in self.shard_states.values()):
            # get the next chunk for each shard (if it has one)
            # ordering determined by shard_names
            old_size = len(self.global_chunks)
            next_chunks = []
            for shard_name, state in self.shard_states.items():
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

                break

            # add the next chunk for each shard to the global ordering
            self.global_chunks += next_chunks

            # Notify any readers that are waiting for this chunk
            for i in range(old_size, len(self.global_chunks)):
                if i in self.reader_promises:
                    self.reader_promises[i].set_result(self.global_chunks[i])
                    del self.reader_promises[i]


@ray.remote
class _ShardCacheManager(Generic[T]):
    """ShardCacheManager is responsible for coordinating the writing and reading of a shard cache. Its main
    job is to create the global ordering of chunks."""

    shard_source: ShardedDataSource[T]
    cache_dir: str
    processor: BatchProcessor[T]

    reader_promises: Dict[int, asyncio.Future[Optional[ChunkMetadata]]]
    writer_tasks: List[ray.ObjectRef]
    handler: ActorHandle

    def __init__(
        self,
        cache_dir: str,
        shard_source: ShardedDataSource[T],
        processor: BatchProcessor[T],
        num_writers: Optional[int] = None,
    ):
        self.cache_dir = cache_dir
        self.shard_source = shard_source
        self.processor = processor

        self.handler = _CacheMessageHandler.remote(shard_source.shard_names)  # type: ignore

        if num_writers is None:
            num_writers = len(ray.nodes())

        # kick off the writer tasks
        shard_assignments = [self.shard_source.shard_names[i::num_writers] for i in range(num_writers)]

        worker_fn = ray.remote(
            num_cpus=processor.num_cpus, num_gpus=processor.num_gpus, resources=processor.resources()
        )(_shard_writer_task)

        writer_tasks = []
        for shard_names in shard_assignments:
            writer_tasks.append(worker_fn.remote(self.handler, processor, cache_dir, shard_source, shard_names))

        self.handler.set_tasks.remote(writer_tasks)

    def _write_ledger(self):
        """Write the ledger file for this shard cache."""
        # TODO: update this to be agnostic to things lkike num_tokens
        ledger_file = os.path.join(self.cache_dir, LEDGER_FILE_NAME)
        with fsspec.open(ledger_file, "w") as w:
            ledger = {
                "files": [
                    {"name": meta.name, "num_tokens": meta.field_counts["input_ids"]} for meta in self.global_chunks
                ]
            }
            json.dump(ledger, w)

    async def is_finished(self):
        """Returns true if all shards have been written."""
        return await self.handler.is_finished.remote()

    async def get_chunk(self, i: int) -> Optional[ChunkMetadata]:
        """Returns the ith chunk in the global ordering. If the chunk is not ready, this will block until
        it is ready."""
        return await self.handler.get_chunk.remote(i)


# TODO: maybe go back to one task per shard? This is a bit tricky on the scheduler.
# We can do one task per chunk but that makes maintaining input shard state tricky
# handler is a _CacheMessageHandler actor (mypy doesn't know this)
def _shard_writer_task(handler, processor, cache_dir, shard_source, shard_names):
    """This should generally be wrapped in a ray.remote, but we don't do that here so that we can
    specify resources in the ray.remote call."""
    try:
        sources: List[Iterator[T]] = [iter(shard_source.open_shard(shard_name)) for shard_name in shard_names]
        writers = [_ShardWriter(cache_dir, name) for name in shard_names]
        # send any finished chunks to the manager
        # TODO: probably better to read from a global list of finished chunks
        i = 0
        while i < len(writers):
            writer = writers[i]
            try:
                print(f"sending {len(writer.written_chunk_metadata)} chunks to manager for {writer.shard_name}")
                for chunk in writer.written_chunk_metadata:
                    print("is it here?")
                    handler.finished_chunk.remote(writer.shard_name, chunk)

                if writer.is_finished:
                    print(f"finished shard {writer.shard_name} early")
                    handler.finished_shard.remote(writer.shard_name)
                    del writers[i]
                    del sources[i]
                else:
                    i += 1
            except Exception as e:
                handler.shard_exception.remote(writer.shard_name, _exc_info())
                raise e

        if len(writers) == 0:
            return

        # now, skip to the first unfinished chunk for each shard
        # do this in threads b/c it's likely io bound for each shard
        # TODO: maybe support seek
        executor = ThreadPoolExecutor(max_workers=len(writers))
        seek_futures: List[Optional[Future]] = [
            executor.submit(_seek_to_row, sources[i], writers[i].rows_written())
            if writers[i].rows_written() > 0
            else None
            for i in range(len(writers))
        ]

        # now, start writing chunks
        # we write one chunk at a time, always the chunk for the furthest behind shard (ties to the first)
        while len(writers) > 0:
            min_chunk_idx = min(range(len(writers)), key=lambda i: writers[i].num_chunks_written)
            min_writer = writers[min_chunk_idx]
            min_source = sources[min_chunk_idx]
            shard_name = min_writer.shard_name
            print(f"writing chunk {min_writer.num_chunks_written} for shard {shard_name}")
            logger.debug(f"Writing chunk {min_writer.num_chunks_written} for shard {min_writer.shard_name}")
            # Our contract is that we always want to make sure we've written chunks evenly from each shard, so we need
            # to wait for the seek to finish before we start writing chunks
            try:
                if seek_futures[min_chunk_idx] is not None:
                    seek_futures[min_chunk_idx].result()
                    logger.info(
                        f"Finished seeking to row {min_writer.rows_written()} for shard {min_writer.shard_name}"
                    )
                    seek_futures[min_chunk_idx] = None

                while True:
                    # write until we have a full chunk or the source is exhausted
                    batch = _take(min_source, processor.batch_size)
                    if len(batch) == 0:
                        break

                    processed = processor(batch)
                    opt_chunk = min_writer.write(processed)

                    # Ray doesn't guarantee message order, so we need to serialize the writes ourselves
                    if opt_chunk is not None:
                        # we have a full chunk
                        handler.finished_chunk.remote(shard_name, opt_chunk)
                        print(f"finished chunk {min_writer.num_chunks_written} for shard {shard_name}")

                # the source is exhausted
                chunk = min_writer.finalize()
                if chunk is not None:
                    print("here")
                    ray.get(handler.finished_chunk.remote(shard_name, chunk))
                    print(f"finished final chunk {min_writer.num_chunks_written} for shard {shard_name}")
                sources.pop(min_chunk_idx)
                writers.pop(min_chunk_idx)
                handler.finished_shard.remote(shard_name)
            except Exception:
                logger.exception(f"Writer process raised an exception while processing shard {min_writer.shard_name}.")
                ray.get(handler.shard_exception.remote(shard_name, _exc_info()))
                raise
    except Exception:
        logger.exception("Writer process raised an exception.")
        handler.other_exception.remote(_exc_info())
        raise


# @ray.remote(num_returns="dynamic", num_cpus=0)  # type: ignore
@ray.remote(num_returns="dynamic")  # type: ignore
def _shard_writer_2(processor, cache_dir, shard_source, shard_name):
    """Similar to _shard_writer_task, but writes from just a single source"""
    # TODO: if/when we switch to forking tasks from this task, we can use this
    # run_batch = ray.remote(num_cpus=processor.num_cpus, num_gpus=processor.num_gpus, resources=processor.resources)(_run_batch)

    try:
        source = shard_source.open_shard(shard_name)
        writer = _ShardWriter(cache_dir, shard_name)

        # send any finished chunks to the manager
        try:
            print(f"sending {len(writer.written_chunk_metadata)} chunks to manager for {writer.shard_name}")
            for chunk in writer.written_chunk_metadata:
                yield chunk

            if writer.is_finished:
                print(f"finished shard {writer.shard_name} early")
                yield None
                return
        except Exception as e:
            logger.exception(f"Writer process raised an exception while processing shard {writer.shard_name}.")
            raise e

        # now, skip to the first unfinished chunk for this shard
        _seek_to_row(source, writer.rows_written())

        # now, start writing chunks
        while True:
            # write until we have a full chunk or the source is exhausted
            batch = _take(source, processor.batch_size)
            if len(batch) == 0:
                break

            processed = processor(batch)
            # processed = ray.get(run_batch.remote(batch))
            opt_chunk = writer.write(processed)

            # Ray doesn't guarantee message order, so we need to serialize the writes ourselves
            if opt_chunk is not None:
                # we have a full chunk
                yield opt_chunk
                print(f"finished chunk {writer.num_chunks_written} for shard {shard_name}")

        # the source is exhausted
        chunk = writer.finalize()
        print(f"finished final chunk {writer.num_chunks_written} for shard {shard_name}")
        yield chunk
        if chunk is not None:
            yield None
    except:
        logger.exception(f"Writer process raised an exception while processing shard {writer.shard_name}.")
        raise


class ShardCacheIterable(Iterable[pa.RecordBatch]):
    """An iterable that reads from a shard cache."""

    def __init__(
        self, cache_dir: str, shard_source: ShardedDataSource[T], processor: BatchProcessor[T], batch_size: int
    ):
        self.cache_dir = cache_dir
        self.shard_source = shard_source
        self.processor = processor

        self._manager = _get_manager_actor(cache_dir, shard_source, processor, None)
        self._batch_size = batch_size

    def __iter__(self):
        i = 0
        while True:
            try:
                chunk = ray.get(self._manager.get_chunk.remote(i))
                i += 1
                if chunk is None:
                    break
                reader = ChunkReader.from_metadata(self.cache_dir, chunk, self._batch_size)
                for batch in reader:
                    yield batch
            except Exception as e:
                logger.exception("Error while reading from shard cache.")
                raise e
