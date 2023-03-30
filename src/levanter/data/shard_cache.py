# Dataset for preprocessing data, tokenizing, and caching to disk.
import asyncio
import json
import logging
import os
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
LEDGER_FILE_NAME = "ledger.json"

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_con = TypeVar("T_con", contravariant=True)
_ExcInfo = Tuple[Optional[BaseException], tblib.Traceback]


class BatchProcessor(Protocol[T_con]):  # type: ignore
    def __call__(self, batch: List[T]) -> pa.RecordBatch:
        raise NotImplementedError

    def resources(self) -> Dict[str, float]:
        raise NotImplementedError

    @property
    def batch_size(self) -> int:
        """Number of items to process in a batch"""
        raise NotImplementedError


class ShardedDataSource(Protocol[T_co]):
    @property
    def shard_names(self) -> Sequence[str]:
        raise NotImplementedError

    def open_shard(self, shard_name: str) -> Iterator[T]:
        raise NotImplementedError

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

        if self.current_chunk_writer.bytes_written >= TARGET_CACHE_SHARD_SIZE:
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
    # reflects the state of the shard in the cache manager
    buffered_chunks: List[ChunkMetadata]  # chunks that have been written but are still awaiting commitment
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
        shard_source: ShardedDataSource[T],
        processor: BatchProcessor[T],
        num_writers: Optional[int] = None,
    ):

        self.shard_source = shard_source
        self.cache_dir = cache_dir
        self.processor = processor

        if num_writers is None:
            num_writers = ray.cluster_resources().get("node", 1)

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

    def _all_done(self):
        return all(state.finished for state in self.shard_states.values())

    def _write_ledger(self):
        """Write the ledger file for this shard cache."""
        # TODO: update this to be agnostic to things lkike num_tokens
        ledger_file = os.path.join(self.cache_dir, LEDGER_FILE_NAME)
        with fsspec.open(ledger_file, "w") as w:
            ledger = {
                "files": [
                    {"name": meta.name, "num_tokens": meta.field_counts["input_ids"]}
                    for meta in self.global_chunks
                ]
            }
            json.dump(ledger, w)

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
    """This should generally be wrapped in a ray.remote, but we don't do that here so that we can
    specify resources in the ray.remote call."""
    try:
        sources: List[Iterator[T]] = [manager.shard_source.open_shard(shard_name) for shard_name in shard_names]
        writers = [_ShardWriter(manager.cache_dir, name) for name in shard_names]
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
                    batch = _take(min_source, manager.processor.batch_size)
                    if len(batch) != 0:
                        processed = manager.processor(batch)
                        opt_chunk = min_writer.write(processed)
                        if opt_chunk is not None:
                            # we have a full chunk
                            manager.finished_chunk(shard_name, opt_chunk)
                            chunk_written = True

                    if len(batch) < manager.processor.batch_size:
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


