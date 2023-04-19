# Dataset for preprocessing data, tokenizing, and caching to disk.
import asyncio
import dataclasses
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Generic, Iterable, Iterator, List, Optional, Protocol, Sequence, Tuple, TypeVar

import fsspec.core
import pyarrow as pa
import pyarrow.parquet as pq
import ray
import tblib
from dataclasses_json import dataclass_json
from ray.exceptions import GetTimeoutError


T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
_ExcInfo = Tuple[Optional[BaseException], tblib.Traceback]

logger = logging.getLogger(__name__)

ROWS_PER_CHUNK = 32 * 1024  # if a doc produces ~1200 tokens, this is ~150MB chunks
LEDGER_FILE_NAME = "ledger.json"


class BatchProcessor(Generic[T], ABC):  # type: ignore
    @abstractmethod
    def __call__(self, batch: List[T]) -> pa.RecordBatch:
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
        return self.open_at_row(shard_name, 0)

    def open_at_row(self, shard_name: str, row: int) -> Iterator[T_co]:
        raise NotImplementedError


def cache_dataset(
    cache_dir: str,
    processor: BatchProcessor[T],
    input_shards: ShardedDataSource[T],
    num_workers: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> "ShardCacheIterable":
    manager = _get_manager_actor(cache_dir, input_shards, processor, num_workers)

    logger.debug(f"Waiting for cache {cache_dir} to be built")
    while True:
        try:
            _ = ray.get(manager.finished_sentinel.remote(), timeout=4)
            break
        except GetTimeoutError:
            pass

    return ShardCacheIterable(cache_dir, input_shards, processor, batch_size or 1)


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
    total_rows_written: int = 0

    is_finished: bool = False


@dataclass_json
@dataclass
class CacheLedger:
    global_chunk_order: List[ChunkMetadata] = dataclasses.field(default_factory=list)
    is_finished: bool = False


def _mk_process_task(processor: BatchProcessor[T]):
    @ray.remote(num_cpus=processor.num_cpus, num_gpus=processor.num_gpus, resources=processor.resources)
    def process_task(batch: List[T]) -> pa.RecordBatch:
        return processor(batch)

    return process_task


@ray.remote(num_cpus=0)
def produce_chunk(batch: List[T], processor: BatchProcessor[T], cache_dir: str, chunk_name: str) -> ChunkMetadata:
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


# NB: "dynamic" num_returns implies that you return a (future of a) generator that itself returns futures
@ray.remote(num_cpus=0, num_returns="dynamic", scheduling_strategy="SPREAD")  # type: ignore
def produce_shard_cache_chunks(
    source: ShardedDataSource[T], processor: BatchProcessor[T], cache_dir: str, shard_name: str
):
    """Produces chunks of preprocessed data from a single shard."""
    # load or create shard metadata (for recovery)
    shard_metadata_path = f"{cache_dir}/{shard_name}.json"
    try:
        with fsspec.open(shard_metadata_path, "r") as file:
            shard_metadata = ShardMetadata.from_json(file.read())  # type: ignore
    except FileNotFoundError:
        shard_metadata = ShardMetadata()

    was_finished = shard_metadata.is_finished

    if not was_finished:
        # fork off a task to produce new chunks from shard
        new_chunk_iter_future = produce_shard_cache_chunks_helper.remote(
            source, shard_name, shard_metadata.total_rows_written, processor, cache_dir
        )

    # yield from existing chunks
    logger.info(f"Yielding {len(shard_metadata.chunks)} chunks from {shard_name}")
    yield from shard_metadata.chunks

    if not was_finished:
        # yield from new chunks, updating the shard metadata (and committing to disk) as we go
        new_chunk_iter = ray.get(new_chunk_iter_future)
        for chunk in new_chunk_iter:
            chunk = ray.get(chunk)
            logger.info(f"Yielding new chunk {chunk.name} from {shard_name}")
            shard_metadata.chunks.append(chunk)
            shard_metadata.total_rows_written += chunk.num_rows
            serialize_json_and_commit(f"{cache_dir}/{shard_name}.json", shard_metadata)
            yield chunk

        # mark shard as finished
        shard_metadata.is_finished = True
        serialize_json_and_commit(f"{cache_dir}/{shard_name}.json", shard_metadata)


def _open_and_seek(shard_name, source, total_rows_written):
    shard = source.open_at_row(shard_name, total_rows_written)
    return shard


@ray.remote(num_cpus=1, num_returns="dynamic", scheduling_strategy="SPREAD")
def produce_shard_cache_chunks_helper(
    source: ShardedDataSource[T], name: str, rows: int, processor: BatchProcessor[T], cache_dir: str
):
    """Produces chunks of preprocessed data from a single shard."""
    count = 0
    shard = source.open_at_row(name, rows)
    batch = []
    for row in shard:
        batch.append(row)
        if len(batch) == ROWS_PER_CHUNK:
            # TODO: don't do a .get here, but spawn a whole bunch of tasks as soon as we can
            # the issue is we need to implement some kind of backpressure or latch-type thing so we don't starve
            # other shards since we want to stream them round-robin
            yield ray.get(produce_chunk.remote(batch, processor, cache_dir, f"{name}/chunk-{count}"))
            batch = []
            count += 1

    if batch:
        yield ray.get(produce_chunk.remote(batch, processor, cache_dir, f"{name}/chunk-{count}"))


@ray.remote(num_cpus=0, num_returns="dynamic")  # type: ignore
def index_all_shards(
    source: ShardedDataSource[T], processor: BatchProcessor[T], cache_dir: str
) -> Iterator[Optional[ChunkMetadata]]:
    """Indexes all shards in a data source and yield/produce the global chunk order."""
    # load or create shard ledger (for recovery)
    cache_ledger = _load_cache_ledger(cache_dir)

    if cache_ledger is None:
        cache_ledger = CacheLedger()

    if cache_ledger.is_finished:
        yield from cache_ledger.global_chunk_order
        return

    # otherwise, we need to index at least some shards
    shard_iterator_futures = {}
    for shard_name in source.shard_names:
        shard_iterator_futures[shard_name] = produce_shard_cache_chunks.remote(
            source, processor, cache_dir, shard_name
        )

    shard_iterators: Dict[str, Iterator[ray.ObjectRef]] = {}
    for shard_name in source.shard_names:
        shard_iterators[shard_name] = iter(ray.get(shard_iterator_futures[shard_name]))

    while len(shard_iterators) > 0:
        # get the next chunk from each shard
        next_chunk_futures = {shard_name: next(shard_iterators[shard_name], None) for shard_name in shard_iterators}
        # remove shards that are finished
        next_chunks = {}
        for shard_name, next_chunk_future in next_chunk_futures.items():
            if next_chunk_future is None:
                del shard_iterators[shard_name]
            else:
                next_chunks[shard_name] = ray.get(next_chunk_future)

        logger.info(f"Yielding {len(next_chunks)} chunks from {next_chunks.keys()}")

        # yield the next chunk from each shard
        for shard_name, chunk in next_chunks.items():
            cache_ledger.global_chunk_order.append(chunk)
            serialize_json_and_commit(f"{cache_dir}/cache_ledger.json", cache_ledger)
            yield chunk

        # mark shard ledger as finished
        cache_ledger.is_finished = True
        serialize_json_and_commit(f"{cache_dir}/cache_ledger.json", cache_ledger)


def serialize_json_and_commit(path, obj):
    # just to be paranoid, we write to a temp file and then rename it
    # TODO: probably we could do better here
    with fsspec.open(f"{path}.tmp", "w") as file:
        file.write(obj.to_json())
    # now copy the old file to a backup
    fs = fsspec.core.url_to_fs(path)[0]
    if fs.exists(path):
        fs.copy(path, f"{path}.bak")
    fs.rename(f"{path}.tmp", path)


def _load_cache_ledger(cache_dir):
    try:
        with fsspec.open(f"{cache_dir}/cache_ledger.json") as file:
            cache_ledger = CacheLedger.from_json(file.read())
        return cache_ledger
    except FileNotFoundError:
        return None


@ray.remote(num_cpus=0)
class ChunkCacheManager:
    """Manages the global order on chunks and vends chunk metadata to readers."""

    _reader_promises: Dict[int, asyncio.Future[ChunkMetadata]]

    def __init__(self, cache_dir: str, source: ShardedDataSource[T], processor: BatchProcessor[T]):
        self._reader_promises = {}
        self._is_finished = False
        self._source = source
        self._processor = processor
        self._cache_dir = cache_dir

        # initialize writer task
        # first see if we need to do anything: check the ledger for is_finished
        cache_ledger = _load_cache_ledger(self._cache_dir)
        if cache_ledger is not None and cache_ledger.is_finished:
            self.chunks = cache_ledger.global_chunk_order
            self._finalize()
        else:
            self.chunks = []
            chunk_generator_future = index_all_shards.remote(self._source, self._processor, self._cache_dir)
            self_ref = ray.runtime_context.get_runtime_context().current_actor
            self._writer_task = _inject_chunks.remote(self_ref, chunk_generator_future)

    def is_finished(self):
        return self._is_finished

    async def finished_sentinel(self):
        if self._is_finished:
            return
        await self._writer_task

    async def get_chunk(self, chunk_idx: int) -> ChunkMetadata:
        if chunk_idx < len(self.chunks):
            return self.chunks[chunk_idx]
        elif self._is_finished:
            raise IndexError(f"chunk index {chunk_idx} out of range")
        else:
            # we don't have this chunk yet, so we need to wait
            if chunk_idx not in self._reader_promises:
                self._reader_promises[chunk_idx] = asyncio.Future()
            return await self._reader_promises[chunk_idx]

    def _append_chunk(self, chunk: ChunkMetadata):
        self.chunks.append(chunk)
        chunk_idx = len(self.chunks) - 1
        if chunk_idx in self._reader_promises:
            self._reader_promises[chunk_idx].set_result(chunk)
            del self._reader_promises[chunk_idx]

    def _writer_exception(self, exc: Exception):
        logger.error(f"Writer task failed with exception {exc}")
        for future in self._reader_promises.values():
            future.set_exception(exc)

    def _finalize(self):
        self._is_finished = True
        for future in self._reader_promises.values():
            future.set_result(None)

        self._reader_promises = {}
        self._writer_task = None


@ray.remote(num_cpus=0)
def _inject_chunks(manager_ref, generator):
    try:
        for chunk in generator:
            chunk = ray.get(chunk)
            ray.get(manager_ref._append_chunk.remote(chunk))

        ray.get(manager_ref._finalize.remote())
    except Exception as e:
        ray.get(manager_ref._writer_exception.remote(e))
        raise e


def _get_manager_actor(cache_dir, input_shards, processor, num_workers):
    return ChunkCacheManager.options(name="lev_cache_manager::" + cache_dir, get_if_exists=True).remote(
        # type: ignore
        cache_dir,
        input_shards,
        processor,
    )


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
                reader = ChunkReader.from_metadata(self.cache_dir, chunk, self._batch_size)
                for batch in reader:
                    yield batch
            except IndexError:
                break
            except Exception as e:
                logger.exception("Error while reading from shard cache.")
                raise e


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
