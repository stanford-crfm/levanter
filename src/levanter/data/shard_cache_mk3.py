# Dataset for preprocessing data, tokenizing, and caching to disk.
import asyncio
import dataclasses
import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import IO, Dict, Generic, Iterable, Iterator, List, Optional, Protocol, Sequence, Tuple, TypeVar

import fsspec.core
import pyarrow as pa
import pyarrow.parquet as pq
import ray
import tblib
from dataclasses_json import dataclass_json
from fsspec import AbstractFileSystem

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
class ShardLedger:
    global_chunk_order: List[ChunkMetadata] = dataclasses.field(default_factory=list)
    is_finished: bool = False


def _mk_process_task(processor: BatchProcessor[T]):
    @ray.remote(num_cpus=processor.num_cpus, num_gpus=processor.num_gpus, resources=processor.resources)
    def process_task(batch: List[T]) -> pa.RecordBatch:
        return processor(batch)

    return process_task


@ray.remote(num_cpus=1)
def produce_chunk(batch: List[T], processor: BatchProcessor[T], cache_dir: str, name: str) -> ChunkMetadata:
    process_task = _mk_process_task(processor)
    record_batch = ray.get(process_task.remote(batch))
    with fsspec.open(f"{cache_dir}/{name}.parquet", "wb") as file:
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

        return ChunkMetadata(name, record_batch.num_rows, field_counts)

@ray.remote(num_cpus=1, num_returns="dynamic")
def produce_shard_cache_chunks(source: ShardedDataSource[T], processor: BatchProcessor[T],
                               cache_dir: str, shard_name: str):
    """Produces chunks of preprocessed data from a single shard."""
    # load or create shard metadata (for recovery)
    shard_metadata_path = f"{cache_dir}/{shard_name}.json"
    fs, shard_metadata_path, _ = fsspec.get_fs_token_paths(shard_metadata_path)
    if fs.exists(shard_metadata_path):
        with fs.open(shard_metadata_path) as file:
            shard_metadata = ShardMetadata.from_json(file.read())
    else:
        shard_metadata = ShardMetadata()

    was_finished = shard_metadata.is_finished

    if not was_finished:
        shard = source.open_at_row(shard_name, shard_metadata.total_rows_written)
        # fork off a task to produce new chunks from shard
        shard_cache_dir = f"{cache_dir}/{shard_name}"
        new_chunk_iter = produce_shard_cache_chunks_helper.remote(shard, processor, shard_cache_dir)

    # yield from existing chunks
    yield from shard_metadata.chunks

    if not was_finished:
        # yield from new chunks, updating the shard metadata (and committing to disk) as we go
        for chunk in new_chunk_iter:
            shard_metadata.chunks.append(chunk)
            shard_metadata.total_rows_written += chunk.num_rows
            # just to be paranoid, we write to a temp file and then rename it
            with fs.open(f"{shard_metadata_path}.tmp", "wb") as file:
                file.write(shard_metadata.to_json().encode("utf-8"))
            fs.rename(f"{shard_metadata_path}.tmp", shard_metadata_path)
            yield chunk

        # mark shard as finished
        shard_metadata.is_finished = True
        with fs.open(f"{shard_metadata_path}.tmp", "wb") as file:
            file.write(shard_metadata.to_json().encode("utf-8"))
        fs.rename(f"{shard_metadata_path}.tmp", shard_metadata_path)


@ray.remote(num_cpus=1, num_returns="dynamic")
def produce_shard_cache_chunks_helper(shard: Iterator[T], processor: BatchProcessor[T], cache_dir: str):
    """Produces chunks of preprocessed data from a single shard."""
    batch = []
    for row in shard:
        batch.append(row)
        if len(batch) == ROWS_PER_CHUNK:
            # TODO: don't do a .get here, but spawn a whole bunch of tasks as soo as we can
            # the issue is we need to implement some kind of backpressure or latch-type thing so we don't starve
            # other shards since we want to stream them round-robin
            yield ray.get(produce_chunk.remote(batch, processor, cache_dir, str(len(batch))))
            batch = []

    if batch:
        yield ray.get(produce_chunk.remote(batch, processor, cache_dir, str(len(batch))))


@ray.remote(num_cpus=0, num_returns="dynamic")
def index_all_shards(source: ShardedDataSource[T], processor: BatchProcessor[T], cache_dir: str):
    """Indexes all shards in a data source and yield/produce the global chunk order."""
    # load or create shard ledger (for recovery)
    shard_ledger_path = f"{cache_dir}/shard_ledger.json"
    fs, shard_ledger_path, _ = fsspec.get_fs_token_paths(shard_ledger_path)
    if fs.exists(shard_ledger_path):
        with fs.open(shard_ledger_path) as file:
            shard_ledger = ShardLedger.from_json(file.read())
    else:
        shard_ledger = ShardLedger()

    was_finished = shard_ledger.is_finished

    if not was_finished:
        # fork off a task to index each shard
        shard_iter = index_all_shards_helper.remote(source, processor, cache_dir)

    # yield from existing chunks
    yield from shard_ledger.global_chunk_order

    if not was_finished:
        # yield from new chunks, updating the shard ledger (and committing to disk) as we go
        for chunk in shard_iter:
            shard_ledger.global_chunk_order.append(chunk)
            # just to be paranoid, we write to a temp file and then rename it
            with fs.open(f"{shard_ledger_path}.tmp", "wb") as file:
                file.write(shard_ledger.to_json().encode("utf-8"))
            fs.rename(f"{shard_ledger_path}.tmp", shard_ledger_path)
            yield chunk

        # mark shard as finished
        shard_ledger.is_finished = True
        with fs.open(f"{shard_ledger_path}.tmp", "wb") as file:
                file.write(shard_ledger.to_json().encode("utf-8"))
        fs.rename(f"{shard_ledger_path}.tmp", shard_ledger_path)



