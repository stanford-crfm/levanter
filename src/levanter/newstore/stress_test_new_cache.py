# Reads an old-style ShardCache and compares to
import asyncio
import logging
import os
import time

import jax.random
import numpy as np
import tensorstore as ts

from levanter.newdata import PermutationDataset
from levanter.newdata.new_text import TokenSeqDataset
from levanter.newstore.cache import LEDGER_FILE_NAME, CacheLedger, TreeCache, _serialize_json_and_commit
from levanter.newstore.tree_store import TreeStoreBuilder
from levanter.tracker import capture_time
from levanter.utils import fsspec_utils


logging.basicConfig(level=logging.INFO)


SEQ_LEN = 1024
BS = 8
BATCHES = 1000

# want to test reading from:
# 1) old cache sequentially
# 2) new cache sequentially
# 3) new cache randomly


def bench_old_cache(old_cache_path):
    from levanter.data.text import TokenizedDocumentCache, TokenSeqDataset

    old_cache = TokenizedDocumentCache.load(old_cache_path, flatten_docs=True)
    ds = TokenSeqDataset(old_cache, SEQ_LEN)
    for i, item in enumerate(ds):
        if i > BS * BATCHES:
            break


def bench_new_cache_serial(exemplar, new_cache_path):
    jagged_array = TreeStoreBuilder.open(exemplar, new_cache_path).tree["input_ids"]
    len_cache = jagged_array.data_size
    new_cache = jagged_array.data
    num_batches = len_cache // SEQ_LEN
    for b in range(BATCHES):
        elems = []
        with ts.Batch():
            for j in range(BS):
                idx = b * BS + j
                idx = idx % num_batches
                arr1 = new_cache[idx * SEQ_LEN : (idx + 1) * SEQ_LEN].read()
                elems.append(arr1)

        for elem in elems:
            elem.result()


def bench_new_cache_random(exemplar, new_cache_path):
    jagged_array = TreeStoreBuilder.open(exemplar, new_cache_path).tree["input_ids"]
    len_cache = jagged_array.data_size
    new_cache = jagged_array.data
    num_batches = len_cache // SEQ_LEN
    for b in range(BATCHES):
        elems = []
        with ts.Batch():
            for j in range(BS):
                idx = np.random.randint(0, num_batches)
                arr1 = new_cache[idx * SEQ_LEN : (idx + 1) * SEQ_LEN].read()
                elems.append(arr1)

        for elem in elems:
            elem.result()


async def bench_new_cache_serial_tokenseq(exemplar, new_cache_path):
    ensure_cache(new_cache_path)
    cache = TreeCache.load(new_cache_path, exemplar)

    ds = TokenSeqDataset(cache, SEQ_LEN)

    num_batches = await ds.async_len()

    for b in range(BATCHES):
        indices = []
        for j in range(BS):
            idx = b * BS + j
            idx = idx % num_batches
            indices.append(idx)
        elems = await ds.get_batch(indices)
        del elems


async def bench_new_cache_permutation_random(exemplar, new_cache_path):
    ensure_cache(new_cache_path)
    cache = TreeCache.load(new_cache_path, exemplar)

    ds = TokenSeqDataset(cache, SEQ_LEN)
    ds = await PermutationDataset.from_dataset(ds, jax.random.PRNGKey(0))

    num_batches = await ds.async_len()

    for b in range(BATCHES):
        indices = []
        for j in range(BS):
            idx = b * BS + j
            idx = idx % num_batches
            indices.append(idx)
        elems = await ds.get_batch(indices)
        del elems


async def bench_new_cache_serial_tokenseq_minisync(exemplar, new_cache_path):
    ensure_cache(new_cache_path)
    cache = TreeCache.load(new_cache_path, exemplar)

    ds = TokenSeqDataset(cache, SEQ_LEN)

    time_in = time.time()
    num_batches = await ds.async_len()
    print(f"Time to get len: {time.time() - time_in}")
    time_in = time.time()

    for b in range(BATCHES):
        indices = []
        for j in range(BS):
            idx = b * BS + j
            idx = idx % num_batches
            indices.append(idx)
        elems = await ds.get_batch(indices)
        del elems

    print(f"Time to get batches: {time.time() - time_in}")


async def bench_new_cache_serial_tokenseq_littlesync(exemplar, new_cache_path):
    ensure_cache(new_cache_path)
    cache = TreeCache.load(new_cache_path, exemplar)

    ds = TokenSeqDataset(cache, SEQ_LEN)

    time_in = time.time()
    num_batches = await ds.async_len()
    print(f"Time to get len: {time.time() - time_in}")
    time_in = time.time()

    for b in range(BATCHES):
        indices = []
        for j in range(BS):
            idx = b * BS + j
            idx = idx % num_batches
            indices.append(idx)
        elems = await ds.get_batch_littlesync(indices)
        del elems

    print(f"Time to get batches: {time.time() - time_in}")


async def bench_new_cache_serial_tokenseq_quasisync(exemplar, new_cache_path):
    ensure_cache(new_cache_path)
    cache = TreeCache.load(new_cache_path, exemplar)

    ds = TokenSeqDataset(cache, SEQ_LEN)

    time_in = time.time()
    num_batches = await ds.async_len()
    print(f"Time to get len: {time.time() - time_in}")
    time_in = time.time()

    for b in range(BATCHES):
        indices = []
        for j in range(BS):
            idx = b * BS + j
            idx = idx % num_batches
            indices.append(idx)
        elems = await ds.get_batch_quasisync(indices)
        del elems

    print(f"Time to get batches: {time.time() - time_in}")


def bench_new_cache_serial_tokenseq_sync(exemplar, new_cache_path):
    ensure_cache(new_cache_path)
    cache = TreeCache.load(new_cache_path, exemplar)

    ds = TokenSeqDataset(cache, SEQ_LEN)

    num_batches = asyncio.run(ds.async_len())

    for b in range(BATCHES):
        indices = []
        for j in range(BS):
            idx = b * BS + j
            idx = idx % num_batches
            indices.append(idx)
        elems = ds.get_batch_sync(indices)
        del elems


def ensure_cache(new_cache_path):
    if not fsspec_utils.exists(os.path.join(new_cache_path, LEDGER_FILE_NAME)):
        ledger = CacheLedger(100000, {}, True)
        _serialize_json_and_commit(os.path.join(new_cache_path, LEDGER_FILE_NAME), ledger)


if __name__ == "__main__":
    import sys

    if not len(sys.argv) == 3:
        print("Usage: convert_to_new_cache.py old_cache_path new_cache_path")
        sys.exit(1)

    for split in ["validation", "train"]:
        print(f"Split: {split}", flush=True)
        in_path = os.path.join(sys.argv[1], split)
        out_path = os.path.join(sys.argv[2], split)
        # convert_to_new_cache(in_path, out_path)
        # with capture_time() as time_fn:
        #     bench_old_cache(in_path)
        # tokens_per_second = SEQ_LEN * BS * BATCHES / time_fn()
        # print(f"Old Cache: {time_fn()} ({tokens_per_second} tps)", flush=True)

        exemplar = {"input_ids": np.zeros((SEQ_LEN,), dtype=np.int32)}

        with capture_time() as time_fn:
            bench_new_cache_serial(exemplar, out_path)
        tokens_per_second = SEQ_LEN * BS * BATCHES / time_fn()
        print(f"New Cache Serial: {time_fn()} ({tokens_per_second} tps)", flush=True)

        with capture_time() as time_fn:
            asyncio.run(bench_new_cache_serial_tokenseq_littlesync(exemplar, out_path))
        tokens_per_second = SEQ_LEN * BS * BATCHES / time_fn()

        print(f"New Cache Serial TokenSeq littlesync: {time_fn()} ({tokens_per_second} tps)", flush=True)

        with capture_time() as time_fn:
            bench_new_cache_serial_tokenseq_sync(exemplar, out_path)
        tokens_per_second = SEQ_LEN * BS * BATCHES / time_fn()

        print(f"New Cache Serial TokenSeq Sync: {time_fn()} ({tokens_per_second} tps)", flush=True)

        with capture_time() as time_fn:
            asyncio.run(bench_new_cache_serial_tokenseq_minisync(exemplar, out_path))
        tokens_per_second = SEQ_LEN * BS * BATCHES / time_fn()

        print(f"New Cache Serial TokenSeq minisync: {time_fn()} ({tokens_per_second} tps)", flush=True)

        with capture_time() as time_fn:
            asyncio.run(bench_new_cache_serial_tokenseq_quasisync(exemplar, out_path))
        tokens_per_second = SEQ_LEN * BS * BATCHES / time_fn()

        print(f"New Cache Serial TokenSeq quasisync: {time_fn()} ({tokens_per_second} tps)", flush=True)

        with capture_time() as time_fn:
            asyncio.run(bench_new_cache_serial_tokenseq(exemplar, out_path))
        tokens_per_second = SEQ_LEN * BS * BATCHES / time_fn()

        print(f"New Cache Serial TokenSeq: {time_fn()} ({tokens_per_second} tps)", flush=True)

        with capture_time() as time_fn:
            bench_new_cache_random(exemplar, out_path)
        tokens_per_second = SEQ_LEN * BS * BATCHES / time_fn()

        print(f"New Cache Random: {time_fn()} ({tokens_per_second} tps)", flush=True)

        with capture_time() as time_fn:
            asyncio.run(bench_new_cache_permutation_random(exemplar, out_path))
        tokens_per_second = SEQ_LEN * BS * BATCHES / time_fn()

        print(f"New Cache Permutation: {time_fn()} ({tokens_per_second} tps)", flush=True)
