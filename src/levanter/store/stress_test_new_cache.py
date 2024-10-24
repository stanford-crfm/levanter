# Reads an old-style ShardCache and compares to
import asyncio
import logging
import os

import jax.random
import numpy as np
import tensorstore as ts

from levanter.data import PermutationDataset
from levanter.data.text import TokenSeqDataset
from levanter.store.cache import LEDGER_FILE_NAME, CacheLedger, TreeCache, _serialize_json_and_commit
from levanter.store.tree_store import TreeStore
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


def bench_new_cache_serial(exemplar, new_cache_path):
    jagged_array = TreeStore.open(exemplar, new_cache_path).tree["input_ids"]
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
    jagged_array = TreeStore.open(exemplar, new_cache_path).tree["input_ids"]
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
    ds = PermutationDataset(ds, jax.random.PRNGKey(0))

    num_batches = await ds.async_len()

    for b in range(BATCHES):
        indices = []
        for j in range(BS):
            idx = b * BS + j
            idx = idx % num_batches
            indices.append(idx)
        elems = await ds.get_batch(indices)
        del elems


def ensure_cache(new_cache_path):
    if not fsspec_utils.exists(os.path.join(new_cache_path, LEDGER_FILE_NAME)):
        ledger = CacheLedger(100000, {}, True)
        _serialize_json_and_commit(os.path.join(new_cache_path, LEDGER_FILE_NAME), ledger)


if __name__ == "__main__":
    import sys

    if not len(sys.argv) == 2:
        print("Usage: convert_to_new_cache.py new_cache_path")
        sys.exit(1)

    for split in ["validation", "train"]:
        print(f"Split: {split}", flush=True)
        cache_path = os.path.join(sys.argv[1], split)
        # convert_to_new_cache(in_path, out_path)
        # with capture_time() as time_fn:
        #     bench_old_cache(in_path)
        # tokens_per_second = SEQ_LEN * BS * BATCHES / time_fn()
        # print(f"Old Cache: {time_fn()} ({tokens_per_second} tps)", flush=True)

        exemplar = {"input_ids": np.zeros((SEQ_LEN,), dtype=np.int32)}

        with capture_time() as time_fn:
            bench_new_cache_serial(exemplar, cache_path)
        tokens_per_second = SEQ_LEN * BS * BATCHES / time_fn()
        print(f"New Cache Serial: {time_fn()} ({tokens_per_second} tps)", flush=True)

        with capture_time() as time_fn:
            asyncio.run(bench_new_cache_serial_tokenseq(exemplar, cache_path))
        tokens_per_second = SEQ_LEN * BS * BATCHES / time_fn()

        print(f"New Cache Serial TokenSeq: {time_fn()} ({tokens_per_second} tps)", flush=True)

        with capture_time() as time_fn:
            bench_new_cache_random(exemplar, cache_path)
        tokens_per_second = SEQ_LEN * BS * BATCHES / time_fn()

        print(f"New Cache Random: {time_fn()} ({tokens_per_second} tps)", flush=True)

        with capture_time() as time_fn:
            asyncio.run(bench_new_cache_permutation_random(exemplar, cache_path))
        tokens_per_second = SEQ_LEN * BS * BATCHES / time_fn()

        print(f"New Cache Permutation: {time_fn()} ({tokens_per_second} tps)", flush=True)
