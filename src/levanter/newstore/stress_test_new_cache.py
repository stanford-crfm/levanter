# Reads an old-style ShardCache and compares to
import os

import numpy as np
import tensorstore as ts

from levanter.newstore.tree_store import TreeStoreBuilder
from levanter.tracker import capture_time


SEQ_LEN = 4096
BS = 128
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
    new_cache = TreeStoreBuilder.open(exemplar, new_cache_path).tree["input_ids"].data
    len_cache = new_cache.shape[0]
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
    new_cache: ts.TensorStore = TreeStoreBuilder.open(exemplar, new_cache_path).tree["input_ids"].data
    len_cache = new_cache.shape[0]
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
        with capture_time() as time_fn:
            bench_old_cache(in_path)
        tokens_per_second = SEQ_LEN * BS * BATCHES / time_fn()
        print(f"Old Cache: {time_fn()} ({tokens_per_second} tps)", flush=True)
        exemplar = {"input_ids": np.zeros((SEQ_LEN,), dtype=np.int32)}

        with capture_time() as time_fn:
            bench_new_cache_serial(exemplar, out_path)
        tokens_per_second = SEQ_LEN * BS * BATCHES / time_fn()
        print(f"New Cache Serial: {time_fn()} ({tokens_per_second} tps)", flush=True)

        with capture_time() as time_fn:
            bench_new_cache_random(exemplar, out_path)
        tokens_per_second = SEQ_LEN * BS * BATCHES / time_fn()

        print(f"New Cache Random: {time_fn()} ({tokens_per_second} tps)", flush=True)
