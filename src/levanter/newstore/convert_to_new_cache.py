# Reads an old-style ShardCache and writes a new-style TreeCache
import os

import jax
import numpy as np
import tqdm

from levanter.data import batched
from levanter.data.shard_cache import DictCacheDataset, ShardCache
from levanter.newstore.jagged_array import JaggedArrayStore
from levanter.newstore.tree_store import TreeStoreBuilder
from levanter.utils.background_iterable import BackgroundIterable


def convert_to_new_cache(old_cache_path, new_cache_path):
    old_cache = ShardCache.load(old_cache_path, 1)
    ds = DictCacheDataset(old_cache)
    item = next(iter(ds))
    item["input_ids"] = item["input_ids"].astype(np.int32)
    new_cache = TreeStoreBuilder.open(item, new_cache_path, mode="w")

    bg_iterable = BackgroundIterable(lambda: iter(ds), max_capacity=32768)

    bi = 0
    for batch in tqdm.tqdm(batched(bg_iterable, 32768)):
        for i in range(len(batch)):
            batch[i]["input_ids"] = batch[i]["input_ids"].astype(np.int32)
        new_cache.extend(batch)
        bi += 1
        if bi > 2000:
            break
        elif bi % 25 == 0:
            print(f"Processed {bi} batches", flush=True)

    new_cache.reload()

    leaves = jax.tree_leaves(new_cache.tree)

    for leaf in leaves:
        assert isinstance(leaf, JaggedArrayStore)

        print(len(leaf))
        print(leaf.data_size)

    new_cache = TreeStoreBuilder.open(item, new_cache_path, mode="r")

    for leaf in jax.tree.leaves(new_cache.tree):
        assert isinstance(leaf, JaggedArrayStore)

        print(len(leaf))
        print(leaf.data_size)


if __name__ == "__main__":
    import sys

    if not len(sys.argv) == 3:
        print("Usage: convert_to_new_cache.py old_cache_path new_cache_path")
        sys.exit(1)

    for split in ["validation", "train"]:
        in_path = os.path.join(sys.argv[1], split)
        out_path = os.path.join(sys.argv[2], split)
        convert_to_new_cache(in_path, out_path)
