# Reads an old-style ShardCache and writes a new-style TreeCache
import os

import jax
import numpy as np

from levanter.data import batched
from levanter.data.shard_cache import DictCacheDataset, ShardCache
from levanter.newstore.jagged_array import JaggedArrayBuilder
from levanter.newstore.tree_store import TreeStoreBuilder


def convert_to_new_cache(old_cache_path, new_cache_path):
    old_cache = ShardCache.load(old_cache_path, 1)
    ds = DictCacheDataset(old_cache)
    item = next(iter(ds))
    item["input_ids"] = item["input_ids"].astype(np.int32)
    new_cache = TreeStoreBuilder.open(item, new_cache_path, mode="w")

    for batch in batched(ds, 32768):
        for i in range(len(batch)):
            batch[i]["input_ids"] = batch[i]["input_ids"].astype(np.int32)
        new_cache.extend(batch)

    new_cache.reload()

    leaves = jax.tree_leaves(new_cache.tree)

    for leaf in leaves:
        assert isinstance(leaf, JaggedArrayBuilder)

        print(len(leaf))
        print(leaf.data_size)


if __name__ == "__main__":
    import sys

    if not len(sys.argv) == 3:
        print("Usage: convert_to_new_cache.py old_cache_path new_cache_path")
        sys.exit(1)

    for split in ["train", "validation"]:
        in_path = os.path.join(sys.argv[1], split)
        out_path = os.path.join(sys.argv[2], split)
        convert_to_new_cache(in_path, out_path)
