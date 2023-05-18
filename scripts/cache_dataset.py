import os

from dataclasses import dataclass

import levanter
from levanter.config import RayConfig
from levanter.data.shard_cache import cache_dataset
from levanter.data.text import BatchTokenizer, LMDatasetConfig
from levanter.logging import init_logger


@dataclass
class RayCachedLMDatasetConfig(LMDatasetConfig, RayConfig):
    pass


@levanter.config.main()
def main(args: RayCachedLMDatasetConfig):
    """Caches two different kinds of datasets. It can cache a dataset from a list of urls, or a dataset from a hf dataset"""
    init_logger("cache_dataset.log")
    args.initialize()

    tokenizer = args.the_tokenizer

    for split in args.splits:
        print(f"Caching {split} to {args.cache_dir}.")
        # connect or start the actor
        batch_tokenizer = BatchTokenizer(tokenizer)
        try:
            source = args.get_shard_source(split)
        except ValueError as e:
            print(f"Skipping {split} because it doesn't exist: {e}")
            continue
        split_cache_dir = os.path.join(args.cache_dir, split)
        cache_dataset(split_cache_dir, source, batch_tokenizer)

        print(f"Finished caching {split} to {split_cache_dir}.")


if __name__ == "__main__":
    main()
