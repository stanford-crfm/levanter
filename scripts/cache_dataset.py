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

    for split in ["train", "validation"]:
        # connect or start the actor
        batch_tokenizer = BatchTokenizer(tokenizer)
        source = args.get_shard_source(split)

        cache_dataset(os.path.join(args.cache_dir, split), source, batch_tokenizer)
        print(f"Finished caching {split} to {args.cache_dir}/{split}.")


if __name__ == "__main__":
    main()
