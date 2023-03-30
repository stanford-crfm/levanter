from dataclasses import dataclass
from typing import Optional

import ray

import levanter
from levanter.data.shard_cache import _ShardCacheManager
from levanter.data.text import BatchTokenizer, CachedLMDatasetConfig

@dataclass
class RayConfig:
    address: Optional[str] = None

@dataclass
class RayCachedLMDatasetConfig(CachedLMDatasetConfig, RayConfig):
    pass


@levanter.config.main()
def main(args: RayCachedLMDatasetConfig):
    """Caches two different kinds of datasets. It can cache a dataset from a list of urls, or a dataset from a hf dataset"""
    ray.init(address=args.address)

    tokenizer = args.the_tokenizer


    for split in ["train", "validation"]:
        # connect or start the actor
        batch_tokenizer = BatchTokenizer(tokenizer, args.train_group_size)
        manager = _ShardCacheManager.options(name="cache_manager", get_if_exists=True)\
            .remote(f"{args.cache_dir}/{split}", args.get_shard_source(split), batch_tokenizer)
        # but for pure-url based ones, shouldn't be hard.
        args.build_or_load_document_cache(split)


if __name__ == "__main__":
    main()