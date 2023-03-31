import time
from dataclasses import dataclass
from typing import Optional

import ray

import levanter
from levanter.data.shard_cache import _ShardCacheManager
from levanter.data.text import BatchTokenizer, LMDatasetConfig


@dataclass
class RayConfig:
    address: Optional[str] = None


@dataclass
class RayCachedLMDatasetConfig(LMDatasetConfig, RayConfig):
    pass


@levanter.config.main()
def main(args: RayCachedLMDatasetConfig):
    """Caches two different kinds of datasets. It can cache a dataset from a list of urls, or a dataset from a hf dataset"""
    ray.init(address=args.address)

    tokenizer = args.the_tokenizer

    for split in ["train", "validation"]:
        # connect or start the actor
        batch_tokenizer = BatchTokenizer(tokenizer, args.train_group_size)
        manager = _ShardCacheManager.options(name="cache_manager", get_if_exists=True).remote(  # type: ignore
            f"{args.cache_dir}/{split}", args.get_shard_source(split), batch_tokenizer
        )

        while not ray.get(manager.is_finished.remote()):
            print("Waiting for cache to be built")
            time.sleep(5)

        print(f"Finished caching {split}")


if __name__ == "__main__":
    main()
