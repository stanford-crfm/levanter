import os

from dataclasses import dataclass

import wandb

import levanter
from levanter.config import RayConfig
from levanter.data.shard_cache import RichMetricsMonitor, WandbMetricsMonitor, cache_dataset
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

    wandb.init(mode="offline")

    for split in ["train", "validation"]:
        # connect or start the actor
        batch_tokenizer = BatchTokenizer(tokenizer)
        source = args.get_shard_source(split)

        cache = cache_dataset(os.path.join(args.cache_dir, split), source, batch_tokenizer, await_finished=False)

        cache.attach_metrics_monitor(RichMetricsMonitor(source.num_shards))
        cache.attach_metrics_monitor(WandbMetricsMonitor("preprocess/" + split, commit=True))

        cache.await_finished()
        print(f"Finished caching {split} to {args.cache_dir}/{split}.")


if __name__ == "__main__":
    main()
