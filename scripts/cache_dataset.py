import os
from dataclasses import dataclass
from typing import Optional

import datasets
import pyrallis
from datasets import Split
from transformers import PreTrainedTokenizerFast, AutoTokenizer

from psithuros.data.text import IndexedDataset, preprocess_dataset


@dataclass
class CacheDatasetArgs:
    id: str
    name: Optional[str] = None
    tokenizer: str = "gpt2"
    enforce_eos: bool = True
    cache_dir: str = "cache/"
    splits: str = "train,validation"
    num_shards: int = 128


@pyrallis.wrap()
def main(args: CacheDatasetArgs):
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.tokenizer)

    dataset = datasets.load_dataset(args.id, name=args.name)
    for split in args.splits.split(","):
        cache_dir = os.path.join(args.cache_dir, args.id, split)
        # seqlen doesn't matter for just building
        preprocess_dataset(dataset[split], tokenizer, seq_len=1024, cache_dir=cache_dir, num_shards=args.num_shards,
                           enforce_eos=args.enforce_eos)


if __name__ == '__main__':
    main()
