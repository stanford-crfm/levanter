import json
import os
from dataclasses import dataclass
from typing import Optional, List

import braceexpand as braceexpand
import datasets
import fsspec
import pyrallis
from fsspec.core import OpenFile
from transformers import PreTrainedTokenizerFast, AutoTokenizer

from psithuros.data.text import preprocess_dataset, batched, tokenize_batch, build_cache


@dataclass
class CacheDatasetArgs:
    id: Optional[str] = None
    name: Optional[str] = None
    urls: List[str] = ()
    tokenizer: str = "gpt2"
    enforce_eos: bool = True
    cache_dir: str = "cache/"
    splits: str = "train,validation"
    num_shards: int = 128
    stream: bool = True


@pyrallis.wrap()
def main(args: CacheDatasetArgs):
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.id is None and len(args.urls) == 0:
        raise ValueError("Either id or urls must be provided")
    elif args.id is not None and len(args.urls) > 0:
        raise ValueError("Either id or urls must be provided, not both")
    elif args.id:
        dataset = datasets.load_dataset(args.id, name=args.name, streaming=args.stream)
        for split in args.splits.split(","):
            cache_dir = os.path.join(args.cache_dir, args.id, split)
            # seqlen doesn't matter for just building
            preprocess_dataset(dataset[split], tokenizer, seq_len=1024, cache_dir=cache_dir, num_shards=args.num_shards,
                               enforce_eos=args.enforce_eos)
    elif args.urls:
        # build a big generator over the files via fsspec
        urls = [u for url in args.urls for u in braceexpand.braceexpand(url)]
        def gen():
            files  = fsspec.open_files(urls, "rb", compression="infer")
            file: OpenFile
            for file in files:
                fin = file.open()
                try:
                    for line in fin.readlines():
                        yield json.loads(line.decode("utf-8"))["text"]
                finally:
                    fin.close()

        token_iter = (tokenize_batch(tokenizer, batch, args.enforce_eos) for batch in batched(gen(), 1000))
        build_cache(token_iter, cache_dir=args.cache_dir, num_shards=args.num_shards)


if __name__ == '__main__':
    main()
