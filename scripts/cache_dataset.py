import json
from dataclasses import dataclass
from typing import Optional, List

import braceexpand as braceexpand
import datasets
import fsspec
import pyrallis
from fsspec.core import OpenFile
from transformers import PreTrainedTokenizerFast, AutoTokenizer

from psithuros.data.text import batched, tokenize_batch, build_cache


@dataclass
class CacheDatasetArgs:
    id: Optional[str] = None
    name: Optional[str] = None
    splits: str = "train,validation"

    urls: List[str] = ()

    tokenizer: str = "gpt2"
    enforce_eos: bool = True
    cache_dir: str = "cache/"
    num_shards: int = 128
    stream: bool = True


@pyrallis.wrap()
def main(args: CacheDatasetArgs):
    """Caches two different kinds of datasets. It can cache a dataset from a list of urls, or a dataset from a hf dataset"""
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.id is None and len(args.urls) == 0:
        raise ValueError("Either id or urls must be provided")
    elif args.id is not None and len(args.urls) > 0:
        raise ValueError("Either id or urls must be provided, not both")
    elif args.id:
        dataset = datasets.load_dataset(args.id, name=args.name, streaming=args.stream)
        for split in args.splits.split(","):
            # seqlen doesn't matter for just building
            data_split = dataset[split]
            text_iter = (x["text"] for x in data_split)
    elif args.urls:
        # build a big generator over the files via fsspec
        urls = [u for url in args.urls for u in braceexpand.braceexpand(url)]
        def gen():
            files  = fsspec.open_files(urls, "rb", compression="infer")
            file: OpenFile
            for file in files:
                with file as f:
                    for line in f.readlines():
                        yield json.loads(line.decode("utf-8"))["text"]

        text_iter = gen()

    token_iter = (tokenize_batch(tokenizer, batch, args.enforce_eos) for batch in batched(text_iter, 1000))

    # TODO: think about doing this on apache beam or something fancy. Maybe nothing fancy we can do for HF datasets, but
    # for pure-url based ones, shouldn't be hard.
    build_cache(token_iter, args.cache_dir, args.num_shards)


if __name__ == '__main__':
    main()
