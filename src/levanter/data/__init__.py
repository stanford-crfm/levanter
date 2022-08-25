import json
import os
from dataclasses import dataclass
from functools import cached_property
from typing import List, Optional

import braceexpand
import datasets
import fsspec
from transformers import AutoTokenizer

from levanter.data.text import TokenizedDocumentCache, tokenize_batch
from levanter.data.utils import batched


@dataclass
class LMDatasetConfig:
    """This class supports loading data both from HF Datasets and from a raw dataset of jsonl urls"""

    id: Optional[str] = None  # id (or path) for hf dataset
    name: Optional[str] = None  # name for hf dataset
    stream: bool = True  # whether to use streaming when doing hf

    train_urls: List[str] = ()  # type: ignore
    validation_urls: List[str] = ()  # type:ignore

    tokenizer: str = "gpt2"
    enforce_eos: bool = True
    text_key: str = "text"  # key for the text field in the jsonl file or hf dataset

    @cached_property
    def the_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.tokenizer)

    def doc_iterator(self, split: str):
        if self.id is not None:
            dataset = datasets.load_dataset(self.id, name=self.name, streaming=self.stream)
            data = dataset[split]
            for doc in data:
                text = doc[self.text_key]
                if self.enforce_eos:
                    text += self.the_tokenizer.eos_token
                yield text
        else:
            if split == "train":
                urls = self.train_urls
            elif split == "validation":
                urls = self.validation_urls
            else:
                raise ValueError(f"Unknown split {split}")

            urls = [url for pat in urls for url in braceexpand.braceexpand(pat)]
            files = fsspec.open_files(urls, "rb", compression="infer")
            for file in files:
                with file as f:
                    for line in f.readlines():
                        text = json.loads(line.decode("utf-8"))[self.text_key]
                        if self.enforce_eos:
                            text += self.the_tokenizer.eos_token
                        yield text

    # def __post_init__(self):
    #     if self.id is None and len(self.train_urls) == 0 and len(self.validation_urls) == 0:
    #         raise ValueError("Either id or urls must be provided")


@dataclass
class CachedLMDatasetConfig(LMDatasetConfig):
    cache_dir: str = "cache/"
    num_train_shards: int = 128
    num_val_shards: int = 32

    train_group_size: int = 1000
    val_group_size: int = 100

    def build_or_load_document_cache(self, split: str):
        cache_dir = os.path.join(self.cache_dir, f"{split}")
        # TODO: think about doing this on apache beam or something fancy. Maybe nothing fancy we can do for HF datasets,
        # but for pure-url based ones, shouldn't be hard.
        doc_iter = self.doc_iterator(split)
        group_size = self.train_group_size if split == "train" else self.val_group_size
        token_iter = (
            tokenize_batch(self.the_tokenizer, batch, self.enforce_eos) for batch in batched(doc_iter, group_size)
        )

        num_shards = self.num_train_shards if split == "train" else self.num_val_shards

        return TokenizedDocumentCache.build_or_load(token_iter, cache_dir, num_shards, self.enforce_eos)


__all__ = ["LMDatasetConfig", "CachedLMDatasetConfig", "batched"]
