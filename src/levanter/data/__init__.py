import json
import os
from dataclasses import dataclass
from functools import cached_property
from typing import List, Optional

import braceexpand
import datasets
import fsspec
import numpy
from transformers import AutoTokenizer

from levanter.data.dataset import Dataset, ShuffleDataset
from levanter.data.text import TokenizedDocumentCache, tokenize_batch
from levanter.data.utils import batched


class PassthroughTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self._vocab_size = vocab_size
        self._eos = self._vocab_size - 1
        self._eos_token = str(self._eos)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def eos_token(self) -> str:
        return self._eos_token

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str, ...]:
        return ()

    def _tokenize(self, text, **kwargs):
        tokens = numpy.fromstring(text, dtype=int, sep=" ")
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return int(token)

    def _convert_id_to_token(self, index: int) -> str:
        return str(index)


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
        if self.tokenizer == "passthrough":
            return PassthroughTokenizer(34026)  # hard-coding the vocab size for now
        else:
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
            files = fsspec.open_files(urls, "r", compression="infer")
            for file in files:
                print(file)
                with file as f:
                    for line in f.readlines():
                        text = json.loads(line)[self.text_key]
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


__all__ = [
    "LMDatasetConfig",
    "CachedLMDatasetConfig",
    "batched",
    "Dataset",
    "ShuffleDataset",
    "TokenizedDocumentCache",
]
