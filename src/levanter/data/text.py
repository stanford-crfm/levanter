# Dataset for preprocessing data, tokenizing, and caching to disk
# The general file format we're going with is an apache parquet file with columns for the output of the tokenizer,
# A row is a single doc. Parquet files are efficient column stores, which means that we can grab token slices from
# multiple docs as a single operation, which makes concatenation much faster (and means we don't need to cache slices).
# (We might add back in file metadata later? though huggingface deletes it)
# We don't want to have one giant file, so we'll split it up into chunks.
# In general, an IndexedDataset is a directory of parquet files plus a metadata file called the ledger.
# The ledger is a json file with the following structure:
# {
#   "files": [{ "file_name": <name>, "num_tokens": <num_tokens>}],
# }
# We don't currently use the num_tokens field, but it's useful for sanity checking.
# The ledger is written last, so we can always check to see if we were interrupted.
import copy
import json
import logging
import os
from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from typing import Iterator, List, Optional, Sequence, Union

import braceexpand
import datasets
import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from jaxtyping import PyTree
from tqdm import tqdm
from transformers import AutoTokenizer, BatchEncoding

from levanter.data.dataset import ShardableDataset
from levanter.data.utils import batched
from levanter.shapes import NamedShapeSpec, ShapeSpec


overwatch = logging.getLogger("levanter.data.text")

# TASKS:
# TODO: figure out directory structure for caching multiple sources
# TODO: if we're super careful we can compute the number of samples (for a given batch size and stride) in advance
#       if we do that, we can implement a Map-style dataset, which is somewhat preferable when not streaming
# TODO: support seeking/serialization/restore in the dataset

LEDGER_FILE = "ledger.json"


class TokenSeqDataset(ShardableDataset[Sequence[int]]):
    """
    A dataset that yields sequences of tokens of fixed length from a TokenizedDocumentCache.
    """

    def __init__(self, doc_cache, seq_len: int, stride: Optional[int] = None):
        self.doc_cache = doc_cache
        self.seq_len = seq_len
        self.stride = stride

    def shard(self, shard_id: int, num_shards: int) -> "TokenSeqDataset":
        """
        Split the dataset into num_processes shards.
        """
        return TokenSeqDataset(self.doc_cache.shard(shard_id, num_shards), self.seq_len, self.stride)

    def __iter__(self) -> Iterator[Sequence[int]]:
        for doc in self.doc_cache:
            for encoded_slice in concatenate_and_group_texts(doc, self.seq_len, self.stride):
                yield encoded_slice["input_ids"]

    @property
    def item_shape(self) -> PyTree:
        return ShapeSpec((self.seq_len,), dtype=np.int32)

    @staticmethod
    def build_or_load(
        token_iter: Iterator[BatchEncoding],
        seq_len: int,
        cache_dir: str,
        num_shards: int,
        stride: Optional[int] = None,
        file_template: str = "docs-{}.parquet",
    ) -> "TokenSeqDataset":
        build_cache(token_iter, cache_dir, num_shards, file_template)
        doc_cache = TokenizedDocumentCache.load(cache_dir, True)
        return TokenSeqDataset(doc_cache, seq_len, stride)


def _load_ledger(cache_dir):
    ledger_path = os.path.join(cache_dir, LEDGER_FILE)

    fs, _, _ = fsspec.get_fs_token_paths(ledger_path)
    if fs.exists(ledger_path):
        with fsspec.open(ledger_path, "r") as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"{cache_dir} is not a complete cache")


class TokenizedDocumentCache(ShardableDataset[BatchEncoding]):
    """
    Represents a tokenized document cache, which is a directory of parquet files with a ledger file.

    The difference between this class and the TokenSeqDataset is that this class yields entire documents,
    while the TokenSeqDataset yields tokens sequences of fixed length from concatenated documents.

    The ledger file is a json file with the following structure:
    {
        "files": [
            {"name": <name>, "num_tokens": <num_tokens>},
            ...
        ]
    }

    The num_tokens field is not currently used, but it's useful for sanity checking.

    The parquet files are a columnar format, which means that we can grab token slices from multiple docs as a single
    operation, which makes concatenation much faster (and means we don't need to cache slices).
    """

    def __init__(self, cache_dir, cache_files, flatten_docs):
        self.cache_dir = cache_dir
        self.cache_files = cache_files
        self.flatten_docs = flatten_docs

    def __iter__(self):
        for cache_file in self.cache_files:
            full_path = os.path.join(self.cache_dir, cache_file)
            for entry in _read_cache_file(full_path, self.flatten_docs):
                yield entry

    @staticmethod
    def load(cache_dir, flatten_docs=True):
        """
        Load a TokenizedDocumentCache from a directory.
        :param cache_dir:
        :param flatten_docs: If true, then multiple documents from a single batch (when the cache was built) will be
        concatenated into a single document. Often one is concatenating documents anyway, so this is a useful option.
        :return:
        """
        ledger = _load_ledger(cache_dir)
        return TokenizedDocumentCache(cache_dir, [e["file_name"] for e in ledger["files"]], flatten_docs)

    @staticmethod
    def build_or_load(
        token_iter: Iterator[BatchEncoding],
        cache_dir: str,
        num_shards,
        flatten_docs: bool,
        file_template: str = "docs-{}.parquet",
    ) -> "TokenizedDocumentCache":
        build_cache(token_iter, cache_dir, num_shards, file_template)
        return TokenizedDocumentCache.load(cache_dir, flatten_docs)

    def shard(self, shard_index, num_shards):
        if num_shards <= shard_index:
            raise ValueError(f"Shard index {shard_index} is out of range")
        if len(self.cache_files) % num_shards != 0:
            raise ValueError(f"Number of shards {num_shards} does not divide evenly into the number of files")

        if num_shards == 1:
            return self

        return TokenizedDocumentCache(self.cache_dir, self.cache_files[shard_index::num_shards], self.flatten_docs)

    @property
    def item_shape(self) -> PyTree[Union[ShapeSpec, NamedShapeSpec]]:
        return {
            "input_ids": ShapeSpec((None,), dtype=np.int32),
        }


def _read_cache_file(file, flatten: bool = False) -> Iterator[BatchEncoding]:
    """Reads the cache files produced by cache_and_group and yields tokenized sequences.
    If flatten is false, this returns the docs as they were presented to the caching process. If flatten is True,
    then the documents returned are actually concatenated documents, where the number is the number of documents
    presented as a batch to the caching process."""
    fs, _, paths = fsspec.get_fs_token_paths(file)
    for b in pq.read_table(file, filesystem=fs).to_batches():
        if flatten:
            # insert a newaxis to the beginning so that it appears to be bs=1
            yield BatchEncoding(
                {
                    b.field(i).name: b.column(i).values.to_numpy(zero_copy_only=True)[np.newaxis, :]
                    for i in range(b.num_columns)
                }
            )
        else:
            yield BatchEncoding(
                {b.field(i).name: b.column(i).to_numpy(zero_copy_only=False) for i in range(b.num_columns)}
            )


def _as_record_batch(doc: BatchEncoding) -> pa.RecordBatch:
    """Converts a document to an arrow-compatible record batch."""
    # for dumb reasons, pa.array doesn't support ndarrays with ndim > 1
    def _as_array(x):
        if isinstance(x, np.ndarray) and x.ndim > 1:
            return [_as_array(y) for y in x]
        elif isinstance(x, np.ndarray):
            return list(x)
        else:
            return pa.array(x)

    names, columns = zip(*[(k, _as_array(v)) for k, v in doc.items()])
    return pa.RecordBatch.from_arrays(list(columns), names)


def build_cache(
    token_iter: Iterator[BatchEncoding],
    cache_dir: str,
    num_shards: int,
    file_template: str = "docs-{}.parquet",
    fsspec_args: Optional[dict] = None,
) -> None:
    ledger_file = os.path.join(cache_dir, LEDGER_FILE)

    fs, _, _ = fsspec.get_fs_token_paths(ledger_file)

    fs.makedirs(cache_dir, exist_ok=True)

    if fs.exists(ledger_file):
        overwatch.info("Found existing indexed dataset at %s", cache_dir)
        return

    file_names = [file_template.format(i) for i in range(num_shards)]
    files_to_open = []
    writers: Optional[Sequence[pq.ParquetWriter]] = None
    tokens_written_to_shard = [0] * num_shards

    tq: tqdm = tqdm(desc="tokens", unit="tk")

    try:
        for tokens in token_iter:
            batch = _as_record_batch(tokens)
            batch_len = sum(len(t) for t in tokens["input_ids"])

            shard_to_write_to = min(range(num_shards), key=lambda i: tokens_written_to_shard[i])
            tokens_written_to_shard[shard_to_write_to] += batch_len

            if writers is None:
                files_to_open = [
                    fsspec.open(os.path.join(cache_dir, f), "wb", **(fsspec_args or {})).open() for f in file_names
                ]
                writers = [
                    pq.ParquetWriter(file, batch.schema, version="2.6", compression="ZSTD") for file in files_to_open
                ]

            writers[shard_to_write_to].write_batch(batch)

            tq.update(batch_len)

        # now close out all the files:
        if writers is not None:
            for w in writers:
                w.close()

            for f in files_to_open:
                f.close()

        # if we successfully wrote the whole iterator, we can write the ledger
        # edge case: if we didn't write any documents, write an empty ledger
        if writers is None:
            with fsspec.open(ledger_file, "w") as w:
                ledger: dict = {"files": []}
                json.dump(ledger, w)
        else:
            with fsspec.open(ledger_file, "w") as w:
                ledger = {
                    "files": [
                        {"file_name": str(name), "num_tokens": count}
                        for name, count in zip(file_names, tokens_written_to_shard)
                    ]
                }
                json.dump(ledger, w)
        return

    except (KeyboardInterrupt, InterruptedError):
        if writers:
            overwatch.error("Interrupted, cleaning up files")
            for w in writers:
                w.close()
            for f in files_to_open:
                f.close()

        raise


def tokenize_batch(tokenizer, texts, enforce_eos: bool) -> BatchEncoding:
    if enforce_eos:
        tokens = tokenizer([t + tokenizer.eos_token for t in texts], return_attention_mask=False)
        assert all(t[-1] == tokenizer.eos_token_id for t in tokens["input_ids"])
        return tokens
    else:
        return tokenizer(texts, return_attention_mask=False)


def preprocess_dataset(dataset, tokenizer, cache_dir, seq_len, num_shards, enforce_eos, doc_group_size=1000):
    data = (x["text"] for x in dataset)

    token_iter = (tokenize_batch(tokenizer, batch, enforce_eos) for batch in batched(data, doc_group_size))
    return TokenSeqDataset.build_or_load(token_iter, seq_len=seq_len, cache_dir=cache_dir, num_shards=num_shards)


def concatenate_and_group_texts(
    encoding: BatchEncoding,
    seq_len: int,
    stride: Optional[int] = None,
    drop_remainder: bool = True,
    mask_stride_overlap=True,
) -> Iterator[BatchEncoding]:
    """Groups texts in a batch together. Typically, you'll want to use this with a fairly large
    set of texts, e.g. 1000 docs.

    You should set mask_stride_overlap to True and drop_remainder to False if you want to use this for test data

    Args:
        encoding: The batch of texts to concatenate and group.
        seq_len: The max length of sequences to emit
        stride: The stride to use when grouping texts. If None, then the stride is set to seq_len.
        mask_stride_overlap: Whether to mask out overlapping tokens if we're using a stride.
        drop_remainder: Whether to drop the last batch if it's not a multiple of the seq_len.

    Returns:
        An iterator of tokenized texts, one at a time.
    """
    concatenated = BatchEncoding(data={k: list(chain(*v)) for k, v in encoding.items()})
    total_length = len(concatenated.input_ids)
    stride = stride or seq_len

    # Drop the "very last" bit of the dataset that doesn't fit into block size...
    if drop_remainder and total_length % stride != 0:
        total_length = ((total_length - seq_len + stride) // stride) * stride

    # Split by Chunks of Maximum Length
    # we want to take chunks up until we've covered all "total_length" tokens with a sliding window of size "stride"
    for begin in range(0, total_length - seq_len + stride, stride):
        data = {k: v[begin : begin + seq_len] for k, v in concatenated.items()}

        if mask_stride_overlap and stride != seq_len:
            labels = data.get("labels", data["input_ids"])
            if begin != 0:
                labels = _mask_overlap(labels, seq_len, stride)
            data["labels"] = labels

        yield BatchEncoding(data=data)


# -100 is pytorch's label mask
def _mask_overlap(labels, target_len, stride, sentinel=-100):
    """Masks out overlapping tokens in a sequence when we're using a stride."""
    labels = copy.deepcopy(labels)
    if isinstance(labels, list):
        for i in range(target_len - stride):
            if i < len(labels):
                labels[i] = sentinel
    else:
        labels[0 : target_len - stride] = sentinel

    return labels


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
            files = fsspec.open_files(urls, "r", compression="infer")
            for file in files:
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
