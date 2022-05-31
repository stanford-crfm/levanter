# Dataset for preprocessing data, tokenizing, and caching to disk
# The general file format we're going with is an apache parquet file with columns for the output of the tokenizer,
# A row is a single doc. Parquet files are efficient column stores, which means that we can grab token slices from
# multiple docs as a single operation, which makes concatenation much faster (and means we don't need to cache slices).
# (We might add back in file metadata later? though huggingface deletes it)
# We don't want to have one giant file, so we'll split it up into chunks.
# In general, an IndexedDataset is a directory of parquet files plus a metadata file called the ledger.
# The ledger is a json file with the following structure:
# {
#   "files": { "file_name": <name>, "num_tokens": <num_tokens>},
# }
# We don't actually use the num_tokens field, but it's useful for sanity checking.
# The ledger is written last, so we can always check to see if we were interrupted.
import copy
import json
import logging
import os
from itertools import chain
from pathlib import Path
from typing import Iterator, Optional, TypeVar, Iterable, List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import BatchEncoding, AutoTokenizer, PreTrainedTokenizerFast

# As a heuristic, we're aiming for files that are around ~250MB
# Typically we're training on sequences of length ~1024 and batch size up to 512, so better to make it divisible by that.
# 4bytes * 512 * 1024 = 2Mi, so we'll go with 128 * 512 * 1024 = 67108864 tokens, which is about 256MiB

NUM_TOKENS_PER_FILE = 67108864

overwatch = logging.getLogger("psithuros.data.text")

# TASKS:
# TODO: figure out directory structure for caching multiple sources
# TODO: if we're super careful we can compute the number of samples (for a given batch size and stride) in advance
#       if we do that, we can implement a Map-style dataset, which is somewhat preferable when not streaming
# TODO: bring in sprucfluo/simultaneous caching and streaming if we want.

LEDGER_FILE = "ledger.json"


class IndexedDataset:
    def __init__(self, cache_dir, seq_len: int, stride: Optional[int] = None):
        self.cache_dir = cache_dir
        self.ledger = self._load_ledger()
        self.seq_len = seq_len
        self.stride = stride

    def _files(self):
        for entry in self.ledger["files"]:
            yield entry["file_name"]

    def __iter__(self) -> Iterator[BatchEncoding]:
        for file_name in self._files():
            for entry in read_cache_file(file_name, flatten=True):
                yield from concatenate_and_group_texts(entry, self.seq_len, self.stride)

    @staticmethod
    def build_or_load(token_iter: Iterator[BatchEncoding],
                      cache_dir: str,
                      seq_len: int,
                      stride: Optional[int] = None,
                      num_tokens_per_file: int = NUM_TOKENS_PER_FILE,
                      file_template: str = 'docs-{}.parquet') -> 'IndexedDataset':
        os.makedirs(cache_dir, exist_ok=True)
        ledger_file = os.path.join(cache_dir, LEDGER_FILE)

        if os.path.exists(ledger_file):
            overwatch.info("Found existing indexed dataset at %s", cache_dir)
            return IndexedDataset(cache_dir, seq_len, stride)

        file_index = 0
        current_writer: Optional[pq.ParquetWriter] = None
        current_num_tokens = 0
        tq: tqdm = tqdm(desc=f"file {file_index} progress", total=num_tokens_per_file, unit="token")
        file_out: Optional[Path] = None

        # list of (file_name, num_tokens), to be output at the end if we finish the whole iterator
        ledger_files = []

        def close_writer():
            nonlocal current_writer, file_out, file_index, current_num_tokens
            if current_writer is not None:
                current_writer.close()
                current_writer = None

            if current_num_tokens > 0:
                ledger_files.append({"file_name": str(file_out), "num_tokens": current_num_tokens})

        try:
            for tokens in token_iter:
                batch = _as_record_batch(tokens)
                batch_len = sum(len(t) for t in tokens["input_ids"])

                if current_writer and current_num_tokens + batch_len > num_tokens_per_file:
                    close_writer()

                if not current_writer:
                    file_out = Path(f"{cache_dir}/{file_template.format(file_index)}")
                    file_out.parent.mkdir(parents=True, exist_ok=True)
                    file_index += 1

                    current_writer = pq.ParquetWriter(file_out, batch.schema, version="2.6", compression="ZSTD")

                    current_num_tokens = 0

                    tq.reset()
                    tq.set_description(f"file {file_index} progress")

                current_writer.write_batch(batch)
                current_num_tokens += batch_len
                tq.update(batch_len)

            if current_writer:
                tq.reset(current_num_tokens)
                tq.update(current_num_tokens)
                close_writer()

            # if we successfully wrote the whole iterator, we can write the ledger
            with open(ledger_file, "w") as f:
                ledger = {"files": ledger_files}
                json.dump(ledger, f)

            return IndexedDataset(cache_dir, seq_len, stride)
        except (KeyboardInterrupt, InterruptedError):
            current_writer.close()
            current_writer = None
            file_out.unlink(missing_ok=True)
            raise

    def _load_ledger(self):
        ledger_path = os.path.join(self.cache_dir, LEDGER_FILE)
        if os.path.exists(ledger_path):
            with open(ledger_path, "r") as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"{self.cache_dir} is not a complete cache")


def read_cache_file(file, flatten: bool = False) -> Iterator[BatchEncoding]:
    """ Reads the cache files produced by cache_and_group and yields tokenized sequences.
    If flatten is false, this returns the docs as they were presented to the caching process. If flatten is True,
    then the documents returned are actually concatenated documents, where the number is the number of documents
    presented as a batch to the caching process."""
    for b in pq.read_table(file).to_batches():
        if flatten:
            # insert a newaxis to the beginning so that it appears to be bs=1
            yield BatchEncoding(
                {b.field(i).name: b.column(i).values.to_numpy(zero_copy_only=True)[np.newaxis, :] for i in
                 range(b.num_columns)}
            )
        else:
            yield BatchEncoding(
                {b.field(i).name: b.column(i).to_numpy(zero_copy_only=False) for i in range(b.num_columns)})


def _as_record_batch(doc: BatchEncoding) -> pa.RecordBatch:
    names, columns = zip(*[(k, pa.array(v)) for k, v in doc.items()])
    return pa.RecordBatch.from_arrays(list(columns), names)


T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)


def batched(iterable: Iterable[T], batch_size: int) -> Iterator[List[T]]:
    """Yields batches of the given size from the given iterable."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []


def concatenate_and_group_texts(encoding: BatchEncoding, seq_len: int,
                                stride: Optional[int] = None,
                                drop_remainder: bool = True,
                                mask_stride_overlap=True) -> Iterator[BatchEncoding]:
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
    if drop_remainder and total_length % stride != 0 :
        total_length = ((total_length - seq_len + stride) // stride) * stride

    # Split by Chunks of Maximum Length
    # we want to take chunks up until we've covered all "total_length" tokens with a sliding window of size "stride"
    for begin in range(0, total_length - seq_len + stride, stride):
        data = {k: v[begin:begin+seq_len] for k, v in concatenated.items()}

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
        labels[0:target_len - stride] = sentinel

    return labels


if __name__ == '__main__':
    import numpy
    import datasets
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained('gpt2')
    dataset = datasets.load_dataset("dlwh/wikitext_103_detokenized", split="train")

    def batch_tokenize(ds: datasets.Dataset, tokenizer, batch_size: int, text_column="text") -> Iterator[BatchEncoding]:
        """Yields batches of tokenized sentences from the given dataset."""
        for batch in batched(ds[text_column], batch_size):
            yield tokenizer(batch)

    token_iter = batch_tokenize(dataset, tokenizer, batch_size=1000)
    indexer = IndexedDataset.build_or_load(batch_tokenize(dataset, tokenizer, batch_size=1000),
                                           "cache/wikitext-103-indexed", seq_len=512, stride=None)

    for i, batch in enumerate(indexer):
        print(i, batch)

