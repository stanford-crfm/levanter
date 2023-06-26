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
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from draccus import field
from jaxtyping import PyTree

import haliax as hax
from haliax import Axis, NamedArray

# intercept the logging nonsense here
from levanter.logging import silence_transformer_nag  # noqa


silence_transformer_nag()  # noqa
from transformers import BatchEncoding, PreTrainedTokenizerBase, PreTrainedTokenizerFast  # noqa

from levanter.compat.hf_checkpoints import load_tokenizer  # noqa
from levanter.data.dataset import ShardableDataset  # noqa
from levanter.data.shard_cache import DEFAULT_ROWS_PER_CHUNK  # noqa
from levanter.data.shard_cache import LEDGER_FILE_NAME as NEW_LEDGER_FILE_NAME  # noqa
from levanter.data.shard_cache import (  # noqa
    BatchProcessor,
    CacheLedger,
    ChunkMetadata,
    LoggerMetricsMonitor,
    MetricsMonitor,
    ShardCache,
    ShardedDataSource,
    WandbMetricsMonitor,
    _load_cache_ledger,
    _serialize_json_and_commit,
    cache_dataset,
)
from levanter.shapes import NamedShapeSpec, ShapeSpec  # noqa


logger = logging.getLogger("levanter.data.text")

# TASKS:
# TODO: figure out directory structure for caching multiple sources
# TODO: consider adding indexing a la Map-style datasets
# TODO: support seeking/serialization/restore in the dataset

LEDGER_FILE = "ledger.json"


class TokenSeqDataset(ShardableDataset[NamedArray]):
    """
    A dataset that yields sequences of tokens of fixed length from a TokenizedDocumentCache.

    :param doc_cache: the TokenizedDocumentCache to draw from
    :param Pos: the axis to use for the sequences. Sequences will be a NamedArray with axis Pos
    """

    def __init__(self, doc_cache, Pos: Axis, stride: Optional[int] = None):
        self.doc_cache = doc_cache
        self.Pos = Pos
        self.stride = stride

    @property
    def seq_len(self) -> int:
        return self.Pos.size

    def shard(self, shard_id: int, num_shards: int) -> "TokenSeqDataset":
        """
        Split the dataset into num_processes shards.
        """
        return TokenSeqDataset(self.doc_cache.shard(shard_id, num_shards), self.Pos, self.stride)

    def __iter__(self) -> Iterator[NamedArray]:
        extra_tokens = None  # BatchEncoding of the last tokens from the previous doc
        for doc in self.doc_cache:
            # TODO: we could be cleverer here, and avoid these expensive copies etc
            # should run some benchmarks to see if it's worth it
            if extra_tokens is not None:
                doc = _stack_batch_encodings(extra_tokens, doc)
                extra_tokens = None

            for encoded_slice in concatenate_and_group_texts(doc, self.seq_len, self.stride, drop_remainder=False):
                if len(encoded_slice["input_ids"]) < self.seq_len:
                    assert extra_tokens is None
                    extra_tokens = encoded_slice
                else:
                    extra_tokens = None
                    ids = encoded_slice["input_ids"]
                    yield hax.named(ids, self.Pos)

    @property
    def item_shape(self) -> PyTree:
        return NamedShapeSpec((self.Pos,), jnp.int32)

    @staticmethod
    def load(pos: Axis, cache_dir: str, stride: Optional[int] = None) -> "TokenSeqDataset":
        doc_cache = TokenizedDocumentCache.load(cache_dir, True)
        return TokenSeqDataset(doc_cache, pos, stride)


def _load_old_ledger(cache_dir):
    ledger_path = os.path.join(cache_dir, LEDGER_FILE)

    fs = fsspec.core.url_to_fs(ledger_path)[0]
    if fs.exists(ledger_path):
        with fsspec.open(ledger_path, "r") as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"{cache_dir} is not a complete cache")


def _convert_to_new_ledger(cache_dir, ledger: dict) -> CacheLedger:
    # The old format looked like {"files": [{"file_name": name, "num_tokens": num_tokens} for name, num_tokens in ledger.items()]}
    # the new format looks like { "chunks": [{"name": name, "num_rows": rows, field_counts: {"input_ids": num_tokens}} for name, rows, num_tokens in ledger.items()]}
    # We unfortunately can't determine num_rows from the old format, so we have to open the chunks to find out

    return CacheLedger(
        chunks=[
            ChunkMetadata(
                name=chunk["file_name"].replace(".parquet", ""),
                num_rows=_open_arrow_table(os.path.join(cache_dir, chunk["file_name"])).num_rows,
                field_counts={"input_ids": chunk["num_tokens"]},
            )
            for chunk in ledger["files"]
        ],
    )


class TokenizedDocumentCache(ShardableDataset[BatchEncoding]):
    """
    Represents a tokenized document cache, which is a directory of parquet files with a ledger file.

    The difference between this class and the TokenSeqDataset is that this class yields entire documents,
    while the TokenSeqDataset yields tokens sequences of fixed length from concatenated documents.
    """

    def __init__(self, chunk_cache: ShardCache, flatten_docs, shard_chunk_offset=0, shard_chunk_stride=1):
        self.chunk_cache = chunk_cache
        self.flatten_docs = flatten_docs
        self.shard_chunk_offset = shard_chunk_offset
        self.shard_chunk_stride = shard_chunk_stride

    def __iter__(self):
        """Reads the cache files produced by cache_and_group and yields tokenized sequences.
        If flatten is false, this returns the docs as they were presented to the caching process. If flatten is True,
        then the documents returned are actually concatenated documents, where the number is the number of documents
        presented as a batch to the caching process."""
        for batch in self._chunks():
            yield _batch_encoding_from_record_batch(batch, self.flatten_docs)

    def _chunks(self):
        return self.chunk_cache.iter_batches_from_chunks(self.shard_chunk_offset, self.shard_chunk_stride)

    @staticmethod
    def build_or_load(
        cache_dir,
        source: ShardedDataSource[str],
        tokenizer: PreTrainedTokenizerBase,
        flatten_docs=True,
        enforce_eos=True,
        batch_size=128,
        rows_per_chunk=DEFAULT_ROWS_PER_CHUNK,
        monitors=None,
        await_finished=True,
    ) -> "TokenizedDocumentCache":
        bt = BatchTokenizer(tokenizer, enforce_eos=enforce_eos)
        monitors = monitors or []
        cache = cache_dataset(
            cache_dir,
            source,
            bt,
            await_finished=await_finished,
            batch_size=batch_size,
            rows_per_chunk=rows_per_chunk,
            monitors=monitors,
        )
        if cache.is_finished:
            logger.info(f"Cache {cache_dir} is complete.")
        else:
            logger.info(
                f"Cache {cache_dir} is incomplete. This will block until at least one chunk per process is complete."
            )

        return TokenizedDocumentCache(cache, flatten_docs=flatten_docs)

    @staticmethod
    def load(cache_dir, batch_size: int = 128, flatten_docs=True):
        """
        Load a TokenizedDocumentCache from a directory. If the ledger file is not present, this will raise a
        FileNotFoundError.

        NOTE: ATM this attempts to migrate old caches to the new format, but this will be removed in the future.

        :param cache_dir:
        :param flatten_docs: If true, then multiple documents from a single batch (when the cache was built) will be
        concatenated into a single document. Often one is concatenating documents anyway, so this is a useful option.
        :return:
        """

        try:
            cache = ShardCache.load(cache_dir, batch_size=batch_size)
            return TokenizedDocumentCache(cache, flatten_docs=flatten_docs)
        except FileNotFoundError:
            logger.info("new cache format not found, trying to convert from old format")
            try:
                ledger = _load_old_ledger(cache_dir)
                logger.info("old cache format found, converting to new format")
                ledger = _convert_to_new_ledger(cache_dir, ledger)
                _serialize_json_and_commit(os.path.join(cache_dir, NEW_LEDGER_FILE_NAME), ledger)
                cache = ShardCache.load(cache_dir, batch_size=batch_size)
                return TokenizedDocumentCache(cache, flatten_docs=flatten_docs)
            except FileNotFoundError:
                logger.warning("old cache format not found, creating new cache")
                raise FileNotFoundError(f"{cache_dir} is not a complete cache")
            except Exception:
                logger.exception("error converting cache")
                raise
        except Exception:
            logger.exception("error loading cache")
            raise

    def shard(self, shard_index, num_shards):
        if num_shards <= shard_index:
            raise ValueError(f"Shard index {shard_index} is out of range")

        if num_shards == 1:
            return self

        combined_offset = self.shard_chunk_offset + shard_index * self.shard_chunk_stride
        combined_stride = self.shard_chunk_stride * num_shards

        return TokenizedDocumentCache(
            self.chunk_cache,
            self.flatten_docs,
            shard_chunk_offset=combined_offset,
            shard_chunk_stride=combined_stride,
        )

    @property
    def item_shape(self) -> PyTree[Union[ShapeSpec, NamedShapeSpec]]:
        return {  # type: ignore
            "input_ids": ShapeSpec((None,), dtype=np.int32),
        }


def _open_arrow_table(path) -> pa.Table:
    fs, _, paths = fsspec.get_fs_token_paths(path)
    return pq.read_table(path, filesystem=fs)


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


def _batch_encoding_from_record_batch(b: pa.RecordBatch, flatten_docs: bool):
    if flatten_docs:
        # insert a newaxis to the beginning so that it appears to be bs=1
        return BatchEncoding(
            {
                b.field(i).name: b.column(i).values.to_numpy(zero_copy_only=False)[np.newaxis, :]
                for i in range(b.num_columns)
            },
            n_sequences=1,
        )
    else:
        return BatchEncoding(
            {b.field(i).name: b.column(i).to_numpy(zero_copy_only=False) for i in range(b.num_columns)},
            n_sequences=b.num_rows,
        )


def _cpu_count():
    """Returns the number of CPUs in the system."""
    try:
        return os.cpu_count()
    except NotImplementedError:
        return 1


def _maybe_force_tokenizer_parallelism(tokenizer: PreTrainedTokenizerBase):
    if tokenizer.is_fast and os.getenv("TOKENIZERS_PARALLELISM") is None:
        # if we're using a fast tokenizer, we want to force parallelism
        # to be the number of CPUs
        os.environ["TOKENIZERS_PARALLELISM"] = "true"


class BatchTokenizer(BatchProcessor[str]):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, enforce_eos=True):
        _maybe_force_tokenizer_parallelism(tokenizer)
        self.tokenizer = tokenizer

        # see if the tokenizer appends eos
        # HF's BPE-based tokenizers do not, but the bert and roberta ones do
        # TODO: this doesn't necessarily ensure it, I guess, but eh
        if enforce_eos:
            input_ids = tokenizer("hi there")["input_ids"]
            should_append_eos = input_ids[-1] != tokenizer.eos_token_id
        else:
            should_append_eos = False

        self._need_to_add_eos = should_append_eos

    def __call__(self, batch: Sequence[str]) -> pa.RecordBatch:
        if self._need_to_add_eos:
            encoding = self.tokenizer(
                [d + " " + self.tokenizer.eos_token for d in batch], return_attention_mask=False, verbose=False
            )
        else:
            encoding = self.tokenizer(batch, return_attention_mask=False, verbose=False)  # type: ignore
        return _as_record_batch(encoding)

    @property
    def num_cpus(self) -> int:
        return max(1, _cpu_count() - 2)


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
    concatenated = BatchEncoding(data={k: np.array(list(chain(*v))) for k, v in encoding.items()})
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


def _stack_batch_encodings(a: BatchEncoding, b: BatchEncoding) -> BatchEncoding:
    """Stacks two batch encodings together, assuming that the keys are the same."""

    def _ensure_batched(x):
        if len(x) == 0:
            return list(x)
        elif isinstance(x[0], Sequence) or isinstance(x[0], np.ndarray):
            return list(x)
        else:
            return [x]

    return BatchEncoding({k: _ensure_batched(a[k]) + _ensure_batched(b[k]) for k in a.keys()})


@dataclass
class LMDatasetConfig:
    """This class supports loading data both from HF Datasets and from a raw dataset of jsonl urls"""

    id: Optional[str] = None  # id (or path) for hf dataset
    name: Optional[str] = None  # name for hf dataset
    stream: bool = True  # whether to use streaming when doing hf

    train_urls: List[str] = ()  # type: ignore
    validation_urls: List[str] = ()  # type:ignore

    # config for the tokenizer
    tokenizer: str = "gpt2"
    text_key: str = "text"  # key for the text field in the jsonl file or hf dataset

    # config related to caching
    cache_dir: str = "cache/"

    enforce_eos: bool = True  # whether to append eos even if the tokenizer doesn't
    splits: List[str] = field(default_factory=lambda: ["train", "validation"])
    rows_per_chunk: int = DEFAULT_ROWS_PER_CHUNK  # number of rows to process and cache per chunk

    @cached_property
    def the_tokenizer(self) -> PreTrainedTokenizerFast:
        return load_tokenizer(self.tokenizer)

    def build_or_load_cache(self, split: str, monitors: Union[bool, List[MetricsMonitor]] = True):
        source = self.get_shard_source(split)
        split_cache_dir = os.path.join(self.cache_dir, split)
        try:
            return TokenizedDocumentCache.load(split_cache_dir, flatten_docs=True)
        except FileNotFoundError:
            logger.info(f"Building cache for {split}...")

        if monitors is True:
            monitors = [
                WandbMetricsMonitor(prefix=f"preprocessing/{split}", commit=False),
                LoggerMetricsMonitor(f"preprocessing.{split}"),
            ]
        elif monitors is False:
            monitors = []

        return TokenizedDocumentCache.build_or_load(
            split_cache_dir,
            source,
            self.the_tokenizer,
            enforce_eos=self.enforce_eos,
            flatten_docs=True,
            rows_per_chunk=self.rows_per_chunk,
            monitors=monitors,
            # TODO: it would be better if we could just prioritize validation higher (we typically want it after the first grad step)
            await_finished=(split == "validation"),
        )

    def doc_iterator(self, split: str):
        if self.id is not None:
            dataset = datasets.load_dataset(self.id, name=self.name, streaming=self.stream)
            data = dataset[split]
            for doc in data:
                yield doc[self.text_key]
        else:
            urls = self.urls_for_split(split)
            yield from self.generate_texts_from_urls(urls)

    def generate_texts_from_urls(self, urls: Sequence[str], skip_to_doc: int = 0) -> Iterator[str]:
        files = fsspec.open_files(urls, "r", compression="infer")
        row = 0
        for file in files:
            with file as f:
                # TODO: would be nice if we could seek faster than this. Right now, all we do is skip json parsing
                # which is not nothing, but not ideal.
                for line in f.readlines():
                    if row >= skip_to_doc:
                        yield json.loads(line)[self.text_key]
                    row += 1

    def urls_for_split(self, split):
        if split == "train":
            urls = self.train_urls
        elif split == "validation":
            urls = self.validation_urls
        else:
            raise ValueError(f"Unknown split {split}")

        def fsspec_expand_glob(url):
            if "*" in url:
                fs = fsspec.core.url_to_fs(url)[0]
                return fs.glob(url)
            else:
                return [url]

        urls = [globbed for pat in urls for url in braceexpand.braceexpand(pat) for globbed in fsspec_expand_glob(url)]
        return urls

    def get_shard_source(self, split) -> ShardedDataSource[str]:
        if self.id is not None:
            return HFDatasetDataSource(self, split)
        return TextDataSource(self, split)


class HFDatasetDataSource(ShardedDataSource[str]):
    """
    This class is responsible for loading a dataset from HuggingFace Datasets and returning the shards.
    Only (some) IterableDatasets are actually sharded in any meaningful way, so we just return a single shard
    for all other datasets.
    """

    def __init__(self, config: LMDatasetConfig, split: str):
        self.config = config
        self.split = split

        self._shard_names = self._compute_shard_names()

    @property
    def shard_names(self) -> Sequence[str]:
        return self._shard_names

    def _compute_shard_names(self):
        dataset = self._load_dataset()
        if isinstance(dataset, datasets.IterableDataset):
            try:
                return [str(i) for i in range(dataset.n_shards)]
            except NotImplementedError:
                return ["data"]
        else:
            return ["data"]

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[str]:
        dataset = self._load_dataset()
        if isinstance(dataset, datasets.IterableDataset) and shard_name != "data":
            shard = dataset._ex_iterable.shard_data_sources([int(shard_name)])
        else:
            shard = dataset

        idx = 0
        for _, doc in shard:
            if idx >= row:
                yield doc[self.config.text_key]
            idx += 1

    def _load_dataset(self):
        # obnoxiously, the dataset loading stuff doesn't work with ray because of multiprocessing and stuff
        # so we have to do this hacky thing where we load the dataset in the worker
        return datasets.load_dataset(
            self.config.id, split=self.split, name=self.config.name, streaming=self.config.stream
        )


class TextDataSource(ShardedDataSource[str]):
    def __init__(self, config: LMDatasetConfig, split: str):
        self.config = config
        self.split = split

        self._shard_name_to_url_mapping = {}

        urls = config.urls_for_split(split)

        # remove common prefix
        if len(urls) == 1:
            common_prefix = os.path.dirname(urls[0])
        else:
            common_prefix = os.path.commonprefix(urls)

        for url in urls:
            # escape the url for the shard name
            shard_name = url
            if common_prefix:
                shard_name = url[len(common_prefix) :]
                if shard_name.startswith("/"):
                    shard_name = shard_name[1:]

            shard_name = shard_name.replace(".", "_")

            self._shard_name_to_url_mapping[shard_name] = url

    @property
    def shard_names(self) -> Sequence[str]:
        return list(self._shard_name_to_url_mapping.keys())

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[str]:
        url = self._shard_name_to_url_mapping[shard_name]
        return self.config.generate_texts_from_urls([url], row)
