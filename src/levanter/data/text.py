import copy
import functools
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
import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from chex import PRNGKey
from draccus import field
from pyarrow._parquet import FileMetaData

import haliax as hax
from haliax import Axis

# intercept the logging nonsense here
from levanter.logging import silence_transformer_nag  # noqa
from levanter.models.attention import AttentionMask
from levanter.models.lm_model import LmExample
from levanter.utils.hf_utils import num_cpus_used_by_tokenizer


silence_transformer_nag()  # noqa
from transformers import BatchEncoding, PreTrainedTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast  # noqa

from levanter.compat.hf_checkpoints import load_tokenizer  # noqa
from levanter.data._preprocessor import BatchProcessor, dict_from_record_batch  # noqa
from levanter.data.dataset import ShardableDataset  # noqa
from levanter.data.shard_cache import DEFAULT_ROWS_PER_CHUNK  # noqa
from levanter.data.shard_cache import CacheLedger  # noqa
from levanter.data.shard_cache import LEDGER_FILE_NAME as NEW_LEDGER_FILE_NAME  # noqa
from levanter.data.shard_cache import (  # noqa
    ChunkMetadata,
    LoggerMetricsMonitor,
    MetricsMonitor,
    ShardCache,
    WandbMetricsMonitor,
    _serialize_json_and_commit,
    build_cache,
)
from levanter.data.sharded_dataset import ShardedDataset, TextUrlDataset, WrappedHFDataset  # noqa
from levanter.shapes import NamedShapeSpec, ShapeSpec  # noqa
from levanter.utils.jax_utils import use_cpu_device  # noqa


logger = logging.getLogger("levanter.data.text")

# TASKS:
# TODO: figure out directory structure for caching multiple sources
# TODO: consider adding indexing a la Map-style datasets
# TODO: support seeking/serialization/restore in the dataset

LEDGER_FILE = "ledger.json"


class CausalLmDataset(ShardableDataset[LmExample]):
    def __init__(
        self,
        dataset: ShardableDataset[np.ndarray],
        QPos: Axis,
        KPos: Axis,
        fcm_prob: float = 0.0,
        key: Optional[PRNGKey] = None,
    ):
        self.dataset = dataset
        self.QPos = QPos
        self.KPos = KPos
        self.fcm_prob = fcm_prob
        self.key = key

        if self.fcm_prob > 0.0 and self.key is None:
            raise ValueError("must provide key if fcm_prob > 0.0")

    def shard(self, shard_id: int, num_shards: int) -> "CausalLmDataset":
        return CausalLmDataset(self.dataset.shard(shard_id, num_shards), self.QPos, self.KPos, self.fcm_prob, self.key)

    def __iter__(self) -> Iterator[LmExample]:
        key = self.key
        for tokens in self.dataset:
            with use_cpu_device():
                example = self._create_lm_example(tokens, key)
                yield example

    @functools.partial(jax.jit, static_argnums=(0))
    def _create_lm_example(self, tokens, key):
        attn_mask = AttentionMask.causal()
        if self.fcm_prob > 0:
            # masks for attention
            # We support forgetful causal masking (FCM) which is a technique that improves training speed by
            # randomly masking out some of the context. This is a bit like dropout, but it's applied to the attention
            # mask instead of the activations. It's described in https://arxiv.org/abs/2210.13432
            assert self.key is not None
            this_key, key = jax.random.split(key)
            fcm_mask = hax.nn.attention.forgetful_causal_mask(self.KPos, self.fcm_prob, key=this_key)
            attn_mask = attn_mask & AttentionMask.explicit(fcm_mask)

        tokens = hax.named(tokens, self.QPos)

        loss_mask = 1 - hax.nn.one_hot(-1, self.QPos, dtype=jnp.float32)

        example = LmExample(tokens=tokens, attn_mask=attn_mask, loss_mask=loss_mask)
        return example


class TokenSeqDataset(ShardableDataset[np.ndarray]):
    """
    A dataset that yields sequences of tokens of fixed length from a TokenizedDocumentCache.

    :param doc_cache: the TokenizedDocumentCache to draw from
    :param Pos: the axis to use for the sequences. Sequences will be a NamedArray with axis Pos
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

    def __iter__(self) -> Iterator[np.ndarray]:
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
                    # yield hax.named(ids, self.Pos)
                    yield ids

    @staticmethod
    def load(seq_len: int, cache_dir: str, stride: Optional[int] = None) -> "TokenSeqDataset":
        doc_cache = TokenizedDocumentCache.load(cache_dir, True)
        return TokenSeqDataset(doc_cache, seq_len, stride)


class BatchEncodingDataset(ShardableDataset[BatchEncoding]):
    """
    A Dataset that yields HF BatchEncodings from a ShardCache.
    This basically yields a dict-of-arrays, just the HF BatchEncoding class version of dict.
    """

    def __init__(self, cache: ShardCache, return_batches: bool = False):
        self.cache = cache
        self.return_batches = return_batches

    def __iter__(self) -> Iterator[BatchEncoding]:
        for batch in self.cache:
            encoding = _batch_encoding_from_record_batch(batch, flatten_docs=False)
            if self.return_batches:
                yield encoding
            else:
                batch_size = 0
                for v in encoding.values():
                    batch_size = len(v)
                    break

                for i in range(batch_size):
                    # this doesn't work for reconstituted batches, so we have to do this
                    # I have no idea why this is the case
                    #     yield encoding[i]
                    yield BatchEncoding({k: v[i] for k, v in encoding.items()})

    def shard(self, shard_id: int, num_shards: int) -> "BatchEncodingDataset":
        return BatchEncodingDataset(self.cache.shard(shard_id, num_shards))

    @staticmethod
    def load(cache_dir: str, return_batches: bool = False, batch_size: Optional[int] = None) -> "BatchEncodingDataset":
        if batch_size is None:
            batch_size = 1
        cache = ShardCache.load(cache_dir, batch_size=batch_size)
        return BatchEncodingDataset(cache, return_batches=return_batches)


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

    def __init__(self, chunk_cache: ShardCache, flatten_docs):
        self.chunk_cache = chunk_cache
        self.flatten_docs = flatten_docs

    def __iter__(self):
        """Reads the cache files produced by cache_and_group and yields tokenized sequences.
        If flatten is false, this returns the docs as they were presented to the caching process. If flatten is True,
        then the documents returned are actually concatenated documents, where the number is the number of documents
        presented as a batch to the caching process."""
        for batch in self._chunks():
            yield _batch_encoding_from_record_batch(batch, self.flatten_docs)

    def _chunks(self):
        return self.chunk_cache.iter_batches_from_chunks()

    @staticmethod
    def build_or_load(
        cache_dir,
        source: ShardedDataset[str],
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
        cache = build_cache(
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

        return TokenizedDocumentCache(self.chunk_cache.shard(shard_index, num_shards), self.flatten_docs)


def _open_arrow_table(path) -> FileMetaData:
    fs, _, paths = fsspec.get_fs_token_paths(path)
    return pq.read_metadata(path, filesystem=fs)


def _batch_encoding_from_record_batch(b: pa.RecordBatch, flatten_docs: bool):
    if flatten_docs:
        # insert a newaxis to the beginning so that it appears to be bs=1
        return BatchEncoding(
            {
                b.field(i).name: b.column(i).values.to_numpy(zero_copy_only=False)[np.newaxis, :]
                for i in range(b.num_columns)
            },
        )
    else:
        return BatchEncoding(dict_from_record_batch(b))


def _maybe_force_tokenizer_parallelism(tokenizer: PreTrainedTokenizerBase):
    if tokenizer.is_fast and os.getenv("TOKENIZERS_PARALLELISM") is None:
        # if we're using a fast tokenizer, we want to force parallelism
        # to be the number of CPUs
        os.environ["TOKENIZERS_PARALLELISM"] = "true"


class BatchTokenizer(BatchProcessor[str]):
    """
    A batch processor that tokenizes a batch of strings using a tokenizer.
    By default, this will append eos to the end of the string, even if the tokenizer doesn't.
    """

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

    def __call__(self, batch: Sequence[str]) -> BatchEncoding:
        if self._need_to_add_eos:
            encoding = self.tokenizer(
                [d + " " + self.tokenizer.eos_token for d in batch], return_attention_mask=False, verbose=False
            )
        else:
            encoding = self.tokenizer(batch, return_attention_mask=False, verbose=False)  # type: ignore
        return encoding

    @property
    def num_cpus(self) -> int:
        return num_cpus_used_by_tokenizer(self.tokenizer)


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

    def token_seq_dataset(self, split: str, seq_len: int, monitors: Union[bool, List[MetricsMonitor]] = True):
        cache = self.build_or_load_cache(split, monitors=monitors)
        return TokenSeqDataset(cache, seq_len)

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

            yield from TextUrlDataset(urls, self.text_key)

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

    def get_shard_source(self, split) -> ShardedDataset[str]:
        if self.id is not None:
            return WrappedHFDataset(self.id, split=split, name=self.name, streaming=self.stream).map(
                lambda x: x[self.text_key]
            )
        else:
            return TextUrlDataset(self.urls_for_split(split), self.text_key)
