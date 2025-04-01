import abc
import asyncio
import copy
import dataclasses
import functools
import json
import logging
import os
import warnings
from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeAlias,
    TypeVar,
    Union,
)

import datasets
import equinox as eqx
import fsspec
import jax
import numpy as np
import regex
import tensorstore as ts
from draccus import field
from jaxtyping import PRNGKeyArray
from tokenizers import normalizers

import haliax as hax
from haliax import Axis

from levanter.data import AsyncDataset
from levanter.data.dataset import MappedAsyncDataset
from levanter.data.mixture import MixtureDataset, StopStrategy, rescale_mixture_schedule_for_batch_schedule
from levanter.models.lm_model import LmExample
from levanter.schedule import BatchSchedule
from levanter.store.cache import CacheOptions, TreeCache
from levanter.store.jagged_array import JaggedArrayStore
from levanter.store.tree_store import TreeStore
from levanter.utils.fsspec_utils import expand_glob
from levanter.utils.hf_utils import HfTokenizer, num_cpus_used_by_tokenizer

# intercept the logging nonsense here
from levanter.utils.logging import silence_transformer_nag  # noqa


silence_transformer_nag()  # noqa
from transformers import BatchEncoding, PreTrainedTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast  # noqa

from levanter.compat.hf_checkpoints import load_tokenizer  # noqa
from levanter.data._preprocessor import BatchProcessor, U, dict_from_record_batch  # noqa
from levanter.data.metrics_monitor import LoggerMetricsMonitor, LoggingMetricsMonitor, MetricsMonitor  # noqa
from levanter.data.sharded_datasource import (  # noqa
    JsonlDataSource,
    ShardedDataSource,
    TextUrlDataSource,
    UrlDataSource,
    WrappedHFDataSource,
    gcs_glob,
)
from levanter.shapes import NamedShapeSpec, ShapeSpec  # noqa
from levanter.store.cache import build_or_load_cache  # noqa
from levanter.utils.jax_utils import key_iterator, use_cpu_device  # noqa


T_co = TypeVar("T_co", covariant=True)

logger = logging.getLogger("levanter.data.text")

# TASKS:
# TODO: consider adding indexing a la Map-style datasets
# TODO: support seeking/serialization/restore in the dataset

LEDGER_FILE = "ledger.json"

DEFAULT_IGNORE_INDEX = -100  # Mirrors pytorch's default ignore index


class EpochDataset(AsyncDataset[T_co]):
    """
    A dataset that wraps another dataset, providing infinite epochs by recycling indices.
    If `max_epochs` is specified, it limits the number of cycles before raising StopIteration.

    :param dataset: The dataset to wrap.
    :param max_epochs: The maximum number of epochs to cycle through. If None, cycle indefinitely.
    """

    def __init__(self, dataset: AsyncDataset[T_co], max_epochs: Optional[int] = None):
        super().__init__()
        self.dataset = dataset
        self.max_epochs = max_epochs

    async def async_len(self) -> int:
        if self.max_epochs is None:
            raise ValueError("Cannot determine length of an infinite dataset without max_epochs.")
        # Return the total number of samples: max_epochs * length of the dataset
        return self.max_epochs * await self.dataset.async_len()

    async def final_length_is_known(self) -> bool:
        return await self.dataset.final_length_is_known()

    def is_finite(self) -> bool:
        # EpochDataset can be finite if max_epochs is set.
        return self.max_epochs is not None

    async def current_len(self) -> Optional[int]:
        # If max_epochs is None, the dataset is effectively infinite.
        if self.max_epochs is None:
            return None

        # If the final length of the dataset is not known, return the current length of the underlying dataset.
        if not await self.dataset.final_length_is_known():
            return await self.dataset.current_len()

        # If the final length is known, return the max_epochs * async_len of the dataset.
        return self.max_epochs * await self.dataset.async_len()

    async def get_batch(self, indices: Sequence[int]) -> Sequence[T_co]:
        # Use self.wait_until_len_at_least to ensure we have enough data for the batch.
        max_index = max(indices)
        ds_len = await self.dataset.wait_until_len_at_least(max_index + 1)

        # Determine the epoch based on the largest index
        epoch = max_index // ds_len

        # If max_epochs is specified, raise an error if the epoch exceeds the allowed number of epochs
        if self.max_epochs is not None and epoch >= self.max_epochs:
            raise StopIteration(
                f"Reached maximum number of epochs: epoch {epoch} exceeds the maximum allowed {self.max_epochs}"
            )

        # Wrap the indices within the bounds of the dataset length
        wrapped_indices = [idx % ds_len for idx in indices]

        # Delegate to the underlying dataset's get_batch
        return await self.dataset.get_batch(wrapped_indices)

    async def wait_until_len_at_least(self, length: int) -> int:
        """
        Returns the length of the dataset once it is at least `length` or if the dataset has a known (finished) length.
        If the dataset's actual length is less than `length`, it returns the minimum of async_len and the current length.
        """
        # Wait until the underlying dataset's length is at least `length`
        if not self.is_finite():
            return length

        if await self.dataset.final_length_is_known():
            base_length = await self.dataset.async_len()
        else:
            base_length = await self.dataset.wait_until_len_at_least(length)

        if base_length < length:
            # hit epoch boundary
            assert self.max_epochs is not None
            return self.max_epochs * base_length

        return base_length


class TokenSeqDataset(AsyncDataset[np.ndarray]):
    """
    A dataset that yields sequences of tokens of fixed length from an underlying TreeCache.

    :param doc_cache: the TreeCache to read from
    :param seq_len: The max length of sequences to emit
    """

    def __init__(self, doc_cache: TreeCache[dict], seq_len: int):
        super().__init__()
        self.doc_cache = doc_cache
        self.seq_len = seq_len
        self._store: Optional[TreeStore] = None
        self._cached_len: Optional[int] = None

    async def async_len(self) -> int:
        await self.doc_cache.finished()
        token_arrays = await self._await_token_cache()
        return token_arrays.data_size // self.seq_len

    async def _await_token_cache(self) -> JaggedArrayStore:
        if self._store is None:
            self._store = await self.doc_cache.store_async()
        return self._store.tree["input_ids"]

    async def final_length_is_known(self) -> bool:
        return await self.doc_cache.final_length_is_known()

    def is_finite(self) -> bool:
        return True

    async def current_len(self) -> Optional[int]:
        store = await self._await_token_cache()
        return store.data_size // self.seq_len

    async def get_batch(self, indices: Sequence[int]) -> Sequence[T_co]:
        token_arrays = await self._await_token_cache()
        # logger.info(f"Time to get token cache: {time.time() - time_in}")
        len = await self.wait_until_len_at_least(max(indices) + 1)
        if len is not None and len < max(indices) + 1:
            raise ValueError("Requested indices beyond the end of the dataset")
        offsets = np.array(indices, dtype=np.int64) * self.seq_len
        with ts.Batch():
            out = []
            for offset in offsets:
                out.append(token_arrays.data[offset : offset + self.seq_len].read())

        out = await asyncio.gather(*out)
        return out

    async def wait_until_len_at_least(self, length: int) -> int:
        # length is brutally slow to compute, so we cache it
        if self._cached_len is not None and self._cached_len >= length:
            return self._cached_len

        # TODO: would be better to listen for cache updates
        length = await super().wait_until_len_at_least(length)
        self._cached_len = length
        return length


class CausalLmDataset(MappedAsyncDataset[np.ndarray, LmExample]):
    def __init__(
        self,
        dataset: AsyncDataset[np.ndarray],
        Pos: Axis,
        *,
        ignore_index: Optional[int] = None,
        eos_id: Optional[int] = None,
    ):
        self.dataset = dataset
        self.Pos = Pos
        self.ignore_id = ignore_index
        self.eos_id = eos_id

        sharding = jax.sharding.SingleDeviceSharding(jax.local_devices(backend="cpu")[0])

        @functools.partial(eqx.filter_jit, out_shardings=sharding)
        def _create_lm_example(tokens):
            tokens = hax.named(tokens, self.Pos)
            example = LmExample.causal(tokens=tokens, ignore_id=self.ignore_id, eos_id=eos_id)

            return example

        super().__init__(self.dataset, _create_lm_example)

    async def async_len(self) -> int:
        return await self.dataset.async_len()


def _maybe_force_tokenizer_parallelism(tokenizer: PreTrainedTokenizerBase):
    if tokenizer.is_fast and os.getenv("TOKENIZERS_PARALLELISM") is None:
        # if we're using a fast tokenizer, we want to force parallelism
        # to be the number of CPUs
        os.environ["TOKENIZERS_PARALLELISM"] = "true"


LONG_STRING_WORKAROUND = 10_000

ws = regex.compile(r"\s")


class BatchTokenizer(BatchProcessor[str, dict]):
    """
    A batch processor that tokenizes a batch of strings using a tokenizer.
    By default, this will append eos to the end of the string, even if the tokenizer doesn't.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        enforce_bos=True,
        enforce_eos=True,
        *,
        override_resources=None,
        _workaround_len=LONG_STRING_WORKAROUND,
        return_attention_mask=False,
        padding=False,
        max_length=None,
    ):
        _maybe_force_tokenizer_parallelism(tokenizer)
        self.tokenizer = tokenizer
        self.override_resources = override_resources
        self.return_attention_mask = return_attention_mask
        self.padding = padding
        if max_length is not None:
            self.max_length = max_length
        else:
            self.max_length = self.tokenizer.model_max_length

        # see if the tokenizer appends bos/eos
        # if we don't have an eos/bos token in the tokenizer, skip
        if tokenizer.bos_token_id is None:
            enforce_bos = False
        if tokenizer.eos_token_id is None:
            enforce_eos = False

        # HF's BPE-based tokenizers do not, but the bert and roberta ones do
        # TODO: this doesn't necessarily ensure it, I guess, but eh
        if enforce_eos or enforce_bos:
            input_ids = tokenizer("hi there")["input_ids"]
            should_append_eos = input_ids[-1] != tokenizer.eos_token_id and enforce_eos
            should_append_bos = input_ids[0] != tokenizer.bos_token_id and enforce_bos
        else:
            should_append_eos = False
            should_append_bos = False

        self._need_to_add_eos = should_append_eos
        self._need_to_add_bos = should_append_bos
        self._workaround_len = _workaround_len

    def __call__(self, batch: Sequence[str]) -> list[dict]:
        if self._need_to_add_bos:
            batch = [self.tokenizer.bos_token + " " + d for d in batch]

        if self._need_to_add_eos:
            batch = [d + " " + self.tokenizer.eos_token for d in batch]

        if self._needs_long_sequence_workaround:
            batch, needs_merge = self._break_for_long_sequences(batch)
        else:
            needs_merge = []

        if self.padding is not False:
            encoding = self.tokenizer(
                batch,
                return_attention_mask=self.return_attention_mask,
                verbose=False,
                padding=self.padding,
                max_length=self.max_length,
                truncation=True,
            )  # type: ignore
        else:
            encoding = self.tokenizer(
                batch, return_attention_mask=self.return_attention_mask, verbose=False
            )  # type: ignore

        if needs_merge:
            new_encoding = self._merge_split_encodings(batch, encoding, needs_merge)
            encoding = BatchEncoding(new_encoding)

        # debatch the encoding
        unbatched = [dict(zip(encoding, t)) for t in zip(*[encoding[k] for k in encoding])]

        return unbatched

    def _break_for_long_sequences(self, batch):
        orig_lengths = [len(d) for d in batch]
        # break any strings that are longer than LONG_STRING_WORKAROUND characters into smaller chunks
        orig_batch = batch
        batch = []
        needs_merge = []
        for i, d in enumerate(orig_batch):
            needs_merge.append(False)
            orig_len = orig_lengths[i]
            while len(d) > self._workaround_len:
                # we'd rather break strings at whitespace, so find the first whitespace
                match = ws.search(d, self._workaround_len)
                # this is vanishingly unlikely, but if we can't find a whitespace, just break it at the limit
                if match is None:
                    split = len(d)
                else:
                    split = match.start()

                batch.append(d[:split])
                needs_merge.append(True)

                d = d[split:]
                orig_len -= split

            batch.append(d)
        return batch, needs_merge

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "tokenizer": self.tokenizer.name_or_path,
            "vocab_size": len(self.tokenizer),
            "return_attention_mask": self.return_attention_mask,
            "padding": self.padding,
            "max_length": self.max_length,
            "append_bos": self._need_to_add_bos,
            "append_eos": self._need_to_add_eos,
        }

    @property
    def output_exemplar(self) -> dict:
        return dict(**self.tokenizer("hi there", return_attention_mask=self.return_attention_mask, verbose=False))

    @property
    def name_or_path(self):
        return self.tokenizer.name_or_path

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @staticmethod
    def _merge_split_encodings(batch, encoding, needs_merge):
        # merge the encodings back together
        # we might need to merge multiple encodings together
        # needs merge marks the first n-1 encodings that need to be merged for each document
        new_encoding = {}
        for k, v in encoding.items():
            if len(v) == 0:
                continue
            if isinstance(v[0], np.ndarray):
                assert len(v) == len(batch)
                v_out = []
                vs_to_merge = []
                for i in range(len(batch)):
                    if not needs_merge[i]:
                        v_out.append(np.concatenate(vs_to_merge))
                        vs_to_merge = []
                    vs_to_merge.append(v[i])

                if len(vs_to_merge) > 0:
                    v_out.append(np.concatenate(vs_to_merge))

                new_encoding[k] = v_out
            elif isinstance(v[0], list):
                v_out = []
                vs_to_merge = []
                for i in range(len(batch)):
                    if not needs_merge[i]:
                        if len(vs_to_merge) > 0:
                            v_out.append(list(chain(*vs_to_merge)))
                        vs_to_merge = []
                    vs_to_merge.append(v[i])

                if len(vs_to_merge) > 0:
                    v_out.append(list(chain(*vs_to_merge)))
                new_encoding[k] = v_out
            else:
                raise ValueError(f"Unknown type {type(v[0])}")
        return new_encoding

    # TODO remove this when it's resolved https://github.com/huggingface/tokenizers/issues/1495
    @cached_property
    def _needs_long_sequence_workaround(self):
        if isinstance(self.tokenizer, PreTrainedTokenizerFast):
            normalizer = self.tokenizer.backend_tokenizer.normalizer
            if normalizer is None:
                return False
            # if there's a "Replace" normalizer, then we need to do the workaround
            # inexplicably there's no way to see inside a Sequence so we also have to assume it needs it
            return isinstance(normalizer, (normalizers.Replace, normalizers.Sequence))
        else:
            return False

    @property
    def num_cpus(self) -> int:
        if self.override_resources is not None:
            cpus = self.override_resources.get("num_cpus", None)
            if cpus is not None:
                return cpus
        return num_cpus_used_by_tokenizer(self.tokenizer)

    @property
    def num_gpus(self) -> int:
        if self.override_resources is not None:
            return self.override_resources.get("num_gpus", 0)
        return 0


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
class LMDatasetSourceConfig:
    """This class represents a dataset source with URLs or hf name/id."""

    tags: Optional[List[str]] = None
    """tags for the dataset. Typically the name of the dataset in the config will be added as a tag as well"""

    id: Optional[str] = None  # id (or path) for hf dataset
    name: Optional[str] = None  # name for hf dataset

    plaintext: bool = False
    stream: bool = True  # whether to use streaming when doing hf
    text_key: str = "text"  # key for the text field in the jsonl file or hf dataset

    train_urls: List[str] = ()  # type: ignore
    validation_urls: List[str] = ()  # type:ignore
    cache_dir: Optional[str] = None  # Optionally override the cache dir for this component

    def get_shard_source(self, split) -> Optional[ShardedDataSource[str]]:
        if self.id is not None:
            try:
                ds = WrappedHFDataSource(self.id, split=split, name=self.name, streaming=self.stream)
            except ValueError as e:
                # if the message starts with Bad split, then just return None
                if str(e).startswith("Bad split"):
                    logger.warning(f"Splits {split} not found for {self.id} {self.name}")
                    return None
                else:
                    raise

            if len(ds.shard_names) == 0:
                return None

            return ds.map(lambda x: x[self.text_key])
        else:
            split_urls = self.urls_for_split(split)
            if len(split_urls) == 0:
                return None
            return TextUrlDataSource(split_urls, self.text_key)

    def doc_iterator(self, split: str):
        if self.id is not None:
            dataset = datasets.load_dataset(self.id, name=self.name, streaming=self.stream)
            data = dataset[split]
            for doc in data:
                yield doc[self.text_key]
        else:
            urls = self.urls_for_split(split)

            yield from TextUrlDataSource(urls, self.text_key)

    def urls_for_split(self, split):
        if split == "train":
            urls = self.train_urls
        elif split == "validation":
            urls = self.validation_urls
        else:
            raise ValueError(f"Unknown split {split}")

        urls = [globbed for url in urls for globbed in expand_glob(url)]
        return urls


@dataclass
class LMTaskConfig(abc.ABC):
    tokenizer: str = "gpt2"
    vocab_size: Optional[int] = None  # if using the passthrough tokenizer, this is required

    # config related to caching
    cache_dir: Optional[str] = "cache/"
    cache_options: CacheOptions = field(default_factory=CacheOptions)
    enforce_eos: bool = True  # whether to append eos even if the tokenizer doesn't

    ignore_token_id: Optional[int] = DEFAULT_IGNORE_INDEX
    shuffle: bool | int = False
    """whether to shuffle the dataset. True means shuffle the whole dataset, False means don't shuffle.
    If you want to shuffle in eras, set this to the era length"""
    permutation_type: Literal["feistel", "linear"] | None = None
    """
    Type of permutation to use for shuffle.

    If None, defaults to linear, but this will change in the future since Feistel is better.
    """

    @cached_property
    def the_tokenizer(self) -> HfTokenizer:
        if self.tokenizer == "passthrough":
            return PassthroughTokenizer(self.vocab_size)
        else:
            return load_tokenizer(self.tokenizer)

    @abc.abstractmethod
    def train_set(
        self,
        Pos: Axis,
        batch_schedule: BatchSchedule,
        monitors: Union[bool, List[MetricsMonitor]] = True,
        *,
        key: PRNGKeyArray,
        epochs: Optional[int] = None,
    ) -> AsyncDataset[LmExample]:
        pass

    @abc.abstractmethod
    def train_sets(
        self,
        Pos: Axis,
        monitors: Union[bool, List[MetricsMonitor]] = True,
        *,
        key: PRNGKeyArray,
        epochs: Optional[int] = None,
    ) -> Mapping[str, AsyncDataset[LmExample]]:
        pass

    @abc.abstractmethod
    def validation_sets(
        self,
        Pos: Axis,
        monitors: Union[bool, List[MetricsMonitor]] = True,
    ) -> Mapping[str, AsyncDataset[LmExample]]:
        pass

    @property
    @abc.abstractmethod
    def sources(self) -> Mapping[str, LMDatasetSourceConfig]:
        pass

    def tagged_eval_sets(
        self, Pos: Axis, monitors: Union[bool, List[MetricsMonitor]] = True
    ) -> list[Tuple[AsyncDataset[LmExample], List[str]]]:
        tags = {name: (config.tags or []) + [name] for name, config in self.sources.items()}
        eval_sets = self.validation_sets(Pos, monitors)

        return [(eval_sets[name], tags[name]) for name in eval_sets]


CANONICAL_INPUT_FIELD = "prompt"
CANONICAL_OUTPUT_FIELD = "response"


class SupervisedSourceConfigBase(Protocol):
    def get_shard_source(self, split: str) -> Optional[ShardedDataSource[dict]]:
        raise NotImplementedError

    input_field: str
    output_field: str
    tags: Optional[List[str]]
    cache_dir: str


@dataclass
class LMSupervisedDatasetConfig(SupervisedSourceConfigBase):
    """Config for supervised fine-tuning datasets"""

    cache_dir: str = "cache/"

    # HF dataset config
    hf_dataset_name: Optional[str] = None  # e.g. "tatsu-lab/alpaca" or "OpenAssistant/oasst1"
    hf_dataset_split: str = "train"  # which split to use

    # Local files config
    validation_urls: List[str] = field(default_factory=list)  # paths to jsonl/json files

    # Field names in the data
    input_field: str = CANONICAL_INPUT_FIELD  # name of the input field
    output_field: str = CANONICAL_OUTPUT_FIELD  # name of output field

    # Optional metadata
    tags: Optional[List[str]] = None

    def __post_init__(self):
        warnings.warn(
            "LMSupervisedDatasetConfig is deprecated. Use SupervisedHfSourceConfig or "
            "SupervisedUrlSourceConfig instead.",
            DeprecationWarning,
        )

    def get_shard_source(self, split: str) -> Optional[ShardedDataSource[dict]]:
        if self.hf_dataset_name is not None:
            return WrappedHFDataSource(self.hf_dataset_name, split=self.hf_dataset_split)
        elif split != "validation":
            raise ValueError("Only validation split is supported for local files")
        else:
            urls = [globbed for url in self.validation_urls for globbed in expand_glob(url)]
            return JsonlDataSource(urls)


@dataclass(frozen=True)
class SupervisedHfSourceConfig(SupervisedSourceConfigBase):
    cache_dir: str
    id: str
    name: str | None = None

    streaming: bool = True

    input_field: str = CANONICAL_INPUT_FIELD
    output_field: str = CANONICAL_OUTPUT_FIELD
    tags: Optional[List[str]] = None

    def get_shard_source(self, split: str) -> Optional[ShardedDataSource[dict]]:
        return WrappedHFDataSource(self.id, split=split, name=self.name, streaming=self.streaming).map(
            lambda x: {CANONICAL_INPUT_FIELD: x[self.input_field], CANONICAL_OUTPUT_FIELD: x[self.output_field]}
        )


@dataclass(frozen=True)
class SupervisedUrlSourceConfig(SupervisedSourceConfigBase):
    cache_dir: str
    train_urls: list[str] = dataclasses.field(default_factory=list)
    validation_urls: list[str] = dataclasses.field(default_factory=list)

    input_field: str = CANONICAL_INPUT_FIELD
    output_field: str = CANONICAL_OUTPUT_FIELD
    tags: Optional[List[str]] = None

    def get_shard_source(self, split: str) -> Optional[ShardedDataSource[dict]]:
        urls = self.train_urls if split == "train" else self.validation_urls
        if not urls:
            return None

        urls = [globbed for url in urls for globbed in expand_glob(url)]

        source = UrlDataSource(urls, columns=[self.input_field, self.output_field])
        return source.map(
            lambda x: {CANONICAL_INPUT_FIELD: x[self.input_field], CANONICAL_OUTPUT_FIELD: x[self.output_field]}
        )


SupervisedSourceConfig: TypeAlias = Union[SupervisedHfSourceConfig, SupervisedUrlSourceConfig]


def _preprocess_supervised_example(
    batch, tokenizer: PreTrainedTokenizerBase, input_field: str, output_field: str
) -> dict:
    sources = [example[input_field] for example in batch]

    targets = [example[output_field] for example in batch]
    # TODO: this seems pretty wasteful since you end up tokenizing twice, but it's how alpaca does it.
    examples = [s + t for s, t in zip(sources, targets)]
    sources_tokenized = tokenizer(sources, padding=False, truncation=True)
    examples_tokenized = tokenizer(examples, padding=False, truncation=True)

    source_lens = [len(s) for s in sources_tokenized["input_ids"]]

    return {
        "input_ids": [np.array(example, dtype=np.int32) for example in examples_tokenized["input_ids"]],
        "sources_len": np.array(source_lens, dtype=np.int32),
    }


def _prepare_supervised_examples(ex: list[dict], tokenizer: PreTrainedTokenizerBase, Pos: hax.Axis) -> list[LmExample]:
    """
    Prepare examples for training. This function converts the (cached) encodings into an LmExample.

    It goes through the following steps:

    1. Pad the batch to the maximum length.
    2. Mask out the input and prompt if requested.
    3. Create an LmExample with the input_ids as the input and the next token as the target.
    """
    lens = np.array([ex["sources_len"] for ex in ex])

    ex_pad = tokenizer.pad(
        ex,
        padding="max_length",
        max_length=Pos.size,
    )

    input_ids = ex_pad["input_ids"]
    truncated = [ids[-Pos.size :] for ids in input_ids]

    out = []
    for ids, len in zip(truncated, lens):
        causal = _mk_sup_example_jit(Pos, hax.named(ids, Pos), len, tokenizer.pad_token_id, tokenizer.eos_token_id)

        out.append(causal)

    return out


@functools.partial(jax.jit, static_argnums=(0, 3, 4))
def _mk_sup_example_jit(Pos, input_ids: hax.NamedArray, sources_len, pad_token_id, eos_id):
    # mask out padding and anything before the start of the target
    loss_mask = hax.arange(Pos) >= sources_len - 1
    # don't predict the padding
    targets = hax.roll(input_ids, -1, Pos)
    loss_mask = loss_mask & (targets != pad_token_id)
    loss_mask = loss_mask & (1 - hax.nn.one_hot(-1, Pos, dtype=jax.numpy.bool_))
    return LmExample.causal(input_ids, loss_mask=loss_mask, eos_id=eos_id)


def mk_supervised_datasets(
    sources: Mapping[str, SupervisedSourceConfigBase] | SupervisedSourceConfigBase,
    split: str,
    tokenizer: PreTrainedTokenizerBase,
    Pos: hax.Axis,
) -> dict[str, tuple[AsyncDataset[LmExample], Sequence[str]]]:
    """
    Create supervised datasets from a set of sources.

    Returns:
        A dictionary of dataset names to tuples of the dataset and the tags associated with the dataset.
    """
    out: dict[str, tuple[AsyncDataset[LmExample], Sequence[str]]] = {}

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if isinstance(sources, Mapping):
        for name, config in sources.items():
            source = config.get_shard_source(split)
            if source is None:
                continue

            ds = _cache_supervised_set(
                source, config.cache_dir, tokenizer, Pos, config.input_field, config.output_field
            )

            if config.tags is None:
                tags = [name]
            else:
                tags = config.tags + [name]

            out[name] = (ds, tags)
    else:
        source = sources.get_shard_source(split)  # type: ignore
        if source is not None:
            ds = _cache_supervised_set(
                source, sources.cache_dir, tokenizer, Pos, sources.input_field, sources.output_field
            )
            tags = sources.tags or []
            if isinstance(sources, SupervisedHfSourceConfig):
                name = sources.id
                if sources.name is not None:
                    name = f"{name}/{sources.name}"

                tags = [name] + tags
            else:
                name = "default"
            out[name] = (ds, tags)

    return out


def mk_supervised_dataset(
    config: SupervisedSourceConfigBase, split: str, tokenizer: HfTokenizer, Pos: hax.Axis
) -> AsyncDataset[LmExample]:

    source = config.get_shard_source(split)

    output_exemplar = {"input_ids": np.zeros((0,), dtype=np.int32), "sources_len": np.zeros((0,), dtype=np.int32)}

    dataset = source.map_batches(  # type: ignore
        lambda ex: _preprocess_supervised_example(ex, tokenizer, config.input_field, config.output_field),
        batch_size=128,
        num_cpus=num_cpus_used_by_tokenizer(tokenizer),
        output_exemplar=output_exemplar,
    )

    cached_dataset: AsyncDataset[dict] = dataset.build_or_load_cache(config.cache_dir, await_finished=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return cached_dataset.map_batches(lambda ex: _prepare_supervised_examples(ex, tokenizer, Pos))


def _cache_supervised_set(source, cache_dir, tokenizer, Pos: hax.Axis, input_field, output_field):
    """
    Cache a supervised dataset into input_ids and sources_len.
    """
    output_exemplar = {"input_ids": np.zeros((0,), dtype=np.int32), "sources_len": np.zeros((0,), dtype=np.int32)}
    dataset = source.map_batches(
        lambda ex: _preprocess_supervised_example(ex, tokenizer, input_field, output_field),
        batch_size=128,
        num_cpus=num_cpus_used_by_tokenizer(tokenizer),
        output_exemplar=output_exemplar,
    )
    cached_dataset: AsyncDataset[dict] = dataset.build_or_load_cache(cache_dir, await_finished=True)
    ds = cached_dataset.map_batches(lambda ex: _prepare_supervised_examples(ex, tokenizer, Pos))
    return ds


@dataclass(frozen=True)
class ChatUrlDataSourceConfig:
    """Config for loading JSONL files in OpenAI chat format for supervised fine-tuning."""

    cache_dir: str
    train_urls: List[str] = field(default_factory=list)
    validation_urls: List[str] = field(default_factory=list)

    # Chat format specific fields
    messages_field: str = "messages"
    input_role: str = "user"
    output_role: str = "assistant"

    def get_shard_source(self, split: str) -> Optional[ShardedDataSource[dict]]:
        """Gets ShardedDataSource for either training or validation data."""
        urls = self.validation_urls if split == "validation" else self.train_urls

        if not urls:
            return None

        # Use the datasource_from_chat_jsonl function from sharded_datasource
        return datasource_from_chat_jsonl(
            urls, messages_field=self.messages_field, input_role=self.input_role, output_role=self.output_role
        )


def preprocess_chat_example(batch, tokenizer: PreTrainedTokenizerBase, should_append_eos: bool) -> dict:
    """
    Preprocess chat examples to match the format of preprocess_supervised_example.
    Returns a dict with input_ids and sources_len like the supervised case.

    Args:
        batch: List of dicts with input/output pairs
        tokenizer: HuggingFace tokenizer
        should_append_eos: Whether we need to manually add EOS (True if tokenizer doesn't do it automatically)
    """
    # Get sources (inputs) and targets (outputs) from the batch
    sources = [example["input"] for example in batch]
    targets = [example["output"] for example in batch]

    # Add EOS only if needed (tokenizer doesn't do it automatically)
    if should_append_eos:
        targets = [t + tokenizer.eos_token for t in targets]

    # Tokenize sources alone first to get the source lengths
    sources_tokenized = tokenizer(sources, padding=False, truncation=True)

    # Combine source and target for full examples
    full_examples = [f"{s}{t}" for s, t in zip(sources, targets)]
    examples_tokenized = tokenizer(full_examples, padding=False, truncation=True)

    # Get source lengths to mask loss appropriately
    source_lens = [len(s) for s in sources_tokenized["input_ids"]]

    return {
        "input_ids": [np.array(example, dtype=np.int32) for example in examples_tokenized["input_ids"]],
        "sources_len": np.array(source_lens, dtype=np.int32),
    }


def mk_cached_sft_dataset(
    config: ChatUrlDataSourceConfig, tokenizer: PreTrainedTokenizerBase, Pos: hax.Axis
) -> AsyncDataset[dict]:
    """Creates a dataset from JSONL files containing chat format data for SFT."""
    source = config.get_shard_source("train")
    if source is None:
        raise ValueError("No training data source found")

    # Set up example structure matching supervised case
    output_exemplar = {"input_ids": np.zeros((0,), dtype=np.int32), "sources_len": np.zeros((0,), dtype=np.int32)}

    input_ids = tokenizer("hi there")["input_ids"]
    should_append_eos = input_ids[-1] != tokenizer.eos_token_id
    logger.info(f"Manual EOS Needed: {should_append_eos}")

    # Process the dataset
    dataset = source.map_batches(
        lambda ex: preprocess_chat_example(ex, tokenizer, should_append_eos),
        batch_size=128,
        num_cpus=num_cpus_used_by_tokenizer(tokenizer),
        output_exemplar=output_exemplar,
    )

    # Cache the processed data
    cached_dataset: AsyncDataset[dict] = dataset.build_or_load_cache(config.cache_dir, await_finished=True)
    return cached_dataset


def mk_chat_sft_dataset(
    config: ChatUrlDataSourceConfig, tokenizer: PreTrainedTokenizerBase, Pos: hax.Axis
) -> AsyncDataset[LmExample]:
    """Creates a dataset from JSONL files containing chat format data for SFT."""
    source = config.get_shard_source("train")
    if source is None:
        raise ValueError("No training data source found")

    # Set up example structure matching supervised case
    output_exemplar = {"input_ids": np.zeros((0,), dtype=np.int32), "sources_len": np.zeros((0,), dtype=np.int32)}

    input_ids = tokenizer("hi there")["input_ids"]
    should_append_eos = input_ids[-1] != tokenizer.eos_token_id
    logger.info(f"Manual EOS Needed: {should_append_eos}")

    # Process the dataset
    dataset = source.map_batches(
        lambda ex: preprocess_chat_example(ex, tokenizer, should_append_eos),
        batch_size=128,
        num_cpus=num_cpus_used_by_tokenizer(tokenizer),
        output_exemplar=output_exemplar,
    )

    # Cache the processed data
    cached_dataset: AsyncDataset[dict] = dataset.build_or_load_cache(config.cache_dir, await_finished=True)

    # Ensure padding token is set (needed by _prepare_supervised_example)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Reuse the supervised prepare function directly
    return cached_dataset.map_batches(lambda ex: _prepare_supervised_examples(ex, tokenizer, Pos))


@dataclass
class LMDatasetConfig(LMDatasetSourceConfig, LMTaskConfig):
    """This class supports loading data both from HF Datasets and from a raw dataset of jsonl urls"""

    cache_dir: Optional[str] = "cache/"

    def train_set(
        self,
        Pos: Axis,
        batch_schedule: BatchSchedule,
        monitors: Union[bool, List[MetricsMonitor]] = True,
        *,
        key: PRNGKeyArray,
        epochs: Optional[int] = None,
    ) -> AsyncDataset[LmExample]:
        del batch_schedule  # unused

        ds: AsyncDataset[np.ndarray] | None = self.token_seq_dataset("train", Pos.size, monitors)

        # add epoch flag here.
        if ds is None:
            raise ValueError("No training set!")

        assert ds is not None

        if epochs:
            logger.info("Wrapping dataset in epoch dataset")
            ds = EpochDataset(ds, max_epochs=epochs)

        perm_type = self.permutation_type
        if perm_type is None:
            logger.warning(
                "Defaulting to linear permutation for shuffling. This will change to Feistel in the future."
            )
            perm_type = "linear"

        if self.shuffle is True:
            ds = ds.shuffle(key, perm_type=perm_type)
        elif isinstance(self.shuffle, int) and self.shuffle > 0:
            ds = ds.era_shuffle(self.shuffle, key=key, perm_type=perm_type)

        return CausalLmDataset(ds, Pos, ignore_index=self.ignore_token_id, eos_id=self.the_tokenizer.eos_token_id)  # type: ignore

    def train_sets(
        self,
        Pos: Axis,
        monitors: Union[bool, List[MetricsMonitor]] = True,
        *,
        key: PRNGKeyArray,
        epochs: Optional[int] = None,
    ) -> Mapping[str, AsyncDataset[LmExample]]:
        return {
            # we don't care about BatchSchedule in this class
            "": self.train_set(Pos, BatchSchedule(32), monitors, key=key, epochs=epochs)
        }

    def validation_set(
        self,
        Pos: Axis,
        monitors: Union[bool, List[MetricsMonitor]] = True,
    ) -> AsyncDataset[LmExample] | None:
        ds = self.token_seq_dataset("validation", Pos.size, monitors)
        if ds is None:
            return None

        return CausalLmDataset(
            ds,
            Pos,
            ignore_index=self.ignore_token_id,
            eos_id=self.the_tokenizer.eos_token_id,
        )

    def validation_sets(
        self, Pos: Axis, monitors: Union[bool, List[MetricsMonitor]] = True
    ) -> Mapping[str, AsyncDataset[LmExample]]:
        validation_set = self.validation_set(Pos, monitors)
        if validation_set is not None:
            return {"": validation_set}
        else:
            return {}

    @property
    def sources(self) -> Mapping[str, LMDatasetSourceConfig]:
        return {"": self}

    @cached_property
    def _has_validation_set(self):
        if len(self.validation_urls) > 0:
            return True

        if self.id is not None:
            dataset = datasets.load_dataset(self.id, name=self.name, streaming=self.stream, split="validation")
            try:
                next(iter(dataset))
                return True
            except StopIteration:
                return False

        return False

    def token_seq_dataset(
        self, split: str, seq_len: int, monitors: Union[bool, List[MetricsMonitor]] = True
    ) -> Optional[TokenSeqDataset]:
        cache = self.build_or_load_cache(split, monitors=monitors)
        if cache is None:
            return None
        return TokenSeqDataset(cache, seq_len)

    def build_or_load_cache(
        self, split: str, monitors: Union[bool, List[MetricsMonitor]] = True, logger_name: Optional[str] = None
    ) -> Optional[TreeCache[BatchEncoding]]:
        if self.cache_dir is None:
            raise ValueError("cache_dir cannot be None")

        split_cache_dir = os.path.join(self.cache_dir, split)
        name = logger_name or os.path.basename(self.cache_dir)

        try:
            # TODO: pass in options
            return TreeCache.load(split_cache_dir, exemplar={"input_ids": np.zeros(0, dtype=np.int32)})
        except FileNotFoundError:
            pass

        source = self.get_shard_source(split)
        if source is None:
            logger.info(f"No data for {split}")
            return None

        logger.info(f"Building cache for {split}...")

        if monitors is True:
            monitors = [
                LoggingMetricsMonitor(prefix=f"preprocessing/{name}/{split}", commit=False),
                LoggerMetricsMonitor(f"preprocessing.{name}.{split}"),
            ]
        elif monitors is False:
            monitors = []

        bt = BatchTokenizer(self.the_tokenizer, enforce_bos=True, enforce_eos=self.enforce_eos)

        return build_or_load_cache(
            split_cache_dir,
            source,
            bt,
            monitors=monitors,
            await_finished=False,
            options=self.cache_options,
            split=split,
        )


class PassthroughTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_size, **kwargs):
        self._vocab = {i: i for i in range(vocab_size)}
        self._vocab_size = vocab_size
        super().__init__(**kwargs)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def get_vocab(self):
        return self._vocab

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str, ...]:
        return ()

    def _tokenize(self, text, **kwargs):
        tokens = np.fromstring(text, dtype=int, sep=" ")
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return int(token)

    def _convert_id_to_token(self, index: int) -> str:
        return str(index)


@dataclass
class LMMixtureDatasetConfig(LMTaskConfig):
    """A mixture of language model datasets that supports dynamic weight changes during training.

    Weights can be specified either as a single dictionary for constant mixing ratios,
    or as a list of (step, weights) tuples to change mixing ratios during training.
    """

    cache_dir: Optional[str] = "cache/"

    configs: Dict[str, LMDatasetSourceConfig] = field(default_factory=dict)
    """ Configuration of each dataset source (urls, hf dataset id, etc.) """

    train_weights: Union[Dict[str, float], List[Tuple[int, Dict[str, float]]]] = field(default_factory=dict)
    """ Dataset mixing weights. Either a constant dict[name->weight] or list of (step, weights) tuples """

    stop_strategy: str = field(default=StopStrategy.RESTART_STRATEGY)

    # Configuration for Simulated Epoching
    target_budget: Optional[int] = None
    experiment_budget: Optional[int] = None

    mixture_block_size: int = 2048
    """Block size for deterministic mixing. In each block, a given dataset will have exactly the same number
    of samples, equal to the expected number of samples in the mixture, rounding in the expected way."""

    def __post_init__(self):
        if len(self.configs) == 0:
            raise ValueError("At least one dataset must be provided")

        if isinstance(self.train_weights, dict):
            if not all(name in self.configs for name in self.train_weights):
                raise ValueError(
                    f"Weight keys {self.train_weights.keys()} must be subset of config keys {self.configs.keys()}"
                )
        elif isinstance(self.train_weights, list):
            for step, weights in self.train_weights:
                if not all(name in self.configs for name in weights):
                    raise ValueError(
                        f"Weight keys {weights.keys()} must be subset of config keys {self.configs.keys()}"
                    )
        else:
            raise ValueError(f"Invalid train_weights type: {type(self.train_weights)}")

    def train_set(
        self,
        Pos: Axis,
        batch_schedule: BatchSchedule,
        monitors: Union[bool, List[MetricsMonitor]] = True,
        *,
        key: PRNGKeyArray,
        epochs: Optional[int] = None,
    ) -> AsyncDataset[LmExample]:
        mix_key, shuffle_key = jax.random.split(key)
        causal_datasets = self.train_sets(Pos, monitors, key=shuffle_key, epochs=epochs)

        weights = self.train_weights
        if isinstance(weights, Sequence):
            weights = rescale_mixture_schedule_for_batch_schedule(weights, batch_schedule)

        mixture = MixtureDataset(
            datasets=causal_datasets,
            weights=weights,
            stop_strategy=self.stop_strategy,
            key=mix_key,
            block_size=self.mixture_block_size,
        )

        return mixture

    def train_sets(
        self,
        Pos: Axis,
        monitors: Union[bool, List[MetricsMonitor]] = True,
        *,
        epochs: Optional[int] = None,
        key: PRNGKeyArray,
    ) -> Mapping[str, AsyncDataset[LmExample]]:
        doc_caches = self.build_caches("train", monitors=monitors)
        token_datasets = self._token_seq_datasets(Pos, doc_caches)
        if epochs:
            raise ValueError("Epochs are not supported for mixture datasets")

        if key is None:
            key = jax.random.PRNGKey(0)

        # We shuffle the components and not the overall mixture because this lets us preserve
        # the "stable batch" property of the mixture dataset.
        perm_type = self.permutation_type
        if perm_type is None and self.shuffle is not False:
            logger.warning(
                "Defaulting to linear permutation for shuffling. This will change to Feistel in the future."
            )
            perm_type = "linear"

        def shuffle_ds(ds, key):
            if self.shuffle is True:
                ds = ds.shuffle(key, perm_type=perm_type)
            elif isinstance(self.shuffle, int) and self.shuffle > 0:
                ds = ds.era_shuffle(self.shuffle, key=key, perm_type=perm_type)

            return ds

        if self.shuffle:
            out_token_datasets = {}
            key_iter = key_iterator(key)
            for name, ds in token_datasets.items():
                out_token_datasets[name] = shuffle_ds(ds, next(key_iter))
            token_datasets = out_token_datasets
        if (
            self.experiment_budget is not None and self.target_budget is not None
        ) and self.experiment_budget > self.target_budget:
            raise ValueError(
                f"Experiment budget should be smaller than target budget, got {self.experiment_budget} >"
                f" {self.target_budget}"
            )
        if self.experiment_budget is not None and self.target_budget is not None:
            simulated_data_ratio = self.experiment_budget / self.target_budget
            sliced_token_datasets: Dict[str, TokenSeqDataset] = {}
            for name, ds in token_datasets.items():
                # Note(Will): This blocks on datasets being fully processed even for small simulated runs making simulating data size slightly latency inducing but I think that's ok
                true_length_of_dataset = len(ds.as_sync_dataset())
                simulated_length_of_dataset = int(true_length_of_dataset * simulated_data_ratio)
                sliced_token_datasets[name] = ds.slice_dataset(end_index=simulated_length_of_dataset)
            token_datasets = sliced_token_datasets
        causal_datasets = {
            name: CausalLmDataset(
                ds,
                Pos,
                ignore_index=self.ignore_token_id,
                eos_id=self.the_tokenizer.eos_token_id,
            )
            for name, ds in token_datasets.items()
        }
        return causal_datasets

    def _token_seq_datasets(self, Pos, doc_caches):
        token_datasets = {name: TokenSeqDataset(cache, Pos.size) for name, cache in doc_caches.items()}
        return token_datasets

    def validation_sets(
        self, Pos: Axis, monitors: Union[bool, List[MetricsMonitor]] = True
    ) -> Mapping[str, AsyncDataset[LmExample]]:
        doc_caches = self.build_caches("validation", monitors=monitors)
        token_datasets = self._token_seq_datasets(Pos, doc_caches)
        return {
            name: CausalLmDataset(
                ds,
                Pos,
                ignore_index=self.ignore_token_id,
                eos_id=self.the_tokenizer.eos_token_id,
            )
            for name, ds in token_datasets.items()
        }

    def build_caches(
        self, split: str, monitors: Union[bool, List[MetricsMonitor]] = True
    ) -> Dict[str, TreeCache[dict]]:
        # this is a bit gross, but we want to forward all "Task" config fields to the LMDatasetConfig for building.
        # We do this by just grabbing all the fields from the LMDatasetConfig and forwarding them to the
        # LMDatasetConfig.build_or_load_cache method. We exclude the cache_dir field.
        task_config_fields = set(x.name for x in dataclasses.fields(LMTaskConfig))
        task_config_dict = {k: v for k, v in self.__dict__.items() if k in task_config_fields and k != "cache_dir"}

        caches = {}
        for name, source_config in self.configs.items():
            # Skip datasets with zero weight in all stages
            if isinstance(self.train_weights, dict):
                has_nonzero_weight = self.train_weights.get(name, 0) > 0
            elif isinstance(self.train_weights, list):
                has_nonzero_weight = any(weights.get(name, 0) > 0 for _, weights in self.train_weights)

            if not has_nonzero_weight and split == "train":
                continue

            source_config_dict = dict(**source_config.__dict__)

            if source_config.cache_dir is None:
                # replace with the main cache dir/{name}
                if self.cache_dir is None:
                    raise ValueError(
                        "If the 'main' cache_dir is None, then all component cache_dirs must be non-None, but"
                        f"{name}'s cache_dir is None."
                    )
                cache_dir = os.path.join(self.cache_dir, name)
                source_config_dict["cache_dir"] = cache_dir

            dataset = LMDatasetConfig(
                **source_config_dict,
                **task_config_dict,
            )
            cache = dataset.build_or_load_cache(split, monitors)
            # drop the data source and corresponding weight if the cache is not built
            if cache is None:
                logger.warning(f"Skipping {name} for split {split} because no source was provided")
            else:
                caches[name] = cache

        # in practice it works best if we block on validation caches
        if split == "validation":
            for cache in caches.values():
                cache.await_finished()

        else:
            logger.info(f"Not waiting for {split} caches to finish building")

        return caches

    @property
    def sources(self) -> Mapping[str, LMDatasetSourceConfig]:
        return self.configs


def datasource_from_chat_jsonl(
    urls: Sequence[str], messages_field: str = "messages", input_role: str = "user", output_role: str = "assistant"
) -> "ShardedDataSource[dict]":
    """Creates a ShardedDataSource from JSONL files containing chat messages.

    Args:
        urls: Sequence of URLs or glob patterns pointing to JSONL files
        messages_field: Field name containing the messages in each JSON object
        input_role: Role identifier for input messages
        output_role: Role identifier for output messages

    Returns:
        ShardedDataSource configured for chat data
    """
    # Expand any glob patterns in the URLs
    expanded_urls = []
    for url in urls:
        if any(c in url for c in "*?[]"):
            expanded_urls.extend(gcs_glob(url))
        else:
            expanded_urls.append(url)

    return ChatJsonlDataSource(expanded_urls, messages_field, input_role, output_role)


# TODO: switch to actual multi-turn
class ChatJsonlDataSource(JsonlDataSource):
    """DataSource that reads JSONL files containing OpenAI chat format messages."""

    def __init__(self, urls: Sequence[str], messages_field: str, input_role: str, output_role: str):
        super().__init__(urls)
        self.messages_field = messages_field
        self.input_role = input_role
        self.output_role = output_role

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[dict]:
        url = self._shard_name_to_url_mapping[shard_name]
        i = 0
        with fsspec.open(url, "r", compression="infer") as f:
            for line in f:
                if i >= row:
                    data = json.loads(line)
                    messages = data[self.messages_field]

                    # Extract input/output from messages
                    input_msg = next(m["content"] for m in messages if m["role"] == self.input_role)
                    output_msg = next(m["content"] for m in messages if m["role"] == self.output_role)

                    yield {"input": input_msg, "output": output_msg}
                i += 1
