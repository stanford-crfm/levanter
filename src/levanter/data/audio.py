import abc
import copy
import dataclasses
import functools
import logging
import os
from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import braceexpand
import datasets
import equinox as eqx
import fsspec
import jax
import numpy as np
import pyarrow as pa
import regex
from draccus import field
from jaxtyping import PRNGKeyArray
from tokenizers import normalizers

import haliax as hax
from haliax import Axis

from levanter.data.mixture import MixtureDataset, StopStrategy

# intercept the logging nonsense here
from levanter.logging import silence_transformer_nag  # noqa
from levanter.models.attention import AttentionMask
from levanter.models.lm_model import LmExample
from levanter.utils.hf_utils import num_cpus_used_by_tokenizer


silence_transformer_nag()  # noqa
from transformers import BatchEncoding, PreTrainedTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast  # noqa

from levanter.compat.hf_checkpoints import load_processor  # noqa
from levanter.data._preprocessor import BatchProcessor, dict_from_record_batch  # noqa
from levanter.data.dataset import ShardableDataset  # noqa
from levanter.data.shard_cache import DEFAULT_ROWS_PER_CHUNK  # noqa
from levanter.data.shard_cache import CacheLedger  # noqa
from levanter.data.shard_cache import LEDGER_FILE_NAME as NEW_LEDGER_FILE_NAME  # noqa
from levanter.data.shard_cache import (  # noqa
    ChunkMetadata,
    LoggerMetricsMonitor,
    LoggingMetricsMonitor,
    MetricsMonitor,
    ShardCache,
    _serialize_json_and_commit,
    build_cache,
)
from levanter.data.sharded_dataset import ShardedDataset, TextUrlDataset, WrappedHFDataset  # noqa
from levanter.shapes import NamedShapeSpec, ShapeSpec  # noqa
from levanter.utils.jax_utils import use_cpu_device  # noqa


logger = logging.getLogger("levanter.data.text")

# TASKS:
# TODO: consider adding indexing a la Map-style datasets
# TODO: support seeking/serialization/restore in the dataset

LEDGER_FILE = "ledger.json"

DEFAULT_IGNORE_INDEX = -100  # Mirrors pytorch's default ignore index


class AudioTextExample(eqx.Module):
    audio: hax.NamedArray
    tokens: hax.NamedArray
    loss_mask: hax.NamedArray
    attn_mask: AttentionMask | NamedArray = AttentionMask.causal()

    @staticmethod
    def causal(
        audio: hax.NamedArray,
        tokens: hax.NamedArray,
        *,
        loss_mask: Optional[hax.NamedArray] = None,
        ignore_id: Optional[int] = None,
    ) -> "AudioTextExample":
        if tokens.ndim != 1:
            raise ValueError("tokens must be a 1D array")

        if not jnp.issubdtype(tokens.dtype, jnp.integer):
            raise ValueError("tokens must be an integer array")

        Pos = tokens.axes[0]

        # don't predict the last token.
        if loss_mask is None:
            loss_mask = 1 - hax.nn.one_hot(-1, Pos, dtype=jnp.float32)

        if ignore_id is not None:
            # we don't compute loss for any tokens matching the ignore index
            ignore_mask = hax.roll(tokens, -1, Pos) != ignore_id
            loss_mask = loss_mask * ignore_mask

        attn_mask = AttentionMask.causal()
        return AudioTextExample(audio=audio, tokens=tokens, loss_mask=loss_mask, attn_mask=attn_mask)


@dataclass
class AudioDatasetSourceConfig:
    """This class represents a dataset source with URLs or hf name/id."""

    id: Optional[str] = None  # id (or path) for hf dataset
    name: Optional[str] = None  # name for hf dataset

    plaintext: bool = False
    stream: bool = True  # whether to use streaming when doing hf
    text_key: str = "sentence"  # key for the text field in the jsonl file or hf dataset
    audio_key: str = "audio"  # key for the text field in the jsonl file or hf dataset

    train_urls: List[str] = ()  # type: ignore
    validation_urls: List[str] = ()  # type:ignore

    def get_shard_source(self, split) -> Optional[ShardedDataset[Tuple[dict, str]]]:
        if self.id is not None:
            try:
                ds = WrappedHFDataset(self.id, split=split, name=self.name, streaming=self.stream)
            except ValueError as e:
                # if the message starts with Bad split, then just return None
                if str(e).startswith("Bad split"):
                    logger.warning(f"Splits {split} not found for {self.id} {self.name}")
                    return None
                else:
                    raise

            if len(ds.shard_names) == 0:
                return None

            return ds.map(lambda x: (x[self.audio_key], x[self.text_key]))
        else:
            split_urls = self.urls_for_split(split)
            if len(split_urls) == 0:
                return None
            return AudioTextUrlDataset(split_urls, self.text_key, self.audio_key)

    def doc_iterator(self, split: str) -> Iterator[Tuple[dict, str]]:
        if self.id is not None:
            dataset = datasets.load_dataset(self.id, name=self.name, streaming=self.stream)
            data = dataset[split]
            for doc in data:
                yield (doc[self.audio_key], doc[self.text_key])
        else:
            urls = self.urls_for_split(split)

            yield from AudioTextUrlDataset(urls, self.text_key, self.audio_key)

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


@dataclass
class AudioTaskConfig(abc.ABC):
    processor: str = "openai/whisper-tiny"

    # config related to caching
    cache_dir: str = "cache/"
    rows_per_chunk: int = DEFAULT_ROWS_PER_CHUNK  # number of rows to process and cache per chunk
    enforce_eos: bool = True  # whether to append eos even if the tokenizer doesn't

    ignore_token_id: Optional[int] = None

    @cached_property
    def the_processor(self) -> PreTrainedTokenizerBase:
        return load_processor(self.processor)

    @cached_property
    def the_tokenizer(self) -> PreTrainedTokenizerBase:
        return self.the_processor.tokenizer

    @cached_property
    def the_feature_extractor(self) -> PreTrainedTokenizerBase:
        return self.the_processor.feature_extractor

    @abc.abstractmethod
    def train_set(
        self, seq_len: int, monitors: Union[bool, List[MetricsMonitor]] = True
    ) -> ShardableDataset[np.ndarray]:
        pass

    @abc.abstractmethod
    def validation_sets(
        self, seq_len: int, monitors: Union[bool, List[MetricsMonitor]] = True
    ) -> Mapping[str, ShardableDataset[np.ndarray]]:
        pass
