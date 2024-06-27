import abc
import functools
import logging
import os
from dataclasses import dataclass
from functools import cached_property
from typing import Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import braceexpand
import datasets
import equinox as eqx
import fsspec
import jax
import numpy as np
from jaxtyping import PRNGKeyArray
from typing_extensions import TypedDict

import haliax as hax
from haliax import Axis

from levanter.compat.hf_checkpoints import load_processor, load_tokenizer
from levanter.data._preprocessor import BatchProcessor, dict_from_record_batch
from levanter.data.dataset import ShardableDataset
from levanter.data.metrics_monitor import LoggerMetricsMonitor, LoggingMetricsMonitor, MetricsMonitor
from levanter.data.shard_cache import DEFAULT_ROWS_PER_CHUNK, ShardCache, build_or_load_cache
from levanter.data.sharded_dataset import AudioTextUrlDataset, ShardedDataset, WrappedHFDataset
from levanter.data.text import BatchTokenizer

# intercept the logging nonsense here
from levanter.logging import silence_transformer_nag
from levanter.models.asr_model import AudioTextExample
from levanter.utils.jax_utils import use_cpu_device


silence_transformer_nag()  # noqa
from transformers import (  # noqa
    BatchEncoding,
    BatchFeature,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    SequenceFeatureExtractor,
)


logger = logging.getLogger("levanter.data.audio")

AudioTextStorageBatch = TypedDict(
    "AudioTextStorageBatch",
    {
        "input_features": np.ndarray,
        "input_ids": np.ndarray,
        "attention_mask": np.ndarray,
        "audio_shape": Sequence[Tuple[int, int]],
    },
)
AudioTextDict = TypedDict(
    "AudioTextDict",
    {
        "input_features": np.ndarray,
        "input_ids": np.ndarray,
        "attention_mask": np.ndarray,
    },
)


class BatchAudioProcessor(BatchProcessor[Tuple[np.ndarray, int, str]]):
    """
    A batch processor that converts raw audio into the expected inputs of a model.
    """

    def __init__(
        self,
        processor: ProcessorMixin,
        tokenizer: PreTrainedTokenizerBase,
        enforce_bos=True,
        enforce_eos=True,
        *,
        batch_size=128,
        override_resources=None,
        max_length=448,
        padding=True,
    ):
        self.feature_extractor: SequenceFeatureExtractor = processor.feature_extractor
        self.bt: PreTrainedTokenizerBase = BatchTokenizer(
            tokenizer,
            enforce_bos=enforce_bos,
            enforce_eos=enforce_eos,
            batch_size=batch_size,
            override_resources=override_resources,
            return_attention_mask=True,
            padding="max_length" if padding else False,
            max_length=max_length,
        )

        self.override_resources = override_resources
        self._batch_size = batch_size

    def __call__(self, batch: Sequence[Tuple[np.ndarray, int, str]]) -> AudioTextStorageBatch:
        """
        Process a batch of data.
        """
        audio_batch: Sequence[np.ndarray]
        sampling_rates: Sequence[int]
        text_batch: Sequence[str]
        audio_batch, sampling_rates, text_batch = list(zip(*batch))
        uniq_sampling_rates: set[int] = set(sampling_rates)
        assert len(uniq_sampling_rates) == 1, "Sampling rates should be standardized"
        audio_features: BatchFeature = self.feature_extractor(audio_batch, sampling_rate=uniq_sampling_rates.pop())
        text_features: BatchEncoding = self.bt(text_batch)
        combined_features = audio_features | text_features
        combined_features["input_ids"] = np.array(combined_features["input_ids"])
        combined_features["attention_mask"] = np.array(combined_features["attention_mask"])
        a_features = np.array(combined_features["input_features"])
        a_shape = a_features.shape
        combined_features["audio_shape"] = [a_shape[1:]] * a_shape[0]
        combined_features["input_features"] = a_features.reshape(a_shape[0], -1)
        return combined_features

    @property
    def num_cpus(self) -> int:
        """The number of CPUs this processor needs to run."""
        return self.bt.num_cpus

    @property
    def num_gpus(self) -> int:
        return self.bt.num_gpus

    @property
    def batch_size(self) -> int:
        return self.bt._batch_size


@dataclass
class AudioDatasetSourceConfig:
    """This class represents a dataset source with URLs or hf name/id."""

    id: Optional[str] = None  # id (or path) for hf dataset
    name: Optional[str] = None  # name for hf dataset

    plaintext: bool = False
    stream: bool = True  # whether to use streaming when doing hf
    text_key: str = "text"  # key for the text field in the jsonl file or hf dataset
    audio_key: str = "audio"  # key for the text field in the jsonl file or hf dataset
    sampling_rate: int = 16_000

    train_urls: List[str] = ()  # type: ignore
    validation_urls: List[str] = ()  # type:ignore

    def get_shard_source(self, split) -> Optional[ShardedDataset[Tuple[np.ndarray, int, str]]]:
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

            def decode(x):
                text = x[self.text_key]
                audio_pointer = x[self.audio_key]
                audio = AudioTextUrlDataset.resolve_audio_pointer(audio_pointer, self.sampling_rate)
                return (audio["array"], audio["sampling_rate"], text)

            return ds.map(decode)
        else:
            split_urls = self.urls_for_split(split)
            if len(split_urls) == 0:
                return None
            return AudioTextUrlDataset(split_urls, self.text_key, self.audio_key, sampling_rate=self.sampling_rate)

    def doc_iterator(self, split: str) -> Iterator[Tuple[np.ndarray, int, str]]:
        if self.id is not None:
            data = datasets.load_dataset(self.id, split=split, name=self.name, streaming=self.stream)
            for doc in data:
                yield (doc[self.audio_key]["array"], doc[self.audio_key]["sampling_rate"], doc[self.text_key])
        else:
            urls = self.urls_for_split(split)

            yield from AudioTextUrlDataset(urls, self.text_key, self.audio_key, sampling_rate=self.sampling_rate)

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
    tokenizer: Optional[str] = None
    # config related to caching
    train_split: str = "train"
    validation_split: str = "validation"
    cache_dir: str = "cache/"
    rows_per_chunk: int = DEFAULT_ROWS_PER_CHUNK  # number of rows to process and cache per chunk
    enforce_bos: bool = True  # whether to append bos even if the tokenizer doesn't
    enforce_eos: bool = True  # whether to append eos even if the tokenizer doesn't

    @cached_property
    def the_processor(self) -> PreTrainedTokenizerBase:
        return load_processor(self.processor)

    @cached_property
    def pad_token_id(self) -> int:
        return self.the_tokenizer.pad_token_id

    @cached_property
    def the_tokenizer(self) -> PreTrainedTokenizerBase:
        if self.tokenizer is None:
            return self.the_processor.tokenizer
        else:
            return load_tokenizer(self.tokenizer)

    @cached_property
    def the_feature_extractor(self) -> PreTrainedTokenizerBase:
        return self.the_processor.feature_extractor

    @abc.abstractmethod
    def train_set(
        self, batch_size: int, monitors: Union[bool, List[MetricsMonitor]] = True
    ) -> ShardableDataset[np.ndarray]:
        pass

    @abc.abstractmethod
    def validation_sets(
        self, batch_size: int, monitors: Union[bool, List[MetricsMonitor]] = True
    ) -> Mapping[str, ShardableDataset[np.ndarray]]:
        pass


class ProcessedAudioCache(ShardableDataset[AudioTextStorageBatch]):
    """
    Represents a cache of data with both pre-processed audio and tokenized text, which is a directory of parquet files with a ledger file.
    """

    def __init__(self, chunk_cache: ShardCache):
        # Separates Batching For Processing from Batching For Training
        self.chunk_cache = chunk_cache.with_batch_size(1)

    def __iter__(self):
        for batch in self._chunks():
            unarrow = dict_from_record_batch(batch)
            # Flatten Singleton Batch Dimension
            singleton_dict = {key: unarrow[key].squeeze() for key in unarrow}
            singleton_dict["input_features"] = singleton_dict["input_features"].reshape(singleton_dict["audio_shape"])
            del singleton_dict["audio_shape"]
            yield singleton_dict

    def _chunks(self):
        return self.chunk_cache.iter_batches_from_chunks()

    @staticmethod
    def build_or_load(
        cache_dir: str,
        source: ShardedDataset[Tuple[np.ndarray, int, str]],
        processor: ProcessorMixin,
        tokenizer: PreTrainedTokenizerBase,
        enforce_bos=True,
        enforce_eos=True,
        batch_size=128,
        rows_per_chunk=DEFAULT_ROWS_PER_CHUNK,
        monitors=None,
        await_finished=True,
        override_resources=None,
    ) -> "ProcessedAudioCache":
        bp: BatchProcessor[Tuple[np.ndarray, int, str]] = BatchAudioProcessor(
            processor,
            tokenizer,
            enforce_bos=enforce_bos,
            enforce_eos=enforce_eos,
            batch_size=batch_size,
            override_resources=override_resources,
        )
        monitors = monitors or []
        cache = build_or_load_cache(
            cache_dir,
            source,
            bp,
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

        return ProcessedAudioCache(cache)

    @staticmethod
    def load(cache_dir, batch_size: int = 128):
        """
        Load a TokenizedDocumentCache from a directory. If the ledger file is not present, this will raise a
        FileNotFoundError.

        NOTE: ATM this attempts to migrate old caches to the new format, but this will be removed in the future.

        :param cache_dir:
        :return:
        """

        try:
            cache = ShardCache.load(cache_dir, batch_size=batch_size)
            return ProcessedAudioCache(cache)
        except FileNotFoundError:
            raise FileNotFoundError(f"{cache_dir} is not a complete cache")
        except Exception:
            logger.exception("error loading cache")
            raise

    def shard(self, shard_index, num_shards):
        if num_shards <= shard_index:
            raise ValueError(f"Shard index {shard_index} is out of range")

        if num_shards == 1:
            return self

        return ProcessedAudioCache(self.chunk_cache.shard(shard_index, num_shards))


@dataclass
class AudioIODatasetConfig(AudioDatasetSourceConfig, AudioTaskConfig):
    """This class supports loading data both from HF Datasets and from a raw dataset of jsonl urls"""

    def train_set(self, batch_size: int, monitors: Union[bool, List[MetricsMonitor]] = True) -> ProcessedAudioCache:
        ds = self.build_or_load_cache(self.train_split, batch_size=batch_size, monitors=monitors)
        if ds is None:
            raise ValueError("No training set!")
        return ds

    def validation_set(
        self, batch_size: int, monitors: Union[bool, List[MetricsMonitor]] = True
    ) -> Optional[ProcessedAudioCache]:
        return self.build_or_load_cache(self.validation_split, batch_size=batch_size, monitors=monitors)

    def validation_sets(
        self, batch_size: int, monitors: Union[bool, List[MetricsMonitor]] = True
    ) -> Mapping[str, ProcessedAudioCache]:
        if self._has_validation_set:
            validation_set = self.validation_set(batch_size, monitors)
            if validation_set is not None:
                return {"": validation_set}
        return {}

    @cached_property
    def _has_validation_set(self):
        if len(self.validation_urls) > 0:
            return True

        if self.id is not None:
            dataset = datasets.load_dataset(
                self.id, name=self.name, streaming=self.stream, split=self.validation_split
            )
            try:
                next(iter(dataset))
                return True
            except StopIteration:
                return False

        return False

    def build_or_load_cache(
        self,
        split: str,
        batch_size: int = 128,
        monitors: Union[bool, List[MetricsMonitor]] = True,
        logger_name: Optional[str] = None,
    ) -> Optional[ProcessedAudioCache]:
        split_cache_dir = os.path.join(self.cache_dir, split)
        name = logger_name or os.path.basename(self.cache_dir)

        try:
            return ProcessedAudioCache.load(split_cache_dir, batch_size=batch_size)
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

        return ProcessedAudioCache.build_or_load(
            split_cache_dir,
            source,
            self.the_processor,
            self.the_tokenizer,
            enforce_bos=self.enforce_bos,
            enforce_eos=self.enforce_eos,
            batch_size=batch_size,
            rows_per_chunk=self.rows_per_chunk,
            monitors=monitors,
            await_finished=(split == "validation"),
        )


class AudioTextDataset(ShardableDataset[AudioTextExample]):
    def __init__(
        self,
        dataset: ShardableDataset[AudioTextStorageBatch],
        TextPos: Axis,
        AudioPos: hax.AxisSelector,
        KPos: Axis,
        key: Optional[PRNGKeyArray] = None,
        ignore_index: Optional[int] = None,
    ):
        self.dataset = dataset
        self.TextPos = TextPos
        self.AudioPos = AudioPos
        self.KPos = KPos
        self.key = key
        self.ignore_id = ignore_index

    def shard(self, shard_id: int, num_shards: int) -> "AudioTextDataset":
        return AudioTextDataset(
            self.dataset.shard(shard_id, num_shards),
            self.TextPos,
            self.AudioPos,
            self.KPos,
            self.key,
            self.ignore_id,
        )

    def __iter__(self) -> Iterator[AudioTextExample]:
        sharding = jax.sharding.SingleDeviceSharding(jax.local_devices(backend="cpu")[0])

        with use_cpu_device():

            @functools.partial(eqx.filter_jit, out_shardings=sharding)
            def _convert_example(inputs: AudioTextDict) -> "AudioTextExample":
                tokens = hax.named(inputs["input_ids"], self.TextPos)
                audio_features = hax.named(inputs["input_features"], self.AudioPos)

                return AudioTextExample.init(audio_features, tokens, ignore_id=self.ignore_id)

            for example in self.dataset:
                converted_example = _convert_example(example)
                yield converted_example
