# Routines for creating UL2R-type objectives: R-denoisers, X-denoisers, S-denoisers
# See https://arxiv.org/pdf/2210.11399v2.pdf
import dataclasses
import functools
import logging
import warnings
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Union

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey
from jaxtyping import PyTree
from transformers import PreTrainedTokenizerBase

import haliax as hax

from levanter.data.dataset import Dataset, ShardableDataset
from levanter.models.lm_model import LmExample
from levanter.shapes import NamedShapeSpec, ShapeSpec
from levanter.utils.jax_utils import use_cpu_device


logger = logging.getLogger(__name__)


class Ul2Example(eqx.Module):
    """A single example for the UL2 Objective for an encoder-decoder objective.
    In general, these should start out unpadded/untruncated"""

    inputs: np.ndarray
    outputs: np.ndarray
    task_token: Optional[int]

    def render(self, tokenizer: PreTrainedTokenizerBase):
        """renders a pretty version of the example as a string"""
        task_str = ""
        if self.task_token is not None:
            task_str = tokenizer.decode(self.task_token) + " "
        return task_str + tokenizer.decode(self.inputs) + "\n<!TARGETS!>\n" + tokenizer.decode(self.outputs)

    def to_decoder_only(self, pad_token_id, QPos, KPos):
        return convert_to_decoder_only(self, pad_token_id, QPos, KPos)


def convert_to_decoder_only(example: Ul2Example, pad_token_id, QPos: hax.Axis, KPos: hax.Axis):
    max_seq_len = QPos.size
    padded, num_inputs, unpadded_length = pack_inputs_and_outputs(example, max_seq_len, pad_token_id)
    padded = hax.named(padded, QPos)

    # Create attention masks
    return _create_prefix_lm_example(QPos, KPos, padded, unpadded_length, num_inputs, pad_token_id)


def pack_inputs_and_outputs(example, max_seq_len, pad_token_id):
    all_tokens = np.full(max_seq_len, pad_token_id or 0)
    used_tokens = 0

    # TODO: it'd be better if we made sure the examples we were creating were of the right length to begin with
    if example.task_token is not None:
        all_tokens[0] = example.task_token
        used_tokens = 1

    total_tokens_to_use = used_tokens + len(example.inputs) + len(example.outputs)

    if total_tokens_to_use > max_seq_len:
        warnings.warn(f"Example is too long to fit in {max_seq_len} tokens. Truncating.")

    # we take some from the inputs and some from the outputs. take from the left of each
    num_inputs_to_take = min(
        max((max_seq_len // 2) - used_tokens, max_seq_len - used_tokens - len(example.outputs)), len(example.inputs)
    )
    all_tokens[used_tokens : used_tokens + num_inputs_to_take] = example.inputs[:num_inputs_to_take]
    used_tokens += num_inputs_to_take
    num_inputs = used_tokens

    num_outputs_to_take = min(max_seq_len - used_tokens, len(example.outputs))
    all_tokens[used_tokens : used_tokens + num_outputs_to_take] = example.outputs[:num_outputs_to_take]
    used_tokens += num_outputs_to_take

    return all_tokens, num_inputs, used_tokens


@functools.partial(jax.jit, static_argnums=(0, 1))
def _create_prefix_lm_example(QPos, KPos, tokens, unpadded_length, num_inputs, pad_token_id):
    # shift back by one since we're predicting the next token
    attention_mask = hax.nn.attention.prefix_lm_mask(QPos, KPos, num_inputs)
    # attention_mask = hax.nn.attention.causal_mask(QPos, KPos)
    # don't compute loss on:
    # 1) task token
    # 2) inputs (except last token of inputs)
    loss_mask = hax.arange(QPos) >= (num_inputs - 1)
    # 3) last token of targets and padding
    loss_mask = loss_mask & (hax.arange(QPos) < unpadded_length - 1)
    # 4) padding
    if pad_token_id is not None:
        loss_mask = loss_mask & (tokens != pad_token_id)
    return LmExample(tokens, loss_mask, attention_mask)


S_TASK_TOKEN = "[S2S]"
R_TASK_TOKEN = "[NLU]"
X_TASK_TOKEN = "[NLG]"


@dataclass(frozen=True)
class DenoisingConfig(draccus.ChoiceRegistry):
    task_token: Optional[str]

    def with_task_token(self, task_token: Optional[str]):
        return dataclasses.replace(self, task_token=task_token)

    def sample(self, key: PRNGKey, tokens: np.ndarray, sentinel_token_ids, task_token_id) -> Ul2Example:
        raise NotImplementedError

    @staticmethod
    def ul2_configs(
        r_task_token: str = R_TASK_TOKEN, x_task_token: str = X_TASK_TOKEN, s_task_token=S_TASK_TOKEN
    ) -> Dict[str, "DenoisingConfig"]:
        return {
            "r1": RDenoisingConfig(r_task_token, 0.15, 3.0),
            "r2": RDenoisingConfig(r_task_token, 0.15, 8.0),
            "x1": XDenoisingConfig(x_task_token, 0.5, 3.0),
            "x2": XDenoisingConfig(x_task_token, 0.5, 8.0),
            "x3": XDenoisingConfig(x_task_token, 0.15, 64.0),
            "x4": XDenoisingConfig(x_task_token, 0.5, 64.0),
            "s": PrefixLmConfig(s_task_token),
        }

    @staticmethod
    def ul2r_configs(
        r_task_token: Optional[str] = "[NLU]", x_task_token: Optional[str] = "[NLG]", s_task_token=S_TASK_TOKEN
    ) -> Dict[str, "DenoisingConfig"]:
        return {
            "r": RDenoisingConfig(r_task_token, 0.15, 3.0, True),
            "x1": XDenoisingConfig(x_task_token, 0.15, 32.0, True),
            "x2": XDenoisingConfig(x_task_token, 0.5, 3.0, True),
            "s": PrefixLmConfig(s_task_token),
        }


@dataclass(frozen=True)
class MaskDenoisingConfig(DenoisingConfig):
    mask_prob: float  # r in the paper
    mean_span_length: float  # mu in the paper
    random_roll: bool = False

    def sample(self, key: PRNGKey, tokens: np.ndarray, sentinel_token_ids, task_token_id) -> Ul2Example:
        """Build a mask denoiser example from a list of tokens"""
        # Slicing.
        # new_length = sliced_length + 2 * num_spans = sliced_length * (1 + 2r / mu) should be at most length (4096)
        length = 1024 - 3 - (task_token_id is not None)  # TODO: change to actual model seqlen
        max_length = int(round(length * self.mean_span_length / (self.mean_span_length + 2 * self.mask_prob)))
        if tokens.shape[0] > max_length:
            tokens = tokens[:max_length]

        this_key, key = jax.random.split(key)
        np_rng = np.random.default_rng(np.array(this_key))
        pivot = int(np_rng.integers(128, max_length))
        tokens = tokens[:pivot]

        # Masking.
        noise_mask = random_spans_noise_mask(
            len(tokens), self.mask_prob, key, self.mean_span_length, random_roll=self.random_roll
        )
        inputs = noise_span_to_unique_sentinel(tokens, noise_mask, sentinel_token_ids)

        # if input start with a blank
        if inputs[0] in sentinel_token_ids:
            targets = nonnoise_span_to_unique_sentinel(tokens, noise_mask, sentinel_token_ids[1:])
            targets = np.concatenate([[sentinel_token_ids[1]], targets])
        else:
            targets = nonnoise_span_to_unique_sentinel(tokens, noise_mask, sentinel_token_ids)
        # if input end with a non-blank (if target ends with a blank)
        if targets[-1] in sentinel_token_ids:
            targets = targets[:-1]
        # optional: put a <sentinel_0> at the end of input to signal the start of infilling task.
        inputs = np.concatenate([inputs, [sentinel_token_ids[0]]])

        return Ul2Example(inputs, targets, task_token_id)


@DenoisingConfig.register_subclass("x")
@dataclass(frozen=True)
class XDenoisingConfig(MaskDenoisingConfig):
    task_token: Optional[str] = X_TASK_TOKEN
    mask_prob: float = 0.5
    mean_span_length: float = 3.0


@DenoisingConfig.register_subclass("r")
@dataclass(frozen=True)
class RDenoisingConfig(MaskDenoisingConfig):
    task_token: Optional[str] = R_TASK_TOKEN
    mask_prob: float = 0.15
    mean_span_length: float = 3.0


@DenoisingConfig.register_subclass("s")
@dataclass(frozen=True)
class PrefixLmConfig(DenoisingConfig):
    task_token: Optional[str] = S_TASK_TOKEN

    def sample(self, key: PRNGKey, tokens: np.ndarray, sentinel_token_ids, task_token_id) -> Ul2Example:
        """Build an S-denoiser example from a list of tokens"""
        # Slicing.
        max_length = 1024 - 1 - (task_token_id is not None)  # TODO: change to actual model seqlen
        if tokens.shape[0] > max_length:
            tokens = tokens[:max_length]

        # choose a random length
        np_rng = np.random.default_rng(np.array(key))
        pivot = int(np_rng.integers(1, len(tokens) + 1))
        inputs = np.concatenate([np.array(tokens[:-pivot]), [sentinel_token_ids[0]]])
        return Ul2Example(inputs, np.array(tokens[-pivot:]), task_token_id)


# these aren't in the UL2(R) papers but they're nice to have
@DenoisingConfig.register_subclass("c")
@dataclass(frozen=True)
class CausalLmConfig(DenoisingConfig):
    """This is just causal language modeling. Technically it's a kind of S-Denoising"""

    task_token: Optional[str] = None

    def sample(self, key: PRNGKey, tokens: np.ndarray, sentinel_token_ids, task_token_id) -> Ul2Example:
        """Build an C-denoiser example from a list of tokens"""
        # Causal language model means we predict all tokens with no context (besides the task token)
        return Ul2Example(np.zeros((0,), dtype=int), np.array(tokens), task_token_id)


# TODO:f denoising
# @DenoisingConfig.register_subclass("f")
# @dataclass(frozen=True)
# class FDenoisingConfig(DenoisingConfig):
#     """This is forgetful causal masking, which we could layer onto S-Denoising, but don't for now"""
#     task_token = None
#     mask_prob = 0.15
#
#     def sample(self, key: PRNGKey, tokens: np.ndarray, sentinel_token_ids, task_token_id) -> Ul2Example:
#         """Build an F-denoiser example from a list of tokens"""
#         # This is the same as causal language model, but we mask some tokens
#         return Ul2Example(np.zeros((0,), dtype=int), np.array(tokens), task_token_id)


@dataclass(frozen=True)
class Ul2rConfig:
    shortcut: str | None = None
    task_configs: Dict[str, DenoisingConfig] | None = None
    task_probs: Dict[str, float] | None = None

    @property
    def task_configs_list(self):
        if self.task_configs is not None:
            task_configs = self.task_configs
        if self.shortcut == "ul2":
            task_configs = DenoisingConfig.ul2_configs()
        elif self.shortcut == "ul2r":
            task_configs = DenoisingConfig.ul2r_configs()
        else:
            raise ValueError("Either task_configs or shortcut should be specified.")

        return list(task_configs.values())

    @property
    def task_weights_list(self):
        if self.shortcut == "ul2r":
            task_probs = {"r": 0.25, "x1": 0.125, "x2": 0.125, "s": 0.5}
            return list(task_probs.values())
        elif self.task_probs is None:
            return None
        else:
            return [self.task_probs[task] for task in self.task_configs]

    def __post_init__(self):
        if self.task_configs is None and self.shortcut != "ul2" and self.shortcut != "ul2r":
            raise ValueError("if task_configs is not provided, shortcut must be either 'ul2' or 'ul2r'.")

        if self.task_probs is not None:
            if self.task_probs.keys() != self.task_configs.keys():
                raise ValueError("task_probs keys must match task_configs keys")

    def build(
        self,
        base_dataset: Dataset[np.ndarray],
        Pos: hax.Axis,
        KPos: hax.Axis,
        tokenizer: PreTrainedTokenizerBase,
        key: PRNGKey,
    ):
        task_configs = self.task_configs_list
        task_weights = self.task_weights_list
        logger.info(f"Building UL2 Datasest with configs {task_configs} and weights {task_weights}")

        return Ul2rDataset(
            base_dataset,
            Pos,
            KPos,
            key,
            tokenizer,
            task_configs,
            task_weights,
        )


class Ul2rDataset(ShardableDataset[LmExample]):
    def __init__(
        self,
        base_dataset: Dataset[np.ndarray],
        Pos: hax.Axis,
        KPos: hax.Axis,
        key: PRNGKey,
        tokenizer: PreTrainedTokenizerBase,
        task_configs: List[DenoisingConfig],
        task_probs: Optional[List[float]] = None,
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self.Pos = Pos
        self.KPos = KPos
        self.initial_key = key

        # reuse the last 1000 subwords as sentinel tokens
        # vocab_size = len(tokenizer.get_vocab())
        # sentinel_tokens = [tokenizer.decode(vocab_size - k) for k in range(1, 1001)]
        sentinel_tokens = [f"<sentinel_{k}>" for k in range(1000)]

        self.generator = Ul2InstanceGenerator(tokenizer, sentinel_tokens, task_configs, task_probs)
        self.tokenizer = tokenizer

    def __iter__(self) -> Iterator[LmExample]:
        key = self.initial_key
        sharding = jax.sharding.SingleDeviceSharding(jax.local_devices(backend="cpu")[0])
        key = jax.device_put(key, sharding)

        # to cpu
        with use_cpu_device():

            def _create_lm_example(tokens, key):
                this_key, key = jax.random.split(key)
                ul2example = self.generator.sample(tokens, this_key)
                decoder_only = convert_to_decoder_only(ul2example, self.tokenizer.pad_token_id, self.Pos, self.KPos)
                return decoder_only, key

            for tokens in self.base_dataset:
                example, key = _create_lm_example(tokens, key)
                yield example

    @property
    def item_shape(self) -> PyTree[Union[ShapeSpec, NamedShapeSpec]]:
        return LmExample(
            tokens=NamedShapeSpec((self.Pos,), jnp.int32),  # type: ignore
            loss_mask=NamedShapeSpec((self.Pos,), jnp.bool_),  # type: ignore
            attn_mask=NamedShapeSpec((self.Pos, self.KPos), jnp.bool_),  # type: ignore
        )

    def shard(self, shard_id: int, num_shards: int) -> "Ul2rDataset":
        return Ul2rDataset(
            self.base_dataset.shard(shard_id, num_shards),  # type: ignore
            self.Pos,
            self.KPos,
            jax.random.fold_in(self.initial_key, shard_id),
            self.tokenizer,
            self.generator.task_configs,
            self.generator.task_weights,
        )


class Ul2InstanceGenerator:
    """A generator for Ul2 instances. Class just because there's so much configuration"""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        sentinel_tokens: List[str],
        task_configs: List[DenoisingConfig],
        task_weights: Optional[List[float]],
    ):
        """Side effect warning: This constructor adds sentinel tokens and task tokens to the tokenizer"""
        self.tokenizer = tokenizer
        self.task_configs = task_configs
        if task_weights is not None:
            self.task_weights = np.array(task_weights) / np.sum(task_weights)
        else:
            self.task_weights = None

        self.tokenizer.add_tokens(sentinel_tokens, special_tokens=True)
        task_tokens = list(set([config.task_token for config in task_configs if config.task_token is not None]))
        self.tokenizer.add_tokens(task_tokens, special_tokens=True)

        self.sentinel_token_ids = np.array([self.tokenizer.convert_tokens_to_ids(token) for token in sentinel_tokens])

        self.denoiser_task_tokens = [
            self.tokenizer.convert_tokens_to_ids(config.task_token) for config in task_configs
        ]

    def sample(self, tokens: jnp.ndarray, key: PRNGKey) -> Ul2Example:
        """Generate a single Ul2Example from a string"""
        choice_key, key = jax.random.split(key)
        np_rng = np.random.default_rng(np.array(choice_key))

        task_id = np_rng.choice(np.arange(len(self.task_configs)), p=self.task_weights)
        task_config = self.task_configs[task_id]
        task_token_id = self.denoiser_task_tokens[task_id]

        return task_config.sample(key, tokens, self.sentinel_token_ids, task_token_id)


# from https://github.com/google-research/text-to-text-transfer-transformer/blob/main/t5/data/preprocessors.py
# Copyright 2022 The T5 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def random_spans_noise_mask(length, noise_density, key: PRNGKey, mean_noise_span_length=3.0, random_roll=False):
    """Noise mask consisting of random spans of noise tokens.
    The number of noise tokens and the number of noise spans and non-noise spans
    are determined deterministically as follows:
      num_noise_tokens = round(length * noise_density)
      num_nonnoise_spans = num_noise_spans = round(
         num_noise_tokens / mean_noise_span_length)
    Spans alternate between non-noise and noise, beginning with non-noise.
    Subject to the above restrictions, all masks are equally likely.
    Args:
      length: an int32 scalar (length of the incoming token sequence)
      noise_density: a float - approximate density of output mask
      seed: an int
      mean_noise_span_length: a number
      random_roll: bool, whether to roll the mask by a random integer offset in
        [0, length). Set random_roll to True to get a more uniform distribution
        of masked positions. Specifically, when random_roll is False (default) and
        a single span is enough to satisfy the noise density requirement, this
        fuction masks only the last few positions.
    Returns:
      a boolean tensor with shape [length]
    """
    if noise_density == 0.0:
        return np.zeros(length, bool)

    orig_length = length
    # increase length to avoid degeneracy
    length = max(length, 2)

    num_noise_tokens = int(round(float(length) * noise_density))
    # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
    num_noise_tokens = np.clip(num_noise_tokens, 1, length - 1)
    num_noise_spans = int(round(float(num_noise_tokens) / mean_noise_span_length))
    # avoid degeneracy by ensuring positive number of noise spans
    num_noise_spans = max(num_noise_spans, 1)
    num_nonnoise_tokens = length - num_noise_tokens

    np_rng = np.random.default_rng(np.array(key))

    # pick the lengths of the noise spans and the non-noise spans
    def _random_segmentation(num_items, num_segments):
        """Partition a sequence of items randomly into non-empty segments.
        Args:
          num_items: an integer scalar > 0
          num_segments: an integer scalar in [1, num_items]
          seed: an integer seed
        Returns:
          a Tensor with shape [num_segments] containing positive integers that add
          up to num_items
        """
        first_in_segment = np.pad(
            np_rng.permutation((np.arange(num_items - 1) < num_segments - 1).astype(int)), [[1, 0]]
        )
        segment_id = np.cumsum(first_in_segment)
        # segment_length = tf.math.segment_sum(tf.ones_like(segment_id), segment_id)
        segment_length = segment_sum(np.ones_like(segment_id), segment_id)

        return segment_length

    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)
    interleaved_span_lengths = np.reshape(
        np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
    )
    span_starts = np.cumsum(interleaved_span_lengths)[:-1]
    span_start_indicator = np.bincount(span_starts, minlength=length)
    span_num = np.cumsum(span_start_indicator)
    is_noise = np.equal(span_num % 2, 1)

    mask = is_noise[:orig_length]

    # if random_roll:
    #     roll_seed = (seeds[0][0] + seeds[1][1], seeds[0][1] - seeds[1][0])  # new seed.
    #     # Roll the mask by a random offset e.g. for offset=2: [1,2,3,4] => [3,4,1,2]
    #     offset = tf.random.stateless_uniform([1], seed=roll_seed, dtype=tf.int32, minval=0, maxval=length)[0]
    #     mask = tf.roll(mask, shift=offset, axis=0)

    if random_roll:
        offset = np_rng.integers(low=0, high=length, dtype=np.int32)
        mask = np.roll(mask, offset, axis=0)

    return mask


def segment_sum(data, segment_ids):
    data = np.asarray(data)
    s = np.zeros((np.max(segment_ids) + 1,) + data.shape[1:], dtype=data.dtype)
    np.add.at(s, segment_ids, data)
    return s


def noise_span_to_unique_sentinel(tokens, noise_mask, sentinel_tokens):
    """Replace each run of consecutive noise tokens with a different sentinel.
    The idea here is to be able to align the dropped spans in the inputs
    with the markers in the targets.
    We want to generate training examples like
    "We hold X to be Y that" -> "X these truths Y self evident Z"

    Args:
      tokens: a 1d integer Tensor
      noise_mask: a boolean Tensor with the same shape as tokens
      sentinel_tokens: an integer Tensor array of token ids
    Returns:
      a Tensor with the same shape and dtype as tokens
    """

    prev_token_is_noise = np.pad(noise_mask[:-1], [[1, 0]])

    first_noise_tokens = np.logical_and(noise_mask, np.logical_not(prev_token_is_noise))
    subsequent_noise_tokens = np.logical_and(noise_mask, prev_token_is_noise)

    segments = np.cumsum(first_noise_tokens.astype(tokens.dtype))
    if int(np.max(segments)) > len(sentinel_tokens):
        warnings.warn(f"Too many noise spans, reusing sentinels: {int(np.max(segments))} > {len(sentinel_tokens)}")
    sentinel = sentinel_tokens[segments % len(sentinel_tokens)]

    tokens = np.where(first_noise_tokens, sentinel, tokens)
    return tokens[np.logical_not(subsequent_noise_tokens)]


def nonnoise_span_to_unique_sentinel(tokens, noise_mask, sentinel_tokens):
    return noise_span_to_unique_sentinel(tokens, np.logical_not(noise_mask), sentinel_tokens)
