# Routines for creating UL2R-type objectives: R-denoisers, X-denoisers, S-denoisers
# See https://arxiv.org/pdf/2210.11399v2.pdf
import collections
import dataclasses
import logging
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Union

import equinox as eqx
import jax
import numpy as np
from jax.random import PRNGKey
from jaxtyping import PyTree
from transformers import PreTrainedTokenizerBase

import haliax as hax
from levanter.data.dataset import Dataset, ShardableDataset
from levanter.shapes import NamedShapeSpec, ShapeSpec


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


class DecoderOnlyExample(eqx.Module):
    tokens: np.ndarray
    targets: np.ndarray
    attn_mask: np.ndarray
    loss_mask: np.ndarray

    def to_named(self, QLen: hax.Axis, KLen: hax.Axis) -> "NamedLmExample":
        return NamedLmExample(
            hax.named(self.tokens, QLen),
            hax.named(self.targets, QLen),
            hax.named(self.attn_mask, (QLen, KLen)),
            hax.named(self.loss_mask, QLen),
        )


class NamedLmExample(eqx.Module):
    tokens: hax.NamedArray
    targets: hax.NamedArray
    attn_mask: hax.NamedArray
    loss_mask: hax.NamedArray


def convert_to_decoder_only(example: Ul2Example, pad_token_id, max_seq_len: int) -> DecoderOnlyExample:
    all_tokens = []
    if example.task_token is not None:
        all_tokens.append(example.task_token)
    all_tokens.extend(example.inputs)
    input_length = len(all_tokens)

    all_tokens.extend(example.outputs)
    all_tokens = np.asarray(all_tokens, dtype=np.int32)

    unpadded_length = len(all_tokens)

    QLen = hax.Axis("QLen", max_seq_len)
    KLen = hax.Axis("KLen", max_seq_len)

    # pad or truncate
    if len(all_tokens) > max_seq_len:
        # take from the beginning
        all_tokens = all_tokens[:max_seq_len]
        input_length = min(input_length, max_seq_len)
        unpadded_length = max_seq_len
        assert input_length <= unpadded_length
    elif len(all_tokens) < max_seq_len:
        num_padding = max_seq_len - len(all_tokens)
        all_tokens = np.pad(all_tokens, (0, num_padding), constant_values=pad_token_id)

    # do on the cpu to avoid dumb roundtrips to gpu/tpu
    with jax.default_device(jax.devices("cpu")[0]):
        # shift back by one since we're predicting the next token
        targets = np.roll(all_tokens, -1)

        attention_mask = hax.nn.attention.prefix_lm_mask(QLen, KLen, input_length).rearrange((QLen, KLen)).array
        loss_mask = _create_decoder_loss_mask(max_seq_len, input_length, unpadded_length)
        assert np.sum(loss_mask) == unpadded_length - input_length

    return DecoderOnlyExample(all_tokens, targets, attention_mask, loss_mask)


def _create_decoder_loss_mask(max_seq_len, input_length, unpadded_length):
    # don't compute loss on:
    # 1) task token
    # 2) inputs (except last token of inputs)
    loss_mask = np.arange(max_seq_len) >= input_length - 1
    # 3) last token of targets
    loss_mask = loss_mask & (np.arange(max_seq_len) < unpadded_length - 1)

    return loss_mask


S_TASK_TOKEN = "[S2S]"
R_TASK_TOKEN = "[NLU]"
X_TASK_TOKEN = "[NLG]"


@dataclass
class DenoisingTaskConfig:
    task_token: Optional[str]
    mask_prob: float = 0.15  # r in the paper
    mean_span_length: float = 3.0  # mu in the paper
    # we don't use their "n" parameter because they only use it for prefix-lm

    def with_task_token(self, task_token: Optional[str]):
        return dataclasses.replace(self, task_token=task_token)

    @staticmethod
    def ul2_configs(r_task_token: str = R_TASK_TOKEN, x_task_token: str = X_TASK_TOKEN) -> List["DenoisingTaskConfig"]:
        return [
            DenoisingTaskConfig(r_task_token, 0.15, 3.0),
            DenoisingTaskConfig(r_task_token, 0.15, 8.0),
            DenoisingTaskConfig(x_task_token, 0.5, 3.0),
            DenoisingTaskConfig(x_task_token, 0.5, 8.0),
            DenoisingTaskConfig(x_task_token, 0.15, 64.0),
            DenoisingTaskConfig(x_task_token, 0.5, 64.0),
        ]

    @staticmethod
    def ul2r_configs(
        r_task_token: Optional[str] = "[NLU]", x_task_token: Optional[str] = "[NLG]"
    ) -> List["DenoisingTaskConfig"]:
        return [
            DenoisingTaskConfig(r_task_token, 0.15, 3.0),
            DenoisingTaskConfig(x_task_token, 0.15, 32.0),
            DenoisingTaskConfig(x_task_token, 0.5, 3.0),
        ]


# the UL2 paper suggests using L/4 for the mean length,
DEFAULT_S_DENOISER_CONFIG = DenoisingTaskConfig(S_TASK_TOKEN, 1.0, 512.0)


class Ul2rDataset(ShardableDataset[NamedLmExample]):
    def __init__(
        self,
        base_dataset: Dataset[Sequence[int]],
        SeqLen: hax.Axis,
        KSeqLen: hax.Axis,
        key: PRNGKey,
        tokenizer: PreTrainedTokenizerBase,
        task_configs: List[DenoisingTaskConfig],
        s_denoiser_config: Optional[DenoisingTaskConfig] = DEFAULT_S_DENOISER_CONFIG,
        s_denoiser_prob: float = 0.5,
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self.SeqLen = SeqLen
        self.KSeqLen = KSeqLen
        self.initial_key = key

        sentinel_tokens = [
            f"<sentinel_{k}>" for k in range(1000)
        ]  # if we need more than 1000, we have bigger problems

        self.generator = Ul2InstanceGenerator(
            tokenizer, sentinel_tokens, task_configs, s_denoiser_config, s_denoiser_prob
        )
        self.tokenizer = tokenizer

    def __iter__(self) -> Iterator[NamedLmExample]:
        key = self.initial_key
        for example in self.base_dataset:
            key, subkey = jax.random.split(key)
            ul2example = self.generator.sample(example, key=subkey)
            try:
                decoder_only = convert_to_decoder_only(
                    ul2example,
                    self.tokenizer.pad_token_id,
                    self.SeqLen.size,
                ).to_named(self.SeqLen, self.KSeqLen)
            except AssertionError:
                print(ul2example.render(self.tokenizer))
                raise

            yield decoder_only

    @property
    def item_shape(self) -> PyTree[Union[ShapeSpec, NamedShapeSpec]]:
        return DecoderOnlyExample(
            tokens=NamedShapeSpec((self.SeqLen,)),  # type: ignore
            targets=NamedShapeSpec((self.SeqLen,)),  # type: ignore
            attn_mask=NamedShapeSpec((self.SeqLen, self.KSeqLen)),  # type: ignore
            loss_mask=NamedShapeSpec((self.SeqLen,)),  # type: ignore
        )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def shard(self, shard_id: int, num_shards: int) -> "ShardableDataset[NamedLmExample]":
        return Ul2rDataset(
            self.base_dataset.shard(shard_id, num_shards),  # type: ignore
            self.SeqLen,
            self.KSeqLen,
            self.initial_key,
            self.tokenizer,
            self.generator.task_configs,
            self.generator.s_denoiser_config,
            self.generator.s_denoiser_prob,
        )


class Ul2InstanceGenerator:
    """A generator for Ul2 instances. Class just because there's so much configuration"""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        sentinel_tokens: List[str],
        task_configs: List[DenoisingTaskConfig],
        s_denoiser_config: Optional[DenoisingTaskConfig] = DEFAULT_S_DENOISER_CONFIG,
        s_denoiser_prob: float = 0.5,
    ):
        """Side effect warning: This constructor adds sentinel tokens and task tokens to the tokenizer"""
        self.tokenizer = tokenizer
        self.task_configs = task_configs
        self.s_denoiser_config = s_denoiser_config
        self.s_denoiser_prob = s_denoiser_prob

        self.tokenizer.add_tokens(sentinel_tokens, special_tokens=True)
        task_tokens = list(set([config.task_token for config in task_configs]))
        if s_denoiser_config is not None:
            task_tokens.append(s_denoiser_config.task_token)
        self.tokenizer.add_tokens(task_tokens, special_tokens=True)

        self.sentinel_token_ids = np.array([self.tokenizer.convert_tokens_to_ids(token) for token in sentinel_tokens])

        if s_denoiser_config is not None:
            self.s_denoiser_token_id = self.tokenizer.convert_tokens_to_ids(s_denoiser_config.task_token)
        else:
            self.s_denoiser_token_id = None

        self.denoiser_task_tokens = [
            self.tokenizer.convert_tokens_to_ids(config.task_token) for config in task_configs
        ]

    def sample(self, tokens: Sequence[int], key: PRNGKey) -> Ul2Example:
        """Generate a single Ul2Example from a string"""
        # first decide if we're doing S-denoiser or not
        # gonna be lazy with keys here
        choice_key, key = jax.random.split(key)
        np_rng = np.random.default_rng(np.array(choice_key))

        if self.s_denoiser_config is not None and np_rng.uniform() < self.s_denoiser_prob:
            return self.sample_s_denoiser(tokens, key)

        # otherwise, pick a denoiser task
        # TODO: do we want to support weights here?
        task_config = np_rng.choice(self.task_configs)

        return self.sample_denoiser(tokens, task_config, key)

    def sample_s_denoiser(self, tokens: Sequence[int], key: PRNGKey) -> Ul2Example:
        """Build an S-denoiser example from a list of tokens"""
        # choose a random length
        np_rng = np.random.default_rng(np.array(key).tolist())
        pivot = int(np_rng.integers(0, len(tokens) + 1))
        return Ul2Example(np.array(tokens[:-pivot]), np.array(tokens[-pivot:]), self.s_denoiser_token_id)  # type: ignore

    def sample_denoiser(self, tokens: Sequence[int], task_config: DenoisingTaskConfig, key: PRNGKey) -> Ul2Example:
        """Build a denoiser example from a list of tokens"""
        # choose a random length
        # Masking.
        noise_mask = random_spans_noise_mask(len(tokens), task_config.mask_prob, key, task_config.mean_span_length)
        tokens = np.array(tokens)
        inputs = noise_span_to_unique_sentinel(tokens, noise_mask, self.sentinel_token_ids)
        targets = nonnoise_span_to_unique_sentinel(tokens, noise_mask, self.sentinel_token_ids)

        return Ul2Example(inputs, targets, self.tokenizer.convert_tokens_to_ids(task_config.task_token))

    def _create_t5_input_output_from_spans(self, output_tokens, masked_spans):
        sentinel_tokens = collections.deque(self.sentinel_token_ids)
        t5_input = []
        t5_output = []
        (start_index, end_index) = (0, None)
        if masked_spans is not None:
            for span in masked_spans:
                try:
                    flag = sentinel_tokens.popleft()
                except IndexError:
                    raise ValueError("Too many spans to mask")

                # Append the same tokens in decoder input and output
                t5_output.append(flag)
                t5_output.extend(span.label)

                end_index = span.index[0]
                t5_input.extend(output_tokens[start_index:end_index])
                t5_input.append(flag)

                # the next start index is the token after the last span token
                start_index = span.index[-1] + 1
        # Add the remaining tokens to the t5 input
        t5_input.extend(output_tokens[start_index:])
        return t5_input, t5_output


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
        logging.warning("Too many noise spans, reusing sentinels")
    sentinel = sentinel_tokens[segments % len(sentinel_tokens)]

    tokens = np.where(first_noise_tokens, sentinel, tokens)
    return tokens[np.logical_not(subsequent_noise_tokens)]


def nonnoise_span_to_unique_sentinel(tokens, noise_mask, sentinel_tokens):
    return noise_span_to_unique_sentinel(tokens, np.logical_not(noise_mask), sentinel_tokens)
