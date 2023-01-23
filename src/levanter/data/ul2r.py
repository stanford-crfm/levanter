# Routines for creating UL2R-type objectives: R-denoisers, X-denoisers, S-denoisers
# See https://arxiv.org/pdf/2210.11399v2.pdf and https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/data/language_modeling/megatron/ul2_dataset.py
import collections
from dataclasses import dataclass
from typing import Iterator, List, Optional, Union

import jax
import numpy as np
import tensorflow as tf
from jax.random import PRNGKey
from jaxtyping import PyTree
from transformers import AutoTokenizer, PreTrainedTokenizerBase

import haliax as hax
from levanter.data.dataset import Dataset, T
from levanter.shapes import NamedShapeSpec, ShapeSpec


@dataclass
class Ul2Example:
    """A single example for the UL2 Objective for an encoder-decoder objective.
    In general, these should start out unpadded/untruncated"""

    task_token: int
    inputs: np.ndarray
    outputs: np.ndarray

    def render(self, tokenizer: PreTrainedTokenizerBase):
        """renders a pretty version of the example as a string"""
        return (
            tokenizer.decode(self.task_token)
            + " "
            + tokenizer.decode(self.inputs)
            + "\n<!TARGETS!>\n"
            + tokenizer.decode(self.outputs)
        )


@dataclass
class DecoderOnlyExample:
    tokens: hax.NamedArray
    targets: hax.NamedArray
    attention_mask: hax.NamedArray
    loss_mask: hax.NamedArray


def convert_to_decoder_only(example: Ul2Example, tokenizer, QLen: hax.Axis, KLen: hax.Axis):
    all_tokens = [example.task_token] + example.inputs.tolist() + [example.outputs.tolist()]
    all_tokens = np.array(all_tokens, dtype=np.int64)
    all_tokens = hax.named(all_tokens, QLen)

    # shift back by one since we're predicting the next token
    targets = hax.roll(all_tokens, -1, QLen)

    initial_length = len(all_tokens)

    max_seq_len = QLen.size

    # pad or truncate
    if len(all_tokens) > max_seq_len:
        all_tokens = all_tokens[:max_seq_len]
    elif len(all_tokens) < max_seq_len:
        num_padding = max_seq_len - len(all_tokens)
        all_tokens = np.pad(all_tokens, (0, num_padding), constant_values=tokenizer.pad_token_id)

    # Create attention masks
    attention_mask = hax.nn.attention.prefix_lm_mask(QLen, KLen, len(example.inputs) + 1)

    # don't compute loss on:
    # 1) task token
    # 2) inputs (except last token of inputs)
    loss_mask = hax.arange(QLen) >= len(example.inputs)
    # 3) last token of targets
    loss_mask = loss_mask & (hax.arange(QLen) < initial_length - 1)
    # 4) padding
    loss_mask = loss_mask & (all_tokens != tokenizer.pad_token_id)

    return DecoderOnlyExample(all_tokens, targets, attention_mask, loss_mask)


S_TASK_TOKEN = "[S2S]"
R_TASK_TOKEN = "[NLU]"
X_TASK_TOKEN = "[NLG]"


@dataclass
class DenoisingTaskConfig:
    task_token: str
    mask_prob: float = 0.15  # r in the paper
    mean_span_length: float = 3.0  # mu in the paper

    # we don't use their "n" parameter because they only use it for prefix-lm

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
    def ul2r_configs(r_task_token: str = "[NLU]", x_task_token: str = "[NLG]") -> List["DenoisingTaskConfig"]:
        return [
            DenoisingTaskConfig(r_task_token, 0.15, 3.0),
            # TODO: S-denoiser
            DenoisingTaskConfig(x_task_token, 0.15, 32.0),
            DenoisingTaskConfig(x_task_token, 0.5, 3.0),
        ]


# the UL2 paper suggests using L/4 for the mean length,
DEFAULT_S_DENOISER_CONFIG = DenoisingTaskConfig(S_TASK_TOKEN, 1.0, 512.0)


class Ul2rDataset(Dataset[DecoderOnlyExample]):
    def __init__(
        self,
        base_dataset: Dataset[List[int]],
        SeqLen: hax.Axis,
        KSeqLen: hax.Axis,
        key: PRNGKey,
        tokenizer: PreTrainedTokenizerBase,
        task_configs: List[DenoisingTaskConfig],
        s_denoiser_config: DenoisingTaskConfig = DEFAULT_S_DENOISER_CONFIG,
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

    def __iter__(self) -> Iterator[T]:
        key = self.initial_key
        for example in self.base_dataset:
            key, subkey = jax.random.split(key)
            ul2example = self.generator.sample(subkey, example)
            decoder_only = convert_to_decoder_only(ul2example, self.tokenizer, self.SeqLen, self.KSeqLen)
            yield decoder_only

    @property
    def item_shape(self) -> PyTree[Union[ShapeSpec, NamedShapeSpec]]:
        return DecoderOnlyExample(
            tokens=(self.SeqLen,),  # type: ignore
            targets=(self.SeqLen,),  # type: ignore
            attention_mask=(self.SeqLen, self.KSeqLen),  # type: ignore
            loss_mask=(self.SeqLen,),  # type: ignore
        )

    def __len__(self) -> int:
        return len(self.base_dataset)


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

    def sample(self, tokens: List[int], key: PRNGKey) -> Ul2Example:
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

    def sample_s_denoiser(self, tokens: List[int], key: PRNGKey) -> Ul2Example:
        """Build an S-denoiser example from a list of tokens"""
        # choose a random length
        np_rng = np.random.default_rng(np.array(key).tolist())
        pivot = int(np_rng.integers(0, len(tokens) + 1))
        return Ul2Example(self.s_denoiser_token_id, np.array(tokens[:-pivot]), np.array(tokens[-pivot:]))  # type: ignore

    def sample_denoiser(self, tokens: List[int], task_config: DenoisingTaskConfig, key: PRNGKey) -> Ul2Example:
        """Build a denoiser example from a list of tokens"""
        # choose a random length
        # Masking.
        seeds = tf.convert_to_tensor(np.array(jax.random.split(key, 2)))  # needs to be 2x2
        noise_mask = random_spans_noise_mask(len(tokens), task_config.mask_prob, seeds, task_config.mean_span_length)
        tokens = tf.convert_to_tensor(tokens)
        inputs = noise_span_to_unique_sentinel(tokens, noise_mask, self.sentinel_token_ids)
        targets = nonnoise_span_to_unique_sentinel(tokens, noise_mask, self.sentinel_token_ids)

        return Ul2Example(self.tokenizer.convert_tokens_to_ids(task_config.task_token), inputs, targets)

        # max_predictions_per_seq = task_config.mask_prob * len(tokens)
        # np_rng = np.random.default_rng(np.array(key))
        # output_tokens, masked_spans = create_masked_lm_predictions(
        #     tokens=tokens,
        #     masked_lm_prob=task_config.mask_prob,
        #     mask_id=self.tokenizer.mask_token_id,
        #     span_length_distribution=task_config.length_distribution,
        #     max_predictions_per_seq=max_predictions_per_seq,
        #     np_rng=np_rng,
        #     max_ngram_size=1024
        # )
        #
        # t5_input, t5_output = self._create_t5_input_output_from_spans(output_tokens, masked_spans)

        # return Ul2Example(self.tokenizer.convert_tokens_to_ids(task_config.task_token), t5_input, t5_output)

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


def random_spans_noise_mask(length, noise_density, seeds, mean_noise_span_length=3.0, random_roll=False):
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
      seeds: an int32 Tensor, shaped (2, 2)
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
        return tf.zeros(length, bool)

    orig_length = length
    # increase length to avoid degeneracy
    length = tf.maximum(length, 2)

    def to_int(x):
        return tf.cast(x, np.int32)

    def to_float(x):
        return tf.cast(x, np.float32)

    num_noise_tokens = to_int(np.round(to_float(length) * noise_density))
    # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
    num_noise_tokens = np.minimum(np.maximum(num_noise_tokens, 1), length - 1)
    num_noise_spans = to_int(np.round(to_float(num_noise_tokens) / mean_noise_span_length))
    # avoid degeneracy by ensuring positive number of noise spans
    num_noise_spans = np.maximum(num_noise_spans, 1)
    num_nonnoise_tokens = length - num_noise_tokens

    # pick the lengths of the noise spans and the non-noise spans
    def _random_segmentation(num_items, num_segments, seed):
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
            tf.random.shuffle(to_int(tf.range(num_items - 1) < num_segments - 1), seed=int(seed[0])), [[1, 0]]
        )
        segment_id = tf.cumsum(first_in_segment)
        segment_length = tf.math.segment_sum(tf.ones_like(segment_id), segment_id)
        return segment_length

    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans, seeds[0])
    nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans, seeds[1])
    interleaved_span_lengths = tf.reshape(
        tf.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
    )
    span_starts = tf.cumsum(interleaved_span_lengths)[:-1]
    span_start_indicator = tf.math.unsorted_segment_sum(tf.ones_like(span_starts), span_starts, length)
    span_num = tf.cumsum(span_start_indicator)
    is_noise = tf.equal(span_num % 2, 1)

    mask = is_noise[:orig_length]

    if random_roll:
        roll_seed = (seeds[0][0] + seeds[1][1], seeds[0][1] - seeds[1][0])  # new seed.
        # Roll the mask by a random offset e.g. for offset=2: [1,2,3,4] => [3,4,1,2]
        offset = tf.random.stateless_uniform([1], seed=roll_seed, dtype=tf.int32, minval=0, maxval=length)[0]
        mask = tf.roll(mask, shift=offset, axis=0)

    return mask


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

    prev_token_is_noise = tf.pad(noise_mask[:-1], [[1, 0]])

    first_noise_tokens = tf.logical_and(noise_mask, tf.logical_not(prev_token_is_noise))
    subsequent_noise_tokens = tf.logical_and(noise_mask, prev_token_is_noise)

    # sentinel = sentinel_id(tokenizer) + 1 - tf.cumsum(tf.cast(first_noise_tokens, tokens.dtype))
    segments = tf.cumsum(tf.cast(first_noise_tokens, tokens.dtype))
    assert int(tf.reduce_max(segments)) <= len(sentinel_tokens)
    sentinel = sentinel_tokens[segments]

    tokens = tf.where(first_noise_tokens, sentinel, tokens)
    return tf.boolean_mask(tokens, tf.logical_not(subsequent_noise_tokens))


# note(dlwh): this is some ninja sh*t right here.
def nonnoise_span_to_unique_sentinel(tokens, noise_mask, sentinel_tokens):
    return noise_span_to_unique_sentinel(tokens, tf.logical_not(noise_mask), sentinel_tokens)


if __name__ == "__main__":
    import time

    from levanter.data.text import TokenSeqDataset

    dataset = TokenSeqDataset.load("cache/train", 2048)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    tokenizer.add_special_tokens({"additional_special_tokens": ["<mask>"]})
    tokenizer.mask_token = "<mask>"

    ul2_generator = Ul2InstanceGenerator(
        tokenizer,
        [f"<mask_{i}>" for i in range(2048)],  # this is excessive but will work
        DenoisingTaskConfig.ul2r_configs(),
    )

    time_in = time.time()

    for i, tokens in zip(range(1000), dataset):
        # print(tokens)
        # print(tokenizer.decode(tokens))
        #
        # print(ul2_generator.sample(tokens, jax.random.PRNGKey(i)).render(tokenizer))
        ul2_generator.sample(tokens, jax.random.PRNGKey(i))  # type: ignore

    print(time.time() - time_in)
