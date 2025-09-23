from dataclasses import dataclass
from haliax import Axis
from jaxtyping import PRNGKeyArray
from typing import Literal, Optional, Dict
import dataclasses
import draccus
import equinox as eqx
import functools
import jax
import jax.numpy as jnp
import numpy as np
import typing
import haliax as hax

from levanter.data.dataset import MappedAsyncDataset
from levanter.data.packing import GreedyPrepackedDataset
from levanter.models.lm_model import LmExample
from levanter.store.cache import TreeCache


# From https://huggingface.co/stanford-crfm/marin-tokenizer/blob/main/tokenizer.json
R_TASK_TOKEN_ID = 128011
R_TASK_TOKEN = "<|reserved_special_token_3|>"
X_TASK_TOKEN_ID = 128012
X_TASK_TOKEN = "<|reserved_special_token_4|>"
S_TASK_TOKEN_ID = 128013
S_TASK_TOKEN = "<|reserved_special_token_5|>"


# Use the last 100 reserved token IDs as sentinel tokens (equivalent to T5's
# extra_ids); <|reserved_special_token_156|> to 255
# https://github.com/google-research/text-to-text-transfer-transformer/blob/e081111ccd51df425aa7124e8e311ed6d403d767/t5/models/gin/objectives/span.gin#L46-L47
num_token_ids = 128256
num_sentinels = 100
SENTINEL_TOKEN_IDS = list(range(num_token_ids - num_sentinels, num_token_ids))


@dataclass(frozen=True)
class DenoisingConfig(draccus.ChoiceRegistry):
    task_token: Optional[str] = None

    def with_task_token(self, task_token: Optional[str]) -> "DenoisingConfig":
        return dataclasses.replace(self, task_token=task_token)

    @staticmethod
    def ul2_configs(
        r_task_token: str = R_TASK_TOKEN,
        x_task_token: str = X_TASK_TOKEN,
        s_task_token: str = S_TASK_TOKEN,
    ) -> typing.Dict[str, "DenoisingConfig"]:
        # Table 1 https://arxiv.org/pdf/2205.05131
        return {
            "r1": RDenoisingConfig(r_task_token, 0.15, 3.0),
            "r2": RDenoisingConfig(r_task_token, 0.15, 8.0),
            "x1": XDenoisingConfig(x_task_token, 0.5, 3.0),
            "x2": XDenoisingConfig(x_task_token, 0.5, 8.0),
            "x3": XDenoisingConfig(x_task_token, 0.15, 64.0),
            "x4": XDenoisingConfig(x_task_token, 0.5, 64.0),
            "s": SDenoisingConfig(s_task_token),
        }

    @staticmethod
    def ul2r_configs(
        r_task_token: Optional[str] = R_TASK_TOKEN,
        x_task_token: Optional[str] = X_TASK_TOKEN,
        s_task_token: str = S_TASK_TOKEN,
    ) -> Dict[str, "DenoisingConfig"]:
        # Section 3.3 Loss Objectives https://arxiv.org/pdf/2210.11399v2
        return {
            "r": RDenoisingConfig(r_task_token, 0.15, 3.0, True),
            "x1": XDenoisingConfig(x_task_token, 0.15, 32.0, True),
            "x2": XDenoisingConfig(x_task_token, 0.5, 3.0, True),
            "s": SDenoisingConfig(s_task_token),
        }

    def to_parameter_tensor(self) -> jnp.ndarray:
        raise NotImplementedError("Not implemented")

    def task_token_id(self) -> int:
        raise NotImplementedError("Not implemented")

    def pad_token_id(self) -> int:
        return 0


@dataclass(frozen=True)
class MaskDenoisingConfig(DenoisingConfig):
    task_kind = 0
    mask_prob: float = 0.15  # r in the paper
    mean_span_length: float = 3.0  # mu in the paper
    random_roll: bool = False

    def to_parameter_tensor(self) -> jnp.ndarray:
        args = [self.task_kind, self.mask_prob, self.mean_span_length]
        return jnp.array(args)


@DenoisingConfig.register_subclass("x")
@dataclass(frozen=True)
class XDenoisingConfig(MaskDenoisingConfig):
    task_token: Optional[str] = X_TASK_TOKEN
    mask_prob: float = 0.5
    mean_span_length: float = 3.0

    # TODO This needs to integrate with the actual tokenizer.
    def task_token_id(self) -> int:
        return X_TASK_TOKEN_ID


@DenoisingConfig.register_subclass("r")
@dataclass(frozen=True)
class RDenoisingConfig(MaskDenoisingConfig):
    task_token: Optional[str] = R_TASK_TOKEN
    mask_prob: float = 0.15
    mean_span_length: float = 3.0

    def task_token_id(self) -> int:
        return R_TASK_TOKEN_ID


@DenoisingConfig.register_subclass("s")
@dataclass(frozen=True)
class SDenoisingConfig(DenoisingConfig):
    task_kind = 1
    task_token: Optional[str] = S_TASK_TOKEN

    def to_parameter_tensor(self) -> jnp.ndarray:
        # should match size of RDenoisingConfig / XDenoisingConfig
        return jnp.array([self.task_kind, 0, 0])

    def task_token_id(self) -> int:
        return S_TASK_TOKEN_ID


@functools.partial(jax.jit, static_argnames=["padded_length"])
def random_segmentation_jax(
    num_items: int, num_segments: int, key: PRNGKeyArray, padded_length: int
) -> jnp.ndarray:
    """
    Generates a random partition of `num_items` items into `num_segments`
    segments described by their lengths. The length of the returned tensor is
    `padded_length`.

    Based on `_random_segmentation` in `random_spans_noise_mask` in T5:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/e081111ccd51df425aa7124e8e311ed6d403d767/t5/data/preprocessors.py#L2992-L3010


    Precondition:
        1 <= num_segments < num_items

    Returns:
        A tensor of segment lengths with shape `(padded_length,)` where the
        first `num_segments` values sum to `num_items` and all other values are
        zero.
    """

    indices = jnp.arange(padded_length)

    perm = jax.random.permutation(key, indices)

    # Threshold is the (num_segments - 1)th smallest value in the first
    # num_items - 1 elements
    perm_masked = jnp.where(indices < num_items - 1, perm, padded_length)
    sorted_perm = jnp.sort(perm_masked)
    threshold = sorted_perm[num_segments - 1]

    # 1 where perm < threshold; so first num_items-1 elements are a random 01
    # tensor, everything else is 0
    first_in_segment = jnp.where(perm < threshold, 1, 0)

    # Roll right by 1; guarantees first element is 0
    first_in_segment = jnp.roll(first_in_segment, 1)

    # Set position num_items to 1 (marks end of all segments)
    first_in_segment = first_in_segment.at[num_items].set(1)

    segment_id = jnp.cumsum(first_in_segment)
    segment_length = jax.ops.segment_sum(
        jnp.ones_like(segment_id), segment_id, padded_length
    )

    return segment_length


@jax.jit
def num_noise_spans_tokens_and_spans(
    length: int, noise_density: float, mean_noise_span_length: float
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    adjusted_length = jnp.maximum(length, 2)

    num_noise_tokens = jnp.round(adjusted_length * noise_density).astype(jnp.int32)
    num_noise_tokens = jnp.clip(num_noise_tokens, 1, adjusted_length - 1)
    num_noise_spans = jnp.maximum(
        jnp.round(num_noise_tokens / mean_noise_span_length).astype(jnp.int32), 1
    )
    num_nonnoise_tokens = adjusted_length - num_noise_tokens

    return num_noise_tokens, num_noise_spans, num_nonnoise_tokens


@functools.partial(jax.jit, static_argnames=["padded_length"])
def random_spans_noise_mask_jax(
    length: int,
    noise_density: float,
    key: PRNGKeyArray,
    mean_noise_span_length: float,
    random_roll: bool,
    padded_length: int,
) -> jnp.ndarray:
    """
    Generates a random 1D Boolean mask tensor where `noise_density` gives the
    fraction of tokens that are 1s, occurring in runs of length
    `mean_noise_span_length`. You must use `random_roll` to make each mask
    equally likely; otherwise the distribution is not uniform and the mask will
    have a prefix of 0s.

    Only the first `length` tokens describe the mask; the contents of the
    remaining `padded_length - length` tokens are 0s.

    Based on `random_spans_noise_mask` in T5:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/e081111ccd51df425aa7124e8e311ed6d403d767/t5/data/preprocessors.py#L2930-L3039

    Returns:
        Boolean mask tensor of shape (padded_length,).
    """
    num_noise_tokens, num_noise_spans, num_nonnoise_tokens = (
        num_noise_spans_tokens_and_spans(length, noise_density, mean_noise_span_length)
    )

    key1, key2, key3 = jax.random.split(key, 3)

    noise_span_lengths = random_segmentation_jax(
        num_noise_tokens, num_noise_spans, key1, padded_length
    )
    nonnoise_span_lengths = random_segmentation_jax(
        num_nonnoise_tokens, num_noise_spans, key2, padded_length
    )

    # Interleave using reshape
    interleaved_span_lengths = jnp.reshape(
        jnp.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
        [padded_length * 2],
    )[:padded_length]

    # Create span_start_indicator using bincount
    span_starts = jnp.cumsum(interleaved_span_lengths)
    span_start_indicator = jnp.bincount(span_starts, length=padded_length)

    span_num = jnp.cumsum(span_start_indicator)
    # Only odd spans less than 2*num_noise_spans are noise
    is_noise = ((span_num % 2) == 1) & (span_num < 2 * num_noise_spans)
    is_noise = is_noise.astype(jnp.bool_)

    # Zero everything at length and after
    indices = jnp.arange(padded_length)
    is_noise = jnp.where(indices < length, is_noise, False)
    is_noise = typing.cast(jnp.ndarray, is_noise)

    def apply_roll(m):
        offset = jax.random.randint(key3, (), 0, length, dtype=jnp.int32)
        # Roll the mask
        rolled = jnp.roll(m, offset, axis=0)
        # Mask out values that wrapped around from beyond length
        # After rolling by offset, positions [0, offset) contain what was at
        # [length, length+offset)
        # We need to zero these out since they're from beyond the valid range
        rolled = jnp.where(
            indices < offset,
            m[jnp.minimum(length + indices, padded_length - 1)],
            rolled,
        )
        rolled = typing.cast(jnp.ndarray, rolled)
        rolled = jnp.where(indices < length, rolled, False)
        return rolled

    mask = jax.lax.cond(random_roll, apply_roll, lambda m: m, is_noise)

    return mask


@jax.jit
def noise_span_to_unique_sentinel_jax(
    tokens: jnp.ndarray,
    noise_mask: jnp.ndarray,
    sentinel_tokens: jnp.ndarray,
    length: int,
    pad_token_id: int,
) -> jnp.ndarray:
    """
    Replace each run of consecutive noise tokens with a different sentinel.
    `length` must be the true length of `tokens`, excluding padding.

    For example:

        tokens = "The longest river in the world is the Amazon"
        noise_mask = [0, 1, 0, ...]
        noise_span_to_unique_sentinel_jax(...) =
            "The <sentinel_0> river in the world is the Amazon <sentinel_0> Amazon"

    Based on `noise_span_to_unique_sentinel` in T5:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/e081111ccd51df425aa7124e8e311ed6d403d767/t5/data/preprocessors.py#L3141

    Returns:
        A tensor with the same shape and dtype as `tokens`.
    """
    padded_length = tokens.shape[0]

    # Identify first noise tokens in each span
    prev_token_is_noise = jnp.roll(noise_mask, 1)
    prev_token_is_noise = prev_token_is_noise.at[0].set(False)
    first_noise_tokens = noise_mask & ~prev_token_is_noise
    subsequent_noise_tokens = noise_mask & prev_token_is_noise

    # Assign segment IDs to each noise span (subtract 1 to make it 0-indexed)
    segments = jnp.cumsum(first_noise_tokens.astype(jnp.int32)) - 1
    max_segments = jnp.max(segments) + 1  # Number of unique noise spans

    # If max_segments > len(sentinel_tokens) we will reuse sentinel tokens which
    # isn't good. Ideally we'd log a warning but we can't do that inside of
    # jax.jit.
    # TODO Warn in the non-JIT wrapper?
    # checkify.checkify(
    #     lambda: checkify.check(
    #         max_segments <= len(sentinel_tokens),
    #         f"Too many noise spans: {max_segments} > {len(sentinel_tokens)}",
    #     ),
    #     errors=checkify.index_checks,
    # )()

    def loop_body(read_pos, state):
        result_arr, write_pos = state

        # Check if current read position is subsequent noise token
        is_subsequent = subsequent_noise_tokens[read_pos]
        is_first_noise = first_noise_tokens[read_pos]

        # Determine what token to write
        sentinel_id = sentinel_tokens[segments[read_pos] % len(sentinel_tokens)]
        token_to_write = jax.lax.select(is_first_noise, sentinel_id, tokens[read_pos])

        # Write to result if not subsequent noise
        new_result = jax.lax.select(
            is_subsequent, result_arr, result_arr.at[write_pos].set(token_to_write)
        )

        # Increment write_pos only if we wrote (not subsequent noise)
        new_write_pos = jax.lax.select(is_subsequent, write_pos, write_pos + 1)

        return new_result, new_write_pos

    result = jnp.full(padded_length, pad_token_id, dtype=tokens.dtype)
    result, _ = jax.lax.fori_loop(0, length, loop_body, (result, 0))

    return result


@jax.jit
def to_ul2r_rx_tokens(
    key: PRNGKeyArray,
    tokens: jnp.ndarray,
    length: int,
    mask_prob: float,
    mean_noise_span_length: float,
    random_roll: bool,
    sentinel_token_ids: jnp.ndarray,
    pad_token_id: int,
    max_length: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply UL2R R/X-denoising to the first `length` elements of `tokens`.

    Based on `span_corruption` in T5:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/e081111ccd51df425aa7124e8e311ed6d403d767/t5/data/preprocessors.py#L1931

    Does not set task tokens.

    Returns:
        - A tensor with the same shape as `tokens` containing
          `[inputs, targets, padding]` where `inputs+targets` are truncated to.
          fit `max_length`.
        - Scalar tensor containing the length of `inputs` (before `targets`).
          For use when generating the loss mask / PrefixLM attention mask.
    """

    padded_length = tokens.shape[0]
    noise_mask = random_spans_noise_mask_jax(
        length,
        mask_prob,
        key,
        mean_noise_span_length,
        random_roll,
        padded_length,
    )

    inputs = noise_span_to_unique_sentinel_jax(
        tokens, noise_mask, sentinel_token_ids, length, pad_token_id
    )
    targets = noise_span_to_unique_sentinel_jax(
        tokens, ~noise_mask, sentinel_token_ids, length, pad_token_id
    )

    indices = jnp.arange(padded_length)
    input_len = jnp.where(inputs == pad_token_id, indices, padded_length).min()
    target_len = jnp.where(targets == pad_token_id, indices, padded_length).min()

    # If `inputs + targets` exceed available `max_length`, truncate both proportionally
    combined_len = input_len + target_len
    overflow = jnp.maximum(combined_len - max_length, 0)

    # Distribute overflow proportionally between `inputs` and `targets`
    drop_inputs = jnp.where(
        combined_len > 0,
        (overflow * input_len) // jnp.maximum(combined_len, 1),
        0,
    )
    drop_targets = overflow - drop_inputs

    new_input_len = jnp.maximum(input_len - drop_inputs, 0)
    new_target_len = jnp.maximum(target_len - drop_targets, 0)

    # Truncate `targets` to the new length; `inputs` are gated by `new_input_len` below
    targets_trunc = jnp.where(indices < new_target_len, targets, pad_token_id)
    targets_trunc = typing.cast(jnp.ndarray, targets_trunc)

    # Roll `targets` to start at position `new_input_len`
    targets_rolled = jnp.roll(targets_trunc, new_input_len)

    result = jnp.where(indices < new_input_len, inputs, targets_rolled)
    result = typing.cast(jnp.ndarray, result)
    return result, new_input_len


@jax.jit
def to_ul2r_s_tokens(
    key: PRNGKeyArray,
    tokens: jnp.ndarray,
    length: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply UL2R S-denoising to the first `length` elements of `tokens`.

    Unlike the T5 version, all start positions for the continuation are equally
    likely given that at least one token is included in the prefix.

    In Tay et al 2008 they mention reusing random_spans_noise_mask and setting
    noise_density to 0.25 and `mean_noise_span_length` to `length / 4`.
    for S-denoising. https://arxiv.org/abs/2210.11399
    But given their code I think this will deterministically just make the
    last quarter of the input noise?!
    https://github.com/google-research/text-to-text-transfer-transformer/blob/e081111ccd51df425aa7124e8e311ed6d403d767/t5/data/preprocessors.py#L2929-L3039
    Here we instead choose a random pivot, including at least one token from
    the beginning of the input.

    Does not set task tokens.
    """

    # TODO support parameters?

    # I'm not sure whether S-denoising examples look like
    #   [S] <prefix> <sentinel_0> <continuation>
    # or
    #   [S] <prefix> <continuation>
    # The latter would be identical to what we had as CausalLmConfig.
    # The code in UL2 mentions noise_span_to_unique_sentinel for R/X-denoising
    # but not S-denoising (see Section 9.2) and a figure in UL2R says
    # "For PaLM and U-PaLM default, we pass the input as-is to the model.
    # For the rest, we prepend one of [S2S], [NLU], or [NLG] to the
    # beginning of # the input, and in the case of [NLU] and [NLG], we add
    # the infill token at the end of the input, as typical for these modes."
    # (implying there is no infill token for S-denoising) but I'm not sure.
    # https://github.com/google-research/text-to-text-transfer-transformer/blob/e081111ccd51df425aa7124e8e311ed6d403d767/t5/data/preprocessors.py#L2033

    pivot = jax.random.randint(key, (), 1, length - 1)
    targets = jnp.roll(tokens, 1)
    indices = jnp.arange(tokens.shape[0])
    result = jnp.where(indices < pivot, tokens, targets)
    result = result.at[pivot].set(SENTINEL_TOKEN_IDS[0])
    return result, pivot


# Returns (inputs_len, denoising_tokens)
# denoising_tokens = [task_token] [inputs] [targets]
@jax.jit
def to_ul2r_tokens(
    key: PRNGKeyArray,
    task_tokens: jnp.ndarray,
    all_task_params: jnp.ndarray,
    tokens: jnp.ndarray,
    length: int,
    pad_token_id: int,
    # TODO maybe we don't actually need the truncation logic in
    # to_ul2r_rx_tokens given that we truncate while packing
    # See slice_strategy.
    # However that slices using offsets[stop] - offsets[start], which can be
    # less than the manually-specified lenghts.
    max_length: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    task_idx = tokens[0]
    task_params = all_task_params[task_idx]
    task_kind = task_params[0]
    task_token = task_tokens[task_idx]

    def rx_tokens():
        noise_density = task_params[1]
        mean_noise_span_length = task_params[2]
        # TODO allow configuring random_roll
        inputs_len, out = to_ul2r_rx_tokens(
            key,
            tokens[1:],
            length,
            noise_density,
            mean_noise_span_length,
            random_roll=True,
            sentinel_token_ids=SENTINEL_TOKEN_IDS,
            pad_token_id=pad_token_id,
            max_length=max_length,
        )
        return inputs_len, jnp.concatenate([jnp.array([task_token]), out])

    def s_tokens():
        inputs_len, out = to_ul2r_s_tokens(key, tokens[1:], length)
        return inputs_len, jnp.concatenate([jnp.array([task_token]), out])

    return jax.lax.cond(task_kind, rx_tokens, s_tokens)


@jax.jit
def ul2r_loss_mask(
    input_mask: jnp.ndarray,
    segment_ids: jnp.ndarray,
    tokens: jnp.ndarray,
    pad_token_id: int,
) -> jnp.ndarray:
    """
    Creates a loss mask for UL2R training.

    Loss is computed only on output tokens (where input_mask is 0),
    excluding the last token in each segment and padding tokens.

    Args:
        - input_mask: Binary mask indicating input positions (1) vs output
          positions (0)
        - segment_ids: Segment IDs for packed sequences (-1 for padding)
        - tokens: Token IDs
        - pad_token_id: Padding token ID

    Returns:
        Loss mask array of same shape as inputs
    """
    #   aaaabbb  segment_ids
    #   1100110  input_mask
    # → 0110010
    #   || ^ no loss on last output
    #   |^ loss on last input / first output
    #   ^ no loss on inputs / task token

    loss_mask = jnp.logical_not(input_mask)
    loss_mask = jnp.roll(loss_mask, -1)

    # Don't compute loss across segment boundaries
    segment_continues = jnp.concatenate(
        [segment_ids[1:] == segment_ids[:-1], jnp.array([True])]
    )
    loss_mask = loss_mask & segment_continues

    # Don't compute loss on padding
    valid_mask = tokens != pad_token_id
    loss_mask = loss_mask & valid_mask

    return loss_mask


@jax.jit
def ul2r_block_diagonal_mask(
    input_masks: jnp.ndarray, segment_ids: jnp.ndarray
) -> jnp.ndarray:
    """
    Creates the prefix portion of an attention mask for UL2R training.

    - Input tokens can attend to all other input tokens bidirectionally within
      the same segment (corresponds to a block diagonal matrix)
    - Causal attention for output tokens isn't handle by this function; combine
      this mask with a causal mask
    - Attention is blocked across segment boundaries

    Args:
        - input_masks: Boolean tensor of shape `(seq_len,)` indicating input
          positions (1) vs output positions (0).
        - segment_ids: Segment IDs of shape `(seq_len,)` for packed sequences
          (-1 for padding).

    Returns:
        An attention mask with shape `(seq_len, seq_len)`.
    """
    segment_mask = segment_ids[:, None] == segment_ids[None, :]
    prefix_pattern = input_masks[:, None] & input_masks[None, :]
    is_not_padding = segment_ids != -1
    prefix_mask = segment_mask & prefix_pattern & is_not_padding
    return prefix_mask


TokenizedDict = typing.TypedDict("TokenizedDict", {"input_ids": np.ndarray})


# [example, seg_ids]: tuple[TokenizedDict, TokenizedDict]
class Ul2rDataset(MappedAsyncDataset[tuple[TokenizedDict, TokenizedDict], LmExample]):
    def __init__(
        self,
        cache: TreeCache[TokenizedDict],
        Pos: Axis,
        task_configs: typing.Dict[str, DenoisingConfig],
        task_probs: Dict[str, float],
        key: PRNGKeyArray,
        max_segments_per_example: int = 64,
        slice_strategy: Literal["left", "right", "raise"] = "left",
    ):
        cache.await_finished()

        # Copied from GreedyPrepackedDataset.__init__
        # TODO factor out?
        # TODO avoid reading store.offsets twice (here and in GreedyPrepackedDataset)
        offsets = jax.tree.map(
            lambda store: store.offsets[0 : store.num_rows + 1].read(), cache.store.tree
        )
        offsets = jax.tree.map(lambda fut: fut.result(), offsets)

        def diff_offsets(offsets: np.ndarray):
            # fine to mutate since we have a copy
            # the array store has the number of rows in the 0th offset
            offsets[0] = 0
            return offsets[1:] - offsets[:-1]

        lengths = jnp.array(jax.tree.map(diff_offsets, offsets))
        n_docs = lengths.shape[0]

        task_items = [
            (config, task_probs[name]) for name, config in task_configs.items()
        ]
        n_tasks = len(task_items)
        task_probs_arr = jnp.array([prob for _, prob in task_items])
        task_indices_key, key = jax.random.split(key)
        task_indices = jax.random.choice(
            task_indices_key, n_tasks, shape=(n_docs,), p=task_probs_arr
        )

        # shape (n_tasks, A)
        # where A = max(num_args(to_ul2r_rx_tokens), num_args(to_ul2r_s_tokens))
        task_params = jnp.array(
            [task.to_parameter_tensor() for task, _prob in task_items]
        )

        # We compute the length of the tokens after denoising because we want
        # to turn each input batch into a denoising batch while still staying
        # under the max sequence length for the model.
        def compute_denoising_length(
            length: jnp.ndarray, task_index: jnp.ndarray
        ) -> jnp.ndarray:
            def rx_length(length: jnp.ndarray, task_index: jnp.ndarray) -> jnp.ndarray:
                noise_density = task_params[task_index, 1]
                mean_noise_span_length = task_params[task_index, 2]
                _num_noise_tokens, num_noise_spans, _num_nonnoise_tokens = (
                    num_noise_spans_tokens_and_spans(
                        length, noise_density, mean_noise_span_length
                    )
                )
                # [task_token] one <sentinel_0> three <sentinel_0> two
                return 1 + 2 * num_noise_spans

            def s_length(length: jnp.ndarray, _task_index: jnp.ndarray) -> jnp.ndarray:
                # [task_token] one <sentinel_0> two three
                return 2 + length

            task_kind = task_params[task_index, 0]
            return jax.lax.cond(task_kind, rx_length, s_length, length, task_index)

        denoising_lengths = jax.vmap(compute_denoising_length)(lengths, task_indices)

        # NB the GreedyPackedDataset returns a tuple, where the first has the
        # packed leaves and the second has the segment ids
        self.packed: GreedyPrepackedDataset[TokenizedDict] = GreedyPrepackedDataset(
            cache.store.tree,
            Pos.size,
            max_segments_per_example=max_segments_per_example,
            slice_strategy=slice_strategy,
            # TODO avoid converting back to numpy
            lengths_for_packing=denoising_lengths.__array__(),
        )
        self.Pos = Pos

        sharding = jax.sharding.SingleDeviceSharding(
            jax.local_devices(backend="cpu")[0]
        )

        task_tokens = jnp.array([t.task_token_id() for t in task_configs.values()])

        # TODO is the type wrong here
        @functools.partial(
            eqx.filter_jit, out_shardings=sharding
        )  # pyright: ignore[reportCallIssue]
        def _create_lm_example(e: tuple[TokenizedDict, TokenizedDict]) -> LmExample:
            example, seg_ids = e
            tokens = hax.named(example["input_ids"], self.Pos)
            segment_ids = hax.named(seg_ids["input_ids"], self.Pos)

            unique_seg_ids = jnp.unique(
                segment_ids.array, size=max_segments_per_example, fill_value=-1
            )

            def process_segment(seg_id: int) -> jnp.ndarray:
                task_idx = task_indices[seg_id]

                mask = (segment_ids.array == seg_id)
                n = mask.shape[0]
                idx = jnp.arange(n)

                segment_start = jnp.min(jnp.where(mask, idx, n))
                segment_end = jnp.max(jnp.where(mask, idx + 1, 0))
                length = segment_end - segment_start

                segment = jnp.roll(tokens.array, -segment_start)

                inputs_len, denoising_tokens = to_ul2r_tokens(
                    key, task_tokens, task_params, segment, length, pad_token_id, Pos.size
                )
                segment = jnp.roll(segment, segment_start)
                pass

            segments = jax.vmap(process_segment)(unique_seg_ids)

            # construct the output example["input_ids"] by ORing together the different segments
            # construct input mask too for loss mask? or do inside process_segment?

            # max_length = tokens.shape[self.Pos.name]

            # task_and_inputs_len = 1 + inputs_len
            # input_mask = jnp.arange(max_length) < task_and_inputs_len

            # loss_mask = ul2r_loss_mask(input_mask, seg_ids, tokens, pad_token_id)

            return LmExample.causal(
                tokens=tokens, loss_mask=loss_mask, segment_ids=segment_ids
            )

        super().__init__(self.packed, _create_lm_example)
