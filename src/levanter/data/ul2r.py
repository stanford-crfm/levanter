import functools
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from jax.experimental import checkify


@functools.partial(jax.jit, static_argnames=["padded_length"])
def random_segmentation_jax(
    num_items: int, num_segments: int, key: PRNGKey, padded_length: int
) -> jnp.ndarray:
    """
    Precondition: 1 <= num_segments < num_items
    Returns:
        Array of segment lengths with shape (padded_length,) where first
        num_segments values sum to num_items
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


@functools.partial(
    jax.jit, static_argnames=["mean_noise_span_length", "random_roll", "padded_length"]
)
def random_spans_noise_mask_jax(
    length: int,
    noise_density: float,
    key: PRNGKey,
    mean_noise_span_length: float,
    random_roll: bool,
    padded_length: int,
) -> jnp.ndarray:
    """
    Returns:
        Boolean mask of shape (padded_length,) indicating noise positions
    """
    adjusted_length = jnp.maximum(length, 2)

    num_noise_tokens = jnp.round(adjusted_length * noise_density).astype(jnp.int32)
    num_noise_tokens = jnp.clip(num_noise_tokens, 1, adjusted_length - 1)
    num_noise_spans = jnp.maximum(
        jnp.round(num_noise_tokens / mean_noise_span_length).astype(jnp.int32), 1
    )
    num_nonnoise_tokens = adjusted_length - num_noise_tokens

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

    def apply_roll(m):
        offset = jax.random.randint(key3, (), 0, adjusted_length, dtype=jnp.int32)
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
        rolled = jnp.where(indices < length, rolled, False)
        return rolled

    mask = jax.lax.cond(random_roll, apply_roll, lambda m: m, is_noise)

    return mask


@functools.partial(jax.jit, static_argnames=["pad_token_id"])
def noise_span_to_unique_sentinel_jax(
    tokens: jnp.ndarray,
    noise_mask: jnp.ndarray,
    sentinel_tokens: jnp.ndarray,
    length: int,
    pad_token_id: int,
) -> jnp.ndarray:
    """
    Replace each run of consecutive noise tokens with a different sentinel.

    Returns:
        an array with the same shape and dtype as tokens
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

    # Check if we have too many noise spans
    err, num_sentinels = checkify.checkify(
        lambda: checkify.check(
            max_segments <= len(sentinel_tokens),
            f"Too many noise spans: {max_segments} > {len(sentinel_tokens)}",
        ),
        errors=checkify.index_checks,
    )()

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


@functools.partial(
    jax.jit,
    static_argnames=[
        "max_length",
        "pad_token_id",
        "mean_noise_span_length",
        "random_roll",
    ],
)
def to_ul2r_rx_tokens(
    key: PRNGKey,
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
    Apply UL2R R/X-denoising.

    Returns:
        - tensor with the same shape as tokens containing
          [inputs, targets, padding] where inputs+targets are truncated to fit
          max_length
        - scalar tensor containing the length of the input (before targets) for
          PrefixLM attention mask
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
    input_len = jnp.argmax(inputs == pad_token_id)
    target_len = jnp.argmax(targets == pad_token_id)

    # If inputs + targets exceed available max_length, truncate both proportionally
    combined_len = input_len + target_len
    overflow = jnp.maximum(combined_len - max_length, 0)

    # Distribute overflow proportionally between inputs and targets
    drop_inputs = jnp.where(
        combined_len > 0,
        (overflow * input_len) // jnp.maximum(combined_len, 1),
        0,
    )
    drop_targets = overflow - drop_inputs

    new_input_len = jnp.maximum(input_len - drop_inputs, 0)
    new_target_len = jnp.maximum(target_len - drop_targets, 0)

    # Truncate targets to the new length; inputs are gated by new_input_len below
    targets_trunc = jnp.where(indices < new_target_len, targets, pad_token_id)

    # Roll targets to start at position new_input_len
    targets_rolled = jnp.roll(targets_trunc, new_input_len)

    result = jnp.where(indices < new_input_len, inputs, targets_rolled)
    return result, new_input_len


@jax.jit
def ul2r_loss_mask(
    input_masks: jnp.ndarray,
    segment_ids: jnp.ndarray,
    tokens: jnp.ndarray,
    pad_token_id: int,
) -> jnp.ndarray:
    """
    Create loss mask for UL2R training.

    Loss is computed only on output tokens (where input_mask is 0),
    excluding the last token in each segment and padding tokens.

    Args:
        input_masks: Binary mask indicating input positions (1) vs output positions (0)
        segment_ids: Segment IDs for packed sequences (-1 for padding)
        tokens: Token IDs
        pad_token_id: Padding token ID

    Returns:
        Loss mask array of same shape as inputs
    """
    #   aaaabbb  segment_ids
    #   1100110  input_masks
    # → 0110010
    #   || ^ no loss on last output
    #   |^ loss on last input / first output
    #   ^ no loss on inputs / task token

    loss_mask = jnp.logical_not(input_masks)
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
    - Input tokens can attend to all other input tokens bidirectionally within
      the same segment
    - Causal attention for output tokens isn't handle by this function; combine
      this mask with a causal mask
    - Attention is blocked across segment boundaries

    Args:
        input_masks: Binary mask of shape (seq_len,) indicating input positions
        (1) vs output positions (0)
        segment_ids: Segment IDs of shape (seq_len,) for packed sequences (-1 for padding)

    Returns:
        Attention mask of shape (seq_len, seq_len) where True allows attention
    """
    # Block diagonal: can only attend within same segment
    segment_mask = segment_ids[:, None] == segment_ids[None, :]

    # Prefix pattern: inputs can attend to all other inputs bidirectionally
    # 1101 -> 1101
    #         1100
    #         1001
    prefix_pattern = input_masks[:, None] & input_masks[None, :]

    is_not_padding = segment_ids != -1

    # Must be in same segment AND outputs attending to inputs
    # Example for input_masks = 1100:
    # 1100  (row 0: input can see all inputs)
    # 1100  (row 1: input can see all inputs)
    # 0000  (row 2: output follows causal, handled separately)
    # 0001  (row 3: output follows causal, handled separately)
    prefix_mask = segment_mask & prefix_pattern & is_not_padding

    return prefix_mask
