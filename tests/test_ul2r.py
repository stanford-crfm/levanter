import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.data.ul2r import (
    noise_span_to_unique_sentinel_jax,
    ul2r_block_diagonal_mask,
    ul2r_loss_mask,
    random_segmentation_jax,
    random_spans_noise_mask_jax,
    to_ul2r_rx_tokens,
    to_ul2r_s_tokens,
    SENTINEL_TOKEN_IDS,
)


def test_random_segmentation_jax():
    padded_length = 100

    test_cases = [
        (10, 3),  # 10 items, 3 segments
        (20, 5),  # 20 items, 5 segments
        (50, 10),  # 50 items, 10 segments
        (5, 2),  # Small case
        (30, 1),  # Single segment
    ]

    for num_items, num_segments in test_cases:
        key = jax.random.PRNGKey(42)

        segment_lengths = random_segmentation_jax(
            num_items, num_segments, key, padded_length
        )

        relevant_lengths = segment_lengths[:num_segments]

        assert (
            jnp.sum(relevant_lengths) == num_items
        ), f"Sum of segments {jnp.sum(relevant_lengths)} != num_items {num_items}"

        assert jnp.all(
            relevant_lengths > 0
        ), f"Found zero-length segments for num_items={num_items}, num_segments={num_segments}"

        # Check determinism - same key should give same result
        segment_lengths2 = random_segmentation_jax(
            num_items, num_segments, key, padded_length
        )
        np.testing.assert_array_equal(segment_lengths, segment_lengths2)

        # Different key should give different result (usually)
        key2 = jax.random.PRNGKey(43)
        segment_lengths3 = random_segmentation_jax(
            num_items, num_segments, key2, padded_length
        )
        # For edge cases like single segment, result might be same
        if num_segments > 1 and num_segments < num_items:
            assert not np.array_equal(
                segment_lengths[:num_segments], segment_lengths3[:num_segments]
            )


def test_random_spans_noise_mask_jax():
    """Test that random_spans_noise_mask_jax works correctly with static shapes."""
    padded_length = 256
    test_cases = [
        (100, 0.15, 3.0, False),  # Standard case without roll
        (100, 0.15, 3.0, True),  # Standard case with roll
        (200, 0.5, 10.0, False),  # Higher density, longer spans
        (50, 0.3, 5.0, True),  # Shorter sequence with roll
        (100, 0.0, 3.0, False),  # Edge case: zero noise density
    ]

    for length, noise_density, mean_span_length, random_roll in test_cases:
        key = jax.random.PRNGKey(42)

        mask = random_spans_noise_mask_jax(
            length, noise_density, key, mean_span_length, random_roll, padded_length
        )

        assert mask.shape == (
            padded_length,
        ), f"Expected shape ({padded_length},), got {mask.shape}"
        assert mask.dtype == jnp.bool_, f"Expected bool dtype, got {mask.dtype}"

        assert jnp.all(mask[length:] == False), "Mask should be False after length"

        # Check noise density approximately matches (for non-zero cases)
        if noise_density > 0:
            actual_density = jnp.sum(mask[:length]) / length
            # Allow for some variance due to rounding
            tolerance = 0.2
            assert (
                abs(actual_density - noise_density) < tolerance
            ), f"Noise density {actual_density} too far from target {noise_density}"
        else:
            assert jnp.sum(mask) == 0, "Zero noise density should produce all zeros"

        # Test that the function is deterministic with same key
        mask2 = random_spans_noise_mask_jax(
            length, noise_density, key, mean_span_length, random_roll, padded_length
        )
        np.testing.assert_array_equal(
            mask, mask2, "Function should be deterministic with same key"
        )

        # Test that different keys produce different results (for non-zero density)
        if noise_density > 0:
            key2 = jax.random.PRNGKey(43)
            mask3 = random_spans_noise_mask_jax(
                length,
                noise_density,
                key2,
                mean_span_length,
                random_roll,
                padded_length,
            )
            assert not jnp.array_equal(
                mask, mask3
            ), "Different keys should produce different masks"


def test_noise_span_to_unique_sentinel_jax():
    """Test that noise_span_to_unique_sentinel_jax works correctly with static shapes."""
    padded_length = 256
    pad_token_id = 0
    sentinel_tokens = jnp.array([100, 101, 102, 103, 104])

    # Test case 1: First token is a noise span (single span)
    tokens = jnp.arange(10, 20)  # [10, 11, 12, ..., 19]
    tokens = jnp.pad(tokens, (0, padded_length - 10), constant_values=pad_token_id)
    noise_mask = jnp.array(
        [
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,  # Span 1: [10, 11, 12]
        ]
    )
    noise_mask = jnp.pad(noise_mask, (0, padded_length - 10), constant_values=False)

    result = noise_span_to_unique_sentinel_jax(
        tokens, noise_mask, sentinel_tokens, 10, pad_token_id
    )

    expected = jnp.array([100, 13, 14, 15, 16, 17, 18, 19])
    np.testing.assert_array_equal(result[:8], expected)
    assert jnp.all(result[8:] == pad_token_id)

    # Test case 2: First token is NOT a noise span (multiple noise spans)
    tokens = jnp.arange(10, 25)  # [10, 11, 12, ..., 24]
    tokens = jnp.pad(tokens, (0, padded_length - 15), constant_values=pad_token_id)
    noise_mask = jnp.array(
        [
            False,
            True,
            True,
            False,
            False,  # Span 1: [11, 12]
            False,
            True,
            False,
            False,
            False,  # Span 2: [16]
            True,
            True,
            True,
            False,
            False,  # Span 3: [20, 21, 22]
        ]
    )
    noise_mask = jnp.pad(noise_mask, (0, padded_length - 15), constant_values=False)

    result = noise_span_to_unique_sentinel_jax(
        tokens, noise_mask, sentinel_tokens, 15, pad_token_id
    )

    expected = jnp.array([10, 100, 13, 14, 15, 101, 17, 18, 19, 102, 23, 24])
    np.testing.assert_array_equal(result[:12], expected)
    assert jnp.all(result[12:] == pad_token_id)

    # Test case 3: Empty noise mask (no noise)
    tokens = jnp.arange(10, 20)
    tokens = jnp.pad(tokens, (0, padded_length - 10), constant_values=pad_token_id)
    noise_mask = jnp.zeros(padded_length, dtype=jnp.bool_)

    result = noise_span_to_unique_sentinel_jax(
        tokens, noise_mask, sentinel_tokens, 10, pad_token_id
    )

    # Should be unchanged except for padding
    np.testing.assert_array_equal(result[:10], jnp.arange(10, 20))
    assert jnp.all(result[10:] == pad_token_id)


def test_to_ul2r_rx_tokens():
    """Test the to_ul2r_rx_tokens function."""
    max_length = 256
    pad_token_id = 0
    sentinel_tokens = jnp.array([100, 101, 102, 103, 104])

    # Test case: Simple sequence with known noise pattern
    tokens = jnp.arange(10, 30)
    tokens = jnp.pad(tokens, (0, max_length - 20), constant_values=pad_token_id)
    length = 20

    key = jax.random.PRNGKey(42)

    result, input_length = to_ul2r_rx_tokens(
        key,
        tokens,
        length,
        mask_prob=0.3,
        mean_noise_span_length=3.0,
        random_roll=False,
        sentinel_token_ids=sentinel_tokens,
        pad_token_id=pad_token_id,
        max_length=max_length,
    )

    assert result.shape == (max_length,)

    assert input_length.shape == ()
    assert input_length > 0
    assert input_length <= length

    inputs_part = result[:input_length]
    contains_sentinels = jnp.any(jnp.isin(inputs_part, sentinel_tokens))
    assert contains_sentinels, "Inputs should contain sentinel tokens"

    # Find where all padding starts (after both inputs and targets)
    # Look for continuous padding at the end
    is_pad = result == pad_token_id
    # Find the last non-padding position
    non_pad_positions = jnp.where(~is_pad, jnp.arange(max_length), -1)
    last_non_pad = jnp.max(non_pad_positions)

    if last_non_pad < max_length - 1:
        assert jnp.all(
            result[last_non_pad + 1 :] == pad_token_id
        ), "Should have continuous padding at the end"

    # Test with random_roll=True
    result_roll, input_length_roll = to_ul2r_rx_tokens(
        key,
        tokens,
        length,
        mask_prob=0.3,
        mean_noise_span_length=3.0,
        random_roll=True,
        sentinel_token_ids=sentinel_tokens,
        pad_token_id=pad_token_id,
        max_length=max_length,
    )

    assert result_roll.shape == (max_length,)
    assert input_length_roll.shape == ()

    is_pad_roll = result_roll == pad_token_id
    non_pad_positions_roll = jnp.where(~is_pad_roll, jnp.arange(max_length), -1)
    last_non_pad_roll = jnp.max(non_pad_positions_roll)
    if last_non_pad_roll < max_length - 1:
        assert jnp.all(
            result_roll[last_non_pad_roll + 1 :] == pad_token_id
        ), "Should have continuous padding at the end with roll"


def test_ul2r_block_diagonal_mask():
    # Test case 1: Single segment with inputs and outputs
    input_masks = jnp.array([1, 1, 0, 0])  # First 2 are inputs, last 2 are outputs
    segment_ids = jnp.array([0, 0, 0, 0])  # All in same segment

    mask = ul2r_block_diagonal_mask(input_masks, segment_ids)

    # Expected: inputs can see each other bidirectionally
    # Row 0 (input): can see positions 0 and 1 (other inputs)
    # Row 1 (input): can see positions 0 and 1 (other inputs)
    # Row 2 (output): only follows causal (handled elsewhere), so no prefix mask
    # Row 3 (output): only follows causal (handled elsewhere), so no prefix mask
    expected = jnp.array([
        [True, True, False, False],
        [True, True, False, False],
        [False, False, False, False],
        [False, False, False, False],
    ])

    np.testing.assert_array_equal(mask, expected)

    # Test case 2: Multiple segments with padding
    input_masks = jnp.array([1, 1, 0, 0, 1, 0, 0, 0])  # Seg 0: inputs, outputs; Seg 1: input, output; then padding
    segment_ids = jnp.array([0, 0, 0, 0, 1, 1, -1, -1])  # Two segments plus padding

    mask = ul2r_block_diagonal_mask(input_masks, segment_ids)

    # Expected: inputs can see other inputs within same segment only, padding should be all False
    expected = jnp.array([
        [True, True, False, False, False, False, False, False],  # Seg 0 input sees seg 0 inputs
        [True, True, False, False, False, False, False, False],  # Seg 0 input sees seg 0 inputs
        [False, False, False, False, False, False, False, False],  # Seg 0 output
        [False, False, False, False, False, False, False, False],  # Seg 0 output
        [False, False, False, False, True, False, False, False],  # Seg 1 input sees only itself
        [False, False, False, False, False, False, False, False],  # Seg 1 output
        [False, False, False, False, False, False, False, False],  # Padding
        [False, False, False, False, False, False, False, False],  # Padding
    ])

    np.testing.assert_array_equal(mask, expected)


def test_ul2r_loss_mask():
    # Test case 1: Simple single segment
    input_masks = jnp.array([1, 1, 0, 0])  # First 2 are inputs
    segment_ids = jnp.array([0, 0, 0, 0])  # All same segment
    tokens = jnp.array([10, 11, 12, 13])  # No padding
    pad_token_id = 0

    mask = ul2r_loss_mask(input_masks, segment_ids, tokens, pad_token_id)

    # Expected: loss on outputs, shifted by 1
    # Original:     1 1 0 0 (input_masks)
    # Inverted:     0 0 1 1
    # Shifted:      0 1 1 0 (roll -1)
    # No boundaries or padding to worry about
    expected = jnp.array([False, True, True, False])

    np.testing.assert_array_equal(mask, expected)

    # Test case 2: Multiple segments with padding
    input_masks = jnp.array([1, 1, 0, 0, 1, 0, 0, 0])
    segment_ids = jnp.array([0, 0, 0, 0, 1, 1, -1, -1])  # Two segments plus padding
    tokens = jnp.array([10, 11, 12, 13, 14, 15, 0, 0])  # Last 2 are padding tokens

    mask = ul2r_loss_mask(input_masks, segment_ids, tokens, pad_token_id)

    # Expected: no loss across segment boundaries or on padding
    # Original:     1 1 0 0 | 1 0 | 0 0 (padding)
    # Inverted:     0 0 1 1 | 0 1 | 1 1
    # Shifted:      0 1 1 0 | 1 1 | 1 0
    # Segment cont: T T T F | T F | F F (boundaries and padding)
    # Valid mask:   T T T T | T T | F F (padding tokens are 0)
    # Final:        0 1 1 0 | 1 0 | 0 0
    expected = jnp.array([False, True, True, False, True, False, False, False])

    np.testing.assert_array_equal(mask, expected)


def test_to_ul2r_rx_tokens_truncates_both_sections_and_contains_sentinels():
    """
    Ensure we truncate from both inputs and outputs and retain sentinels in each section.

    We create a long padded input (padded_length > max_length) and compare the
    pre-truncation lengths to the output of to_ul2r_rx_tokens with a small
    max_length to force truncation.
    """
    padded_length = 512
    max_length = 400
    pad_token_id = 0
    sentinel_tokens = jnp.arange(100, 140)  # plenty of sentinels

    length = 450  # true sequence length (greater than max_length)
    tokens = jnp.arange(1, length + 1)
    tokens = jnp.pad(tokens, (0, padded_length - length), constant_values=pad_token_id)

    key = jax.random.PRNGKey(123)
    mask_prob = 0.3
    mean_noise_span_length = 3.0
    random_roll = False

    # First, run with a large max_length to observe untruncated section lengths
    result_full, input_len_full = to_ul2r_rx_tokens(
        key,
        tokens,
        length,
        mask_prob=mask_prob,
        mean_noise_span_length=mean_noise_span_length,
        random_roll=random_roll,
        sentinel_token_ids=sentinel_tokens,
        pad_token_id=pad_token_id,
        max_length=padded_length,
    )
    # Compute outputs length (non-padding after inputs)
    outputs_full = result_full[input_len_full:]
    outputs_nonpad_full = jnp.sum(outputs_full != pad_token_id)

    # Now run with a smaller max_length to force truncation
    result_small, input_len_small = to_ul2r_rx_tokens(
        key,
        tokens,
        length,
        mask_prob=mask_prob,
        mean_noise_span_length=mean_noise_span_length,
        random_roll=random_roll,
        sentinel_token_ids=sentinel_tokens,
        pad_token_id=pad_token_id,
        max_length=max_length,
    )

    assert result_small.shape == (padded_length,)
    assert input_len_small > 0
    assert input_len_small < input_len_full  # inputs truncated

    # Outputs section length and checks
    outputs_slice = result_small[input_len_small:]
    outputs_nonpad = jnp.sum(outputs_slice != pad_token_id)
    assert outputs_nonpad > 0
    assert outputs_nonpad < outputs_nonpad_full  # outputs truncated

    # Sentinel presence in both sections
    inputs_slice = result_small[:input_len_small]
    has_sentinel_inputs = jnp.any(jnp.isin(inputs_slice, sentinel_tokens))
    has_sentinel_outputs = jnp.any(
        jnp.isin(outputs_slice[:outputs_nonpad], sentinel_tokens)
    )
    assert has_sentinel_inputs
    assert has_sentinel_outputs


def test_to_ul2r_s_tokens():
    # Test case 1: Basic functionality with simple sequence
    tokens = jnp.arange(10, 20)  # [10, 11, 12, ..., 19]
    padded_tokens = jnp.pad(tokens, (0, 256 - 10), constant_values=0)
    length = 10
    key = jax.random.PRNGKey(42)

    result, pivot = to_ul2r_s_tokens(key, padded_tokens, length)

    assert result.shape == padded_tokens.shape
    assert pivot.shape == ()
    assert 1 <= pivot < length - 1, f"Pivot {pivot} should be between 1 and {length-2}"
    assert result[pivot] == SENTINEL_TOKEN_IDS[0]

    # Check prefix is unchanged (before pivot)
    np.testing.assert_array_equal(result[:pivot], padded_tokens[:pivot])

    # Check continuation (after pivot) is shifted from original
    # The continuation should be tokens[pivot:] starting at position pivot+1
    expected_continuation = padded_tokens[pivot:]
    # Shift by 1 to account for sentinel at pivot
    np.testing.assert_array_equal(result[pivot + 1 :], expected_continuation[:-1])

    # Test case 2: Determinism - same key should give same result
    result2, pivot2 = to_ul2r_s_tokens(key, padded_tokens, length)
    np.testing.assert_array_equal(result, result2)
    assert pivot == pivot2

    # Test case 3: Different keys should give different pivots (usually)
    key2 = jax.random.PRNGKey(43)
    result3, pivot3 = to_ul2r_s_tokens(key2, padded_tokens, length)
    assert pivot != pivot3, "Different keys should produce different pivots"
