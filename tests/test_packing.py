import jax.numpy as jnp
import numpy as np
import pytest

import haliax as hax

from levanter.data.packing import (
    PromptCompletion,
    SequencePacker,
    pack_prompt_completions,
    per_segment_correct,
    per_segment_loss,
)
from levanter.models.attention import AttentionMask
from levanter.models.lm_model import LmExample


def test_per_segment_loss():
    Pos = hax.Axis("pos", size=10)
    packer = SequencePacker(Pos=Pos, max_pack_size=10, pad_token=0)

    # Add two sequences
    packer.add_example(ids=[1, 2, 3], loss_mask=[1, 1, 1], segment_id=None)
    packer.add_example(ids=[4, 5], loss_mask=[1, 1], segment_id=None)

    # Pack into LmExample
    packed = packer.pack()

    losses = hax.named(jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]), Pos)

    Segments = hax.Axis("segments", size=3)

    unique_ids, segment_losses = per_segment_loss(packed, losses, max_Segments=Segments)

    assert list(unique_ids.array) == [-1, 0, 1]
    assert list(segment_losses.array) == [0.0, 0.6, 0.9]


def test_can_pack_simple_case():
    Pos = hax.Axis("pos", size=10)
    packer = SequencePacker(Pos=Pos, max_pack_size=2, pad_token=0)

    assert packer.can_pack([1, 2, 3]) is True
    packer.add_example(ids=[1, 2, 3], loss_mask=[1, 1, 1])
    assert packer.can_pack([4, 5]) is True
    assert packer.can_pack(list(range(6, 16))) is False  # Exceeds Pos size


def test_add_example_and_pack():
    Pos = hax.Axis("pos", size=10)
    packer = SequencePacker(Pos=Pos, max_pack_size=2, pad_token=0)

    packer.add_example([1, 2, 3], [1, 1, 1])
    packed = packer.pack()

    expected_tokens = [1, 2, 3, 0, 0, 0, 0, 0, 0, 0]
    expected_segment_ids = [0, 0, 0, -1, -1, -1, -1, -1, -1, -1]
    expected_loss_mask = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]

    np.testing.assert_array_equal(packed.tokens.array, expected_tokens)
    np.testing.assert_array_equal(packed.attn_mask.segment_ids.array, expected_segment_ids)
    np.testing.assert_array_equal(packed.loss_mask.array, expected_loss_mask)


def test_exceed_max_pack_size():
    Pos = hax.Axis("pos", size=10)
    packer = SequencePacker(Pos=Pos, max_pack_size=2, pad_token=0)

    packer.add_example([1, 2, 3], [1, 1, 1])
    packer.add_example([4, 5, 6], [1, 1, 1])

    with pytest.raises(ValueError, match="Too many segments"):
        packer.add_example([7, 8], [1, 1])  # Exceeds max pack size


def test_empty_sequence():
    Pos = hax.Axis("pos", size=10)
    packer = SequencePacker(Pos=Pos, max_pack_size=2, pad_token=0)

    with pytest.raises(ValueError, match="ids and loss_mask must have the same length"):
        packer.add_example([], [1])  # Mismatched lengths

    packer.add_example([], [])  # Adding an empty sequence is allowed but does nothing
    packed = packer.pack()

    expected_tokens = [0] * 10
    expected_segment_ids = [-1] * 10
    expected_loss_mask = [0] * 10

    np.testing.assert_array_equal(packed.tokens.array, expected_tokens)
    np.testing.assert_array_equal(packed.attn_mask.segment_ids.array, expected_segment_ids)
    np.testing.assert_array_equal(packed.loss_mask.array, expected_loss_mask)


def test_packing_multiple_examples():
    Pos = hax.Axis("pos", size=10)
    packer = SequencePacker(Pos=Pos, max_pack_size=2, pad_token=0)

    # First example
    packer.add_example([1, 2], [1, 1])
    # Second example
    packer.add_example([3, 4, 5], [1, 1, 1])

    packed = packer.pack()

    expected_tokens = [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]
    expected_segment_ids = [0, 0, 1, 1, 1, -1, -1, -1, -1, -1]
    expected_loss_mask = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    np.testing.assert_array_equal(packed.tokens.array, expected_tokens)
    np.testing.assert_array_equal(packed.attn_mask.segment_ids.array, expected_segment_ids)
    np.testing.assert_array_equal(packed.loss_mask.array, expected_loss_mask)


def test_pack_prompt_completions_simple():
    Pos = hax.Axis("pos", size=10)
    pad_token = 0
    max_pack_size = 2
    max_buffered_examples = 2

    sequences = [
        PromptCompletion(ids=[1, 2, 3], prompt_length=2),
        PromptCompletion(ids=[4, 5], prompt_length=1),
        PromptCompletion(ids=[6, 7, 8], prompt_length=1),
    ]

    results = list(pack_prompt_completions(Pos, sequences, pad_token, max_pack_size, max_buffered_examples))

    assert len(results) == 2  # Expect two packed LmExamples

    # Check the first packed example
    packed_1 = results[0]
    expected_tokens_1 = [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]
    expected_segment_ids_1 = [0, 0, 0, 1, 1, -1, -1, -1, -1, -1]
    expected_loss_mask_1 = [0, 1, 0, 1, 0, 0, 0, 0, 0, 0]

    np.testing.assert_array_equal(packed_1.tokens.array, expected_tokens_1)
    np.testing.assert_array_equal(packed_1.attn_mask.segment_ids.array, expected_segment_ids_1)
    np.testing.assert_array_equal(packed_1.loss_mask.array, expected_loss_mask_1)

    # Check the second packed example
    packed_2 = results[1]
    expected_tokens_2 = [6, 7, 8, 0, 0, 0, 0, 0, 0, 0]
    expected_segment_ids_2 = [0, 0, 0, -1, -1, -1, -1, -1, -1, -1]
    expected_loss_mask_2 = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

    np.testing.assert_array_equal(packed_2.tokens.array, expected_tokens_2)
    np.testing.assert_array_equal(packed_2.attn_mask.segment_ids.array, expected_segment_ids_2)
    np.testing.assert_array_equal(packed_2.loss_mask.array, expected_loss_mask_2)


def test_pack_prompt_completions_exceed_max_buffered_examples():
    Pos = hax.Axis("pos", size=10)
    pad_token = 0
    max_pack_size = 1
    max_buffered_examples = 1

    sequences = [
        PromptCompletion(ids=[1, 2, 3], prompt_length=2),
        PromptCompletion(ids=[4, 5], prompt_length=1),
        PromptCompletion(ids=[6, 7, 8], prompt_length=1),
    ]

    results = list(pack_prompt_completions(Pos, sequences, pad_token, max_pack_size, max_buffered_examples))

    assert len(results) == 3

    # Check the first packed example
    packed_1 = results[0]
    expected_tokens_1 = [1, 2, 3, 0, 0, 0, 0, 0, 0, 0]
    expected_segment_ids_1 = [0, 0, 0, -1, -1, -1, -1, -1, -1, -1]
    expected_loss_mask_1 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

    np.testing.assert_array_equal(packed_1.tokens.array, expected_tokens_1)
    np.testing.assert_array_equal(packed_1.attn_mask.segment_ids.array, expected_segment_ids_1)
    np.testing.assert_array_equal(packed_1.loss_mask.array, expected_loss_mask_1)

    # Check the second packed example
    packed_2 = results[1]
    expected_tokens_2 = [4, 5, 0, 0, 0, 0, 0, 0, 0, 0]
    expected_segment_ids_2 = [0, 0, -1, -1, -1, -1, -1, -1, -1, -1]
    expected_loss_mask_2 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    np.testing.assert_array_equal(packed_2.tokens.array, expected_tokens_2)
    np.testing.assert_array_equal(packed_2.attn_mask.segment_ids.array, expected_segment_ids_2)
    np.testing.assert_array_equal(packed_2.loss_mask.array, expected_loss_mask_2)

    # Check the third packed example
    packed_3 = results[2]
    expected_tokens_3 = [6, 7, 8, 0, 0, 0, 0, 0, 0, 0]
    expected_segment_ids_3 = [0, 0, 0, -1, -1, -1, -1, -1, -1, -1]
    expected_loss_mask_3 = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

    np.testing.assert_array_equal(packed_3.tokens.array, expected_tokens_3)
    np.testing.assert_array_equal(packed_3.attn_mask.segment_ids.array, expected_segment_ids_3)
    np.testing.assert_array_equal(packed_3.loss_mask.array, expected_loss_mask_3)


def test_segment_correct():
    # Mock segment_ids and loss_mask
    Pos = hax.Axis("pos", size=10)
    tokens = hax.named(jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), Pos)
    segment_ids = hax.named(jnp.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2]), Pos)
    loss_mask = hax.named(jnp.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]), Pos)

    # Create a packed example
    attn_mask = AttentionMask.causal().with_segment_ids(segment_ids=segment_ids)
    packed_example = LmExample(tokens=tokens, loss_mask=loss_mask, attn_mask=attn_mask)

    # Mock correctness array (True for correct, False for incorrect)
    correct = hax.named(jnp.array([True, True, True, False, True, False, True, True, True, True]), Pos)

    max_Segments = hax.Axis("segments", size=4)

    # Call the function
    unique_ids, segment_correct = per_segment_correct(packed_example, correct, max_Segments)

    assert list(unique_ids.array) == [0, 1, 2, -1]
    assert list(segment_correct.array) == [True, False, True, True]


if __name__ == "__main__":
    pytest.main()
