from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pytest

import haliax as hax

from levanter.data.packing import (
    GreedyPrepackedDataset,
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


####################################
# tests for GreedyPrepackedDataset #
####################################


# Dummy implementations for testing.


@dataclass
class DummyFuture:
    value: np.ndarray

    def result(self):
        return self.value


class DummyOffsets:
    def __init__(self, offsets: np.ndarray):
        self._offsets = offsets

    def read(self):
        return DummyFuture(self._offsets)


class DummyJaggedArrayStore:
    def __init__(self, offsets: np.ndarray):
        self.offsets = DummyOffsets(offsets)

    # In production there would be more (like data and shapes).


# Import GreedyPrepackedDataset from your module.
# from your_module import GreedyPrepackedDataset

# TEST 1: Single-leaf dataset.
@pytest.fixture
def simple_dataset():
    # Create a single JaggedArrayStore with four documents.
    # Let document lengths be: 100, 200, 150, 150 tokens.
    offsets = np.array([0, 100, 300, 450, 600])
    store = DummyJaggedArrayStore(offsets)
    # Build dataset as a PyTree; here a dict with one key.
    dataset = {"store": store}
    # Set allowed max tokens (per leaf) as 300.
    max_length = {"store": 300}
    return dataset, max_length, offsets


def test_simple_pack(simple_dataset):
    dataset, max_length, offsets = simple_dataset
    tester = GreedyPrepackedDataset(dataset, max_length)
    packs = tester._pack_indices
    # We expect, given document lengths [100,200,150,150] and budget 300,
    # that the first pack covers docs 0 and 1: token range = [offsets[0], offsets[2]) = [0,300),
    # and the second pack covers docs 2 and 3: [offsets[2], offsets[4]) = [300,600).
    assert len(packs) == 2
    pack0 = packs[0]  # Expect {"store": range(0,300)}
    pack1 = packs[1]  # Expect {"store": range(300,600)}
    assert list(pack0["store"]) == list(range(0, 300))
    assert list(pack1["store"]) == list(range(300, 600))


def test_simple_pack_max_examples(simple_dataset):
    dataset, max_length, offsets = simple_dataset
    tester = GreedyPrepackedDataset(dataset, max_length, max_segments_per_example=1)
    packs = tester._pack_indices
    # We expect, given document lengths [100,200,150,150] and budget 300,
    # that the first pack covers docs 0 and 1: token range = [offsets[0], offsets[2]) = [0,300),
    # and the second pack covers docs 2 and 3: [offsets[2], offsets[4]) = [300,600).
    assert len(packs) == 4
    assert list(packs[0]["store"]) == list(range(0, 100))
    assert list(packs[1]["store"]) == list(range(100, 300))
    assert list(packs[2]["store"]) == list(range(300, 450))
    assert list(packs[3]["store"]) == list(range(450, 600))


# TEST 2: Multi-leaf dataset.
@pytest.fixture
def multi_leaf_dataset():
    # Create two leaves.
    # Leaf1: document lengths: [100,200,150,150] => offsets: [0,100,300,450,600]
    # Leaf2: document lengths: [90,190,150,150]  => offsets: [0,90,280,430,580]
    offsets1 = np.array([0, 100, 300, 450, 600])
    offsets2 = np.array([0, 90, 280, 430, 580])
    store1 = DummyJaggedArrayStore(offsets1)
    store2 = DummyJaggedArrayStore(offsets2)
    dataset = {"store1": store1, "store2": store2}
    # Allowed max per leaf: for store1: 300 tokens, for store2: 250 tokens.
    max_length = {"store1": 300, "store2": 250}
    return dataset, max_length, (offsets1, offsets2)


def test_multi_leaf_pack(multi_leaf_dataset):
    dataset, max_length, _ = multi_leaf_dataset
    tester = GreedyPrepackedDataset(dataset, max_length)
    packs = tester._pack_indices
    # Here the effective allowed max is computed per leaf:
    # For store1: budget = 300, for store2: budget = 250. Thus the pack must satisfy both.
    # Document lengths for store1: [100,200,150,150]; for store2: [90,190,150,150].
    # For pack 1: starting at doc0, adding doc0 yields (100 and 90) which are within allowed.
    # Trying to add doc1: store1 becomes 100+200=300 (okay) but store2 becomes 90+190=280 (>250);
    # so pack 1 covers doc0 only.
    # Similarly, pack 2 covers doc1 only; pack 3 covers doc2 only; pack 4 covers doc3 only.
    assert len(packs) == 4
    expected_ranges_store1 = [range(0, 100), range(100, 300), range(300, 450), range(450, 600)]
    expected_ranges_store2 = [range(0, 90), range(90, 280), range(280, 430), range(430, 580)]
    for pack, er1, er2 in zip(packs, expected_ranges_store1, expected_ranges_store2):
        assert list(pack["store1"]) == list(er1)
        assert list(pack["store2"]) == list(er2)


# TEST 3: max_segments_per_example constraint.
@pytest.fixture
def dataset_with_segments():
    # Create a single-leaf dataset of 4 documents (as in test 1) with offsets [0,100,300,450,600].
    offsets = np.array([0, 100, 300, 450, 600])
    store = DummyJaggedArrayStore(offsets)
    dataset = {"store": store}
    max_length = {"store": 1000}  # large enough so that token count doesn't force a break.
    return dataset, max_length, offsets


def test_max_segments_constraint(dataset_with_segments):
    dataset, max_length, offsets = dataset_with_segments
    # With max_segments_per_example set to 1, each pack must cover exactly one document.
    tester = GreedyPrepackedDataset(dataset, max_length, max_segments_per_example=1)
    packs = tester._pack_indices
    # There are 4 documents so expect 4 packs.
    assert len(packs) == 4
    # For each document, the returned range should be from offsets[i] to offsets[i+1].
    for i, pack in enumerate(packs):
        expected = range(offsets[i], offsets[i + 1])
        assert list(pack["store"]) == list(expected)


def test_too_long_to_pack(multi_leaf_dataset):
    dataset, _, _ = multi_leaf_dataset
    max_length = {"store1": 10, "store2": 5000}

    with pytest.raises(ValueError, match="exceeds allowed capacity"):
        GreedyPrepackedDataset(dataset, max_length)

    pack2 = GreedyPrepackedDataset(dataset, max_length, slice_too_long_examples=True)._pack_indices[2]
    assert list(pack2["store1"]) == list(range(440, 450))
    assert list(pack2["store2"]) == list(range(280, 430))


if __name__ == "__main__":
    pytest.main()
