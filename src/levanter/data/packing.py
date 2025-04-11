"""
Implements sequence packing, mostly for doing evaluation on lots of short sequences.

Our strategy is basically to maintain a pool of SequencePackers, each of which can hold a fixed number of tokens
(and a maximum number of segments). We then iterate over the sequences, adding them to the packers if they fit, and
yielding the packed examples when they are full.

This achieves about a 90% "real token" rate, compared to like 10% without packing.
"""
import asyncio
from dataclasses import dataclass
from typing import Awaitable, Iterable, Iterator, Optional, Sequence, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import tensorstore as ts
from jaxtyping import PyTree

import haliax as hax

from levanter.data import AsyncDataset
from levanter.models.attention import AttentionMask
from levanter.models.lm_model import LmExample
from levanter.store.jagged_array import JaggedArrayStore
from levanter.utils import jax_utils
from levanter.utils.jax_utils import local_cpu_mesh


# cf https://github.com/tensorflow/tensor2tensor/blob/bafdc1b67730430d38d6ab802cbd51f9d053ba2e/tensor2tensor/data_generators/generator_utils.py#L623

# todo should we use something like this: https://arxiv.org/pdf/2107.02027?

T = TypeVar("T", bound=PyTree)
L = TypeVar("L")


# Python 3.10 can't handle this
# @dataclass(frozen=True)
# class LeafType:
#     leaf_type: type
#
#     def __class_getitem__(cls, item):
#         return cls(item)
#
#
# WithLeaf: TypeAlias = Annotated[T, LeafType[L]]


class SequencePacker:
    """
    Packs sequences into a single LmExample.
    """

    def __init__(self, Pos: hax.Axis, max_pack_size: int, pad_token: int):
        self.Pos = Pos
        self._ids: list[int] = []
        self._segment_ids: list[int] = []
        self._loss_mask: list[int] = []
        self.num_segments = 0
        self.pad_token = pad_token
        self.max_pack_size = max_pack_size
        assert pad_token is not None, "pad_token must be set"

    def can_pack(self, ids: list[int]) -> bool:
        return len(ids) + len(self._ids) <= self.Pos.size and self.num_segments < self.max_pack_size

    def add_example(self, ids: list[int], loss_mask: list[int] | np.ndarray, segment_id: int | None = None):
        if len(ids) != len(loss_mask):
            raise ValueError("ids and loss_mask must have the same length")

        if len(ids) == 0:
            return

        if len(ids) + len(self._ids) > self.Pos.size:
            raise ValueError("Too many tokens")

        if self.num_segments >= self.max_pack_size:
            raise ValueError("Too many segments")

        self._ids.extend(ids)
        if segment_id is None:
            segment_id = self.num_segments

        self.num_segments += 1

        self._segment_ids.extend([segment_id] * len(ids))

        self._loss_mask.extend(loss_mask)

    def pack(self) -> LmExample:
        ids = self._ids + [self.pad_token] * (self.Pos.size - len(self._ids))

        segment_ids = self._segment_ids + [-1] * (self.Pos.size - len(self._segment_ids))

        loss_mask = self._loss_mask + [0] * (self.Pos.size - len(self._loss_mask))

        with local_cpu_mesh():
            tokens = hax.named(ids, self.Pos).astype(jnp.int32)
            segment_ids = hax.named(segment_ids, self.Pos).astype(jnp.int32)
            loss_mask = hax.named(loss_mask, self.Pos).astype(jnp.int32)

            attn_mask = AttentionMask.causal().with_segment_ids(segment_ids)

            return LmExample(tokens=tokens, loss_mask=loss_mask, attn_mask=attn_mask)


@dataclass(frozen=True)
class PromptCompletion:
    ids: list[int]
    prompt_length: int
    segment_id: int | None = None

    def __post_init__(self):
        if len(self.ids) == 0:
            raise ValueError("PromptCompletion must have at least one token")

        # check that there is at least one token in the response
        if len(self.ids) <= self.prompt_length:
            raise ValueError(
                f"PromptCompletion must have strictly more tokens than the prompt length. Got {len(self.ids)} tokens"
                f" and prompt length {self.prompt_length}"
            )


def pack_prompt_completions(
    Pos: hax.Axis,
    sequences: Iterable[PromptCompletion],
    pad_token: int,
    max_segments_per_example: int = 64,
    max_buffered_examples: int = 64,
) -> Iterator[LmExample]:
    """
    Packs a list of prompt completions into LmExamples using the SequencePacker
    """

    packers = [SequencePacker(Pos, max_segments_per_example, pad_token)]

    for sequence in sequences:
        loss_mask = np.arange(len(sequence.ids)) >= sequence.prompt_length - 1
        loss_mask[-1] = 0
        assert np.any(loss_mask)

        for packer in packers:
            if packer.can_pack(sequence.ids):
                packer.add_example(sequence.ids, loss_mask, sequence.segment_id)

                if packer.num_segments == max_segments_per_example:
                    yield packer.pack()
                    packers.remove(packer)
                break
        else:
            # no packer could fit the example, create a new one
            packer = SequencePacker(Pos, max_segments_per_example, pad_token)
            packer.add_example(sequence.ids, loss_mask, sequence.segment_id)
            packers.append(packer)

        while len(packers) >= max_buffered_examples:
            yield packers.pop(0).pack()

    for packer in packers:
        yield packer.pack()


def per_segment_loss(
    packed_example: LmExample, losses: hax.NamedArray, max_Segments: hax.Axis
) -> tuple[hax.NamedArray, hax.NamedArray]:
    """
    Returns a pair of arrays of shape (Segments,), where:

    * the first array is segment ids
    * the second is loss per segment.

    This code is designed to run in a jit-compiled function, meaning we have to careful of shapes
    """

    assert packed_example.attn_mask.segment_ids is not None, "segment_ids must be set in the AttentionMask"

    segment_ids = packed_example.attn_mask.segment_ids
    assert (
        segment_ids.ndim == 1
    ), f"Expected segment_ids to be 1D, got {segment_ids.ndim}. Use vmap if you have multiple examples"
    Pos = packed_example.tokens.axes[0]

    # mask out padding etc
    masked_losses = losses * packed_example.loss_mask

    # sum the losses for each segment
    unique_segment_ids = _unique_segment_ids(max_Segments, segment_ids)

    # Create a mask matrix where each row corresponds to a unique segment
    segment_mask = unique_segment_ids == segment_ids.broadcast_axis(max_Segments)

    segment_mask = segment_mask.astype(masked_losses.dtype)

    segment_losses = hax.dot(segment_mask, masked_losses, axis=Pos)

    return unique_segment_ids, segment_losses


def _unique_segment_ids(max_Segments, segment_ids):
    # Extract unique segment IDs with padding
    # TODO: add unique to haliax
    unique_segment_ids = jnp.unique(segment_ids.array, size=max_Segments.size, fill_value=-1)
    unique_segment_ids = hax.named(unique_segment_ids, max_Segments)
    return unique_segment_ids


def per_segment_correct(
    packed_example: LmExample, correct: hax.NamedArray, max_Segments: hax.Axis
) -> tuple[hax.NamedArray, hax.NamedArray]:
    """
    Returns a pair of arrays of shape (max_segments,), where:

    * the first array is segment ids
    * the second is whether all tokens in the segment are correct.

    This code is designed to run in a jit-compiled function, meaning we have to careful of shapes

    correct is a boolean array of the same shape as the losses array indicating whether the token was correct
    """

    assert packed_example.attn_mask.segment_ids is not None, "segment_ids must be set in the AttentionMask"

    segment_ids = packed_example.attn_mask.segment_ids
    assert (
        segment_ids.ndim == 1
    ), f"Expected segment_ids to be 1D, got {segment_ids.ndim}. Use vmap if you have multiple examples"

    Pos = packed_example.tokens.axes[0]

    # mask out padding etc
    masked_correct = hax.logical_or(correct, hax.logical_not(packed_example.loss_mask))

    # sum the losses for each segment
    # Extract unique segment IDs with padding
    unique_segment_ids = _unique_segment_ids(max_Segments, segment_ids)

    # Create a mask matrix where each row corresponds to a unique segment
    segment_mask = unique_segment_ids == segment_ids.broadcast_axis(max_Segments)

    segment_mask = segment_mask.astype(masked_correct.dtype)

    segment_correct = hax.all(hax.where(segment_mask, masked_correct, True), axis=Pos)

    return unique_segment_ids, segment_correct


class GreedyPrepackedDataset(AsyncDataset[T]):
    """
    Prepacks a dataset into a new dataset where examples are packed into a single example.

    As per usual, I can't help but make this generic.

    Args:
        datastore: The dataset to pack. Can be a single JaggedArrayStore or a PyTree of JaggedArrayStores.
        max_length: The maximum length of each packed example. Can be a single int or a PyTree of ints.
        pad_with_zeros: Whether to pad the packed examples with zeros. Defaults to True.
        slice_too_long_examples: Whether to slice examples that are too long. Keeps from right. Defaults to False.
        max_segments_per_example: The maximum number of segments per example. Defaults to None.
    """

    def __init__(
        self,
        # datastore: WithLeaf[T, JaggedArrayStore],
        # max_length: WithLeaf[T, int],
        datastore: T,
        max_length: T | int,
        *,
        pad_with_zeros: bool = True,
        slice_too_long_examples: bool = False,
        max_segments_per_example: int | None = None,
    ):
        super().__init__()
        self.dataset = datastore
        self.max_segments_per_example = max_segments_per_example
        self.max_length = max_length
        self.pad_with_zeros = pad_with_zeros
        self.slice_too_long_examples = slice_too_long_examples

        # _pack_indices is a list of pytrees, one for each packed example.
        # The leaf of each pytree is a range of data indices into the JaggedArrayStore
        # Because it's greedy, the examples in a packed example are contiguous
        self._pack_indices: list[PyTree[range]] = self._build_pack()

    def is_finite(self) -> bool:
        return True

    async def async_len(self) -> int:
        return len(self._pack_indices)

    async def final_length_is_known(self) -> bool:
        return True

    async def current_len(self) -> Optional[int]:
        return len(self._pack_indices)

    async def get_batch(self, indices: Sequence[int]) -> Sequence[PyTree[np.ndarray]]:
        async def get_data_for_leaf(store: JaggedArrayStore, max_length, *s: range):
            out = []
            with ts.Batch():
                for r in s:
                    assert r.stop - r.start <= max_length
                    out.append(store._data[r.start : r.stop].read())
            out = await asyncio.gather(*out)

            if self.pad_with_zeros:
                out = [np.pad(x, (0, max_length - x.shape[0])) for x in out]

            return out

        # For each leaf dataset, we get all of the data for the packed examples
        leaf_batch_futures: PyTree[Awaitable[list[np.ndarray]]] = jax.tree.map(
            get_data_for_leaf, self.dataset, self.max_length, *[self._pack_indices[i] for i in indices]
        )

        leaf_batch_futures_leaves, structure = jax.tree.flatten(leaf_batch_futures)

        leaf_batch = await asyncio.gather(*leaf_batch_futures_leaves)

        return [jax.jax.tree.unflatten(structure, [leaf[i] for leaf in leaf_batch]) for i in range(len(indices))]

    def _build_pack(self) -> list[PyTree[range]]:
        """
        Pack the dataset using a greedy, contiguous strategy.

        For each JaggedArrayStore leaf in 'dataset' (which represent aligned documents),
        we obtain its offsets array and also the allowed max length from the corresponding leaf in the max_length pytree.
        Then we iterate document by document (documents assumed to be contiguous) and accumulate tokens
        until adding the next document would exceed the allowed number of tokens for any leaf
        or the number of segments would exceed max_segments_per_example (if provided).

        For each pack, we return a PyTree (with the same structure as dataset) where every leaf is a range(start, end)
        indicating that for that leaf, documents from index start to (end-1) are included.
        """
        # attempt to broadcast max_length to match dataset
        max_length = jax_utils.tree_broadcast_to(self.max_length, self.dataset)

        dataset_leaves, tree_def = jax.tree.flatten(self.dataset)
        max_length_leaves, _ = jax.tree.flatten(max_length)

        if len(dataset_leaves) != len(max_length_leaves):
            raise ValueError("Dataset and max_length PyTrees must have the same number of leaves.")

        # For each dataset leaf, fetch its offsets array.
        offsets_list = []
        for store in dataset_leaves:
            offs = store.offsets.read()
            offsets_list.append(offs)

        offsets_list = [fut.result() for fut in offsets_list]

        n_docs = None
        for offs in offsets_list:
            if n_docs is None:
                n_docs = len(offs) - 1
            elif len(offs) - 1 != n_docs:
                raise ValueError("All JaggedArrayStore leaves must have the same number of documents.")

        assert n_docs is not None, "Could not determine number of documents"

        # Now, pack documents greedily.
        pack_indices = []
        i = 0
        while i < n_docs:
            start = i
            total_segments = 0
            # We will increase i greedily until any leaf would overflow its allowed max.
            while i < n_docs:
                # Check optional segment constraint: if adding one more document would exceed max_segments_per_example.
                if self.max_segments_per_example is not None and (total_segments + 1) > self.max_segments_per_example:
                    break
                # For each leaf, check if adding document i would keep the token count within allowed capacity.
                valid = True
                for offs, allowed in zip(offsets_list, max_length_leaves):
                    # Compute total tokens from start to document i+1.

                    # JaggedArrayStore uses offsets[0] to store the number of documents.
                    start_ = offs[start]
                    if start == 0:
                        start_ = 0

                    token_sum = offs[i + 1] - start_
                    if token_sum > allowed:
                        valid = False
                        break
                if not valid:
                    break
                total_segments += 1
                i += 1
            # If no document could be added (should not happen unless a single document exceeds capacity),
            # raise
            if i == start:
                if not self.slice_too_long_examples:
                    raise ValueError(f"Document {start} exceeds allowed capacity.")
                else:
                    i += 1

            # Build the PyTree for this pack: replace every leaf with
            # pack_leaves = [ range(offsets[start], offsets[i]) for offsets in offsets_list]
            pack_leaves = []
            for offs, allowed in zip(offsets_list, max_length_leaves, strict=True):
                start_, end = offs[start], offs[i]
                if start == 0:
                    start_ = 0

                # keep from the right
                start_ = max(end - allowed, start_)

                pack_leaves.append(range(start_, end))
                assert end - start_ <= allowed
                assert end > start_

            pack = jax.tree.unflatten(tree_def, pack_leaves)
            pack_indices.append(pack)
        return pack_indices


if __name__ == "__main__":
    # demo the GreedyPrepackedDataset
    import time

    import numpy as np

    path = "gs://marin-us-central2/tokenized/tulu_sft_v3_llama3_tokenizer-f88fdb/input_ids/"

    store = JaggedArrayStore.open(path, mode="r", dtype=np.uint32, cache_metadata=True)

    time_in = time.time()
    packed = GreedyPrepackedDataset(store, max_length=4096, pad_with_zeros=True, slice_too_long_examples=True)
    time_out = time.time()
    print(f"Took {time_out - time_in:.2f}s to build pack")

    packed_sync = packed.as_sync_dataset()

    padding_count = 0
    total_tokens = 0

    for i in range(10):
        example_batch = packed_sync.get_batch(range(i * 100, (i + 1) * 100))

        for example in example_batch:
            padding_count += np.sum(example == 0)
            total_tokens += example.size

    print(f"Padding rate: {padding_count / total_tokens:.2f}")
