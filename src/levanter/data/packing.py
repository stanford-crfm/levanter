# Implements sequence packing
from dataclasses import dataclass
from typing import Iterator

import jax.numpy as jnp
import numpy as np

import haliax as hax

from levanter.models.attention import AttentionMask
from levanter.models.lm_model import LmExample


# cf https://github.com/tensorflow/tensor2tensor/blob/bafdc1b67730430d38d6ab802cbd51f9d053ba2e/tensor2tensor/data_generators/generator_utils.py#L623

# todo should we use something like this: https://arxiv.org/pdf/2107.02027?


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

        tokens = hax.named(ids, self.Pos)
        segment_ids = hax.named(segment_ids, self.Pos)
        loss_mask = hax.named(loss_mask, self.Pos)

        attn_mask = AttentionMask.causal().with_segment_ids(segment_ids)

        return LmExample(tokens=tokens, loss_mask=loss_mask, attn_mask=attn_mask)


@dataclass
class PromptCompletion:
    ids: list[int]
    prompt_length: int
    segment_id: int | None = None


def pack_prompt_completions(
    Pos: hax.Axis,
    sequences: list[PromptCompletion],
    pad_token: int,
    max_pack_size: int = 64,
    max_buffered_examples: int = 64,
) -> Iterator[LmExample]:
    """
    Packs a list of prompt completions into LmExamples using the SequencePacker
    """

    packers = [SequencePacker(Pos, max_pack_size, pad_token)]

    for sequence in sequences:
        loss_mask = np.arange(len(sequence.ids)) >= sequence.prompt_length - 1
        loss_mask[-1] = 0

        for packer in packers:
            if packer.can_pack(sequence.ids):
                packer.add_example(sequence.ids, loss_mask, sequence.segment_id)

                if packer.num_segments == max_pack_size:
                    yield packer.pack()
                    packers.remove(packer)
                break
        else:
            # no packer could fit the example, create a new one
            packer = SequencePacker(Pos, max_pack_size, pad_token)
            packer.add_example(sequence.ids, loss_mask, sequence.segment_id)
            packers.append(packer)

        while len(packers) >= max_buffered_examples:
            yield packer.pack()
            packers.pop(0)

    for packer in packers:
        yield packer.pack()


def per_segment_loss(
    packed_example: LmExample, losses: hax.NamedArray, max_segments: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Returns a pair of arrays of shape (max_segments,), where the first array is segment ids
    and the second is loss per segment.

    This code is designed to run in a jit-compiled function, meaning we have to careful of shapes
    """

    assert packed_example.attn_mask.segment_ids is not None, "segment_ids must be set in the AttentionMask"

    segment_ids = packed_example.attn_mask.segment_ids.array
    assert (
        segment_ids.ndim == 1
    ), f"Expected segment_ids to be 1D, got {segment_ids.ndim}. Use vmap if you have multiple examples"

    # mask out padding etc
    masked_losses = losses * packed_example.loss_mask

    # sum the losses for each segment
    # Extract unique segment IDs with padding
    unique_segment_ids = jnp.unique(segment_ids, size=max_segments, fill_value=-1)

    # Create a mask matrix where each row corresponds to a unique segment
    segment_mask = unique_segment_ids[:, None] == segment_ids[None, :]  # [segment, len]

    segment_mask = segment_mask.astype(masked_losses.dtype)

    # segment_losses = jnp.esum(losses * segment_mask, axis=1) # [segment]
    segment_losses = jnp.einsum("ij,j->i", segment_mask, masked_losses.array)

    return unique_segment_ids, segment_losses
