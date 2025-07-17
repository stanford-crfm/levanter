import dataclasses

import equinox as eqx
import haliax as hax
from haliax import NamedArray, haxtyping as ht
from jax import numpy as jnp, random as jrandom
from jaxtyping import PRNGKeyArray


def masked_set(dest: NamedArray, selector, axis, start, src, num_to_copy) -> NamedArray:
    """
    jit-safe masked memcpy-like operation.
    Copy into dest[selector, axis, start:start+num_to_copy] the values from src[axis, :num_to_copy].

    Probably faster to not use an arange (which lowers to a scatter)
    """

    axis = dest.resolve_axis(axis)
    src_axis = src.resolve_axis(axis.name)
    src_slice = src[src_axis, hax.ds(0, num_to_copy)]
    setter = {**selector, axis: hax.ds(start, num_to_copy)}
    return dest.at[setter].set(src_slice, mode="drop")


class JitScheduler(eqx.Module):
    """
    inside-JIT scheduler for sequences. We assume there is an outer scheduler that manages all sequences, and this
    scheduler handles the sequences in a single macro-round of prefill/decodes. That is, we assume something like:

    ```
        # in an outer thread
        outer_scheduler.enqueue_new_sequences(...)

        while outer_scheduler.has_sequences():
            # add new sequences to the jit scheduler
            jit_scheduler = outer_scheduler.get_next_macro_round(jit_scheduler)
            # do iterative prefill/decode
            jit_scheduler = do_generate(jit_scheduler)
            generated_tokens, generated_seq_ids = jit_scheduler.generated_tokens, jit_scheduler.generated_seq_ids

    """
    generated_tokens: ht.i32[NamedArray, "position"]  # all tokens we have generated so far
    generated_seq_ids: ht.i32[NamedArray, "position"]  # seq ids for each token in generated_tokens
    num_generated_tokens: jnp.int32  # total number of tokens generated so far
    queued_tokens: ht.i16[NamedArray, "position"]  # number of tokens ready to be processed in each sequence.
    queued_seq_ids: ht.i32[NamedArray, "position"]
    num_queued_tokens: jnp.int32
    finished: ht.bool_[NamedArray, "seq"]  # whether the sequence is finished
    key: PRNGKeyArray  # batched to seq

    # TODO: per-seq sampling params

    @staticmethod
    def init(max_tokens: int, max_seqs: int, key: PRNGKeyArray) -> "JitScheduler":
        """Create a ``JitScheduler`` with empty buffers."""
        Pos = hax.Axis("position", max_tokens)
        Seq = hax.Axis("seq", max_seqs)
        return JitScheduler(
            generated_tokens=hax.full(Pos, -1, dtype=jnp.int32),
            generated_seq_ids=hax.full(Pos, -1, dtype=jnp.int32),
            num_generated_tokens=jnp.array(0, dtype=jnp.int32),
            queued_tokens=hax.full(Pos, -1, dtype=jnp.int16),
            queued_seq_ids=hax.full(Pos, -1, dtype=jnp.int32),
            num_queued_tokens=jnp.array(0, dtype=jnp.int32),
            finished=hax.zeros(Seq, dtype=jnp.bool_),
            key=jrandom.split(key, max_seqs),
        )

    @property
    def max_seqs(self) -> int:
        """Maximum number of sequences in this batch."""
        return self.finished.axis_size("seq")

    def enqueue_tokens(
        self,
        new_tokens: ht.i32[NamedArray, "position"],  # type: ignore[name-defined]
        new_seq_ids: ht.i32[NamedArray, "position"],  # type: ignore[name-defined]
        num_new_tokens: int,
    ) -> "JitScheduler":
        """Append ``new_tokens`` and ``new_seq_ids`` to the queue."""

        new_q_tokens = masked_set(
            self.queued_tokens,
            {},
            "position",
            self.num_queued_tokens,
            new_tokens,
            num_new_tokens,
        )
        new_q_seq_ids = masked_set(
            self.queued_seq_ids,
            {},
            "position",
            self.num_queued_tokens,
            new_seq_ids,
            num_new_tokens,
        )

        return dataclasses.replace(
            self,
            queued_tokens=new_q_tokens,
            queued_seq_ids=new_q_seq_ids,
            num_queued_tokens=self.num_queued_tokens + num_new_tokens,
        )
    def update_after_sampling(
        self,
        new_tokens: ht.i32[NamedArray, "position"],  # type: ignore[name-defined]
        new_token_seq_ids: ht.i32[NamedArray, "position"],  # type: ignore[name-defined]
        num_new_tokens: int,
    ) -> "JitScheduler":
        """
        Append new tokens to generated tokens and update the scheduler state.

        generated tokens should also be enqueued in the scheduler for processing in the next round.
        """

        new_g_tokens = masked_set(
            self.generated_tokens,
            {},
            "position",
            self.num_generated_tokens,
            new_tokens,
            num_new_tokens,
        )
        new_g_seq_ids = masked_set(
            self.generated_seq_ids,
            {},
            "position",
            self.num_generated_tokens,
            new_token_seq_ids,
            num_new_tokens,
        )

        updated = dataclasses.replace(
            self,
            generated_tokens=new_g_tokens,
            generated_seq_ids=new_g_seq_ids,
            num_generated_tokens=self.num_generated_tokens + num_new_tokens,
        )

        updated = updated.enqueue_tokens(new_tokens, new_token_seq_ids, num_new_tokens)

        return updated

    def pack_next_sequence(
        self, max_tokens: int
    ) -> tuple[
        "JitScheduler",
        NamedArray,
        NamedArray,
    ]:
        """Remove up to ``max_tokens`` tokens from the queue and return them."""

        pos_axis = self.queued_tokens.resolve_axis("position")
        num = jnp.minimum(self.num_queued_tokens, max_tokens)

        tokens = self.queued_tokens["position", hax.ds(0, max_tokens)]  # type: ignore[name-defined]
        seq_ids = self.queued_seq_ids["position", hax.ds(0, max_tokens)]  # type: ignore[name-defined]

        rolled_tokens = hax.roll(self.queued_tokens, -num, "position")
        rolled_seq_ids = hax.roll(self.queued_seq_ids, -num, "position")
        idx = hax.arange(pos_axis)
        mask = idx >= (pos_axis.size - num)
        filler_tokens = hax.where(mask, hax.full_like(idx, -1), rolled_tokens)
        filler_seq_ids = hax.where(mask, hax.full_like(idx, -1), rolled_seq_ids)

        new_q_tokens = filler_tokens
        new_q_seq_ids = filler_seq_ids

        return (
            dataclasses.replace(
                self,
                queued_tokens=new_q_tokens,
                queued_seq_ids=new_q_seq_ids,
                num_queued_tokens=self.num_queued_tokens - num,
            ),
            tokens,
            seq_ids,
        )
