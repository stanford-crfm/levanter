import equinox as eqx
import haliax as hax
import jax
from haliax import NamedArray, haxtyping as ht
from jax import numpy as jnp, random as jrandom
from jaxtyping import PRNGKeyArray


def masked_set(dest: NamedArray, selector, axis, start, src, num_to_copy) -> NamedArray:
    """
    jit-safe masked memcpy-like operation.
    Copy into dest[selector, axis, start:start+num_to_copy] the values from src[axis, :num_to_copy].

    Probably faster to not use an arange (which lowers to a scatter)
    """

    dest_size = dest.axis_size(axis)
    dest_range = hax.arange(axis, start=start)
    dest_range = hax.where(dest_range < dest_size, dest_range, dest_size)

    axis = src.resolve_axis(axis)

    setter = {**selector, axis: dest_range}
    return dest.at[setter].set(src, mode="drop")


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

    @property
    def max_seqs(self) -> int:
        """Maximum number of sequences in this batch. Raises if not batched over "seq" axis."""
        return self.tokens.axis_size("seq")

    def enqueue_tokens(self,
                       new_tokens: ht.i32[NamedArray, "position"],
                       new_seq_ids: ht.i32[NamedArray, "position"],
                       num_new_tokens: int) -> "JitScheduler":
        """
        Enqueue new tokens to the scheduler to be processed.

        An unchecked assumption is that there are no already enqueued tokens for the sequences in `new_seq_ids`.
        """



    def update_after_sampling(self, new_tokens: ht.i32[NamedArray, "position"],
                              new_token_seq_ids: ht.i32[NamedArray, "position"],
                              num_new_tokens: int) -> "JitScheduler":
        """
        Append new tokens to generated tokens and update the scheduler state.

        generated tokens should also be enqueued in the scheduler for processing in the next round.
        """

        # TODO: implement this

        # We assume that new_tokens is a 1D array with one token per sequence
        # dests = self.num_tokens.at["seq", new_token_seq_ids].get(mode="fill", fill_value=self.max_seqs)
        # new_tokens = self.tokens.at["seq", new_token_seq_ids, "dests", dests].set(new_tokens)
        #
        # new_num_tokens = self.num_tokens.at["seq", new_token_seq_ids].add(1, mode="drop")
        # new_queued_tokens = self.queued_tokens.at["seq", new_token_seq_ids].set(1, mode="drop")
        #
        # # update key for each updated sequence
        # new_keys = jax.vmap(jrandom.split)(self.key)
        # is_updated_seq = new_num_tokens != self.num_tokens
        # new_keys = jnp.where(is_updated_seq, new_keys, self.key)
        #
        # return JitScheduler(
        #     tokens=new_tokens,
        #     num_tokens=new_num_tokens,
        #     seq_ids=self.seq_ids,
        #     queued_tokens=new_queued_tokens,
        #     key=new_keys,
        # )

    def pack_next_sequence(self,
                           max_tokens: int
                           ) -> tuple["JitScheduler", ht.i32[NamedArray, "position"], ht.i32[NamedArray, "position"]]:
        """
        Pack the next sequence to process.
        This should dequeue the next sequence max_sequence to process, and return the tokens and seq_ids to process.
        """
