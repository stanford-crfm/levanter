import dataclasses

import equinox as eqx
import haliax as hax
from haliax import NamedArray, haxtyping as ht
from jax import numpy as jnp
import jax

from levanter.inference.utils import INVALID, masked_set, is_valid


class PackedSequence(eqx.Module):
    """
    A sequence of tokens packed into a single array, with
    This is used to pack sequences into a single array for efficient processing.

    Sequence boundaries are stored as indices in the `boundary_idx` array, which is used to
    determine sequence end points (which in turn are used for sampling)
    Note that this is basically just wherever the sequence id changes, except that if the last sequence is not full,
    it will not have a boundary index.

    (Sequence can be not-full in the case of chunked prefill.)
    """

    tokens: ht.i32[NamedArray, "position"]  # packed tokens
    seq_ids: ht.i32[NamedArray, "position"]  # sequence ids for each token
    num_tokens: jax.Array  # number of tokens in the packed sequence
    is_boundary: ht.bool_[NamedArray, "position"]  # boolean mask for sequence boundaries

    def boundary_indices(self, max_boundaries: int) -> ht.i32[NamedArray, "position"]:  # type: ignore[name-defined]
        """
        Returns the indices of the sequence boundaries in the packed sequence.
        The boundaries are determined by the `is_boundary` mask.
        """
        # Get the positions where the boundary is True
        axis = self.is_boundary.resolve_axis("position").resize(max_boundaries)
        boundary_positions = hax.where(self.is_boundary, fill_value=INVALID, new_axis=axis)[0]
        return boundary_positions


class JitScheduler(eqx.Module):
    """
    inside-JIT scheduler for sequences. We assume there is an outer scheduler that manages all sequences, and this
    scheduler handles the sequences in a single macro-round of prefill/decodes. That is, we assume something like:

    ```
        # in an outer thread
        outer_scheduler.enqueue_new_sequences(...)

        # intiialization
        page_table = PageTable.init(...)
        cache = model.initial_cache(page_table, dtype=jnp.bfloat16)
        jit_scheduler = JitScheduler.init(...)

        while outer_scheduler.has_sequences():
            # add new sequences to the jit scheduler
            jit_scheduler = outer_scheduler.get_next_macro_round(jit_scheduler)
            # do iterative prefill/decode
            jit_scheduler = do_generate(jit_scheduler, page_table, cache, MAX_STEPS)
            generated_tokens = jit_scheduler.generated_tokens

        do_generate might look like this:
        def do_generate(sched: JitScheduler, cache, page_table,  max_steps: int) -> JitScheduler:
            def cond(state):
                _sched, *_ , step = state
                return (step < max_new_tokens) & (~jnp.all(_sched.finished.array))

            def body(state):
                sched: JitScheduler
                sched, page_table, cache, key, step = state

                # Pack the next chunk from the queue
                sched, chunk_tokens, chunk_seq_ids = sched.pack_next_sequence(max_tokens_per_round)

                # Allocate cache pages for this chunk
                page_table, binfo = page_table.allocate_for_seq(token_seq_ids=chunk_seq_ids)

                # Decode logits and sample new tokens
                logits, cache = model.decode(chunk_tokens, cache, binfo, binfo.pos_ids)
                sample_key, key = jrandom.split(key)
                logits = logits["position", binfo.last_token_idx]
                new_tokens, _ = sampler(logits, temps, key=sample_key)

                num_new_tokens = hax.sum(binfo.last_token_idx != -1).scalar()

                # Update scheduler with the freshly sampled tokens
                sched = sched.update_after_sampling(
                    new_tokens=new_tokens,
                    new_token_seq_ids=chunk_seq_ids,
                    num_new_tokens=num_new_tokens,
                )
                return sched, page_table, cache, key, step + 1

            init_state = (sched, page_table, cache, key, jnp.array(0, dtype=jnp.int32))
            sched, page_table, cache, key, _ = jax.lax.while_loop(cond, body, init_state)
            return sched, cache, page_table, key

    """
    # Notes:
    # - generate_tokens and queued_tokens is stored "flat" with seq_ids per token
    generated_tokens: ht.i32[NamedArray, "seq position"]  # tokens generated per sequence
    num_generated_tokens: ht.i32[NamedArray, "seq"]  # number of generated tokens per sequence
    queued_tokens: ht.i32[NamedArray, "position"]  # number of tokens ready to be processed in each sequence.
    queued_seq_ids: ht.i32[NamedArray, "position"]
    num_queued_tokens: jax.Array

    # TODO: per-seq sampling params

    @property
    def empty_generated_space(self) -> jnp.ndarray:
        """How many tokens can be generated in the generated tokens buffer."""
        total_cap = self.generated_tokens.axis_size("position") * self.generated_tokens.axis_size("seq")
        return total_cap - hax.sum(self.num_generated_tokens).array

    @property
    def empty_queue_space(self) -> jnp.ndarray:
        """How many tokens can be enqueued in the queue."""
        return self.queued_tokens.axis_size("position") - self.num_queued_tokens

    @property
    def max_queued_tokens(self) -> int:
        """Maximum number of tokens that can be buffered in the queue."""
        return self.queued_tokens.axis_size("position")

    @staticmethod
    def init(max_seqs: int, max_queued_tokens: int, max_buffered_tokens: int) -> "JitScheduler":
        """Create a ``JitScheduler`` with empty buffers."""
        return JitScheduler(
            generated_tokens=hax.full({"seq": max_seqs, "position": max_buffered_tokens}, INVALID, dtype=jnp.int32),
            num_generated_tokens=hax.zeros({"seq": max_seqs}, dtype=jnp.int32),
            queued_tokens=hax.full({"position": max_queued_tokens}, INVALID, dtype=jnp.int32),
            queued_seq_ids=hax.full({"position": max_queued_tokens}, INVALID, dtype=jnp.int32),
            num_queued_tokens=jnp.array(0, dtype=jnp.int32),
        )

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

        generated tokens will also be enqueued in the scheduler for processing in the next round.
        """

        # first sort tokens by seq id so we can append them per sequence
        sort_order = hax.argsort(new_token_seq_ids, axis="position")
        tokens = new_tokens["position", sort_order]
        seq_ids = new_token_seq_ids["position", sort_order]

        def body(i, state):
            g_tokens, g_counts = state
            seq_id = seq_ids["position", i].scalar()

            def update(state):
                g_tokens, g_counts = state
                pos = g_counts["seq", seq_id].scalar()
                g_tokens = g_tokens.at["seq", seq_id, "position", pos].set(tokens["position", i])
                g_counts = g_counts.at["seq", seq_id].add(1)
                return g_tokens, g_counts

            return jax.lax.cond(is_valid(seq_id), update, lambda s: s, state)

        init_state = (self.generated_tokens, self.num_generated_tokens)
        new_g_tokens, new_counts = jax.lax.fori_loop(0, num_new_tokens, body, init_state)

        updated = dataclasses.replace(
            self,
            generated_tokens=new_g_tokens,
            num_generated_tokens=new_counts,
        )

        updated = updated.enqueue_tokens(new_tokens, new_token_seq_ids, num_new_tokens)

        return updated

    def pack_next_sequence(
        self, max_tokens: int
    ) -> tuple["JitScheduler", PackedSequence]:  # type: ignore[name-defined]
        """
        Dequeue up to ``max_tokens`` tokens from the queue and return them.

        Returns the updated scheduler, the tokens, sequence ids and number of actual tokens that were dequeued.
        """

        pos_axis = self.queued_tokens.resolve_axis("position")
        num = jnp.minimum(self.num_queued_tokens, max_tokens)

        tokens = self.queued_tokens["position", hax.ds(0, max_tokens)]  # type: ignore[name-defined]
        seq_ids = self.queued_seq_ids["position", hax.ds(0, max_tokens)]  # type: ignore[name-defined]

        rolled_tokens = hax.roll(self.queued_tokens, -num, "position")
        rolled_seq_ids = hax.roll(self.queued_seq_ids, -num, "position")
        idx = hax.arange(pos_axis)
        mask = idx >= (pos_axis.size - num)
        filler_tokens = hax.where(mask, hax.full_like(idx, INVALID), rolled_tokens)
        filler_seq_ids = hax.where(mask, hax.full_like(idx, INVALID), rolled_seq_ids)

        new_q_tokens = filler_tokens
        new_q_seq_ids = filler_seq_ids

        new_scheduler = dataclasses.replace(
            self,
            queued_tokens=new_q_tokens,
            queued_seq_ids=new_q_seq_ids,
            num_queued_tokens=self.num_queued_tokens - num
        )

        # boundary if seq_id changes and not INVALID
        is_boundary: hax.NamedArray = (seq_ids != hax.roll(seq_ids, -1, "position")) & (seq_ids != INVALID)

        last_idx = num - 1
        next_after_last = rolled_seq_ids["position", 0]
        boundary_last = (
                (seq_ids["position", last_idx] != next_after_last)
                & (seq_ids["position", last_idx] != INVALID)
        )
        is_boundary = is_boundary.at["position", last_idx].set(boundary_last)

        # now ensure seqids are sorted

        seqids_sort_order = hax.argsort(seq_ids, axis="position")
        tokens = tokens["position", seqids_sort_order]
        seq_ids = seq_ids["position", seqids_sort_order]
        is_boundary = is_boundary["position", seqids_sort_order]

        sequence = PackedSequence(
            tokens=tokens,
            seq_ids=seq_ids,
            num_tokens=num,
            is_boundary=is_boundary
        )

        return new_scheduler, sequence

    @eqx.filter_jit(donate="all")
    def extract_all_generated_tokens(self) -> tuple["JitScheduler", PackedSequence]:
        """
        Extract all generated tokens and sequence ids, returning a new scheduler with empty buffers.
        This is used to finalize the generation process and retrieve all generated tokens.
        """
        P = self.generated_tokens.axis_size("position")
        S = self.generated_tokens.axis_size("seq")

        out_tokens = hax.full({"position": P * S}, INVALID, dtype=jnp.int32)
        out_seq_ids = hax.full({"position": P * S}, INVALID, dtype=jnp.int32)

        def body(i, state):
            toks, ids, start = state
            count = self.num_generated_tokens["seq", i].scalar()

            def tok_loop(j, carry):
                toks, ids = carry
                tok = self.generated_tokens["seq", i, "position", j]
                toks = toks.at["position", start + j].set(tok)
                ids = ids.at["position", start + j].set(jnp.array(i, dtype=jnp.int32))
                return toks, ids

            toks, ids = jax.lax.fori_loop(0, count, tok_loop, (toks, ids))
            return toks, ids, start + count

        init = (out_tokens, out_seq_ids, 0)
        out_tokens, out_seq_ids, total = jax.lax.fori_loop(0, S, body, init)

        out = PackedSequence(
            tokens=out_tokens,
            seq_ids=out_seq_ids,
            num_tokens=total,
            is_boundary=hax.zeros_like(out_tokens, dtype=jnp.bool_),
        )

        updated = dataclasses.replace(
            self,
            generated_tokens=hax.full_like(self.generated_tokens, INVALID),
            num_generated_tokens=hax.zeros_like(self.num_generated_tokens),
        )

        return updated, out

    def purge_queue_of_seq(self, seq_id) -> "JitScheduler":
        """
        Remove all tokens from the queue that belong to the given sequence ID.
        Slides remaining tokens to the front of the queue.
        """

        is_seq_id = self.queued_seq_ids == seq_id
        new_num_queued_tokens = self.num_queued_tokens - hax.sum(is_seq_id).scalar()
        remaining_seq_id_pos = hax.where(~is_seq_id, fill_value=INVALID, new_axis=self.queued_seq_ids.resolve_axis("position"))[0]
        new_seq_ids = self.queued_seq_ids.at["position", remaining_seq_id_pos].get(mode="fill", fill_value=INVALID)
        new_tokens = self.queued_tokens.at["position", remaining_seq_id_pos].get(mode="fill", fill_value=INVALID)


        return dataclasses.replace(
            self,
            queued_tokens=new_tokens,
            queued_seq_ids=new_seq_ids,
            num_queued_tokens=new_num_queued_tokens,
        )


    def extract_generated_tokens(
        self,
        sequence_ids: ht.i32[NamedArray, "seq"],  # type: ignore[name-defined]
        max_tokens: int,
    ) -> tuple["JitScheduler", ht.i32[NamedArray, "seq position"]]:
        """
        Extract *at most* ``max_tokens`` tokens for each requested sequence, pad with
        INVALID, and clear them from ``generated_tokens``.
        """
        out_shape = {"seq": sequence_ids.axis_size("seq"), "position": max_tokens}
        out = hax.full(out_shape, INVALID, dtype=jnp.int32)

        def body(i, state):
            g_tokens, g_counts, out_tokens = state
            seq_id = sequence_ids["seq", i].scalar()

            def do(state):
                g_tokens, g_counts, out_tokens = state
                available = g_counts["seq", seq_id].scalar()
                n = jnp.minimum(available, max_tokens)

                def tok_loop(j, carry):
                    g_tokens, out_tokens = carry
                    tok = g_tokens["seq", seq_id, "position", j]
                    out_tokens = out_tokens.at["seq", i, "position", j].set(tok)
                    g_tokens = g_tokens.at["seq", seq_id, "position", j].set(INVALID)
                    return g_tokens, out_tokens

                g_tokens, out_tokens = jax.lax.fori_loop(0, n, tok_loop, (g_tokens, out_tokens))
                # shift remaining tokens to the front
                total_pos = g_tokens.axis_size("position")
                rolled = hax.roll(g_tokens["seq", seq_id], -n, "position")
                idx = hax.arange(g_tokens.resolve_axis("position"))
                mask = idx >= (total_pos - n)
                rolled = hax.where(mask, hax.full_like(idx, INVALID), rolled)
                g_tokens = g_tokens.at["seq", seq_id].set(rolled)
                g_counts = g_counts.at["seq", seq_id].add(-n)
                return g_tokens, g_counts, out_tokens

            return jax.lax.cond(is_valid(seq_id), do, lambda s: s, state)

        init_state = (self.generated_tokens, self.num_generated_tokens, out)
        new_tokens, new_counts, out = jax.lax.fori_loop(0, sequence_ids.axis_size("seq"), body, init_state)

        updated = dataclasses.replace(
            self,
            generated_tokens=new_tokens,
            num_generated_tokens=new_counts,
        )

        return updated, out

    def cleared(self) -> "JitScheduler":
        """
        Returns a new JitScheduler with all buffers cleared.
        This is useful for resetting the scheduler state.
        """
        return JitScheduler.init(
            max_seqs=self.generated_tokens.axis_size("seq"),
            max_queued_tokens=self.queued_tokens.axis_size("position"),
            max_buffered_tokens=self.generated_tokens.axis_size("position"),
        )


    def debug_print(self, prefix: str = ""):

        def callback(self):
            print(f"{prefix}JitScheduler State:")
            print(f"{prefix}Generated Tokens: {self.generated_tokens}")
            print(f"{prefix}Num Generated Tokens: {self.num_generated_tokens}")
            print(f"{prefix}Queued Tokens: {self.queued_tokens}")
            print(f"{prefix}Queued Seq IDs: {self.queued_seq_ids}")
            print(f"{prefix}Num Queued Tokens: {self.num_queued_tokens}")

        jax.experimental.io_callback(callback,  None, ordered=True, self=self)
