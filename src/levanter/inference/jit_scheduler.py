import dataclasses

import equinox as eqx
import haliax as hax
from haliax import NamedArray, haxtyping as ht
from jax import numpy as jnp
import jax


def masked_set(dest: NamedArray, selector, axis, start, src, num_to_copy) -> NamedArray:
    """
    jit-safe masked memcpy-like operation.
    Copy into dest[selector, axis, start:start+num_to_copy] the values from src[axis, :num_to_copy].

    Probably faster to not use an arange (which lowers to a scatter) and use blocks? Probably not a bottleneck

    num_to_copy may be dynamic
    """

    src_arange = hax.arange(src.resolve_axis(axis))
    dest_axis_size = dest.axis_size(axis)
    # mask out the tail
    dest_arange = hax.where(src_arange >= num_to_copy, dest_axis_size, src_arange + start)
    src_arange = hax.where(src_arange >= num_to_copy, src_arange.size, src_arange)

    return dest.at[{**selector, axis: dest_arange}].set(src[axis, src_arange], mode="drop")


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
        boundary_positions = hax.where(self.is_boundary, fill_value=-1, new_axis=axis)[0]
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
            generated_tokens, generated_seq_ids = jit_scheduler.generated_tokens, jit_scheduler.generated_seq_ids

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
    generated_tokens: ht.i32[NamedArray, "position"]  # all tokens we have generated so far
    generated_seq_ids: ht.i32[NamedArray, "position"]  # seq ids for each token in generated_tokens
    num_generated_tokens: jax.Array  # total number of tokens generated so far
    queued_tokens: ht.i32[NamedArray, "position"]  # number of tokens ready to be processed in each sequence.
    queued_seq_ids: ht.i32[NamedArray, "position"]
    num_queued_tokens: jax.Array

    # TODO: per-seq sampling params

    @property
    def empty_generated_space(self) -> jnp.ndarray:
        """How many tokens can be generated in the generated tokens buffer."""
        return self.generated_tokens.axis_size("position") - self.num_generated_tokens

    @property
    def empty_queue_space(self) -> jnp.ndarray:
        """How many tokens can be enqueued in the queue."""
        return self.queued_tokens.axis_size("position") - self.num_queued_tokens

    @property
    def max_queued_tokens(self) -> int:
        """Maximum number of tokens that can be buffered in the queue."""
        return self.queued_tokens.axis_size("position")

    @staticmethod
    def init(max_queued_tokens: int, max_buffered_tokens: int) -> "JitScheduler":
        """Create a ``JitScheduler`` with empty buffers."""
        return JitScheduler(
            generated_tokens=hax.full({"position": max_buffered_tokens}, -1, dtype=jnp.int32),
            generated_seq_ids=hax.full({"position": max_buffered_tokens}, -1, dtype=jnp.int32),
            num_generated_tokens=jnp.array(0, dtype=jnp.int32),
            queued_tokens=hax.full({"position": max_queued_tokens}, -1, dtype=jnp.int32),
            queued_seq_ids=hax.full({"position": max_queued_tokens}, -1, dtype=jnp.int32),
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
        filler_tokens = hax.where(mask, hax.full_like(idx, -1), rolled_tokens)
        filler_seq_ids = hax.where(mask, hax.full_like(idx, -1), rolled_seq_ids)

        new_q_tokens = filler_tokens
        new_q_seq_ids = filler_seq_ids

        new_scheduler = dataclasses.replace(
            self,
            queued_tokens=new_q_tokens,
            queued_seq_ids=new_q_seq_ids,
            num_queued_tokens=self.num_queued_tokens - num
        )

        # boundary if seq_id changes and not -1
        is_boundary: hax.NamedArray = (seq_ids != hax.roll(seq_ids, -1, "position")) & (seq_ids != -1)

        last_idx = num - 1
        next_after_last = rolled_seq_ids["position", 0]
        boundary_last = (
                (seq_ids["position", last_idx] != next_after_last)
                & (seq_ids["position", last_idx] != -1)
        )

        is_boundary = is_boundary.at["position", last_idx].set(boundary_last)

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
        out = PackedSequence(
            tokens=self.generated_tokens,
            seq_ids=self.generated_seq_ids,
            num_tokens=self.num_generated_tokens,
            is_boundary=hax.zeros_like(self.generated_tokens, dtype=jnp.bool_),
        )

        updated = dataclasses.replace(
            self,
            generated_tokens=hax.full_like(self.generated_tokens, -1),
            generated_seq_ids=hax.full_like(self.generated_seq_ids, -1),
            num_generated_tokens=jnp.zeros((), dtype=jnp.int32),
        )

        return updated, out

    def extract_generated_tokens(
        self,
        sequence_ids: ht.i32[NamedArray, "seq"],  # type: ignore[name-defined]
        max_tokens: int,
    ) -> tuple["JitScheduler", ht.i32[NamedArray, "seq position"]]:
        """
        Extract *at most* `max_tokens` tokens for each requested sequence, pad with
        –1, and remove exactly those tokens from `generated_tokens` / `generated_seq_ids`.
        Shapes stay static, so the method is fully JIT-safe.
        """
        # ---------- shorthands ----------
        P = self.generated_tokens.axis_size("position")  # buffer size
        num_valid = self.num_generated_tokens  # scalar
        tok_buf = self.generated_tokens.array  # (P,)
        id_buf = self.generated_seq_ids.array  # (P,)
        req_ids = sequence_ids.array  # (S,)
        pos_idx = jnp.arange(P, dtype=jnp.int32)  # (P,)
        valid_mask = pos_idx < num_valid  # (P,)

        # ---------- 1. mark tokens to TAKE (first `max_tokens` per seq) ----------
        # seq_matches[s, p] == True  iff  slot p belongs to req_ids[s] and is valid
        seq_matches = (id_buf[None, :] == req_ids[:, None]) & valid_mask[None, :]  # (S,P)
        # rank within its sequence: 0,1,2,… via prefix-sum
        prefix = jnp.cumsum(seq_matches.astype(jnp.int32), axis=1) - 1  # (S,P)
        take_mask_per_seq = seq_matches & (prefix < max_tokens)  # (S,P)
        removal_mask = jnp.any(take_mask_per_seq, axis=0)  # (P,)
        keep_mask = valid_mask & (~removal_mask)  # (P,)

        # ---------- 2. build the [seq, position] output ----------
        def gather_row(mask_row):
            key = jnp.where(mask_row, pos_idx, P + pos_idx)  # stable order
            idx = jnp.argsort(key)[:max_tokens]  # indices of taken tokens
            vals = tok_buf[idx]
            good = mask_row[idx]
            return jnp.where(good, vals, -1)

        gathered = jax.vmap(gather_row)(take_mask_per_seq)  # (S, max_tokens)
        out_named = hax.named(gathered, axis=("seq", "position"))

        # ---------- 3. repack remaining tokens ----------
        key_keep = jnp.where(keep_mask, pos_idx, P + pos_idx)
        order_keep = jnp.argsort(key_keep)  # moves kept → front
        new_toks = tok_buf[order_keep]
        new_ids = id_buf[order_keep]

        new_num = keep_mask.sum()  # how many kept tokens
        tail_mask = pos_idx >= new_num
        new_toks = jnp.where(tail_mask, -1, new_toks)
        new_ids = jnp.where(tail_mask, -1, new_ids)

        updated = dataclasses.replace(
            self,
            generated_tokens=hax.named(new_toks, axis=self.generated_tokens.axes),
            generated_seq_ids=hax.named(new_ids, axis=self.generated_seq_ids.axes),
            num_generated_tokens=new_num,
        )

        return updated, out_named
