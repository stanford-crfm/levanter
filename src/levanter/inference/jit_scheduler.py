import dataclasses

import equinox as eqx
import haliax as hax
from haliax import NamedArray, haxtyping as ht
from jax import numpy as jnp, random as jrandom
import jax
from jaxtyping import PRNGKeyArray


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
            jit_scheduler = do_generate(jit_scheduler)
            generated_tokens, generated_seq_ids = jit_scheduler.generated_tokens, jit_scheduler.generated_seq_ids

        do_generate might look like this:
        def do_generate(sched: JitScheduler, max_steps: int) -> JitScheduler:

            def body(state):
                sched, tokens, seq_ids = state
                # pack next sequence
                sched, tokens, seq_ids, binfo = sched.pack_next_sequence(max_tokens=32)
                # compute logits
                logits, cache = model.decode(tokens, cache, binfo.pos_ids)
                # sample new tokens
                last_token_idxes_per_seq = get_last_tokens_per_seq(tokens, seq_ids) # padded with max_pos out to max_seq
                new_seq_ids = seq_ids["position", last_token_idxes_per_seq]
                keys = jnp.take(sched.key, new_seq_ids.array, axis=0)
                new_tokens = sample(logits, last_token_idxes_per_seq)
                # update the scheduler with the new tokens
                actual_num_new_tokens = hax.sum(last_token_idxes_per_seq <= max_pos).scalar()
                sched = sched.update_after_sampling(new_tokens, new_seq_ids, num_new_tokens= actual_num_new_tokens)
                return sched, tokens, seq_ids


            while not sched.finished.all():
                # pack next sequence
                sched, tokens, seq_ids, binfo = sched.pack_next_sequence(max_tokens=32)
                # compute logits
                logits = model.decode(tokens, cache, binfo.pos_ids)
                # sample new tokens
                new_tokens, new_seq_ids = sample(tokens, seq_ids)
                # update the scheduler with the new tokens
                sched = sched.update_after_sampling(new_tokens, new_seq_ids, num_new_tokens=len(new_tokens))
            return sched, cache

    """
    # Notes:
    # - generate_tokens and queued_tokens is stored "flat" with seq_ids per token
    generated_tokens: ht.i32[NamedArray, "position"]  # all tokens we have generated so far
    generated_seq_ids: ht.i32[NamedArray, "position"]  # seq ids for each token in generated_tokens
    num_generated_tokens: jax.Array  # total number of tokens generated so far
    queued_tokens: ht.i32[NamedArray, "position"]  # number of tokens ready to be processed in each sequence.
    queued_seq_ids: ht.i32[NamedArray, "position"]
    num_queued_tokens: jax.Array
    finished: ht.bool_[NamedArray, "seq"]  # whether the sequence is finished
    key: PRNGKeyArray  # batched to seq

    # TODO: per-seq sampling params

    @staticmethod
    def init(max_tokens: int, max_seqs: int, key: PRNGKeyArray) -> "JitScheduler":
        """Create a ``JitScheduler`` with empty buffers."""
        return JitScheduler(
            generated_tokens=hax.full({"position": max_tokens}, -1, dtype=jnp.int32),
            generated_seq_ids=hax.full({"position": max_tokens}, -1, dtype=jnp.int32),
            num_generated_tokens=jnp.array(0, dtype=jnp.int32),
            queued_tokens=hax.full({"position": max_tokens}, -1, dtype=jnp.int32),
            queued_seq_ids=hax.full({"position": max_tokens}, -1, dtype=jnp.int32),
            num_queued_tokens=jnp.array(0, dtype=jnp.int32),
            finished=hax.zeros({"seq": max_seqs}, dtype=jnp.bool_),
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
    ) -> tuple["JitScheduler", ht.i32[NamedArray, "position"], ht.i32[NamedArray, "position"]]:  # type: ignore[name-defined]
        """Remove up to ``max_tokens`` tokens from the queue and return them.

        Additional sequences are packed only if all their currently queued tokens
        fit in ``max_tokens``. Shapes stay static inside ``jit``.
        """

        P = self.queued_tokens.axis_size("position")

        def compute_take(q_num: jax.Array) -> jax.Array:
            # which slots currently contain queued tokens?
            pos_idx = jnp.arange(P, dtype=jnp.int32)
            valid_mask = pos_idx < q_num

            ids = self.queued_seq_ids.array
            ones = valid_mask.astype(jnp.int32)

            seg_lens = jax.lax.segment_sum(ones, ids, num_segments=self.max_seqs)

            first_pos = jnp.full(self.max_seqs, P, dtype=jnp.int32)
            first_pos = first_pos.at[ids].min(pos_idx, where=valid_mask, mode="drop")

            sorted_ids = jnp.argsort(first_pos)
            lens_sorted = seg_lens[sorted_ids]
            valid_sorted = lens_sorted > 0
            lens_masked = jnp.where(valid_sorted, lens_sorted, 0)

            cums = jnp.cumsum(lens_masked)
            full_mask = (cums <= max_tokens) & valid_sorted
            num_full = full_mask.sum()

            first_len = lens_masked[0]
            take_cnt = jnp.where(num_full > 0, cums[num_full - 1], jnp.minimum(first_len, max_tokens))
            take_cnt = jnp.minimum(take_cnt, q_num)
            take_cnt = jnp.minimum(take_cnt, max_tokens)
            return take_cnt

        take_cnt = jax.lax.cond(
            self.num_queued_tokens > 0,
            compute_take,
            lambda q: jnp.array(0, dtype=jnp.int32),
            self.num_queued_tokens,
        )

        # Fixed-shape slice for return value
        tokens_full = self.queued_tokens["position", hax.ds(0, max_tokens)]  # type: ignore[name-defined]
        seq_ids_full = self.queued_seq_ids["position", hax.ds(0, max_tokens)]  # type: ignore[name-defined]

        idx = hax.arange(tokens_full.resolve_axis("position"))
        mask_out = idx >= take_cnt
        tokens = hax.where(mask_out, hax.full_like(idx, -1), tokens_full)
        seq_ids = hax.where(mask_out, hax.full_like(idx, -1), seq_ids_full)

        to_take = take_cnt.astype(int)
        rest_tokens = jnp.concatenate(
            [self.queued_tokens.array[to_take:P], -1 * jnp.ones((to_take,), dtype=self.queued_tokens.dtype)],
            axis=0,
        )
        rest_ids = jnp.concatenate(
            [self.queued_seq_ids.array[to_take:P], -1 * jnp.ones((to_take,), dtype=self.queued_seq_ids.dtype)],
            axis=0,
        )

        new_sched = dataclasses.replace(
            self,
            queued_tokens=hax.named(rest_tokens, axis=self.queued_tokens.axes),
            queued_seq_ids=hax.named(rest_ids, axis=self.queued_seq_ids.axes),
            num_queued_tokens=self.num_queued_tokens - take_cnt,
        )

        return new_sched, tokens, seq_ids

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

    # def extract_generated_tokens(
    #     self,
    #     sequence_ids: ht.i32[NamedArray, "seq"],
    #     max_tokens: int,
    # ) -> tuple["JitScheduler", ht.i32[NamedArray, "seq position"]]:
    #     """
    #     JIT‐safe: Extract up to `max_tokens` for each seq in `sequence_ids`,
    #     remove exactly those tokens from the buffer, and return a [seq,position]
    #     array (padded with -1).
    #     """
    #     # aliases
    #     P       = self.generated_tokens.axis_size("position")
    #     S       = sequence_ids.axis_size("seq")
    #     num_raw = self.num_generated_tokens
    #     tok_buf = self.generated_tokens.array    # shape (P,)
    #     id_buf  = self.generated_seq_ids.array   # shape (P,)
    #     req_ids = sequence_ids.array             # shape (S,)
    #
    #     # 1) compute per‐token “local position” within its sequence:
    #     #    we do a segment‐cumsum of 1s, then subtract 1 to get 0-based indices.
    #     ones    = jnp.where(positions < num_raw, 1, 0)
    #     raw_cum = jax.lax.segment_sum(ones, id_buf, num_segments=self.max_seqs)
    #     # BUT segment_sum gives total per-seq at each slot, not prefix sums.
    #     # Instead we can do a trick: sort-by-(seq,id,position) then cumsum then undo the sort.
    #     # For brevity here assume we have an efficient segment_cumsum in Haliax:
    #     local_pos = hax.segment_cumsum(ones, id_buf) - 1      # shape (P,)
    #
    #     # 2) build a [P, max_tokens] one‐hot for each local_pos < max_tokens
    #     in_req      = (id_buf[:,None] == req_ids[None,:]).any(axis=1)  # which slots *could* be extracted
    #     can_extract = in_req & (local_pos < max_tokens)               # mask exactly the first `max_tokens`
    #     # we’ll use this later for removal.
    #
    #     oh = jax.nn.one_hot(local_pos, max_tokens, dtype=tok_buf.dtype)  # (P, T)
    #     masked = tok_buf[:,None] * oh                                     # (P, T)
    #     # zero out any slots beyond num_raw
    #     masked = jnp.where(positions[:,None] < num_raw, masked, 0)
    #
    #     # 3) segment_sum into a [max_seqs, max_tokens] matrix
    #     dense = lax.segment_sum(masked, id_buf, num_segments=self.max_seqs)
    #     # pick only the requested seqs and pad with -1
    #     out = dense[req_ids]                     # (S, T)
    #     out = jnp.where(out==0, -1, out)         # 0→-1 sentinel
    #     out_named = hax.named(out, axes=("seq","position"))
    #
    #     # 4) remove exactly those slots we just extracted:
    #     removal_mask = can_extract                # shape (P,)
    #     keep_mask    = (positions < num_raw) & (~removal_mask)
    #     # pack‐to‐front via argsort of a key
    #     key          = jnp.where(keep_mask, positions, P+positions)
    #     order        = jnp.argsort(key)
    #     new_toks     = tok_buf[order]
    #     new_ids      = id_buf[order]
    #     new_num      = keep_mask.sum()
    #
    #     # zero out the tail
    #     tail = positions >= new_num
    #     new_toks = jnp.where(tail, -1, new_toks)
    #     new_ids  = jnp.where(tail, -1, new_ids)
    #
    #     sched2 = dataclasses.replace(
    #         self,
    #         generated_tokens  = hax.named(new_toks,  axes=self.generated_tokens.axes),
    #         generated_seq_ids = hax.named(new_ids,   axes=self.generated_seq_ids.axes),
    #         num_generated_tokens = new_num,
    #     )
    #     return sched2, out_named
    #
    #
    #
    #
