import dataclasses

import equinox as eqx
import haliax as hax
import jaxtyping
from haliax import NamedArray, haxtyping as ht
from jax import numpy as jnp
import jax

from levanter.inference.utils import INVALID, masked_set, is_valid, is_stop_signal, purge


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

    def token_counts_per_sequence(self, max_sequences: int) -> ht.i32[NamedArray, "seq"]:  # type: ignore[name-defined]
        """
        Returns the number of tokens per sequence in the packed sequence.
        The result is a vector of size `max_sequences`, where each entry corresponds to a sequence ID.
        """
        raw_seq_ids = self.seq_ids.array
        weights = jnp.where(jnp.arange(len(raw_seq_ids)) < self.num_tokens, 1, 0)
        counts = jnp.bincount(raw_seq_ids, weights=weights, length=max_sequences)

        return hax.named(counts, axis=("seq",))

    def boundary_indices(self, max_boundaries: int) -> ht.i32[NamedArray, "position"]:  # type: ignore[name-defined]
        """
        Returns the indices of the sequence boundaries in the packed sequence.
        The boundaries are determined by the `is_boundary` mask.
        """
        # Get the positions where the boundary is True
        axis = self.is_boundary.resolve_axis("position").resize(max_boundaries)
        boundary_positions = hax.where(self.is_boundary, fill_value=INVALID, new_axis=axis)[0]
        return boundary_positions


class SeqDecodingParams(eqx.Module):
    """Per-sequence decoding parameters."""
    max_num_tokens: jnp.ndarray
    stop_tokens: ht.i32[NamedArray, "stop_seq position"] | None
    temperature: jnp.ndarray
    key: jaxtyping.PRNGKeyArray

    @staticmethod
    def default() -> "SeqDecodingParams":
        """
        Returns a default SeqDecodingParams with the given number of stop sequences and maximum stop tokens.
        """
        max_int_jnp = jnp.iinfo(jnp.int32).max
        return SeqDecodingParams(
            max_num_tokens=jnp.array(max_int_jnp - 100000, dtype=jnp.int32),
            stop_tokens=None,
            temperature=jnp.array(0.0, dtype=jnp.float32),
            key=jax.random.key(0)
        )


class DecodeState(eqx.Module):
    """
    State of sequences during decoding. This manages a "hot set" of sequences that are currently being decoded.

    * `seq_id` is a buffer of sequence IDs, which is used to identify sequences in the `tokens` buffer. It is
    the "global" sequence ID. (That is, there might be more sequences than `seq_id.size`, but only the ones that are
    currently being decoded are stored in this buffer.)
    * `tokens` is a buffer of tokens for each sequence. It includes any prompt/prefix.
    * `num_tokens` is a buffer of the number of tokens generated in the current cycle for each sequence.
    * `prefix_len` is a buffer of prefix lengths for each sequence. This is the length of tokens in the `tokens` buffer
      that were provided and not generated in the current cycle.
    * `logprobs` is an optional buffer of log probabilities for the tokens. If not None, it should have the same shape
       as `tokens`, i.e. `logprobs["seq", i, "position", j]` is the log probability of the token at position `j` in
       sequence `i`. It is kept in sync with `tokens`, i.e. if a token is generated, its log probability is also
       generated. We don't currently compute log probabilities for the prefix tokens, so `logprobs` is set to nan for
       those positions.
    """
    seq_id: ht.i32[NamedArray, "seq"]  # sequence ID. This is the "global" sequence ID
    tokens: ht.i32[NamedArray, "seq position"]
    """ most recent tokens generated for each sequence. Should always start at a page boundary. """
    logprobs: ht.Float[NamedArray, "seq position"] | None  # log probabilities of the tokens
    num_tokens: ht.i32[NamedArray, "seq"]
    """Number of tokens in the buffer right now."""
    prefix_len: ht.i32[NamedArray, "seq"]
    """ Length of the prefix for each sequence."""

    kv_pages: ht.i32[NamedArray, "seq page"]
    """Key-value pages for each sequence. This is used to store the key-value pairs for the sequences."""
    page_size: int = eqx.field(static=True)

    # Per sequence sampling parameters
    max_num_tokens: ht.i32[NamedArray, "seq"]
    """
    Maximum number of tokens for each sequence. This is used to limit the number of tokens generated.
    This is inclusive of the prefix length, i.e. the total number of tokens that can be generated for each sequence.
    """
    stop_tokens: ht.i32[NamedArray, "seq stop_seq position"] | None
    """Stop sequences for each sequence. If None, no stop sequences are used. **Left padded** with pad_token_id."""
    temperature: ht.Float[NamedArray, "seq"]
    """temperature for sampling. 0 means greedy sampling"""
    prng_keys: jaxtyping.PRNGKeyArray
    """one per sequence, used for sampling. This is a JAX PRNG key, so it can be split to get new keys."""

    def prng_keys_for(self, seq_ids: ht.i32[NamedArray, "position"], pos_ids: ht.i32[NamedArray, "position"]) -> jaxtyping.PRNGKeyArray:  # type: ignore[name-defined]
        """
        Get the PRNG keys for the given sequence IDs and positions.
        This is used to sample new tokens for the given sequence IDs and positions.
        """
        # We assume that seq_ids and pos_ids are aligned
        per_pos_keys = self.prng_keys[seq_ids.array]
        return jax.vmap(jax.random.fold_in)(per_pos_keys, pos_ids.array)

    @property
    def max_seqs(self) -> int:
        """Number of sequences in the buffer."""
        return self.seq_id.axis_size("seq")

    @property
    def max_tokens(self) -> int:
        """Maximum number of tokens that can be generated for each sequence, including any prefix tokens."""
        return self.tokens.axis_size("position")

    @property
    def max_stop_seq_len(self) -> int:
        """Maximum number of stop sequences for each sequence."""
        if self.stop_tokens is None:
            return 0
        return self.stop_tokens.axis_size("position")

    @eqx.filter_jit
    def assign_seq(self,
                   local_seq_id: int,
                   global_seq_id: int,
                   kv_pages: ht.i32[NamedArray, "page"],  # type: ignore[name-defined]
                   tokens: ht.i32[NamedArray, "position"],  # type: ignore[name-defined]
                   prefix_len: int,
                   seq_params: SeqDecodingParams | None = None) -> "DecodeState":
        """Assign a new sequence to the given local slot."""
        num = tokens.axis_size("position")

        row_tokens = self.tokens["seq", local_seq_id]
        row_tokens = masked_set(row_tokens, "position", 0, tokens, num)
        new_tokens = self.tokens.at["seq", local_seq_id].set(row_tokens)

        new_state = dataclasses.replace(
            self,
            kv_pages=self.kv_pages.at["seq", local_seq_id].set(kv_pages),
            seq_id=self.seq_id.at["seq", local_seq_id].set(global_seq_id),
            tokens=new_tokens,
            # set log probs to nan for the prefix tokens
            logprobs=self.logprobs.at["seq", local_seq_id, "position", 0:prefix_len].set(jnp.nan) if self.logprobs is not None else None,
            num_tokens=self.num_tokens.at["seq", local_seq_id].set(prefix_len),
            prefix_len=self.prefix_len.at["seq", local_seq_id].set(prefix_len),
        )

        if seq_params is not None:
            new_state = dataclasses.replace(
                new_state,
                max_num_tokens=new_state.max_num_tokens.at["seq", local_seq_id].set(seq_params.max_num_tokens),
                temperature=new_state.temperature.at["seq", local_seq_id].set(seq_params.temperature),
                prng_keys=self.prng_keys.at[local_seq_id].set(seq_params.key)  # type: ignore[name-defined]
            )
            match (new_state.stop_tokens, seq_params.stop_tokens):
                case (None, None):
                    pass
                case (None, _):
                    raise ValueError("DecodeState was initialized without stop token storage")
                case (stops, None):
                    # this is fine, just fill this sequence with the pad token
                    assert stops is not None  # make mypy happy
                    new_stop_tokens = stops.at["seq", local_seq_id].set(INVALID)
                    new_state = dataclasses.replace(new_state, stop_tokens=new_stop_tokens)
                case (stops, seq_stops):
                    # too fancy, but we allow for different stop sequences per sequence etc.
                    # Probably better to do this in python outside of the jit loop
                    assert stops is not None  # make mypy happy
                    assert seq_stops is not None  # make mypy happy
                    seq_num_stops = seq_stops.axis_size("stop_seq")
                    seq_stop_len = seq_stops.axis_size("position")
                    this_row_full = hax.full_like(stops["seq", local_seq_id], INVALID)
                    this_row_full = this_row_full.at["stop_seq", 0:seq_num_stops, "position", -seq_stop_len:].set(seq_stops)
                    new_stops = stops.at["seq", local_seq_id].set(this_row_full)
                    new_state = dataclasses.replace(new_state, stop_tokens=new_stops)

        return new_state

    def update_tokens(self,
                      local_seq_ids:  ht.i32[NamedArray, "position"],  # type: ignore[name-defined]
                      new_tokens: ht.i32[NamedArray, "position"],  # type: ignore[name-defined]
                      new_log_probs: ht.Float[NamedArray, "position"],  # type: ignore[name-defined]
                      num_new_tokens: jnp.ndarray  # scalar
                      ) -> "DecodeState":
        """
        Update the tokens and (optional) log probabilities for the given local sequence IDs.
        """
        tokens = self.tokens
        logprobs = self.logprobs
        counts = self.num_tokens

        def body(i, state):
            sid = local_seq_ids["position", i].scalar()

            def update(state):
                tkns, lps, cnts = state
                pos = cnts["seq", sid].scalar()
                tkns = tkns.at["seq", sid, "position", pos].set(new_tokens["position", i])
                if lps is not None:
                    lps = lps.at["seq", sid, "position", pos].set(new_log_probs["position", i])
                cnts = cnts.at["seq", sid].add(1)
                return tkns, lps, cnts

            return jax.lax.cond(is_valid(sid), update, lambda s: s, state)

        tokens, logprobs, counts = jax.lax.fori_loop(0, num_new_tokens, body, (tokens, logprobs, counts))

        return dataclasses.replace(self, tokens=tokens, logprobs=logprobs, num_tokens=counts)

    def is_finished(self, seq_id: jnp.ndarray) -> jnp.ndarray:
        """
        Check if the sequence or sequences with the given local ID is finished.
        A sequence is finished if it has reached its maximum number of tokens or hit its stop sequence.

        See is_stop_signal for stop sequence checking.

        Returns jnp.ndarray with the same shape as seq_id, where each entry is True if the sequence is finished.
        """

        # if it's a scalar, we need to make it a vector for the vmap to work properly
        if seq_id.ndim == 0:
            seq_id = jnp.expand_dims(seq_id, axis=0)

        def body(i):
            sid = seq_id[i]

            done = ((self.num_tokens["seq", sid] != INVALID) &
                    (self.num_tokens["seq", sid] >= self.max_num_tokens["seq", sid])
                    ).scalar()

            if self.stop_tokens is not None:
                stop_len = self.stop_tokens.axis_size("position")
                num = self.num_tokens["seq", sid].scalar()
                tokens_row = self.tokens["seq", sid].array
                padded_tokens = jnp.concatenate([
                    jnp.full((stop_len,), INVALID, dtype=jnp.int32),
                    tokens_row,
                ])
                tail = jax.lax.dynamic_slice(padded_tokens, (num,), (stop_len,))
                stop = is_stop_signal(
                    hax.named(tail, axis=("position",)),
                    self.stop_tokens["seq", sid],
                ).array
                done |= stop

            # If the sequence ID is INVALID, we consider it not finished.
            return done & (self.seq_id["seq", sid] != INVALID).scalar()

        return jax.vmap(body)(seq_id)

    def debug_print(self):
        jax.debug.print(
            """
DecodeState:
seq_id: {seq_id}
num_tokens: {num_tokens}
prefix_len: {prefix_len}
finished: {finished}
tokens: {tokens}
stop_tokens: {stop_tokens}
kv_pages: {kv_pages}
logprobs: {logprobs}
max_num_tokens: {max_num_tokens}
""",
            seq_id=self.seq_id,
            num_tokens=self.num_tokens,
            prefix_len=self.prefix_len,
            finished=self.is_finished(jnp.arange(self.max_seqs, dtype=jnp.int32)),
            tokens=self.tokens,
            stop_tokens=self.stop_tokens,
            kv_pages=self.kv_pages,
            logprobs=self.logprobs if self.logprobs is not None else "None",
            max_num_tokens=self.max_num_tokens,
        )


    @staticmethod
    def init(
        max_seqs: int,
        max_pages: int,
        page_size: int,
        max_tokens: int,
        pad_token_id: int = INVALID,
        max_stop_seqs: int = 0,
        max_stop_tokens: int = 16,
    ) -> "DecodeState":
        """
        Initialize a DecodeState with empty buffers.
        """
        return DecodeState(
            kv_pages=hax.full({"seq": max_seqs, "page": max_pages}, INVALID, dtype=jnp.int32),
            page_size=page_size,
            seq_id=hax.full({"seq": max_seqs}, INVALID, dtype=jnp.int32),
            tokens=hax.full({"seq": max_seqs, "position": max_tokens}, pad_token_id, dtype=jnp.int32),
            logprobs=None,
            prefix_len=hax.zeros({"seq": max_seqs}, dtype=jnp.int32),
            num_tokens=hax.zeros({"seq": max_seqs}, dtype=jnp.int32),
            max_num_tokens=hax.full({"seq": max_seqs}, 0, dtype=jnp.int32),
            stop_tokens=hax.full(
                {"seq": max_seqs, "stop_seq": max_stop_seqs, "position": max_stop_tokens},
                INVALID,
                dtype=jnp.int32,
            ) if max_stop_tokens > 0 else None,
            temperature=hax.ones({"seq": max_seqs}, dtype=jnp.float32),
            prng_keys=jax.vmap(jax.random.PRNGKey, axis_size=max_seqs, in_axes=None)(0)
        )


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

        do_generate might look like this:
        def do_generate(sched: JitScheduler, cache, page_table,  max_steps: int) -> JitScheduler:
            def cond(state):
                _sched, *_ , step = state
                return (step < max_new_tokens) & (~jnp.all(_sched.is_finished(jnp.arange(_sched.seq_id.size))))

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
                sched = sched.enqueue_tokens(
                    new_tokens,
                    chunk_seq_ids,
                    num_new_tokens,
                )
                return sched, page_table, cache, key, step + 1

            init_state = (sched, page_table, cache, key, jnp.array(0, dtype=jnp.int32))
            sched, page_table, cache, key, _ = jax.lax.while_loop(cond, body, init_state)
            return sched, cache, page_table, key

    """
    # Notes:
    # - ``queued_tokens`` are stored "flat" with accompanying ``queued_seq_ids``
    queued_tokens: ht.i32[NamedArray, "position"]  # tokens queued for decoding
    queued_seq_ids: ht.i32[NamedArray, "position"]
    num_queued_tokens: jax.Array

    # TODO: per-seq sampling params

    @property
    def empty_queue_space(self) -> jnp.ndarray:
        """How many tokens can be enqueued in the queue."""
        return self.queued_tokens.axis_size("position") - self.num_queued_tokens

    @property
    def max_queued_tokens(self) -> int:
        """Maximum number of tokens that can be buffered in the queue."""
        return self.queued_tokens.axis_size("position")

    @staticmethod
    def init(max_queued_tokens: int) -> "JitScheduler":
        """Create a ``JitScheduler`` with empty buffers."""
        return JitScheduler(
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
            "position",
            self.num_queued_tokens,
            new_tokens,
            num_new_tokens,
        )
        new_q_seq_ids = masked_set(
            self.queued_seq_ids,
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

    def purge_queue_of_seq(self, seq_id: hax.NamedArray | int) -> "JitScheduler":
        """
        Remove all tokens from the queue that belong to the given sequence IDs or sequence ids
        Slides remaining tokens to the front of the queue.
        """

        if isinstance(seq_id, hax.NamedArray):
            is_seq_id = hax.einsum(" -> position", self.queued_seq_ids.broadcast_axis(seq_id.axes) == seq_id)
        else:
            is_seq_id = self.queued_seq_ids == seq_id
        new_seq_ids = purge(self.queued_seq_ids, is_seq_id)
        new_tokens = purge(self.queued_tokens, is_seq_id)
        new_queued = hax.sum(new_seq_ids != INVALID).scalar()

        return dataclasses.replace(
            self,
            queued_tokens=new_tokens,
            queued_seq_ids=new_seq_ids,
            num_queued_tokens=new_queued,
        )

    def cleared(self) -> "JitScheduler":
        """
        Returns a new JitScheduler with all buffers cleared.
        This is useful for resetting the scheduler state.
        """
        return JitScheduler.init(
            max_queued_tokens=self.queued_tokens.axis_size("position"),
        )

    def debug_print(self, prefix: str = ""):

        def callback(self):
            print(f"{prefix}JitScheduler State:")
            print(f"{prefix}Queued Tokens: {self.queued_tokens}")
            print(f"{prefix}Queued Seq IDs: {self.queued_seq_ids}")
            print(f"{prefix}Num Queued Tokens: {self.num_queued_tokens}")

        jax.experimental.io_callback(callback,  None, ordered=True, self=self)
