from collections import deque
from typing import Deque, List

from .sequence import Sequence, SequenceStatus


class Scheduler:
    """Very small scheduler managing sequence state."""

    def __init__(self, eos: int):
        self.eos = eos
        self.waiting: Deque[Sequence] = deque()
        self.running: Deque[Sequence] = deque()

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def is_finished(self) -> bool:
        return not self.waiting and all(s.is_finished for s in self.running)

    def schedule(self) -> tuple[List[Sequence], bool]:
        if self.waiting:
            seq = self.waiting.popleft()
            self.running.append(seq)
            return [seq], True
        seqs = [s for s in self.running if not s.is_finished]
        return seqs, False

    def postprocess(self, seqs: List[Sequence], token_ids: List[int]) -> None:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(int(token_id))
            if (
                (not seq.sampling_params.ignore_eos and token_id == self.eos)
                or seq.num_completion_tokens >= seq.sampling_params.max_tokens
            ):
                seq.status = SequenceStatus.FINISHED


class JittedScheduler:
    """Ring buffer scheduler designed for jitted execution.

    This scheduler stores sequence state in JAX arrays so it can run entirely
    inside a ``lax`` loop.  Previously this relied on ``lax.infeed``/``outfeed``
    to stream sequences to and from the device, but those APIs are now
    deprecated.  Instead we execute for a fixed number of steps and return the
    resulting state to the host once the loop completes.
    """

    def __init__(self, max_seqs: int, max_len: int, eos: int):
        self.max_seqs = max_seqs
        self.max_len = max_len
        self.eos = eos

    # ------------------------------------------------------------------
    def init_state(self):
        import jax.numpy as jnp
        from dataclasses import dataclass

        @dataclass
        class State:
            token_ids: jnp.ndarray  # (max_seqs, max_len)
            lengths: jnp.ndarray  # (max_seqs,)
            active: jnp.ndarray  # (max_seqs,)
            head: jnp.ndarray  # ()
            tail: jnp.ndarray  # ()

        return State(
            token_ids=jnp.full((self.max_seqs, self.max_len), self.eos, dtype=jnp.int32),
            lengths=jnp.zeros((self.max_seqs,), dtype=jnp.int32),
            active=jnp.zeros((self.max_seqs,), dtype=jnp.bool_),
            head=jnp.array(0, dtype=jnp.int32),
            tail=jnp.array(0, dtype=jnp.int32),
        )

    # ------------------------------------------------------------------
    def add_sequence(self, state, tokens, length):
        """Add a new sequence to ``state``. Evicts the oldest if full."""
        import jax.numpy as jnp
        from jax import lax

        def _evict(state):
            idx = state.tail
            state.active = state.active.at[idx].set(False)
            state.tail = (state.tail + 1) % self.max_seqs
            return state

        def _add(state):
            idx = state.head
            state.token_ids = state.token_ids.at[idx, :length].set(tokens[:length])
            state.lengths = state.lengths.at[idx].set(length)
            state.active = state.active.at[idx].set(True)
            state.head = (state.head + 1) % self.max_seqs
            return state

        need_evict = jnp.logical_and(state.active[state.head], True)
        state = lax.cond(need_evict, _evict, lambda s: s, state)
        state = _add(state)
        return state

    # ------------------------------------------------------------------
    def decode_step(self, state, decode_fn):
        """Decode one token for the sequence at ``state.tail``."""
        import jax.numpy as jnp
        from jax import lax

        idx = state.tail
        tokens = state.token_ids[idx]
        length = state.lengths[idx]
        prev = tokens[length - 1]
        new_tok = decode_fn(prev)
        state.token_ids = state.token_ids.at[idx, length].set(new_tok)
        state.lengths = state.lengths.at[idx].set(length + 1)

        finished = jnp.logical_or(new_tok == self.eos, length + 1 >= self.max_len)

        def _finish(st):
            st.active = st.active.at[idx].set(False)
            st.tail = (st.tail + 1) % self.max_seqs
            return st

        state = lax.cond(finished, _finish, lambda s: s, state)
        return state

    # ------------------------------------------------------------------
    def collect(self, state):
        """Return the decoded sequences as Python lists."""
        return [
            [int(t) for t in state.token_ids[i, : int(state.lengths[i])].tolist()]
            for i in range(self.max_seqs)
            if int(state.lengths[i]) > 0
        ]


def run(state, scheduler: JittedScheduler, decode_fn, max_steps: int):
    """Execute ``scheduler.decode_step`` ``max_steps`` times inside a JAX loop."""
    from jax import lax

    def body(_, st):
        return scheduler.decode_step(st, decode_fn)

    return lax.fori_loop(0, max_steps, body, state)
