import jax.numpy as jnp
import haliax as hax

from levanter.inference.jit_scheduler import TokenQueue
from levanter.inference.utils import INVALID


def test_pack_next_sequence_no_boundary_when_queue_drained_single_seq():
    # Queue entirely filled with a single sequence id; pack exactly all tokens.
    capacity = 8
    tq = TokenQueue.init(capacity)

    tokens = hax.named(jnp.arange(capacity, dtype=jnp.int32), axis=("position",))
    seq_ids = hax.named(jnp.full((capacity,), 3, dtype=jnp.int32), axis=("position",))

    tq = tq.enqueue_tokens(tokens, seq_ids, capacity)

    tq2, packed = tq.pack_next_sequence(capacity)

    # Queue should be empty now
    assert tq2.num_queued_tokens == 0

    # Packed sequence should have no boundaries since it's a single ongoing sequence
    # and the queue was drained (no next token to indicate an end-of-sequence boundary).
    assert int(packed.num_tokens) == capacity

    # All boundary flags should be False
    is_boundary = packed.is_boundary.array
    assert is_boundary.dtype == jnp.bool_
    # Only the first `num_tokens` entries are meaningful; others are ignored by num_tokens
    assert not bool(is_boundary[:capacity].any())

    # boundary indices should be INVALID (no boundaries found)
    bidx = packed.boundary_indices(max_boundaries=capacity).array
    assert (bidx == INVALID).all()


def test_pack_next_sequence_boundary_between_sequences():
    # Two sequences back-to-back; boundary should appear at the transition inside the packed slice.
    capacity = 6
    tq = TokenQueue.init(capacity)

    tokens = hax.named(jnp.array([10, 11, 12, 20, 21, 22], dtype=jnp.int32), axis=("position",))
    seq_ids = hax.named(jnp.array([1, 1, 1, 2, 2, 2], dtype=jnp.int32), axis=("position",))

    tq = tq.enqueue_tokens(tokens, seq_ids, capacity)

    tq2, packed = tq.pack_next_sequence(capacity)

    assert int(packed.num_tokens) == capacity
    # There should be exactly one boundary at position 2 (0-based) marking end of seq 1
    ib = packed.is_boundary.array
    assert ib.dtype == jnp.bool_
    assert bool(ib[2]) is True
    # No boundary at the very end just because the queue drained
    assert bool(ib[capacity - 1]) is False
