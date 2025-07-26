import jax.numpy as jnp
import haliax as hax
import equinox as eqx

from levanter.inference.jit_scheduler import JitScheduler
from levanter.inference.utils import INVALID


def test_enqueue_and_pack():
    sched = JitScheduler.init(max_queued_tokens=8, max_buffered_tokens=8)
    toks = hax.named(jnp.array([1, 2], dtype=jnp.int32), "position")
    seqs = hax.named(jnp.array([0, 1], dtype=jnp.int32), "position")
    sched = eqx.filter_jit(sched.enqueue_tokens)(toks, seqs, 2)

    pack = eqx.filter_jit(lambda s: s.pack_next_sequence(2))
    sched, packed_seq = pack(sched)
    ptoks = packed_seq.tokens
    pseqs = packed_seq.seq_ids
    assert jnp.array_equal(ptoks.array, jnp.array([1, 2], dtype=jnp.int32))
    assert jnp.array_equal(pseqs.array, jnp.array([0, 1], dtype=jnp.int32))
    assert sched.num_queued_tokens == 0


def test_enqueue_and_pack_over_length():
    sched = JitScheduler.init(max_queued_tokens=8, max_buffered_tokens=8)
    toks = hax.named(jnp.array([1, 2], dtype=jnp.int32), "position")
    seqs = hax.named(jnp.array([0, 1], dtype=jnp.int32), "position")
    sched = eqx.filter_jit(sched.enqueue_tokens)(toks, seqs, 2)

    pack = eqx.filter_jit(lambda s: s.pack_next_sequence(4))

    sched, packed_seq = pack(sched)
    ptoks = packed_seq.tokens
    pseqs = packed_seq.seq_ids
    assert jnp.array_equal(ptoks.array, jnp.array([1, 2, INVALID, INVALID], dtype=jnp.int32))
    assert jnp.array_equal(pseqs.array, jnp.array([0, 1, INVALID, INVALID], dtype=jnp.int32))



def test_update_after_sampling():
    sched = JitScheduler.init(max_queued_tokens=8, max_buffered_tokens=16)
    toks = hax.named(jnp.array([5], dtype=jnp.int32), "position")
    seqs = hax.named(jnp.array([0], dtype=jnp.int32), "position")

    sched = eqx.filter_jit(sched.update_after_sampling)(toks, seqs, 1)
    assert jnp.array_equal(sched.generated_tokens["position", hax.ds(0, 1)].array, jnp.array([5], dtype=jnp.int32))
    assert sched.num_generated_tokens == 1
    assert sched.num_queued_tokens == 1


def test_partial_dequeue():
    sched = JitScheduler.init(max_queued_tokens=8, max_buffered_tokens=16)
    toks = hax.named(jnp.array([1, 2, 3], dtype=jnp.int32), "position")
    seqs = hax.named(jnp.array([0, 0, 0], dtype=jnp.int32), "position")
    sched = eqx.filter_jit(sched.enqueue_tokens)(toks, seqs, 3)

    assert sched.num_queued_tokens == 3
    assert jnp.array_equal(sched.queued_tokens["position", 0:3].array, jnp.array([1, 2, 3], dtype=jnp.int32))
    assert jnp.array_equal(sched.queued_seq_ids["position", 0:3].array, jnp.array([0, 0, 0], dtype=jnp.int32))

    sched, packed = eqx.filter_jit(lambda s: s.pack_next_sequence(2))(sched)
    ptoks = packed.tokens
    pseqs = packed.seq_ids
    assert jnp.array_equal(ptoks.array, jnp.array([1, 2], dtype=jnp.int32))
    assert jnp.array_equal(pseqs.array, jnp.array([0, 0], dtype=jnp.int32))
    assert sched.num_queued_tokens == 1


def _make_scheduler_with_tokens(max_tokens=8, max_buffered_tokens=16):
    # Build a scheduler and push four tokens: [10,20,30,40]
    # with seq‐ids [0,1,0,1].
    sched = JitScheduler.init(max_queued_tokens=max_tokens, max_buffered_tokens=max_buffered_tokens)
    toks = hax.named(jnp.array([10, 20, 30, 40], dtype=jnp.int32), axis=("position",))
    seqs = hax.named(jnp.array([ 0,  1,  0,  1], dtype=jnp.int32), axis=("position",))
    return eqx.filter_jit(sched.update_after_sampling)(toks, seqs, 4)


def test_extract_single_sequence():
    sched = _make_scheduler_with_tokens(max_buffered_tokens=16)
    # extract up to 3 tokens for seq=0
    seq_ids = hax.named(jnp.array([0], dtype=jnp.int32), axis=("seq",))
    sched2, out = eqx.filter_jit(sched.extract_generated_tokens)(seq_ids, 3)

    # seq 0 produced [10,30] then pad INVALID
    expected = jnp.array([[10, 30, INVALID]], dtype=jnp.int32)
    assert jnp.array_equal(out.array, expected)

    # buffer should now only have the tokens for seq=1 in order [20,40]
    remaining = sched2.generated_tokens["position", hax.ds(0,2)].array
    remaining_ids = sched2.generated_seq_ids["position", hax.ds(0,2)].array
    assert jnp.array_equal(remaining, jnp.array([20, 40], dtype=jnp.int32))
    assert jnp.array_equal(remaining_ids, jnp.array([1, 1], dtype=jnp.int32))
    # count updated
    assert sched2.num_generated_tokens == 2


def test_extract_multiple_sequences():
    sched = _make_scheduler_with_tokens(max_buffered_tokens=16)
    # extract up to 2 tokens for seq=0 and seq=1
    seq_ids = hax.named(jnp.array([0,1], dtype=jnp.int32), axis=("seq",))
    sched2, out = eqx.filter_jit(sched.extract_generated_tokens)(seq_ids, 2)

    # row 0 → seq 0: [10,30], row 1 → seq 1: [20,40]
    expected = jnp.array([[10, 30], [20, 40]], dtype=jnp.int32)
    assert jnp.array_equal(out.array, expected)

    # buffer now empty
    assert sched2.num_generated_tokens == 0
    assert jnp.all(sched2.generated_tokens.array == INVALID)
    assert jnp.all(sched2.generated_seq_ids.array == INVALID)


def test_extract_partial_sequence():
    sched = _make_scheduler_with_tokens(max_buffered_tokens=16)
    # request seq-id 0, but only 1 token available
    seq_ids = hax.named(jnp.array([0], dtype=jnp.int32), axis=("seq",))
    sched2, out = eqx.filter_jit(sched.extract_generated_tokens)(seq_ids, 1)

    expected = jnp.array([[10]], dtype=jnp.int32)
    assert jnp.array_equal(out.array, expected)

    # buffer should now have 1 token for seq 0 and 2 for seq 1
    remaining = sched2.generated_tokens["position", hax.ds(0,3)].array
    remaining_ids = sched2.generated_seq_ids["position", hax.ds(0,3)].array
    assert jnp.array_equal(remaining, jnp.array([20, 30, 40], dtype=jnp.int32))
    assert jnp.array_equal(remaining_ids, jnp.array([1, 0, 1], dtype=jnp.int32))


def test_extract_nonexistent_sequence_leaves_buffer():
    sched = _make_scheduler_with_tokens(max_buffered_tokens=16)
    # request seq-id 2 (never inserted)
    seq_ids = hax.named(jnp.array([2], dtype=jnp.int32), axis=("seq",))
    sched2, out = eqx.filter_jit(sched.extract_generated_tokens)(seq_ids, 3)

    # output should be all INVALID
    assert jnp.array_equal(out.array, jnp.full((1,3), INVALID, dtype=jnp.int32))

    # buffer remains exactly as before
    assert sched2.num_generated_tokens == sched.num_generated_tokens
    assert jnp.array_equal(
        sched2.generated_tokens["position", hax.ds(0,4)].array,
        sched.generated_tokens["position", hax.ds(0,4)].array
    )
    assert jnp.array_equal(
        sched2.generated_seq_ids["position", hax.ds(0,4)].array,
        sched.generated_seq_ids["position", hax.ds(0,4)].array
    )


def test_purge_queue_of_seq():
    sched = JitScheduler.init(max_queued_tokens=8, max_buffered_tokens=8)
    # Enqueue tokens with seq_ids: [0, 1, 0, 2, 1, 2]
    toks = hax.named(jnp.array([10, 20, 30, 40, 50, 60], dtype=jnp.int32), "position")
    seqs = hax.named(jnp.array([0, 1, 0, 2, 1, 2], dtype=jnp.int32), "position")
    sched = eqx.filter_jit(sched.enqueue_tokens)(toks, seqs, 6)

    # Purge seq_id 1
    sched2 = eqx.filter_jit(sched.purge_queue_of_seq)(1)
    # After purge, tokens with seq_id 1 (positions 1 and 4) should be INVALID
    expected_tokens = jnp.array([10, 30, 40, 60, INVALID, INVALID, INVALID, INVALID], dtype=jnp.int32)
    expected_seqids = jnp.array([0, 0, 2, 2, INVALID, INVALID, INVALID, INVALID], dtype=jnp.int32)
    assert jnp.array_equal(sched2.queued_tokens.array, expected_tokens)
    assert jnp.array_equal(sched2.queued_seq_ids.array, expected_seqids)
    # num_queued_tokens should be 4 (6 - 2 purged)
    assert sched2.num_queued_tokens == 4

    # Purge seq_id 0
    sched3 = eqx.filter_jit(sched2.purge_queue_of_seq)(0)
    expected_tokens2 = jnp.array([40, 60, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID], dtype=jnp.int32)
    expected_seqids2 = jnp.array([2, 2, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID], dtype=jnp.int32)
    assert jnp.array_equal(sched3.queued_tokens.array, expected_tokens2)
    assert jnp.array_equal(sched3.queued_seq_ids.array, expected_seqids2)
    assert sched3.num_queued_tokens == 2

    # Purge seq_id 2 (should clear all remaining tokens)
    sched4 = eqx.filter_jit(sched3.purge_queue_of_seq)(2)
    assert jnp.all(sched4.queued_tokens.array == INVALID)
    assert jnp.all(sched4.queued_seq_ids.array == INVALID)
    assert sched4.num_queued_tokens == 0
