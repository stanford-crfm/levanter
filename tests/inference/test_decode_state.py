import dataclasses
import jax.numpy as jnp
import equinox as eqx
import haliax as hax

from levanter.inference.jit_scheduler import DecodeState, SeqDecodingParams
from levanter.inference.utils import INVALID


def _make_state(max_tokens=4, num_stop_sequences=2, max_stop_tokens=2):
    """Create a DecodeState with specific parameters for testing."""
    return DecodeState.init(
        max_seqs=1,
        max_pages=2,
        page_size=2,
        max_tokens=max_tokens,
        max_stop_seqs=num_stop_sequences,
        max_stop_tokens=max_stop_tokens,
    )


def test_assign_seq():
    ds = _make_state()
    kv = hax.named(jnp.array([0, 1], dtype=jnp.int32), axis=("page",))
    toks = hax.named(jnp.array([11, 12], dtype=jnp.int32), axis=("position",))
    stop = hax.named(jnp.array([[INVALID, 99]], dtype=jnp.int32), axis=("stop_seq", "position"))
    params = SeqDecodingParams(
        max_num_tokens=jnp.array(4, dtype=jnp.int32),
        stop_tokens=stop,
        temperature=jnp.array(0.5, dtype=jnp.float32),
    )
    ds = eqx.filter_jit(ds.assign_seq)(0, 42, kv, toks, 2, params)
    assert ds.seq_id["seq", 0] == 42
    assert jnp.array_equal(ds.kv_pages["seq", 0].array, kv.array)
    assert jnp.array_equal(ds.tokens["seq", 0, "position", hax.ds(0, 2)].array, jnp.array([11, 12], dtype=jnp.int32))
    assert ds.num_tokens["seq", 0] == 2
    assert ds.max_num_tokens["seq", 0] == 4
    assert ds.temperature["seq", 0] == 0.5
    assert jnp.array_equal(
        ds.stop_tokens["seq", 0, "stop_seq", 0, "position", hax.ds(ds.max_stop_seq_len - 2, 2)].array,
        jnp.array([INVALID, 99], dtype=jnp.int32),
    )


def test_update_tokens_and_finish_max():
    ds = _make_state()
    kv = hax.named(jnp.array([0, 1], dtype=jnp.int32), axis=("page",))
    toks = hax.named(jnp.array([1, 2], dtype=jnp.int32), axis=("position",))
    params = SeqDecodingParams(max_num_tokens=jnp.array(4, dtype=jnp.int32), stop_tokens=None, temperature=jnp.array(1.0, dtype=jnp.float32))
    ds = eqx.filter_jit(ds.assign_seq)(0, 0, kv, toks, 2, params)

    logprobs = hax.full({"seq": 1, "position": 4}, 0.0, dtype=jnp.float32)
    ds = dataclasses.replace(ds, logprobs=logprobs)

    new_toks = hax.named(jnp.array([3, 4], dtype=jnp.int32), axis=("position",))
    seqs = hax.named(jnp.array([0, 0], dtype=jnp.int32), axis=("position",))
    new_lps = hax.named(jnp.array([-0.1, -0.2], dtype=jnp.float32), axis=("position",))
    ds = eqx.filter_jit(ds.update_tokens)(seqs, new_toks, new_lps, jnp.array(2, dtype=jnp.int32))

    assert jnp.array_equal(ds.tokens["seq", 0, "position", hax.ds(0, 4)].array, jnp.array([1, 2, 3, 4], dtype=jnp.int32))
    assert jnp.allclose(
        ds.logprobs["seq", 0].array,
        jnp.array([0.0, 0.0, -0.1, -0.2], dtype=jnp.float32),
    )
    assert ds.num_tokens["seq", 0] == 4
    assert eqx.filter_jit(ds.is_finished)(jnp.array(0, dtype=jnp.int32))


def test_stop_sequence_triggers_finished():
    ds = _make_state(max_tokens=6, max_stop_tokens=1)
    kv = hax.named(jnp.array([0, 1], dtype=jnp.int32), axis=("page",))
    toks = hax.named(jnp.array([10, 11], dtype=jnp.int32), axis=("position",))
    stop = hax.named(jnp.array([[4]], dtype=jnp.int32), axis=("stop_seq", "position"))
    params = SeqDecodingParams(max_num_tokens=jnp.array(6, dtype=jnp.int32), stop_tokens=stop, temperature=jnp.array(1.0, dtype=jnp.float32))
    ds = eqx.filter_jit(ds.assign_seq)(0, 0, kv, toks, 2, params)

    new_toks = hax.named(jnp.array([3, 4], dtype=jnp.int32), axis=("position",))
    seqs = hax.named(jnp.array([0, 0], dtype=jnp.int32), axis=("position",))
    new_lps = hax.named(jnp.zeros(2, dtype=jnp.float32), axis=("position",))
    ds = eqx.filter_jit(ds.update_tokens)(seqs, new_toks, new_lps, jnp.array(2, dtype=jnp.int32))

    assert eqx.filter_jit(ds.is_finished)(jnp.array(0, dtype=jnp.int32))
