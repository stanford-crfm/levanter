import dataclasses
import jax
import jax.numpy as jnp
import haliax as hax

from levanter.main.sample_lm import _handle_clones
from levanter.inference.utils import INVALID


@dataclasses.dataclass
class DummyDecodeState:
    temperature: hax.NamedArray  # axis: seq
    last_new_tokens: hax.NamedArray | None = None
    last_seq_ids: hax.NamedArray | None = None
    last_log_probs: hax.NamedArray | None = None
    last_num_new: int | None = None

    def prng_keys_for(self, seq_ids, pos_ids):
        # Return dummy keys; sampler ignores in tests
        n = seq_ids.axis_size("position")
        return jax.vmap(jax.random.PRNGKey, in_axes=None, axis_size=n)(0)

    def update_tokens(self, new_tokens, local_seq_ids, new_log_probs, num_new_tokens):
        # Mutate for tests (function under test is not JIT-compiled)
        self.last_new_tokens = new_tokens
        self.last_seq_ids = local_seq_ids
        self.last_log_probs = new_log_probs
        self.last_num_new = int(num_new_tokens)
        return self


@dataclasses.dataclass
class DummyGenState:
    decode_state: DummyDecodeState


def dummy_sampler(logits, temp, key):
    # Pick argmax token id, ignore temp and key
    tok = hax.argmax(logits, axis="vocab")
    return tok, hax.zeros((), dtype=jnp.float32)


def _named(arr, axis_name):
    return hax.named(jnp.asarray(arr), axis_name)


def test_do_multisample_basic_two_clones():
    # Positions for two sequences 1 and 2
    V = hax.Axis("vocab", 10)
    P = hax.Axis("position", 3)

    # logits with distinct argmax per position
    logits = hax.full((P, V), -10.0, dtype=jnp.float32)
    logits = logits.at[P, 0, V, 5].set(3.0)
    logits = logits.at[P, 1, V, 7].set(2.0)
    logits = logits.at[P, 2, V, 4].set(1.0)

    seq_ids = _named([1, 2, INVALID], P)
    pos_ids = _named([0, 0, INVALID], P)

    # Clone sources match the first two positions
    clone_sources = _named([1, 2], "seq")
    clone_targets = _named([10, 11], "seq")

    temps = hax.zeros({"seq": 32}, dtype=jnp.float32)
    dstate = DummyDecodeState(temperature=temps)
    gstate = DummyGenState(decode_state=dstate)

    new_state, sampled_mask = _handle_clones(gstate, logits, seq_ids, pos_ids, clone_sources, clone_targets, dummy_sampler)

    assert bool(hax.all(sampled_mask).scalar())
    # Expect tokens [5, 7] sampled for targets [10, 11]
    assert new_state.decode_state.last_num_new == 2
    assert jnp.array_equal(new_state.decode_state.last_seq_ids.array, jnp.asarray([10, 11]))
    assert jnp.array_equal(new_state.decode_state.last_new_tokens.array, jnp.asarray([5, 7]))


def test_do_multisample_missing_source_skips():
    # Only one source appears in this slice; the other is missing
    V = hax.Axis("vocab", 8)
    P = hax.Axis("position", 2)

    logits = hax.full((P, V), -10.0, dtype=jnp.float32)
    logits = logits.at[P, 0, V, 6].set(1.0)
    logits = logits.at[P, 1, V, 1].set(1.0)

    seq_ids = _named([3, INVALID], P)
    pos_ids = _named([0, INVALID], P)

    Clone = hax.Axis("seq", 2)
    clone_sources = _named([3, 5], Clone)  # 5 is missing
    clone_targets = _named([20, 21], Clone)

    temps = hax.zeros({"seq": 32}, dtype=jnp.float32)
    dstate = DummyDecodeState(temperature=temps)
    gstate = DummyGenState(decode_state=dstate)

    new_state, sampled_mask = _handle_clones(gstate, logits, seq_ids, pos_ids, clone_sources, clone_targets, dummy_sampler)

    # Only the first clone should be sampled
    assert jnp.array_equal(sampled_mask.array, jnp.asarray([True, False]))
    assert new_state.decode_state.last_num_new == 1
    # Only the first entry is meaningful; others are ignored by num_new
    assert jnp.array_equal(new_state.decode_state.last_seq_ids.array[:1], jnp.asarray([20]))
    assert jnp.array_equal(new_state.decode_state.last_new_tokens.array[:1], jnp.asarray([6]))


def test_do_multisample_invalid_target_skips():
    V = hax.Axis("vocab", 6)
    P = hax.Axis("position", 1)

    logits = hax.full((P, V), -10.0, dtype=jnp.float32)
    logits = logits.at[P, 0, V, 2].set(5.0)

    seq_ids = _named([7], P)
    pos_ids = _named([0], P)

    Clone = hax.Axis("seq", 2)
    clone_sources = _named([7, 8], Clone)
    clone_targets = _named([INVALID, 30], Clone)  # first target invalid, second source missing

    temps = hax.zeros({"seq": 64}, dtype=jnp.float32)
    dstate = DummyDecodeState(temperature=temps)
    gstate = DummyGenState(decode_state=dstate)

    new_state, sampled_mask = _handle_clones(gstate, logits, seq_ids, pos_ids, clone_sources, clone_targets, dummy_sampler)

    # Both clones should be skipped: one invalid target, one missing source
    assert jnp.array_equal(sampled_mask.array, jnp.asarray([False, False]))
    # update_tokens should not be called; last_num_new remains None or 0 if called with zero
    assert new_state.decode_state.last_num_new in (None, 0)
