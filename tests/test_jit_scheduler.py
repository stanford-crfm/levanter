import jax
import jax.numpy as jnp
import haliax as hax
import equinox as eqx

from levanter.inference.jit_scheduler import JitScheduler


def test_enqueue_and_pack():
    sched = JitScheduler.init(max_tokens=8, max_seqs=2, key=jax.random.PRNGKey(0))
    toks = hax.named(jnp.array([1, 2], dtype=jnp.int32), "position")
    seqs = hax.named(jnp.array([0, 1], dtype=jnp.int32), "position")
    sched = sched.enqueue_tokens(toks, seqs, 2)

    pack = eqx.filter_jit(lambda s: s.pack_next_sequence(2))
    sched, ptoks, pseqs = pack(sched)
    assert jnp.array_equal(ptoks.array, jnp.array([1, 2], dtype=jnp.int32))
    assert jnp.array_equal(pseqs.array, jnp.array([0, 1], dtype=jnp.int32))
    assert sched.num_queued_tokens == 0


def test_update_after_sampling():
    sched = JitScheduler.init(max_tokens=8, max_seqs=1, key=jax.random.PRNGKey(0))
    toks = hax.named(jnp.array([5], dtype=jnp.int32), "position")
    seqs = hax.named(jnp.array([0], dtype=jnp.int32), "position")

    sched = sched.update_after_sampling(toks, seqs, 1)
    assert jnp.array_equal(sched.generated_tokens["position", hax.ds(0, 1)].array, jnp.array([5], dtype=jnp.int32))
    assert sched.num_generated_tokens == 1
    assert sched.num_queued_tokens == 1
