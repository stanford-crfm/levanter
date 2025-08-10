import jax.numpy as jnp
import jax.random as jrandom
from chex import assert_trees_all_close

import haliax as hax
from haliax import Axis
import haliax.nn as hnn

from levanter.layers.sampler import Sampler


def test_sampler_greedy():
    B = Axis("batch", 3)
    V = Axis("vocab", 5)

    # logits so that argmax is known (batch index)
    logits = hax.arange((B, V), start=0, dtype=jnp.float32)
    # make each row increasing with vocab index so argmax is last index
    logits = logits.astype(jnp.float32)
    temps = hax.zeros((B,), dtype=jnp.float32)

    sampler = Sampler(V)
    tokens, logp = sampler(logits, temps, key=jrandom.PRNGKey(0))

    # greedy should pick last token (vocab size - 1)
    expected = hax.full(B, V.size - 1, dtype=jnp.int32)
    assert_trees_all_close(tokens.array, expected.array)
    # log probs for greedy should match log softmax of logits
    ref_logp = hax.nn.log_softmax(logits.astype(jnp.float32), axis=V)
    ref_selected = hax.sum(ref_logp * hnn.one_hot(tokens, V), axis=V)
    assert_trees_all_close(logp.array, ref_selected.array)


def test_sampler_sampling_changes_with_temp():
    B = Axis("batch", 1024)
    V = Axis("vocab", 2)

    # logits favor token 0 strongly
    logits = hax.zeros((B, V), dtype=jnp.float32)
    logits = logits.at[V, 0].set(5.0)
    logits = logits.at[V, 1].set(0.0)

    temps = hax.full((B,), 1.0, dtype=jnp.float32)
    sampler = Sampler(V)
    tokens, _ = sampler(logits, temps, key=jrandom.PRNGKey(42))

    # With temperature 1, probability ~0.993 for token 0
    prob_0 = jnp.mean((tokens.array == 0).astype(jnp.float32))
    prob_1 = jnp.mean((tokens.array == 1).astype(jnp.float32))
    assert prob_0 > 0.9
    assert prob_1 < 0.1
