import tempfile

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

# TODO: chex doesn't seem to respect custom nodes...
from chex import assert_trees_all_close
from utils import arrays_only, assert_trees_not_close

from levanter.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_simple():
    key0 = jax.random.PRNGKey(0)
    key1 = jax.random.PRNGKey(1)

    def make_state(key):
        model = nn.MLP(in_size=2, out_size=1, width_size=2, depth=3, key=key)
        optim = optax.adam(1e-4)
        opt_state = optim.init(arrays_only(model))

        return model, opt_state, key

    initial_model, initial_opt_state, initial_key = make_state(key0)
    rep_model, rep_state, rep_key = make_state(key1)

    assert_trees_not_close(initial_model, rep_model)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_checkpoint(
            initial_model,
            (initial_opt_state, initial_key),
            step=10,
            checkpoint_path=tmpdir,
            exist_ok=True,
        )
        restored_model, (restored_optstate, rkey), step = load_checkpoint(
            rep_model,
            (rep_state, rep_key),
            checkpoint_path=tmpdir,
            discover_latest=False,
        )

        assert_trees_all_close(
            jax.tree_util.tree_leaves(arrays_only(restored_model)),
            jax.tree_util.tree_leaves(arrays_only(initial_model)),
        )
        assert all(np.isclose(rkey, initial_key))
        assert step == 10


def test_checkpoint_steps():
    key0 = jax.random.PRNGKey(0)
    key1 = jax.random.PRNGKey(1)

    optim = optax.adam(1e-4)

    def make_state(key):
        model = nn.MLP(in_size=2, out_size=1, width_size=2, depth=3, key=key)
        opt_state = optim.init(arrays_only(model))

        return model, opt_state, key

    initial_model, initial_opt_state, initial_key = make_state(key0)
    data = jax.random.uniform(key0, (2, 2))

    @eqx.filter_grad
    def loss_fn(model, data):
        m = jax.vmap(model)
        return jnp.mean(jnp.square(m(data)))

    model, state = initial_model, initial_opt_state
    for i in range(3):
        grad = loss_fn(model, data)
        updates, state = optim.update(grad, state)
        model = eqx.apply_updates(model, updates)

    assert_trees_not_close(model, initial_model)
    assert_trees_not_close(state, initial_opt_state)

    rep_model, rep_state, rep_key = make_state(key1)
    assert_trees_not_close(model, rep_model)
    assert_trees_not_close(state, rep_state)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_checkpoint(model, state, step=3, checkpoint_path=tmpdir, exist_ok=True)
        restored_model, restored_optstate, step = load_checkpoint(
            rep_model, rep_state, checkpoint_path=tmpdir, discover_latest=False
        )

        assert_trees_all_close(
            jax.tree_util.tree_leaves(arrays_only(restored_model)),
            jax.tree_util.tree_leaves(arrays_only(model)),
        )
        assert_trees_all_close(
            jax.tree_util.tree_leaves(arrays_only(restored_optstate)),
            jax.tree_util.tree_leaves(arrays_only(state)),
        )
        assert step == 3
