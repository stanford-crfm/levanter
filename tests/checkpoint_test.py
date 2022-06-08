from typing import List, Callable, Optional

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import tempfile

import numpy as np
from equinox import static_field

from psithuros.checkpoint import load_checkpoint, save_checkpoint
import optax

# TODO: chex doesn't seem to respect custom nodes...
from chex import assert_trees_all_close

def assert_trees_not_close(a, b):
    try:
        assert_trees_all_close(jax.tree_leaves(arrays_only(a)), jax.tree_leaves(arrays_only(b)))
    except AssertionError:
        pass
    else:
        raise AssertionError("Trees are equal")
class MLP(eqx.Module):
    """slightly less annoying MLP"""
    layers: List[nn.Linear]
    activation: Callable = eqx.static_field()
    final_activation: Callable = eqx.static_field()
    in_size: int = static_field()
    out_size: int = static_field()
    width_size: int = static_field()
    depth: int = static_field()

    def __init__(
        self,
        in_size: int,
        out_size: int,
        width_size: int,
        depth: int,
        activation: Callable = jax.nn.relu,
        final_activation: Callable = lambda x: x,
        *,
        key: "jax.random.PRNGKey",
        **kwargs
    ):
        """**Arguments**:

        - `in_size`: The size of the input layer.
        - `out_size`: The size of the output layer.
        - `width_size`: The size of each hidden layer.
        - `depth`: The number of hidden layers.
        - `activation`: The activation function after each hidden layer. Defaults to
            ReLU.
        - `final_activation`: The activation function after the output layer. Defaults
            to the identity.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """

        super().__init__(**kwargs)
        keys = jax.random.split(key, depth + 1)
        layers = []
        if depth == 0:
            layers.append(nn.Linear(in_size, out_size, key=keys[0]))
        else:
            layers.append(nn.Linear(in_size, width_size, key=keys[0]))
            for i in range(depth - 1):
                layers.append(nn.Linear(width_size, width_size, key=keys[i + 1]))
            layers.append(nn.Linear(width_size, out_size, key=keys[-1]))
        self.layers = layers
        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth
        self.activation = activation
        self.final_activation = final_activation

    def __call__(
        self, x, *, key: Optional["jax.random.PRNGKey"] = None
    ):
        """**Arguments:**

        - `x`: A JAX array with shape `(in_size,)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array with shape `(out_size,)`.
        """
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        x = self.final_activation(x)
        return x

def arrays_only(x):
    return eqx.filter(x, eqx.is_inexact_array_like)


def test_checkpoint_simple():
    key0 = jax.random.PRNGKey(0)
    key1 = jax.random.PRNGKey(1)

    def make_state(key):
        model = nn.MLP(in_size=2, out_size=1, width_size=2, depth=3, key=key)
        optim = optax.adam(1E-4)
        opt_state = optim.init(arrays_only(model))

        return model, opt_state, key

    initial_model, initial_opt_state, initial_key = make_state(key0)
    rep_model, rep_state, rep_key = make_state(key1)

    assert_trees_not_close(initial_model, rep_model)


    with tempfile.TemporaryDirectory() as tmpdir:
        save_checkpoint(initial_model, (initial_opt_state, initial_key), step=10, checkpoint_path=tmpdir, exist_ok=True)
        restored_model, (restored_optstate, rkey), step = load_checkpoint(rep_model, (rep_state, rep_key), checkpoint_path=tmpdir, discover_latest=False)

        assert_trees_all_close(jax.tree_leaves(arrays_only(restored_model)), jax.tree_leaves(arrays_only(initial_model)))
        assert all(np.isclose(rkey, initial_key))
        assert step == 10


def test_checkpoint_steps():
    key0 = jax.random.PRNGKey(0)
    key1 = jax.random.PRNGKey(1)

    optim = optax.adam(1E-4)

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
        restored_model, restored_optstate, step = load_checkpoint(rep_model, rep_state, checkpoint_path=tmpdir, discover_latest=False)

        assert_trees_all_close(jax.tree_leaves(arrays_only(restored_model)), jax.tree_leaves(arrays_only(model)))
        assert_trees_all_close(jax.tree_leaves(arrays_only(restored_optstate)), jax.tree_leaves(arrays_only(state)))
        assert step == 3

