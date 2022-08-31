import jax.numpy as jnp

import haliax as hax
from haliax import Axis


def test_one_hot():
    i = Axis("i", 3)
    c = Axis("c", 3)
    actual = hax.nn.one_hot(hax.NamedArray(jnp.array([0, 1, 2]), (i,)), c)
    expected = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    assert actual.axes == (i, c)
    assert jnp.all(jnp.isclose(actual.array, expected))

    actual = hax.nn.one_hot(hax.NamedArray(jnp.array([1, 2, 0]), (i,)), c)
    expected = jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    assert actual.axes == (i, c)
    assert jnp.all(jnp.isclose(actual.array, expected))


def test_one_hot_out_of_bound():
    i = Axis("i", 2)
    c = Axis("c", 3)
    actual = hax.nn.one_hot(hax.NamedArray(jnp.array([-1, 3]), (i,)), c)
    expected = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert jnp.all(jnp.isclose(actual.array, expected))
