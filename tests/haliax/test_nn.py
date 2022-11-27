import equinox as eqx
import jax.nn
import jax.random as jrandom
from jax import numpy as jnp

import haliax as hax
from haliax import Axis, NamedArray


def _compare_eqx_and_haliax(hax_mod: eqx.Module, eqx_mod: eqx.Module):
    def f(x: NamedArray, *args, **kwargs):
        unnamed_x = x.array
        hax_out = hax_mod(x, *args, **kwargs)  # type: ignore
        eqx_out = eqx_mod(unnamed_x, *args, **kwargs)  # type: ignore

        assert jnp.allclose(hax_out.array, eqx_out)
        return hax_out

    return f


def test_layer_norm():
    H = Axis("H", 10)
    hax_ln = hax.nn.LayerNorm(H)
    eqx_ln = eqx.nn.LayerNorm(shape=(H.size,))

    f = _compare_eqx_and_haliax(hax_ln, eqx_ln)
    out = f(hax.random.uniform(jrandom.PRNGKey(0), (H,)))

    assert out.axes == (H,)


def test_dropout():
    H = Axis("H", 10)
    key = jrandom.PRNGKey(0)
    hax_dropout = hax.nn.Dropout(0.5)
    eqx_dropout = eqx.nn.Dropout(0.5)

    f = _compare_eqx_and_haliax(hax_dropout, eqx_dropout)
    out = f(hax.random.uniform(jrandom.PRNGKey(0), (H,)), key=key, inference=False)

    assert out.axes == (H,)


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


def test_standardize():
    b = Axis("b", 2)
    c = Axis("c", 3)
    actual = hax.nn.standardize(hax.NamedArray(jnp.array([0, 1, 2]), (c,)), c)
    expected = jax.nn.standardize(jnp.array([0, 1, 2]), axis=0)

    assert actual.axes == (c,)
    assert jnp.all(jnp.isclose(actual.array, expected))

    actual = hax.nn.standardize(hax.NamedArray(jnp.array([[0, 1, 2], [3, 4, 5]]), (b, c)), c)
    expected = jax.nn.standardize(jnp.array([[0, 1, 2], [3, 4, 5]]), axis=1)

    assert actual.axes == (b, c)
    assert jnp.all(jnp.isclose(actual.array, expected))

    actual = hax.nn.standardize(hax.NamedArray(jnp.array([[0, 1, 2], [3, 4, 5]]), (b, c)), b)
    expected = jax.nn.standardize(jnp.array([[0, 1, 2], [3, 4, 5]]), axis=0)

    assert actual.axes == (b, c)
    assert jnp.all(jnp.isclose(actual.array, expected))

    # test passing in where
    mask = hax.NamedArray(jnp.array([True, False, True]), (c,))
    actual = hax.nn.standardize(hax.NamedArray(jnp.array([[0, 1, 2], [3, 4, 5]]), (b, c)), b, where=mask)
    expected = jax.nn.standardize(jnp.array([[0, 1, 2], [3, 4, 5]]), axis=0, where=mask.array)

    assert actual.axes == (b, c)
    assert jnp.all(jnp.isclose(actual.array, expected) | jnp.isnan(expected))

    # now mean/variance
    input = hax.NamedArray(jnp.array([[0, 1, 2], [3, 4, 5]]), (b, c))
    mean = hax.mean(input, c)
    variance = hax.var(input, c)
    actual = hax.nn.standardize(input, c, mean=mean, variance=variance)
    expected = jax.nn.standardize(
        input.array, axis=1, mean=mean.array.reshape(-1, 1), variance=variance.array.reshape(-1, 1)
    )

    assert actual.axes == (b, c)
    assert jnp.all(jnp.isclose(actual.array, expected))
