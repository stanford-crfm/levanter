import jax
import jax.numpy as jnp
import pytest
from jax.random import PRNGKey

import haliax as hax
from haliax import NamedArray, Axis


def test_dot():
    Height = Axis("Height", 2)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    m1 = NamedArray(jnp.ones((Height.size, Width.size, Depth.size)), (Height, Width, Depth))
    m2 = NamedArray(jnp.ones((Depth.size, Width.size, Height.size)), (Depth, Width, Height))

    assert jnp.all(jnp.equal(hax.dot(Height, m1, m2).array, jnp.einsum("ijk,kji->jk", m1.array, m2.array)))
    assert jnp.all(jnp.equal(hax.dot((Height, Width), m1, m2).array, jnp.einsum("ijk,kji->k", m1.array, m2.array)))
    assert jnp.all(jnp.equal(hax.dot((Height, Width, Depth), m1, m2).array, jnp.einsum("ijk,kji->", m1.array, m2.array)))


def test_unary_np_functions():
    Height = Axis("Height", 2)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    m1 = NamedArray(jnp.ones((Height.size, Width.size, Depth.size)), (Height, Width, Depth))

    assert jnp.all(jnp.equal(hax.abs(m1).array, jnp.abs(m1.array)))
    assert jnp.all(jnp.equal(hax.absolute(m1).array, jnp.absolute(m1.array)))
    assert jnp.all(jnp.equal(hax.angle(m1).array, jnp.angle(m1.array)))
    assert jnp.all(jnp.equal(hax.arccos(m1).array, jnp.arccos(m1.array)))
    assert jnp.all(jnp.equal(hax.arccosh(m1).array, jnp.arccosh(m1.array)))
    assert jnp.all(jnp.equal(hax.arcsin(m1).array, jnp.arcsin(m1.array)))
    assert jnp.all(jnp.equal(hax.arcsinh(m1).array, jnp.arcsinh(m1.array)))
    assert jnp.all(jnp.equal(hax.arctan(m1).array, jnp.arctan(m1.array)))
    assert jnp.all(jnp.equal(hax.arctanh(m1).array, jnp.arctanh(m1.array)))
    assert jnp.all(jnp.equal(hax.around(m1).array, jnp.around(m1.array)))
    assert jnp.all(jnp.equal(hax.cbrt(m1).array, jnp.cbrt(m1.array)))


def test_reduction_functions():
    Height = Axis("Height", 2)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    rand_m = jax.random.uniform(PRNGKey(0), (Height.size, Width.size, Depth.size))

    m1 = NamedArray(rand_m, (Height, Width, Depth))

    # sum out everything
    assert jnp.all(jnp.equal(hax.sum(m1).array, jnp.sum(m1.array)))
    # ensure it's a scalar

    assert jnp.all(jnp.equal(hax.sum(m1, axis=Height).array, jnp.sum(m1.array, axis=0)))
    assert jnp.all(jnp.equal(hax.sum(m1, axis=Width).array, jnp.sum(m1.array, axis=1)))

    # sum out two axes
    assert jnp.all(jnp.equal(hax.sum(m1, axis=(Height, Width)).array, jnp.sum(m1.array, axis=(0, 1))))
    assert jnp.all(jnp.equal(hax.sum(m1, axis=(Width, Height)).array, jnp.sum(m1.array, axis=(1, 0))))
    assert jnp.all(jnp.equal(hax.sum(m1, axis=(Height, Depth)).array, jnp.sum(m1.array, axis=(0, 2))))

    # sum out three axes
    assert jnp.all(jnp.equal(hax.sum(m1, axis=(Height, Width, Depth)).array, jnp.sum(m1.array, axis=(0, 1, 2))))
    assert jnp.all(jnp.equal(hax.sum(m1, axis=(Width, Height, Depth)).array, jnp.sum(m1.array, axis=(1, 0, 2))))


def test_split():
    Height = Axis("Height", 2)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    D10 = Axis("Depth", Depth.size * 10)

    rand_m = hax.random.uniform(PRNGKey(0), (Height, Width, D10))
    m = rand_m.array

    splits = hax.split(rand_m, axis=D10, new_axes=[Depth] * 10)

    assert splits[0].axes == (Height, Width, Depth)
    assert len(splits) == 10

    usplits = jnp.split(m, 10, axis=2)

    for i in range(10):
        assert jnp.all(jnp.equal(splits[i].array, usplits[i]))


def test_take():
    Height = Axis("Height", 2)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)
    named1 = hax.random.uniform(PRNGKey(0), (Height, Width, Depth))

    assert jnp.all(jnp.equal(hax.take(named1, Height, 0).array, named1.array[0]))

    Index = Axis("Index", 5)
    indices = hax.ones(Index, dtype=jnp.int32)

    named2 = hax.take(named1, Height, indices)
    assert named2.axes == (Index, Width, Depth)

    named2 = hax.take(named1, Width, indices)
    assert named2.axes == (Height, Index, Depth)

    named2 = hax.take(named1, Depth, indices)
    assert named2.axes == (Height, Width, Index)

    Index2 = Axis("Index2", 3)

    indices2 = hax.ones((Index, Index2), dtype=jnp.int32)

    named2 = hax.take(named1, Height, indices2)
    assert named2.axes == (Index, Index2, Width, Depth)

    named2 = hax.take(named1, Width, indices2)
    assert named2.axes == (Height, Index, Index2, Depth)

    named2 = hax.take(named1, Depth, indices2)
    assert named2.axes == (Height, Width, Index, Index2)


def test_cumsum_etc():
    Height = Axis("Height", 2)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    named1 = hax.random.uniform(PRNGKey(0), (Height, Width, Depth))

    assert jnp.all(jnp.equal(hax.cumsum(named1, axis=Height).array, jnp.cumsum(named1.array, axis=0)))
    assert hax.cumsum(named1, axis=Height).axes == (Height, Width, Depth)

    assert jnp.all(jnp.equal(hax.cumsum(named1, axis=Width).array, jnp.cumsum(named1.array, axis=1)))
    assert hax.cumsum(named1, axis=Width).axes == (Height, Width, Depth)

    assert jnp.all(jnp.equal(hax.cumsum(named1, axis=Depth).array, jnp.cumsum(named1.array, axis=2)))
    assert hax.cumsum(named1, axis=Depth).axes == (Height, Width, Depth)

    assert jnp.all(jnp.equal(hax.cumprod(named1, axis=Height).array, jnp.cumprod(named1.array, axis=0)))
    assert hax.cumprod(named1, axis=Height).axes == (Height, Width, Depth)

    assert jnp.all(jnp.equal(hax.cumprod(named1, axis=Width).array, jnp.cumprod(named1.array, axis=1)))
    assert hax.cumprod(named1, axis=Width).axes == (Height, Width, Depth)

    assert jnp.all(jnp.equal(hax.cumprod(named1, axis=Depth).array, jnp.cumprod(named1.array, axis=2)))
    assert hax.cumprod(named1, axis=Depth).axes == (Height, Width, Depth)