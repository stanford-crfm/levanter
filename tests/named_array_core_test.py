import jax
import jax.numpy as jnp
import pytest
from jax.random import PRNGKey

import hapax as hpx
from hapax import NamedArray, Axis


def test_dot():
    Height = Axis("Height", 2)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    m1 = NamedArray(jnp.ones((Height.size, Width.size, Depth.size)), [Height, Width, Depth])
    m2 = NamedArray(jnp.ones((Depth.size, Width.size, Height.size)), [Depth, Width, Height])

    assert jnp.all(jnp.equal(hpx.dot(Height, m1, m2).array, jnp.einsum("ijk,kji->jk", m1.array, m2.array)))
    assert jnp.all(jnp.equal(hpx.dot( (Height, Width), m1, m2).array, jnp.einsum("ijk,kji->k", m1.array, m2.array)))
    assert jnp.all(jnp.equal(hpx.dot( (Height, Width, Depth), m1, m2).array, jnp.einsum("ijk,kji->", m1.array, m2.array)))


def test_unary_np_functions():
    Height = Axis("Height", 2)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    m1 = NamedArray(jnp.ones((Height.size, Width.size, Depth.size)), [Height, Width, Depth])

    assert jnp.all(jnp.equal(hpx.abs(m1).array, jnp.abs(m1.array)))
    assert jnp.all(jnp.equal(hpx.absolute(m1).array, jnp.absolute(m1.array)))
    assert jnp.all(jnp.equal(hpx.angle(m1).array, jnp.angle(m1.array)))
    assert jnp.all(jnp.equal(hpx.arccos(m1).array, jnp.arccos(m1.array)))
    assert jnp.all(jnp.equal(hpx.arccosh(m1).array, jnp.arccosh(m1.array)))
    assert jnp.all(jnp.equal(hpx.arcsin(m1).array, jnp.arcsin(m1.array)))
    assert jnp.all(jnp.equal(hpx.arcsinh(m1).array, jnp.arcsinh(m1.array)))
    assert jnp.all(jnp.equal(hpx.arctan(m1).array, jnp.arctan(m1.array)))
    assert jnp.all(jnp.equal(hpx.arctanh(m1).array, jnp.arctanh(m1.array)))
    assert jnp.all(jnp.equal(hpx.around(m1).array, jnp.around(m1.array)))
    assert jnp.all(jnp.equal(hpx.cbrt(m1).array, jnp.cbrt(m1.array)))


def test_reduction_functions():
    Height = Axis("Height", 2)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    rand_m = jax.random.uniform(PRNGKey(0), (Height.size, Width.size, Depth.size))

    m1 = NamedArray(rand_m, [Height, Width, Depth])

    # sum out everything
    assert jnp.all(jnp.equal(hpx.sum(m1).array, jnp.sum(m1.array)))
    # ensure it's a scalar

    assert jnp.all(jnp.equal(hpx.sum(m1, axis=Height).array, jnp.sum(m1.array, axis=0)))
    assert jnp.all(jnp.equal(hpx.sum(m1, axis=Width).array, jnp.sum(m1.array, axis=1)))

    # sum out two axes
    assert jnp.all(jnp.equal(hpx.sum(m1, axis=(Height, Width)).array, jnp.sum(m1.array, axis=(0, 1))))
    assert jnp.all(jnp.equal(hpx.sum(m1, axis=(Width, Height)).array, jnp.sum(m1.array, axis=(1, 0))))
    assert jnp.all(jnp.equal(hpx.sum(m1, axis=(Height, Depth)).array, jnp.sum(m1.array, axis=(0, 2))))

    # sum out three axes
    assert jnp.all(jnp.equal(hpx.sum(m1, axis=(Height, Width, Depth)).array, jnp.sum(m1.array, axis=(0, 1, 2))))
    assert jnp.all(jnp.equal(hpx.sum(m1, axis=(Width, Height, Depth)).array, jnp.sum(m1.array, axis=(1, 0, 2))))

