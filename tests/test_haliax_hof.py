import jax.numpy as jnp
from jax.random import PRNGKey

import haliax as hax
from haliax import Axis


def test_scan():
    Height = Axis("Height", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)
    named1 = hax.random.uniform(PRNGKey(0), (Height, Width, Depth))

    def scan_fun(acc, x):
        return acc + jnp.sum(x.array), x.take(Width, 2)

    total, selected = hax.scan(scan_fun, Height, 0.0, named1)

    assert jnp.all(jnp.isclose(total, jnp.sum(named1.array, axis=(0, 1, 2))))
    assert jnp.all(jnp.equal(selected.array, named1.take(Width, 2).array))


def test_scan_not_0th_axis():
    Height = Axis("Height", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)
    named1 = hax.random.uniform(PRNGKey(0), (Height, Width, Depth))

    def scan_fun(acc, x):
        return acc + jnp.sum(x.array), x.take(Width, 2)

    total, selected = hax.scan(scan_fun, Depth, 0.0, named1)

    assert jnp.all(jnp.isclose(total, jnp.sum(named1.array, axis=(0, 1, 2))))
    assert jnp.all(jnp.equal(selected.array, named1.take(Width, 2).rearrange(selected.axes).array))
