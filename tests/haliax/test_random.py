import jax

import haliax as hax
from haliax.random import generate_sharded


def test_empty_shape():
    key = jax.random.PRNGKey(0)
    hax.random.uniform(key, shape=())


def test_uniform_with_bounds_scalar():
    key = jax.random.PRNGKey(0)
    Height = hax.Axis("Height", 4)
    Width = hax.Axis("Width", 8)
    u = hax.random.uniform(key, shape=(Height, Width), minval=-3.0, maxval=0.5)

    assert u.axes == (Height, Width)

    assert hax.all(u >= -3.0)
    assert hax.all(u <= 0.5)


def test_uniform_with_bounds_broadcast():
    key = jax.random.PRNGKey(0)
    Height = hax.Axis("Height", 4)
    Width = hax.Axis("Width", 8)
    lb = hax.full(Height, -3.0)
    ub = hax.full(Width, 0.5)
    u = hax.random.uniform(key, shape=(Height, Width), minval=lb, maxval=ub)

    assert u.axes == (Height, Width)

    assert hax.all(u >= -3.0)
    assert hax.all(u <= 0.5)


def test_uniform_with_bounds_broadcast_and_scalar():
    key = jax.random.PRNGKey(0)
    Height = hax.Axis("Height", 4)
    Width = hax.Axis("Width", 8)
    lb = hax.full(Height, -3.0)
    ub = 0.5
    u = hax.random.uniform(key, shape=(Height, Width), minval=lb, maxval=ub)

    assert u.axes == (Height, Width)

    assert hax.all(u >= -3.0)
    assert hax.all(u <= 0.5)


def test_sharded_uniform_with_bounds_broadcast_and_scalar():
    hax.random._enforce_sharded_generate = True
    try:
        key = jax.random.PRNGKey(0)
        Height = hax.Axis("Height", 4)
        Width = hax.Axis("Width", 8)
        lb = hax.full(Height, -3.0)
        ub = 0.5
        u = generate_sharded(hax.random.uniform, axis=Height)(key, shape=(Height, Width), minval=lb, maxval=ub)

        assert u.axes == (Height, Width)

        assert hax.all(u >= -3.0)
        assert hax.all(u <= 0.5)
    finally:
        hax.random._enforce_sharded_generate = False

    # now just assert that this does in fact change the randomness
    u2 = hax.random.uniform(key, shape=(Height, Width), minval=lb, maxval=ub)
    assert not hax.all(u == u2)
