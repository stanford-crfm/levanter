import jax.numpy as jnp
import pytest
from jax.random import PRNGKey

import haliax as hax
from haliax import Axis, NamedArray


def test_trace():
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)
    named1 = hax.random.uniform(PRNGKey(0), (Width, Depth))
    trace1 = hax.trace(named1, Width, Depth)
    assert jnp.all(jnp.isclose(trace1.array, jnp.trace(named1.array)))
    assert len(trace1.axes) == 0

    trace1 = hax.trace(named1, "Width", "Depth")
    assert jnp.all(jnp.isclose(trace1.array, jnp.trace(named1.array)))
    assert len(trace1.axes) == 0

    Height = Axis("Height", 10)
    named2 = hax.random.uniform(PRNGKey(0), (Height, Width, Depth))
    trace2 = hax.trace(named2, Width, Depth)
    assert jnp.all(jnp.isclose(trace2.array, jnp.trace(named2.array, axis1=1, axis2=2)))
    assert trace2.axes == (Height,)

    trace2 = hax.trace(named2, "Width", "Depth")
    assert jnp.all(jnp.isclose(trace2.array, jnp.trace(named2.array, axis1=1, axis2=2)))
    assert trace2.axes == (Height,)


def test_add():
    Height = Axis("Height", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    named1 = hax.random.uniform(PRNGKey(0), (Height, Width, Depth))
    named2 = hax.random.uniform(PRNGKey(1), (Height, Width, Depth))

    named3 = named1 + named2
    assert jnp.all(jnp.isclose(named3.array, named1.array + named2.array))

    named2_reorder = named2.rearrange((Width, Height, Depth))
    named4 = named1 + named2_reorder
    named4 = named4.rearrange((Height, Width, Depth))
    assert jnp.all(jnp.isclose(named4.array, named1.array + named2.array))


def test_add_broadcasting():
    Height = Axis("Height", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    named1 = hax.random.uniform(PRNGKey(0), (Height, Width, Depth))
    named2 = hax.random.uniform(PRNGKey(1), (Width, Depth))

    named3 = named1 + named2
    assert jnp.all(jnp.isclose(named3.array, named1.array + named2.array))

    named2_reorder = named2.rearrange((Depth, Width))
    named4 = named1 + named2_reorder
    named4 = named4.rearrange((Height, Width, Depth))

    assert jnp.all(jnp.isclose(named4.array, named1.array + named2.array))

    # now for the broadcasting we don't like
    named5 = hax.random.uniform(PRNGKey(1), (Height, Depth))
    named6 = hax.random.uniform(PRNGKey(2), (Width, Depth))

    with pytest.raises(ValueError):
        _ = named5 + named6


def test_add_scalar():
    Height = Axis("Height", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    named1 = hax.random.uniform(PRNGKey(0), (Height, Width, Depth))
    named2 = named1 + 1.0
    assert jnp.all(jnp.isclose(named2.array, named1.array + 1.0))

    named3 = 1.0 + named1
    assert jnp.all(jnp.isclose(named3.array, named1.array + 1.0))


def test_add_no_overlap():
    Height = Axis("Height", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    named1: NamedArray = hax.random.uniform(PRNGKey(0), (Height))
    named2 = hax.random.uniform(PRNGKey(1), (Width, Depth))

    with pytest.raises(ValueError):
        _ = named1 + named2

    named3 = named1.broadcast_to((Height, Width, Depth)) + named2

    assert jnp.all(
        jnp.isclose(named3.array, named1.array.reshape((-1, 1, 1)) + named2.array.reshape((1,) + named2.array.shape))
    )


# TODO: tests for other ops:


def test_where():
    Height = Axis("Height", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    named1 = hax.random.uniform(PRNGKey(0), (Height, Width, Depth))
    named2 = hax.random.uniform(PRNGKey(1), (Height, Width, Depth))

    named3 = hax.where(named1 > named2, named1, named2)
    assert jnp.all(jnp.isclose(named3.array, jnp.where(named1.array > named2.array, named1.array, named2.array)))

    named2_reorder = named2.rearrange((Width, Height, Depth))
    named4 = hax.where(named1 > named2_reorder, named1, named2_reorder)
    named4 = named4.rearrange((Height, Width, Depth))
    assert jnp.all(jnp.isclose(named4.array, jnp.where(named1.array > named2.array, named1.array, named2.array)))

    # now some broadcasting
    named5 = hax.random.uniform(PRNGKey(1), (Height, Width))
    named6 = hax.random.uniform(PRNGKey(2), Width)

    named7 = hax.where(named5 > named6, named5, named6)
    named7 = named7.rearrange((Height, Width))
    assert jnp.all(jnp.isclose(named7.array, jnp.where(named5.array > named6.array, named5.array, named6.array)))

    # now for the broadcasting we don't like
    named5 = hax.random.uniform(PRNGKey(1), (Height, Depth))
    named6 = hax.random.uniform(PRNGKey(2), (Width, Depth))

    with pytest.raises(ValueError):
        _ = hax.where(named5 > named6, named5, named6)


def test_clip():
    Height = Axis("Height", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    named1 = hax.random.uniform(PRNGKey(0), (Height, Width, Depth))
    named2 = hax.clip(named1, 0.3, 0.7)
    assert jnp.all(jnp.isclose(named2.array, jnp.clip(named1.array, 0.3, 0.7)))

    named2_reorder = named2.rearrange((Width, Height, Depth))
    named3 = hax.clip(named2_reorder, 0.3, 0.7)
    named3 = named3.rearrange((Height, Width, Depth))
    assert jnp.all(jnp.isclose(named3.array, jnp.clip(named2.array, 0.3, 0.7)))

    # now some interesting broadcasting
    lower = hax.full((Height, Width), 0.3)
    upper = hax.full((Width, Depth), 0.7)
    named4 = hax.clip(named1, lower, upper)
    named4 = named4.rearrange((Height, Width, Depth))

    assert jnp.all(
        jnp.isclose(
            named4.array,
            jnp.clip(
                named1.array,
                lower.array.reshape((Height.size, Width.size, 1)),
                upper.array.reshape((1, Width.size, Depth.size)),
            ),
        )
    )


def test_tril_triu():
    Height = Axis("Height", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    for hax_fn, jnp_fn in [(hax.tril, jnp.tril), (hax.triu, jnp.triu)]:
        named1 = hax.random.uniform(PRNGKey(0), (Height, Width, Depth))
        named2 = hax_fn(named1, Width, Depth)
        assert jnp.all(jnp.isclose(named2.array, jnp_fn(named1.array)))

        named3 = hax_fn(named1, Width, Depth, k=1)
        assert jnp.all(jnp.isclose(named3.array, jnp_fn(named1.array, k=1)))

        named4 = hax_fn(named1, Width, Depth, k=-1)
        assert jnp.all(jnp.isclose(named4.array, jnp_fn(named1.array, k=-1)))

        named5 = hax_fn(named1, Height, Depth)
        expected5 = jnp_fn(named1.array.transpose([1, 0, 2]))
        assert jnp.all(jnp.isclose(named5.array, expected5))
