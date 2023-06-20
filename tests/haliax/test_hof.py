import jax.numpy as jnp
from jax.random import PRNGKey

import haliax as hax
from haliax import Axis, NamedArray
from haliax.util import is_named_array


def test_scan():
    Height = Axis("Height", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)
    named1 = hax.random.uniform(PRNGKey(0), (Height, Width, Depth))

    def scan_fun(acc, x):
        return acc + jnp.sum(x.array), x.take(Width, 2)

    total, selected = hax.scan(scan_fun, Height)(0.0, named1)

    assert jnp.all(jnp.isclose(total, jnp.sum(named1.array, axis=(0, 1, 2))))
    assert jnp.all(jnp.equal(selected.array, named1.take(Width, 2).array))

    total, selected = hax.scan(scan_fun, "Height")(0.0, named1)

    assert jnp.all(jnp.isclose(total, jnp.sum(named1.array, axis=(0, 1, 2))))
    assert jnp.all(jnp.equal(selected.array, named1.take("Width", 2).array))


def test_scan_not_0th_axis():
    Height = Axis("Height", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)
    named1 = hax.random.uniform(PRNGKey(0), (Height, Width, Depth))

    def scan_fun(acc, x):
        return acc + jnp.sum(x.array), x.take(Width, 2)

    total, selected = hax.scan(scan_fun, Depth)(0.0, named1)

    assert jnp.all(jnp.isclose(total, jnp.sum(named1.array, axis=(0, 1, 2))))
    assert jnp.all(jnp.equal(selected.array, named1.take(Width, 2).rearrange(selected.axes).array))


def test_scan_str_arg():
    Height = Axis("Height", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)
    named1 = hax.random.uniform(PRNGKey(0), (Height, Width, Depth))

    def scan_fun(acc, x):
        return acc + jnp.sum(x.array), x.take("Width", 2)

    total, selected = hax.scan(scan_fun, "Height")(0.0, named1)

    assert jnp.all(jnp.isclose(total, jnp.sum(named1.array, axis=(0, 1, 2))))
    assert jnp.all(jnp.equal(selected.array, named1.take(Width, 2).array))


def test_scan_static_args():
    Height = Axis("Height", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)
    named1 = hax.random.uniform(PRNGKey(0), (Height, Width, Depth))

    def scan_fun(acc, x, static1, *, static2):
        assert static1 is True
        assert static2 is False
        return acc + jnp.sum(x.array), x.take(Width, 2)

    total, selected = hax.scan(scan_fun, Depth, is_scanned=is_named_array)(0.0, named1, True, static2=False)

    assert jnp.all(jnp.isclose(total, jnp.sum(named1.array, axis=(0, 1, 2))))
    assert jnp.all(jnp.equal(selected.array, named1.take(Width, 2).rearrange(selected.axes).array))


def test_reduce():
    Height = Axis("Height", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)
    named1 = hax.random.uniform(PRNGKey(0), (Height, Width, Depth))

    acc = hax.zeros((Height, Width))

    total = hax.fold(lambda x, y: x + y, Depth)(acc, named1)

    assert jnp.all(jnp.isclose(total.rearrange(acc.axes).array, jnp.sum(named1.array, axis=2)))


def test_reduce_str_args():
    Height = Axis("Height", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)
    named1 = hax.random.uniform(PRNGKey(0), (Height, Width, Depth))

    acc = hax.zeros((Height, Width))

    total = hax.fold(lambda x, y: x + y, "Depth")(acc, named1)

    assert jnp.all(jnp.isclose(total.rearrange(acc.axes).array, jnp.sum(named1.array, axis=2)))


def test_reduce_static_args():
    Height = Axis("Height", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)
    named1 = hax.random.uniform(PRNGKey(0), (Height, Width, Depth))

    def fold_fun(acc, x, static1, *, static2):
        assert static1 is True
        assert static2 is False
        return NamedArray(acc.array + x.rearrange(acc.axes).array, acc.axes)

    acc = hax.zeros((Height, Width))

    total = hax.fold(fold_fun, Depth)(acc, named1, True, static2=False)

    assert jnp.all(jnp.isclose(total.rearrange(acc.axes).array, jnp.sum(named1.array, axis=2)))


def test_vmap_unmapped_args():
    Batch = Axis("Batch", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)
    named1 = hax.random.uniform(PRNGKey(0), (Width, Depth))

    def vmap_fun(x):
        return x.take(Width, 2)

    selected = hax.vmap(vmap_fun, Batch)(named1)

    expected_jax = jnp.array([named1.take(Width, 2).array for _ in range(Batch.size)])
    expected_names = (Batch, Depth)

    assert jnp.all(jnp.equal(selected.array, expected_jax))
    assert selected.axes == expected_names


def test_vmap_mapped_args():
    Batch = Axis("Batch", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)
    named1 = hax.random.uniform(PRNGKey(0), (Batch, Width, Depth))

    def vmap_fun(x):
        return x.sum(Width)

    selected = hax.vmap(vmap_fun, Batch)(named1)

    expected_jax = jnp.array([named1.sum(Width).array for _ in range(Batch.size)])
    expected_names = (Batch, Depth)

    assert jnp.all(jnp.equal(selected.array, expected_jax))
    assert selected.axes == expected_names


def test_vmap_str_args():
    Batch = Axis("Batch", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)
    named1 = hax.random.uniform(PRNGKey(0), (Batch, Width, Depth))

    def vmap_fun(x):
        return x.sum(Width)

    selected = hax.vmap(vmap_fun, "Batch")(named1)

    expected_jax = jnp.array([named1.sum(Width).array for _ in range(Batch.size)])
    expected_names = (Batch, Depth)

    assert jnp.all(jnp.equal(selected.array, expected_jax))
    assert selected.axes == expected_names

    # also ensure that this works when we return a non-haliax array
    def vmap_fun(x):
        return x.sum(Width).array

    selected = hax.vmap(vmap_fun, "Batch")(named1)

    assert jnp.all(jnp.equal(selected, expected_jax))


def test_vmap_mapped_kwarg():
    Batch = Axis("Batch", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)
    named1 = hax.random.uniform(PRNGKey(0), (Batch, Width, Depth))

    def vmap_fun(x):
        return x.sum(Width)

    selected = hax.vmap(vmap_fun, Batch)(x=named1)

    expected_jax = jnp.array([named1.sum(Width).array for _ in range(Batch.size)])
    expected_names = (Batch, Depth)

    assert jnp.all(jnp.equal(selected.array, expected_jax))
    assert selected.axes == expected_names


def test_vmap_static_args():
    Batch = Axis("Batch", 10)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)
    named1 = hax.random.uniform(PRNGKey(0), (Batch, Width, Depth))

    def vmap_fun(x, y):
        return x.sum(Width) if y else x

    selected = hax.vmap(vmap_fun, Batch)(named1, True)

    expected = hax.sum(named1, Width)

    assert jnp.all(jnp.equal(selected.array, expected.array))
    assert selected.axes == expected.axes
