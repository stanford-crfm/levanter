from typing import Tuple

import jax
import optax

import jax.numpy as jnp
import pytest
from jax import tree_structure, tree_flatten, tree_unflatten
import jax.random as jrandom
from jax.experimental.maps import xmap
from psithuros.axis_names import *
import equinox as eqx

from psithuros.axis_names import infer_named_axes_from_module, UnnamedAxes


class MyModule(eqx.Module):
    named: Array["x", "y"]
    unnamed1: Array
    unnamed2: Array[...]
    partially_named: Array["x", ...]
    static_field: int = eqx.static_field()

def show_example(structured):
  flat, tree = tree_flatten(structured)
  unflattened = tree_unflatten(tree, flat)
  print("structured={}\n  flat={}\n  tree={}\n  unflattened={}".format(
      structured, flat, tree, unflattened))


def test_infer_named_axes():
    mod = MyModule(named=jnp.ones((2, 3)), unnamed1=jnp.ones((2)), unnamed2=jnp.ones((2)), partially_named=jnp.ones((2, 3)), static_field=1)

    axes = infer_named_axes_from_module(mod)

    structure = jax.tree_structure(mod)
    expected_axes = jax.tree_unflatten(structure, [AxisNames(("x", "y")), UnnamedAxes, UnnamedAxes, AxisNames(("x", ...,))])

    assert axes == expected_axes


def test_auto_xmap_identity():
    def identity(x: Array["x", "y"]) -> Array["x", "y"]:
        return x

    arr = jnp.ones((2, 3))

    fun = auto_xmap(identity)
    assert jnp.all(jnp.isclose(fun(arr), arr))


def test_auto_xmap_identity_module():
    mod = MyModule(named=jnp.ones((2, 3)), unnamed1=jnp.ones((2)), unnamed2=jnp.ones((2)), partially_named=jnp.ones((2, 3)), static_field=1)

    def identity(x) -> MyModule:
        return x

    fun = auto_xmap(identity)
    assert fun(mod) == mod


@pytest.mark.skip(reason="doesn't work and possibly can't be fixed. Need to use type params or something")
def test_auto_xmap_optax():
    mod = MyModule(named=jnp.ones((2, 3)), unnamed1=jnp.ones((2)), unnamed2=jnp.ones((2)), partially_named=jnp.ones((2, 3)), static_field=1)
    adam = optax.adam(1E-4)
    opt = adam.init(mod)

    def identity(x, opt) -> Tuple[MyModule, MyModule]:
        return x, opt

    fun = auto_xmap(identity)
    assert fun(mod, opt) == (mod, opt)


class MyModuleInit(eqx.Module):
    named: Array["x", "y"]
    unnamed1: Array
    unnamed2: Array[...]
    partially_named: Array["x", ...]
    static_field: int = eqx.static_field()

    def __init__(self):
        self.named = jnp.ones(())
        self.unnamed1 = jnp.ones(())
        self.unnamed2 = jnp.ones(())
        self.partially_named = jnp.ones(10)
        self.static_field = 1


def test_xmap_class_init():
    mod = xmapped_init(MyModuleInit, axis_sizes={"x": 2, "y": 3})()
    assert(isinstance(mod, MyModuleInit))
    assert(mod.named.shape == (2, 3))
    assert(mod.unnamed1.shape == ())
    assert(mod.unnamed2.shape == ())
    assert(mod.partially_named.shape == (2, 10))


def test_xmap_class_nested_init():
    class Mod2(eqx.Module):
        inner: MyModuleInit

        def __init__(self):
            self.inner = MyModuleInit()

    mod2 = xmapped_init(Mod2, axis_sizes={"x": 2, "y": 3})()
    assert(isinstance(mod2, Mod2))
    mod = mod2.inner
    assert(mod.named.shape == (2, 3))
    assert(mod.unnamed1.shape == ())
    assert(mod.unnamed2.shape == ())
    assert(mod.partially_named.shape == (2, 10))


def test_xmap_class_init_with_args():
    class ModWithArgs(eqx.Module):
        array: Array["x", "y"]
        array2: Array["x", ...]

        def __init__(self, in_array: Array["x", "y"]):
            self.array = in_array
            self.array2 = jnp.zeros(10)

    mod = xmapped_init(ModWithArgs)(jnp.ones((2, 3)))
    assert(isinstance(mod, ModWithArgs))
    assert(mod.array.shape == (2, 3))
    assert(mod.array2.shape == (2, 10))

def test_xmap_class_init_with_args_partial_dim():
    class ModWithArgs(eqx.Module):
        array: Array["x", "y"]
        array2: Array["x", ...]
        unaxised: Array[...]

        def __init__(self, in_array: Array["y"], unaxised: Array[...]):
            self.array = in_array
            self.array2 = jnp.zeros(10)
            self.unaxised = unaxised

    in_array = jnp.ones(3)
    unaxised = jnp.ones(10)
    mod = xmapped_init(ModWithArgs, axis_sizes={"x": 2})(in_array, unaxised)
    assert(isinstance(mod, ModWithArgs))
    assert(mod.array.shape == (2, 3))
    assert(mod.array2.shape == (2, 10))
    assert(mod.unaxised.shape == (10, ))


def test_xmap_class_with_shaped_annotation():
    class Mod(eqx.Module):
        shaped_linear: Shaped["shard", eqx.nn.Linear]
        array: Shaped["y", Array["x", ...]]
        multi_array: Shaped[("shard", "y"), Array]

        def __init__(self, key):
            self.shaped_linear = eqx.nn.Linear(5, 1, key=key)
            self.array = jnp.zeros(10)
            self.multi_array = jnp.zeros(2)

    mod = xmapped_init(Mod, axis_sizes={"x": 2, "y": 3, "shard": 4})(jrandom.PRNGKey(0))
    assert(isinstance(mod, Mod))
    assert(mod.shaped_linear.weight.shape == (4, 1, 5))
    assert(mod.shaped_linear.bias.shape == (4, 1))
    assert(mod.array.shape == (3, 2, 10))
    assert(mod.multi_array.shape == (4, 3, 2))
