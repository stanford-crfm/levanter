from typing import Tuple

import jax
import optax
import pytest

import jax.numpy as jnp
from jax import tree_structure, tree_flatten, tree_unflatten
from jax.experimental.maps import xmap
from psithuros.named_tensors import *
import equinox as eqx


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
    show_example(mod)

    axes = infer_named_axes_from_module(mod)

    structure = jax.tree_structure(mod)
    expected_axes = jax.tree_unflatten(structure, [("x", "y"), (..., ), (...,), ("x", ...,)])

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
    XMappedModule = xmapped_class(MyModuleInit)
    mod = XMappedModule(axis_sizes={"x": 2, "y": 3})
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

    XMappedModule = xmapped_class(Mod2)
    mod2 = XMappedModule(axis_sizes={"x": 2, "y": 3})
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

    XMappedModule = xmapped_class(ModWithArgs)
    in_array = jnp.ones((2, 3))
    mod = XMappedModule(in_array)
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

    XMappedModule = xmapped_class(ModWithArgs)
    in_array = jnp.ones(3)
    unaxised = jnp.ones(10)
    mod = XMappedModule(in_array, unaxised, axis_sizes={"x": 2})
    assert(isinstance(mod, ModWithArgs))
    assert(mod.array.shape == (2, 3))
    assert(mod.array2.shape == (2, 10))
    assert(mod.unaxised.shape == (10, ))
