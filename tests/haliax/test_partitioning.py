import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxtyping import Array
from test_utils import skip_if_not_enough_devices

import haliax as hax
from haliax import Axis, NamedArray
from haliax.partitioning import ResourceAxis, axis_mapping, infer_resource_partitions, named_jit


class MyModule(eqx.Module):
    named: NamedArray
    unnamed1: Array
    static_field: int = eqx.static_field()


Dim1 = Axis("dim1", 8)
Dim2 = Axis("dim2", 16)
Dim3 = Axis("dim3", 32)

resource_map = {
    "dim2": ResourceAxis.DATA,
    "dim3": ResourceAxis.MODEL,
}


def test_infer_named_axes():
    mesh = Mesh(np.array(jax.devices()).reshape(-1, 1), (ResourceAxis.DATA, ResourceAxis.MODEL))
    with axis_mapping(resource_map), mesh:
        mod = MyModule(named=hax.ones((Dim1, Dim2, Dim3)), unnamed1=jnp.ones(Dim2.size), static_field=1)

        axes: MyModule = infer_resource_partitions(mod)

        spec = PartitionSpec(None, ResourceAxis.DATA, ResourceAxis.MODEL)

        assert axes.named.array == NamedSharding(mesh, spec)
        assert axes.unnamed1.is_fully_replicated


class MyModuleInit(eqx.Module):
    named: NamedArray
    unnamed1: Array
    named2: NamedArray
    static_field: int = eqx.static_field()

    def __init__(self):
        self.named = hax.ones((Dim2, Dim3))
        self.unnamed1 = jnp.ones(())
        self.named2 = hax.ones(Dim3)
        self.static_field = 1


@skip_if_not_enough_devices(4)
def test_pjit_class_init():
    with axis_mapping(resource_map):
        devices = jax.devices()
        with Mesh(np.array(devices).reshape(-1, 2), (ResourceAxis.DATA, ResourceAxis.MODEL)):
            mod = named_jit(MyModuleInit)()

        assert mod.named.array.shape == (Dim2.size, Dim3.size)

        assert mod.unnamed1.shape == ()
        assert mod.named2.array.shape == (Dim3.size,)


@skip_if_not_enough_devices(4)
def test_pjit_class_nested_init():
    with axis_mapping(resource_map):

        class Mod2(eqx.Module):
            inner: MyModuleInit

            def __init__(self):
                self.inner = MyModuleInit()

        devices = jax.devices()
        with Mesh(np.array(devices).reshape(-1, 2), (ResourceAxis.DATA, ResourceAxis.MODEL)):
            mod2 = named_jit(Mod2)()

        mod = mod2.inner
        assert mod.named.array.shape == (Dim2.size, Dim3.size)
        assert mod.unnamed1.shape == ()
        assert mod.named2.array.shape == (Dim3.size,)


def test_pjit_class_init_with_args():
    with axis_mapping(resource_map):

        class ModWithArgs(eqx.Module):
            array: NamedArray
            array2: NamedArray

            def __init__(self, in_array: NamedArray):
                self.array = in_array
                self.array2 = hax.zeros(Dim3)

        devices = jax.devices()
        with Mesh(np.array(devices).reshape(-1, 1), (ResourceAxis.DATA, ResourceAxis.MODEL)):
            mod = named_jit(ModWithArgs)(hax.ones((Dim1, Dim2)))
        assert isinstance(mod, ModWithArgs)
        assert mod.array.array.shape == (Dim1.size, Dim2.size)
        assert mod.array2.array.shape == (Dim3.size,)


def test_infer_resource_partition_gda_bug():
    devices = jax.devices()
    with Mesh(np.array(devices).reshape(-1, 1), (ResourceAxis.DATA, ResourceAxis.MODEL)):
        jax.config.update("jax_parallel_functions_output_gda", True)
        try:

            def foo():
                return hax.zeros((Dim1, Dim2, Dim3))

            pjit_foo = named_jit(foo, resource_map)
            r = pjit_foo()
            assert r.axes == (Dim1, Dim2, Dim3)

            def bar(x):
                return x

            # this won't work with GDAs
            pjit_bar = named_jit(bar, resource_map)
            r = pjit_bar(r)
            assert r.axes == (Dim1, Dim2, Dim3)

        finally:
            jax.config.update("jax_parallel_functions_output_gda", False)


@skip_if_not_enough_devices(4)
def test_shard_with_axis_mapping_outside_pjit():
    devices = jax.devices()
    with Mesh(np.array(devices).reshape(-1, 2), (ResourceAxis.DATA, ResourceAxis.MODEL)) as mesh:
        x = hax.ones((Dim1, Dim2))
        y = hax.ones((Dim2, Dim3))

        x = hax.shard_with_axis_mapping(x, resource_map)
        assert x.array.sharding == NamedSharding(mesh, PartitionSpec(None, ResourceAxis.DATA))

        y = hax.shard_with_axis_mapping(y, resource_map)
        assert y.array.sharding == NamedSharding(mesh, PartitionSpec(ResourceAxis.DATA, ResourceAxis.MODEL))


def test_named_jit_works_without_axis_resources():
    devices = jax.devices()
    with Mesh(np.array(devices).reshape(-1, 1), (ResourceAxis.DATA, ResourceAxis.MODEL)) as mesh:

        def foo(x):
            return x

        pjit_foo = named_jit(foo)
        r = pjit_foo(hax.ones((Dim1, Dim2)))

        assert r.array.sharding.is_fully_replicated

        def foo2(x):
            return hax.shard_with_axis_mapping(x, resource_map)

        pjit_foo2 = named_jit(foo2)
        r2 = pjit_foo2(hax.ones((Dim1, Dim2)))

        assert r2.array.sharding.is_equivalent_to(NamedSharding(mesh, PartitionSpec(None, ResourceAxis.DATA)), ndim=2)


@skip_if_not_enough_devices(4)
def test_shard_with_axis_mapping_inside_jit():
    devices = jax.devices()
    with Mesh(np.array(devices).reshape(-1, 2), (ResourceAxis.DATA, ResourceAxis.MODEL)) as mesh:
        x = hax.ones((Dim1, Dim2))
        y = hax.ones((Dim2, Dim3))

        def assert_inside_pjit(arr, expected: NamedSharding):
            def assert_eq(x, y):
                assert x == y

            jax.debug.inspect_array_sharding(arr.array, callback=lambda x: assert_eq(x, expected))

        @named_jit(in_axis_resources={}, out_axis_resources=resource_map)
        def do_shard(x, y):
            x = hax.shard_with_axis_mapping(x, resource_map)
            assert_inside_pjit(x, NamedSharding(mesh, PartitionSpec(None, ResourceAxis.DATA)))

            y = hax.shard_with_axis_mapping(y, resource_map)
            assert_inside_pjit(y, NamedSharding(mesh, PartitionSpec(ResourceAxis.DATA, ResourceAxis.MODEL)))

            return x, y

        x, y = do_shard(x, y)

        assert x.array.sharding == NamedSharding(mesh, PartitionSpec(None, ResourceAxis.DATA))
        assert y.array.sharding == NamedSharding(mesh, PartitionSpec(ResourceAxis.DATA, ResourceAxis.MODEL))


def test_shard_scalar_in_module():
    with axis_mapping(resource_map):

        class MyModule(eqx.Module):
            scalar: jnp.ndarray

            def __init__(self):
                self.scalar = jnp.zeros(
                    (),
                )

        devices = jax.devices()
        with Mesh(np.array(devices).reshape(-1, 1), (ResourceAxis.DATA, ResourceAxis.MODEL)):
            mod = named_jit(MyModule)()
            assert mod.scalar.sharding.is_fully_replicated


def test_shard_plain_array_in_module():
    with axis_mapping(resource_map):

        class MyModule(eqx.Module):
            array: jnp.ndarray

            def __init__(self):
                self.array = jnp.zeros((8, 8))

        devices = jax.devices()
        with Mesh(np.array(devices).reshape(-1, 1), (ResourceAxis.DATA, ResourceAxis.MODEL)):
            mod = named_jit(MyModule)()
            assert mod.array.sharding.is_fully_replicated
