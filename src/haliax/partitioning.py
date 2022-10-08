import contextlib
import functools
import threading
import typing
from math import prod
from typing import List, Mapping, Optional, Sequence, TypeVar, Union

import equinox as eqx
import jax
from equinox import is_array
from equinox.compile_utils import compile_cache, get_fun_names, hashable_combine, hashable_partition
from jax.experimental.global_device_array import GlobalDeviceArray
from jax.experimental.pjit import FROM_GDA, pjit, with_sharding_constraint
from jax.interpreters.pxla import PartitionSpec
from jaxtyping import PyTree

from .core import Axis, AxisSpec, NamedArray
from .jax_utils import filter_eval_shape
from .util import StringHolderEnum, ensure_tuple, is_jax_array_like, is_named_array


LogicalAxisName = str
PhysicalAxis = str
PhysicalAxisSpec = Union[PhysicalAxis, Sequence[PhysicalAxis]]
ResourceMapping = Mapping[LogicalAxisName, PhysicalAxisSpec]
"""Mapping from logical axis names to physical axis names"""


class ResourceAxis(StringHolderEnum):
    """Standard names for physical axes"""

    MODEL = "model"
    DATA = "data"


class _ResourceMappingHolder:
    """Global resource mapping, used with a context manager to give dynamic scoping to resource mappings"""

    def __init__(self):
        self.thread_data = threading.local()
        self.thread_data.resource_mapping = None


_mapping_holder = _ResourceMappingHolder()


@contextlib.contextmanager
def axis_mapping(mapping: ResourceMapping, *, merge: bool = True, **kwargs):
    """Context manager for setting the global resource mapping"""
    mapping = dict(mapping)

    old_mapping = _mapping_holder.thread_data.resource_mapping
    if merge:
        mapping.update(old_mapping or {})

    if len(kwargs):
        mapping.update(kwargs)

    _mapping_holder.thread_data.resource_mapping = mapping
    yield
    _mapping_holder.thread_data.resource_mapping = old_mapping


T = TypeVar("T", bound=PyTree)


def auto_sharded(x: T) -> T:
    """
    Shard a PyTree using the global axis mapping. NamedArrays in the PyTree are sharded using the axis mapping
     and the names in the tree.

    If there is no axis mapping, the global axis mapping, this function is a no-op.
    """
    mapping = _mapping_holder.thread_data.resource_mapping

    if mapping is None:
        return x

    return shard_with_axis_mapping(x, mapping)


def shard_with_axis_mapping(x: T, mapping: ResourceMapping) -> T:
    """
    Shard a PyTree using the provided axis mapping. NamedArrays in the PyTree are sharded using the axis mapping.
    Other arrays are not sharded.

    :param x:
    :param mapping:
    :return:
    """

    def _as_pspec(x):
        if isinstance(x, NamedArray):
            physical_names: List[Optional[PhysicalAxisSpec]] = [mapping.get(a.name, None) for a in x.axes]
        elif is_array(x):
            physical_names = [None] * len(x.shape)
        else:
            return None

        spec = PartitionSpec(
            *tuple(tuple(p) if not (isinstance(p, str)) and isinstance(p, Sequence) else p for p in physical_names)
        )
        return spec

    pspec = jax.tree_util.tree_map(_as_pspec, x, is_leaf=is_named_array)
    return with_sharding_constraint(x, pspec)


def infer_resource_partitions(tree: PyTree, resource_mapping: Optional[ResourceMapping] = None) -> PyTree:
    """
    Infer the resource partitions for a module, to be used with pjit.
    The basic idea is to tree all NamedArrays as leaves for the purposes of this function,
    and to create PartitionSpecs from those names plus the resource_mapping.

    If resource_mapping is not provided, this function attempts to use the global resource mapping.
    """
    if resource_mapping is None:
        resource_mapping = _mapping_holder.thread_data.resource_mapping

    if resource_mapping is None:
        raise ValueError("No resource mapping found")

    _resource_mapping = typing.cast(ResourceMapping, resource_mapping)  # for mypy

    def partition_spec(node: typing.Any):
        if isinstance(node, NamedArray):
            if isinstance(node.array, GlobalDeviceArray):
                # TODO: should probably check for compatibility
                return FROM_GDA
            else:
                return NamedArray(
                    PartitionSpec(*tuple(_resource_mapping.get(axis.name, None) for axis in node.axes)),  # type: ignore
                    node.axes,
                )
        elif isinstance(node, GlobalDeviceArray):
            return FROM_GDA
        # TODO: jax.Array
        else:
            return None

    return jax.tree_util.tree_map(partition_spec, tree, is_leaf=is_named_array)


def eval_resource_partitions(fn):
    """
    Similar to jax.eval_shape but for resource partitions. It returns a PyTree of PartitionSpecs.
    """

    def f(*args, **kwargs):
        out_shape = jax.eval_shape(fn, *args, **kwargs)
        return infer_resource_partitions(out_shape)

    return f


# This is more or less copy-pasted from Equinox's similar functions (pmap, vmap, etc), but
# it's not really explained there so we'll explain it here.
# Many jax functions work by compiling functions to XLA. The compilation process is expensive,
# so we want to cache the compiled functions. However, the compiled functions are tied to the
# "static" arguments to the functions. This is particularly important for a library like Equinox,
# which Haliax is built on top of, because Equinox uses pytrees extensively for modules, and mixes "static"
# configuration with "dynamic" data.
# Thus we need to carefully partition the arguments to the function into "static" and "dynamic" arguments,
# and cache our compiled functions based on the static arguments.
# In Equinox conceptually there are three types of "arguments": positional, named, and the function itself.
# All of these are pytrees, and we need to partition them into static and dynamic arguments.
# Inside the function, we then combine the arguments into a single pytree, and pass that to the original function.
# With pjit we also have "donated" arguments, which are arguments that we promise not to use after the function
# returns. This is useful for conserving memory, but we also have to splice them back in.
# Also recall that a "pytree" can split into leaves and a "treedef", which can then be reconstructed.
@compile_cache
def _named_pjit_cache(fun_names, **jitkwargs):
    def fun_wrapped(dynamic_donated, dynamic_reserved, static):
        dynamic = eqx.combine(dynamic_donated, dynamic_reserved)
        dynamic_fun, dynamic_spec = dynamic

        (
            static_fun_treedef,
            static_fun_leaves,
            static_spec_treedef,
            static_spec_leaves,
        ) = static

        fun = hashable_combine(dynamic_fun, static_fun_leaves, static_fun_treedef)
        args, kwargs = hashable_combine(dynamic_spec, static_spec_leaves, static_spec_treedef)
        out = fun(*args, **kwargs)
        return out

    fun_name, fun_qualname = fun_names
    fun_wrapped.__name__ = fun_name
    fun_wrapped.__qualname__ = fun_qualname

    return pjit(fun_wrapped, static_argnums=2, donate_argnums=0, **jitkwargs)


def named_pjit(
    fn=None,
    axis_resources: Optional[ResourceMapping] = None,
    *,
    in_axis_resources: Optional[ResourceMapping] = None,
    out_axis_resources: Optional[ResourceMapping] = None,
    donate_args: Optional[PyTree] = None,
    donate_kwargs: Optional[PyTree] = None,
    **pjit_args,
):
    """
    A version of pjit that uses NamedArrays, GlobalDeviceArrays, and the provided resource mapping to infer the
    resource partitions.

    If no resource mapping is provided, this function attempts to use the global resource mapping.
    If either of in_axis_resources or out_axis_resources is provided, then both must be provided.
    If axis_resources is provided, then in_axis_resources and out_axis_resources must not be provided.

    :param fn: The function to be pjit'd
    :param axis_resources: A mapping from logical axis names to physical axis names
    :param in_axis_resources: A mapping from logical axis names to physical axis names for the input
    :param out_axis_resources: A mapping from logical axis names to physical axis names for the output
    :param donate_args: A PyTree of booleans or function leaf->bool, indicating whether to donate arguments to the
     computation
    :param donate_kwargs: A PyTree of booleans or function leaf->bool, indicating whether to donate keyword arguments to
        the computation
    """
    # TODO: support jax.Array

    if in_axis_resources is not None or out_axis_resources is not None:
        if axis_resources is not None:
            raise ValueError("Cannot provide both axis_resources and in_axis_resources/out_axis_resources")
        if in_axis_resources is None or out_axis_resources is None:
            raise ValueError("Must provide both in_axis_resources and out_axis_resources")

    if fn is None:
        return functools.partial(
            named_pjit,
            axis_resources=axis_resources,
            in_axis_resources=in_axis_resources,
            out_axis_resources=out_axis_resources,
            donate_args=donate_args,
            donate_kwargs=donate_kwargs,
            **pjit_args,
        )

    if axis_resources is None and in_axis_resources is None:
        axis_resources = _mapping_holder.thread_data.resource_mapping

    if axis_resources is not None:
        in_axis_resources = axis_resources
        out_axis_resources = axis_resources

    if in_axis_resources is None or out_axis_resources is None:
        raise ValueError(
            "Must provide in_axis_resources and out_axis_resources, or axis_resources, or have a global "
            "mapping vis axis_mapping"
        )

    dynamic_fun, static_fun_leaves, static_fun_def = hashable_partition(fn, is_jax_array_like)

    @functools.wraps(fn)
    def f(*args, **kwargs):
        dynamic_argspec, static_argspec, static_arg_def = hashable_partition((args, kwargs), is_jax_array_like)
        dynamic = (dynamic_fun, dynamic_argspec)

        if donate_args is not None or donate_kwargs is not None:
            dargs = donate_args or (False,) * len(args)
            dkwargs = donate_kwargs or {k: False for k in kwargs}
            dynamic_donated, dynamic_reserved = eqx.partition(dynamic, (False, (dargs, dkwargs)))
        else:
            dynamic_donated = jax.tree_util.tree_map(lambda _: None, dynamic)
            dynamic_reserved = dynamic

        static = (
            static_fun_def,
            static_fun_leaves,
            static_arg_def,
            static_argspec,
        )

        in_resources = infer_resource_partitions((dynamic_donated, dynamic_reserved), in_axis_resources)
        shapes = filter_eval_shape(fn, *args, **kwargs)
        out_resources = infer_resource_partitions(shapes, out_axis_resources)

        my_pjit_args = dict(**pjit_args)
        my_pjit_args["in_axis_resources"] = in_resources
        my_pjit_args["out_axis_resources"] = out_resources
        cached_pjitted_fun = _named_pjit_cache(get_fun_names(fn), **my_pjit_args)

        return cached_pjitted_fun(dynamic_donated, dynamic_reserved, static)

    return f


def physical_axis_name(axis: Axis) -> Optional[PhysicalAxis]:
    """Get the physical axis name for a logical axis"""
    mapping = _mapping_holder.thread_data.resource_mapping
    if mapping is None:
        return None
    else:
        return mapping.get(axis.name, None)


def physical_axis_size(axis: Axis) -> Optional[int]:
    """Get the physical axis size for a logical axis. This is the product of the size of all physical axes
    that this logical axis is mapped to."""
    # TODO: shouldn't be accessing this internal api, but...
    from jax.experimental.maps import thread_resources

    try:
        mesh_shape = thread_resources.env.shape
    except AttributeError:
        raise ValueError("No resource mapping found")

    name: Union[None, str, Sequence[str]] = physical_axis_name(axis)
    if name is None:
        return None
    elif isinstance(name, str):
        name = (name,)

    return prod([mesh_shape[n] for n in name])


def pspec_for_axis(axis: AxisSpec) -> PartitionSpec:
    """Get the PartitionSpec for a single axis"""
    axis = ensure_tuple(axis)
    return PartitionSpec(*(physical_axis_name(a) for a in axis))


def round_axis_for_partitioning(axis: Axis) -> Axis:
    """Round an axis so that it's divisible by the size of the partition it's on"""
    size = physical_axis_size(axis)
    if size is None:
        return axis
    else:
        new_size = (axis.size + size - 1) // size * size
        return Axis(axis.name, new_size)


__all__ = [
    "LogicalAxisName",
    "PhysicalAxis",
    "PhysicalAxisSpec",
    "ResourceAxis",
    "ResourceMapping",
    "axis_mapping",
    "auto_sharded",
    "infer_resource_partitions",
    "eval_resource_partitions",
    "named_pjit",
    "physical_axis_name",
    "pspec_for_axis",
    "round_axis_for_partitioning",
]
