import functools
import typing
from typing import Dict

import jax
from equinox.custom_types import PyTree
from jax.experimental.pjit import pjit
from jax.interpreters.pxla import PartitionSpec

from haliax import NamedArray
from levanter.python_utils import StringHolderEnum


# Predefined axis names
# in compliance with https://github.com/google-research/t5x/blob/main/t5x/partitioning.py
class ResourceAxis(StringHolderEnum):
    MODEL = "model"
    DATA = "data"


class LogicalAxis(StringHolderEnum):
    BATCH = "batch"
    PARAMS = "params"


T = typing.TypeVar('T')


def infer_resource_partitions(tree: PyTree, axis_resources: Dict[str, str]) -> PyTree:
    """
    Infer the resource partitions for a module, to be used with pjit.
    The basic idea is to tree all NamedArrays as leaves for the purposes of this function,
    and to create PartitionSpecs from those names plus axis_resources
    """

    def named_array_is_leaf(x):
        return isinstance(x, NamedArray)

    def partition_spec(node: typing.Any):
        if isinstance(node, NamedArray):
            return PartitionSpec(*tuple(axis_resources.get(axis.name, None) for axis in node.axes))
        else:
            return None

    return jax.tree_map(partition_spec, tree, is_leaf=named_array_is_leaf)


def named_pjit_init(cls: typing.Type[T], axis_resources, **pjit_args):
    """Uses NamedArrays to infer the resource partitions for a module when creating it """
    @functools.wraps(cls.__new__)
    def init(*args, **kwargs):
        inst = cls(*args, **kwargs)
        return inst

    # two passes, one using eval_shape and another that actually makes the class
    @functools.wraps(cls.__new__)
    def make(*args, **kwargs):
        shapes = jax.eval_shape(init, *args, **kwargs)
        out_resources = infer_resource_partitions(shapes, axis_resources)
        in_resources = infer_resource_partitions((args, kwargs), axis_resources)

        fn = pjit(lambda args, kwargs: init(*args, **kwargs), in_resources, out_resources, **pjit_args)
        return fn(args, kwargs)

    return make


__all__ = ["LogicalAxis", "ResourceAxis", "infer_resource_partitions", "named_pjit_init"]
