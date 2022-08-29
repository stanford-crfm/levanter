import contextlib
import threading
from typing import List, Mapping, Optional, Sequence, TypeVar, Union

import jax
from equinox import is_array
from equinox.custom_types import PyTree
from jax.experimental import pjit
from jax.interpreters.pxla import PartitionSpec

from .core import NamedArray
from .util import ensure_tuple, is_named_array


LogicalAxisName = str
PhysicalAxis = str
PhysicalAxisSpec = Union[PhysicalAxis, Sequence[PhysicalAxis]]
ResourceMapping = Mapping[LogicalAxisName, PhysicalAxisSpec]
"""Mapping from logical axis names to physical axis names"""


class _ResourceMappingHolder:
    """Global resource mapping, used with a context manager to give dynamic scoping to resource mappings"""

    def __init__(self):
        self.thread_data = threading.local()
        self.thread_data.resource_mapping = None


_mapping_holder = _ResourceMappingHolder()


@contextlib.contextmanager
def resource_mapping(mapping: ResourceMapping, **kwargs):
    """Context manager for setting the global resource mapping"""
    if len(kwargs):
        mapping = dict(mapping)
        mapping.update(kwargs)

    old_mapping = _mapping_holder.thread_data.resource_mapping
    _mapping_holder.thread_data.resource_mapping = mapping
    yield
    _mapping_holder.thread_data.resource_mapping = old_mapping


T = TypeVar("T", bound=PyTree)


def logically_sharded(x: T, logical_axes: Optional[PyTree] = None) -> T:
    """
    Shard a PyTree using the global resource mapping. NamedArrays in the PyTree are sharded using the resource mapping
     and the names in the tree. Non-NamedArrays are sharded using the logical_axes argument, if provided.

    If there is no global resource mapping, this function is a no-op.
    """
    mapping = _mapping_holder.thread_data.resource_mapping

    if mapping is None:
        return x

    def _as_pspec(x, logical_axis=None):
        if isinstance(x, NamedArray):
            physical_names: List[Optional[PhysicalAxisSpec]] = [mapping.get(a.name, None) for a in x.axes]
        elif logical_axis is not None:
            physical_names: List[Optional[PhysicalAxisSpec]] = [
                mapping.get(a, None) for a in ensure_tuple(logical_axis)
            ]
        elif is_array(x):
            physical_names = [None] * len(x.shape)
        else:
            return None

        spec = PartitionSpec(
            *tuple(tuple(p) if not (isinstance(p, str)) and isinstance(p, Sequence) else p for p in physical_names)
        )
        return spec

    if logical_axes is None:
        # TODO: support logical_axes as a tree prefix. jax doesn't seem to have a good utility for this.
        pspec = jax.tree_util.tree_map(_as_pspec, x, is_leaf=is_named_array)
    else:
        pspec = jax.tree_util.tree_map(_as_pspec, x, logical_axes, is_leaf=is_named_array)

    print(x, pspec)
    return pjit.with_sharding_constraint(x, pspec)
