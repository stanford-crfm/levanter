import dataclasses
from typing import Any, Callable, Tuple, TypeVar

import jax
import jax.lax as lax

from .core import Axis, NamedArray


Carry = TypeVar("Carry")
X = TypeVar("X")
Y = TypeVar("Y")


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class _ScannedArrayResult:
    """With scan in NamedArray, we can't just have the scan tree prepend the scanned axis to the result, because
    we don't have a way to feed it the name of the scanned axis. This class is a 'chill' version of NamedArray
    that doesn't check invariants until we're ready to create the result
    """

    array: jax.numpy.ndarray
    main_axes: Tuple[Axis, ...]

    def to_named_array(self, scan_axis: Axis):
        return NamedArray(self.array, (scan_axis,) + self.main_axes)

    def tree_flatten(self) -> Any:
        return ((self.array,), self.main_axes)

    @classmethod
    def tree_unflatten(cls, aux, tree: Any) -> Any:
        assert len(tree) == 1
        return cls(tree[0], main_axes=aux)


def scan(f: Callable[[Carry, X], Tuple[Carry, Y]], axis: Axis, init: Carry, xs: X, reverse=False, unroll=1):
    """
    Scan over a named axis. Arrays that are not part of a named axis will have their 0th dim scanned over
    """
    if axis.size == 0:
        return init, xs

    # This implementation is a bit tricky.
    # First we have to hoist the axis we're scanning over to the front of the array.
    # Then we have to scan over the 0th dim of the arrays (as flattened non-pytrees)
    # We have to be careful that we don't try to create NamedArrays that have the shape of the scanned result
    # but don't yet have the scanned axis as ones of `axes`, so we use _ScannedArrayResult that doesn't check
    # invariants until we're ready to create the result.

    def ensure_first(leaf):
        if isinstance(leaf, NamedArray):
            return leaf.rearrange((axis, ...))
        else:
            return leaf

    def named_array_leaf(leaf):
        return isinstance(leaf, NamedArray)

    axis_first_xs = jax.tree_util.tree_map(ensure_first, xs, is_leaf=named_array_leaf)

    # now get a template for where we fold over the axis in question
    def select_0th(leaf):
        if isinstance(leaf, NamedArray):
            return leaf.take(axis, 0)
        else:
            # other leaves don't matter
            return leaf

    x_elem = jax.tree_util.tree_map(select_0th, axis_first_xs, is_leaf=named_array_leaf)
    x_elem_structure = jax.tree_util.tree_structure(x_elem)

    def chill_named_arrays(leaf):
        if isinstance(leaf, NamedArray):
            return _ScannedArrayResult(leaf.array, leaf.axes)
        else:
            return leaf

    # now we can fold over the axis
    def wrapped_fn(carry, x):
        x = jax.tree_util.tree_unflatten(x_elem_structure, x)
        carry, y = f(carry, x)
        y = jax.tree_util.tree_map(chill_named_arrays, y, is_leaf=named_array_leaf)
        return carry, y

    leaves = jax.tree_util.tree_leaves(axis_first_xs)
    carry, ys = lax.scan(wrapped_fn, init, leaves, reverse=reverse, unroll=unroll)

    def unchill_named_arrays(leaf):
        if isinstance(leaf, _ScannedArrayResult):
            return leaf.to_named_array(axis)
        else:
            return leaf

    ys = jax.tree_util.tree_map(unchill_named_arrays, ys, is_leaf=lambda arr: isinstance(arr, _ScannedArrayResult))

    return carry, ys


__all__ = ["scan"]
