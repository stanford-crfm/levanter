import dataclasses
from functools import wraps
from typing import Any, Callable, Sequence, Tuple, TypeVar, Union

import equinox
import jax
import jax.lax as lax

from .core import Axis, NamedArray
from .partitioning import physical_axis_name
from .util import ensure_tuple, is_named_array


Carry = TypeVar("Carry")
X = TypeVar("X")
Y = TypeVar("Y")


def scan(f: Callable[[Carry, X], Tuple[Carry, Y]], axis: Axis, init: Carry, xs: X, reverse=False, unroll=1):
    """
    Scan over a named axis. Arrays that are not part of a named axis will have their 0th dim scanned over
    """

    # This implementation is a bit tricky.
    # First we have to hoist the axis we're scanning over to the front of the array, because that's what scan expects.
    # Then we have to scan over the 0th dim of the arrays (as flattened non-pytrees)
    # We have to be careful that we don't try to create NamedArrays that have the shape of the scanned result
    # but don't yet have the scanned axis as ones of `axes`, so we use _ScannedArrayResult that doesn't check
    # invariants until we're ready to create the result.

    axis_first_xs = jax.tree_util.tree_map(_ensure_first(axis), xs, is_leaf=is_named_array)

    # now get a template of an element of "X"
    x_elem = jax.tree_util.tree_map(_select_0th(axis), axis_first_xs, is_leaf=is_named_array)
    x_elem_structure = jax.tree_util.tree_structure(x_elem)

    # now we can fold over the axis
    @wraps(f)
    def wrapped_fn(carry, x):
        x = jax.tree_util.tree_unflatten(x_elem_structure, x)
        carry, y = f(carry, x)
        y = jax.tree_util.tree_map(_pacify_named_arrays, y, is_leaf=is_named_array)
        return carry, y

    leaves = jax.tree_util.tree_leaves(axis_first_xs)
    carry, ys = lax.scan(wrapped_fn, init, leaves, reverse=reverse, unroll=unroll)
    ys = jax.tree_util.tree_map(_to_active_named_array(axis), ys, is_leaf=_is_passive_array)

    return carry, ys


def reduce(
    fn: Callable[[Carry, X], Carry], axis: Axis, init: Carry, xs: X, reverse: bool = False, unroll: int = 1
) -> Carry:
    """
    Slightly simpler implementation of scan that folds over the first axis of the array, not returning intermediates.

    If reverse is True, the fold is actually a fold_right
    """
    if axis.size == 0:
        return init

    # This implementation is a bit tricky.
    # First we have to hoist the axis we're scanning over to the front of the array.
    # Then we have to scan over the 0th dim of the arrays (as flattened non-pytrees)
    # We have to be careful that we don't try to create NamedArrays that have the shape of the scanned result
    # but don't yet have the scanned axis as ones of `axes`, so we use _ScannedArrayResult that doesn't check
    # invariants until we're ready to create the result.

    axis_first_xs = jax.tree_util.tree_map(_ensure_first(axis), xs, is_leaf=is_named_array)

    # now get a template for where we fold over the axis in question
    x_elem = jax.tree_util.tree_map(_select_0th(axis), axis_first_xs, is_leaf=is_named_array)
    x_elem_structure = jax.tree_util.tree_structure(x_elem)

    # now we can fold over the axis
    @wraps(fn)
    def wrapped_fn(carry, x):
        x = jax.tree_util.tree_unflatten(x_elem_structure, x)
        return fn(carry, x), None

    leaves = jax.tree_util.tree_leaves(axis_first_xs)
    carry, _ = lax.scan(wrapped_fn, init, leaves, reverse=reverse, unroll=unroll)

    return carry


def vmap(
    fn,
    axis: Axis,
    unmapped_argnums: Union[int, Sequence[int]] = (),
):
    """
    NamedArray aware version of jax.vmap. Normal arrays are mapped over the 0th axis.
    unmapped_argnums are the argnums of the function that are not batched over the axis.
    """
    unmmapped_argnums = ensure_tuple(unmapped_argnums)

    def _index_of_batch_axis(array):
        if isinstance(array, NamedArray):
            return array.axes.index(axis)
        elif equinox.is_array(array):
            return 0
        else:
            return None

    # TODO: do fancier things with kwargs and signature and such
    # TODO: allow specifying spmd axis name and/or inferring it from resource_mapping
    # TODO: allow other axes to be mapped over
    # TODO: maybe implement equinox-style filtering vmap
    # TODO: tests to exercise this more
    @wraps(fn)
    def wrapped_vmap_fn(*args):
        # TODO: this probably results in a lot of compilation misses. Need to think about it.
        mapped_axes = []
        chilled_args = []

        for i, arg in enumerate(args):
            if i in unmmapped_argnums:
                mapped_axes.append(None)
            else:
                chilled_arg = jax.tree_util.tree_map(_pacify_named_arrays, arg, is_leaf=is_named_array)
                chilled_args.append(chilled_arg)
                mapped_axis = jax.tree_util.tree_map(_index_of_batch_axis, chilled_arg, is_leaf=_is_passive_array)
                mapped_axes.append(mapped_axis)

        def wrapped_fn(*args):
            unchilled_args = jax.tree_util.tree_map(_to_active_named_array(axis), args, is_leaf=is_named_array)
            r = fn(*unchilled_args)
            chilled = jax.tree_util.tree_map(_pacify_named_arrays, r, is_leaf=is_named_array)
            return chilled

        spmd_axis_name = physical_axis_name(axis)

        result = jax.vmap(
            wrapped_fn, in_axes=mapped_axes, out_axes=0, axis_size=axis.size, spmd_axis_name=spmd_axis_name
        )(*args)
        result = jax.tree_util.tree_map(_to_active_named_array(axis), result, is_leaf=_is_passive_array)
        return result

    return wrapped_vmap_fn


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class _PassiveNamedArray:
    """For some higher-order jax manipulations, jax will insert/remove an axis at the beginning of the array (or
    sometimes elsewhere). Jax doesn't know about our NamedArray wrapper, so we need a variant of NamedArray
    that doesn't care about the axis names. This is that variant.This class is a 'chill' version of NamedArray
    that doesn't check invariants until we're ready to create the result

    For example, with scan in NamedArray, we can't just have the scan tree prepend the scanned axis to the result,
    because we don't have a way to feed it the name of the scanned axis.
    """

    array: jax.numpy.ndarray
    main_axes: Tuple[Axis, ...]

    def as_scanned_result(self, scan_axis: Axis):
        return NamedArray(self.array, (scan_axis,) + self.main_axes)

    def tree_flatten(self) -> Any:
        return ((self.array,), self.main_axes)

    @classmethod
    def tree_unflatten(cls, aux, tree: Any) -> Any:
        assert len(tree) == 1
        return cls(tree[0], main_axes=aux)


def _is_passive_array(arr):
    return isinstance(arr, _PassiveNamedArray)


def _to_active_named_array(leading_axis):
    def to_active_named_array(leaf):
        if isinstance(leaf, _PassiveNamedArray):
            return leaf.as_scanned_result(leading_axis)
        else:
            return leaf

    return to_active_named_array


def _pacify_named_arrays(leaf):
    if isinstance(leaf, NamedArray):
        return _PassiveNamedArray(leaf.array, leaf.axes)
    elif isinstance(leaf, _PassiveNamedArray):
        assert False, "PassiveNamedArray should not be present in the tree"
    else:
        return leaf


def _select_0th(axis):
    def select_0th(leaf):
        if isinstance(leaf, NamedArray):
            return leaf.take(axis, 0)
        elif isinstance(leaf, _PassiveNamedArray):
            assert False, "PassiveNamedArray should not be present in the tree"
        else:
            # other leaves don't matter
            return leaf

    return select_0th


def _ensure_first(axis):
    def ensure_first(leaf):
        if isinstance(leaf, NamedArray):
            return leaf.rearrange((axis, ...))
        elif isinstance(leaf, _PassiveNamedArray):
            assert False, "PassiveNamedArray should not be present in the tree"
        else:
            return leaf

    return ensure_first


__all__ = ["scan", "reduce", "vmap"]
