import contextlib
import functools as ft
import typing
import warnings
from dataclasses import dataclass
from math import prod
from types import EllipsisType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, cast, overload

import jax
import jax.numpy as jnp
import numpy as np

import haliax
from haliax.jax_utils import is_jax_array_like
from haliax.util import ensure_tuple, index_where, py_slice, slice_t

from .types import Axis, AxisSelection, AxisSelector, AxisSpec, PrecisionLike, Scalar


NamedOrNumeric = Union[Scalar, "NamedArray"]

_ENABLE_SHAPE_CHECKS = True


@contextlib.contextmanager
def enable_shape_checks(enabled):
    """
    Sometimes we end up in situations where an array that jax makes is passed into the NamedArray constructor that
    doesn't conform to the shape we expect. This shows up in particular when we are using jax.vmap or jax.scan,
    and we sometimes have weird situations with deserialization

    Yields the old value because we sometimes want to nest this
    """
    global _ENABLE_SHAPE_CHECKS
    old = _ENABLE_SHAPE_CHECKS
    _ENABLE_SHAPE_CHECKS = enabled
    try:
        yield old
    finally:
        _ENABLE_SHAPE_CHECKS = old


def are_shape_checks_enabled():
    return _ENABLE_SHAPE_CHECKS


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class NamedArray:
    array: jnp.ndarray
    axes: Tuple[Axis, ...]

    def __post_init__(self):
        if not isinstance(self.axes, tuple):
            object.__setattr__(self, "axes", tuple(self.axes))
        # ensure axes are all Axis objects
        # TODO: anonymous positional axes?
        for axis in self.axes:
            if not isinstance(axis, Axis):
                raise TypeError(f"Expected Axis, got {type(axis)}")

        # ensure unique axes for now
        if len(set(a.name for a in self.axes)) != len(self.axes):
            raise ValueError(f"Axes must be unique, but {self.axes} are not")

        if are_shape_checks_enabled():
            self._ensure_shape_matches_axes()

    def _ensure_shape_matches_axes(self):
        """This is typically called automatically, but sometimes we need to call it manually if
        are_shape_checks_enabled() is False"""
        if is_jax_array_like(self.array):
            s = jnp.shape(self.array)
            if s != tuple(a.size for a in self.axes):
                raise ValueError(f"Shape of underlying array {s} does not match shape of axes {self.axes}")

    def item(self):
        return self.array.item()

    def scalar(self) -> jnp.ndarray:
        """
        Returns a scalar array corresponding to the value of this NamedArray.
        Raises an error if the NamedArray is not scalar.

        We sometimes use this to convert a NamedArray to a scalar for returning a loss or similar. Losses
        have to be jnp.ndarrays, not NamedArrays, so we need to convert them. item doesn't work inside jitted
        functions because it returns a python scalar.

        You could just call array, but that's not as clear and doesn't assert.
        """
        if self.array.ndim != 0:
            raise ValueError(f"Expected scalar, got {self.array.ndim}-dimensional array")
        return self.array

    # shape = property(lambda self: self.array.shape)
    dtype = property(lambda self: self.array.dtype)
    ndim = property(lambda self: self.array.ndim)
    size = property(lambda self: self.array.size)
    nbytes = property(lambda self: self.array.nbytes)

    def tree_flatten(self) -> Any:
        return ((self.array,), self.axes)

    @classmethod
    def tree_unflatten(cls, aux, tree: Any) -> Any:
        assert len(tree) == 1
        return cls(tree[0], axes=aux)

    def has_axis(self, axis: AxisSelection) -> bool:
        return self._lookup_indices(axis) is not None

    @overload
    def axis_size(self, axis: AxisSelector) -> int:  # type: ignore
        ...

    @overload
    def axis_size(self, axis: Sequence[AxisSelector]) -> Tuple[int, ...]:  # type: ignore
        ...

    def axis_size(self, axis: AxisSelection) -> Union[int, Tuple[int, ...]]:
        """
        Returns the size of the given axis, or a tuple of sizes if given multiple axes.
        """
        indices = self._lookup_indices(axis)
        if isinstance(indices, int):
            return self.axes[indices].size
        elif indices is None:
            raise ValueError(f"Axis {axis} not found")
        else:
            result = []
            for i in indices:
                if i is None:
                    raise ValueError(f"Axis {axis} not found")
                result.append(self.axes[i].size)
            return tuple(result)

    @overload
    def resolve_axis(self, axis: AxisSelector) -> Axis:  # type: ignore
        ...

    @overload
    def resolve_axis(self, axis: Sequence[AxisSelector]) -> Tuple[Axis, ...]:  # type: ignore
        ...

    def resolve_axis(self, axes: AxisSelection) -> AxisSpec:  # type: ignore
        """Returns the axes corresponding to the given axis selection."""
        indices = self._lookup_indices(axes)
        if isinstance(indices, int):
            return self.axes[indices]
        elif indices is None:
            raise ValueError(f"Axis {axes} not found")
        else:
            result = []
            for i in indices:
                if i is None:
                    raise ValueError(f"Axis {axes} not found")
                result.append(self.axes[i])
            return tuple(result)

    @overload
    def _lookup_indices(self, axis: AxisSelector) -> Optional[int]:  # type: ignore
        ...

    @overload
    def _lookup_indices(self, axis: Sequence[AxisSelector]) -> Tuple[Optional[int], ...]:
        ...

    def _lookup_indices(self, axis: AxisSelection) -> Union[Optional[int], Tuple[Optional[int], ...]]:
        """
        For a single axis, returns an int corresponding to the index of the axis.
        For multiple axes, returns a tuple of ints corresponding to the indices of the axes.

        If the axis is not present, returns None for that position
        """
        if isinstance(axis, Axis):
            ax_name = axis.name
            try:
                return self.axes.index(axis)
            except ValueError:
                try:
                    axis_index = index_where(lambda a: a.name == ax_name, self.axes)
                    if axis_index >= 0:
                        warnings.warn("Found axis with same name but different size.", UserWarning)
                    return axis_index
                except ValueError:
                    return None
        elif isinstance(axis, str):
            try:
                return index_where(lambda a: a.name == axis, self.axes)
            except ValueError:
                return None
        elif isinstance(axis, str):
            try:
                return index_where(lambda a: a.name == axis, self.axes)
            except ValueError:
                return None
        else:
            return tuple(self._lookup_indices(a) for a in axis)

    # Axis rearrangement
    def rearrange(self, axis: Sequence[Union[AxisSelector, EllipsisType]]) -> "NamedArray":
        return rearrange(self, axis)

    def broadcast_to(self, axes: AxisSpec) -> "NamedArray":
        axes = ensure_tuple(axes)
        return haliax.broadcast_to(self, axes=axes)

    def broadcast_axis(self, axis: AxisSpec) -> "NamedArray":
        return haliax.broadcast_axis(self, axis=axis)

    def split(self, axis: AxisSelector, new_axes: Sequence[Axis]) -> Sequence["NamedArray"]:
        return haliax.split(self, axis=axis, new_axes=new_axes)

    def flatten_axes(self, old_axes: AxisSelection, new_axis: AxisSelector) -> "NamedArray":
        return haliax.flatten_axes(self, old_axes=old_axes, new_axis=new_axis)

    def unflatten_axis(self, axis: AxisSelector, new_axes: AxisSpec) -> "NamedArray":
        return haliax.unflatten_axis(self, axis=axis, new_axes=new_axes)

    def unbind(self, axis: AxisSelector) -> Sequence["NamedArray"]:
        return haliax.unbind(self, axis=axis)

    def rename(self, renames: Mapping[AxisSelector, AxisSelector]) -> "NamedArray":
        return haliax.rename(self, renames=renames)

    # slicing

    # TOOD: AxisSelector-ify new_axis
    def slice(self, axis: AxisSelector, new_axis: Axis, start: int = 0) -> "NamedArray":
        return haliax.slice(self, axis=axis, new_axis=new_axis, start=start)

    def take(self, axis: AxisSelector, index: Union[int, "NamedArray"]) -> "NamedArray":
        return haliax.take(self, axis=axis, index=index)

    @overload
    def __getitem__(self, item: Tuple[str, Union[int, slice_t, "NamedArray"]]) -> Union["NamedArray", jnp.ndarray]:
        ...  # pragma: no cover

    @overload
    def __getitem__(
        self, item: Tuple[str, Union[int, slice_t, "NamedArray"], str, Union[int, slice_t, "NamedArray"]]
    ) -> Union["NamedArray", jnp.ndarray]:
        ...

    @overload
    def __getitem__(self, item: Mapping[str, Union[int, slice_t, "NamedArray"]]) -> Union["NamedArray", jnp.ndarray]:
        ...

    def __getitem__(self, idx) -> Union["NamedArray", jnp.ndarray]:
        """Syntactic sugar for slice_nd, which is the actual implementation.

        Supports indexing like:

        >>> X = Axis("x", 10)
        >>> Y = Axis("y", 20)
        >>> arr = haliax.random.randint(jax.random.PRNGKey(0), (X, Y), 0, X.size)
        # slice with ints or slices
        >>> arr[{"x": 1, "y": slice(0, 10, 2)}]
        >>> Z = Axis("z", 3)
        # so-called "advanced indexing" with NamedArrays.
        >>> index_arr = NamedArray(np.array([1, 2, 3]), Z)
        >>> arr[{"x": 1, "y": index_arr}]

        A shorthand is provided that works with Python's slicing syntax:
        >>> arr["x", :] == arr[{"x": slice(None, None, None)}]
        >>> arr["y", slice(0, 10, 2)] == arr[{"y": slice(0, 10, 2)}]

        Advanced indexing is implemented by broadcasting all index arrays to the same shape (using Haliax's
        usual broadcasting rules).

        This returns a NamedArray if any axes remain, or a scalar (0-dimensional) jnp.ndarray if all axes are indexed out.
        """
        if isinstance(idx, tuple):
            if len(idx) == 1:
                idx = idx[0]
            else:
                if len(idx) % 2 != 0:
                    raise ValueError(
                        "Must provide an even number of arguments to __getitem__ when using the shorthand syntax."
                    )
                idx = {idx[i]: idx[i + 1] for i in range(0, len(idx), 2)}

        return slice_nd(self, idx)

    # np.ndarray methods:
    def all(self, axis: Optional[AxisSelection] = None) -> "NamedArray":
        return haliax.all(self, axis=axis)

    def any(self, axis: Optional[AxisSelection] = None) -> "NamedArray":
        return haliax.any(self, axis=axis)

    # TODO: test
    def argmax(self, axis: AxisSelector) -> "NamedArray":
        return haliax.argmax(self, axis=axis)

    def argmin(self, axis: AxisSelector) -> "NamedArray":
        return haliax.argmin(self, axis=axis)

    def argsort(self, axis: AxisSelector) -> "NamedArray":
        return haliax.argsort(self, axis=axis)

    def astype(self, dtype) -> "NamedArray":
        return NamedArray(self.array.astype(dtype), self.axes)

    def clip(self, a_min=None, a_max=None) -> Any:
        return haliax.clip(self, a_min=a_min, a_max=a_max)

    # TODO
    # def compress(self, condition, axis: Optional[int] = None) -> Any:
    #     ...

    def conj(self) -> "NamedArray":
        return NamedArray(self.array.conj(), self.axes)

    def conjugate(self) -> "NamedArray":
        return NamedArray(self.array.conjugate(), self.axes)

    def copy(self) -> "NamedArray":
        return NamedArray(self.array.copy(), self.axes)

    def cumprod(self, axis: Optional[AxisSelection] = None, *, dtype=None) -> "NamedArray":
        return haliax.cumprod(self, axis=axis, dtype=dtype)

    def cumsum(self, axis: Optional[AxisSelection] = None, *, dtype=None) -> "NamedArray":
        return haliax.cumsum(self, axis=axis, dtype=dtype)

    def dot(self, axis: AxisSelection, b, *, precision: PrecisionLike = None) -> "NamedArray":
        return dot(axis, self, b, precision=precision)

    @property
    def imag(self) -> "NamedArray":
        return NamedArray(self.array.imag, self.axes)

    def max(
        self,
        axis: Optional[AxisSelection] = None,
        *,
        initial=None,
        where=None,
    ) -> "NamedArray":
        return haliax.max(self, axis=axis, initial=initial, where=where)

    def mean(
        self,
        axis: Optional[AxisSelection] = None,
        *,
        dtype=None,
        where=None,
    ) -> "NamedArray":
        return haliax.mean(self, axis=axis, dtype=dtype, where=where)

    def min(
        self,
        axis: Optional[AxisSelection] = None,
        *,
        initial=None,
        where=None,
    ) -> "NamedArray":
        return haliax.min(self, axis=axis, initial=initial, where=where)

    def prod(
        self,
        axis: Optional[AxisSelection] = None,
        *,
        dtype=None,
        initial=None,
        where=None,
    ) -> "NamedArray":
        return haliax.prod(
            self,
            axis=axis,
            dtype=dtype,
            initial=initial,
            where=where,
        )

    def ptp(self, axis: AxisSelection = None) -> "NamedArray":
        return haliax.ptp(self, axis=axis)

    # TODO: implement ravel. Can only do if we either ask for an axis or add ProductAxis or something
    # def ravel(self, order='C') -> Any:
    #     ...

    @property
    def real(self) -> "NamedArray":
        return NamedArray(self.array.real, self.axes)

    # TODO: what should reshape look like?
    # def reshape(self, *args, order='C') -> Any:
    #     ...

    def round(self, decimals=0) -> "NamedArray":
        return haliax.round(self, decimals=decimals)

    def sort(self, axis: AxisSelector, kind="quicksort") -> Any:
        return haliax.sort(self, axis=axis, kind=kind)

    def std(self, axis: Optional[AxisSelection] = None, *, dtype=None, ddof=0, where=None) -> "NamedArray":
        return haliax.std(self, axis=axis, dtype=dtype, ddof=ddof, where=where)

    def sum(
        self,
        axis: Optional[AxisSelection] = None,
        *,
        dtype=None,
        initial=None,
        where=None,
    ) -> "NamedArray":
        return haliax.sum(
            self,
            axis=axis,
            dtype=dtype,
            initial=initial,
            where=where,
        )

    def tobytes(self, order="C") -> Any:
        return self.array.tobytes(order=order)

    def tolist(self) -> Any:
        return self.array.tolist()

    def trace(self, axis1: AxisSelector, axis2: AxisSelector, offset=0, dtype=None) -> "NamedArray":
        return haliax.trace(self, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype)

    def var(
        self,
        axis: Optional[AxisSelection] = None,
        dtype=None,
        ddof=0,
        *,
        where=None,
    ) -> "NamedArray":
        return haliax.var(
            self,
            axis=axis,
            dtype=dtype,
            ddof=ddof,
            where=where,
        )

    # operators

    # Comparisons
    def __lt__(self, other) -> "NamedArray":
        return haliax.less(self, other)

    def __le__(self, other) -> "NamedArray":
        return haliax.less_equal(self, other)

    def __eq__(self, other):
        return haliax.equal(self, other)

    def __ne__(self, other):
        return haliax.not_equal(self, other)

    def __gt__(self, other) -> "NamedArray":
        return haliax.greater(self, other)

    def __ge__(self, other) -> "NamedArray":
        return haliax.greater_equal(self, other)

    # Unary arithmetic

    def __neg__(self) -> "NamedArray":
        return haliax.negative(self)

    def __pos__(self) -> "NamedArray":
        return haliax.positive(self)

    def __abs__(self) -> "NamedArray":
        return haliax.absolute(self)

    def __invert__(self) -> "NamedArray":
        return haliax.invert(self)

    # Binary arithmetic

    def __add__(self, other) -> "NamedArray":
        return haliax.add(self, other)

    def __sub__(self, other) -> "NamedArray":
        return haliax.subtract(self, other)

    def __mul__(self, other) -> "NamedArray":
        return haliax.multiply(self, other)

    def __matmul__(self, other) -> "NamedArray":
        raise ValueError("matmul is too ambiguous with NamedArrays. Use dot instead.")

    def __truediv__(self, other) -> "NamedArray":
        return haliax.true_divide(self, other)

    def __floordiv__(self, other) -> "NamedArray":
        return haliax.floor_divide(self, other)

    def __mod__(self, other) -> "NamedArray":
        return haliax.mod(self, other)

    def __divmod__(self, other) -> "NamedArray":
        return haliax.divmod(self, other)

    def __pow__(self, other) -> "NamedArray":
        return haliax.power(self, other)

    def __lshift__(self, other) -> "NamedArray":
        return haliax.left_shift(self, other)

    def __rshift__(self, other) -> "NamedArray":
        return haliax.right_shift(self, other)

    def __and__(self, other) -> "NamedArray":
        return haliax.bitwise_and(self, other)

    def __xor__(self, other) -> "NamedArray":
        return haliax.bitwise_xor(self, other)

    def __or__(self, other) -> "NamedArray":
        return haliax.bitwise_or(self, other)

    def __radd__(self, other) -> "NamedArray":
        return haliax.add(other, self)

    def __rsub__(self, other) -> "NamedArray":
        return haliax.subtract(other, self)

    def __rmul__(self, other) -> "NamedArray":
        return haliax.multiply(other, self)

    def __rmatmul__(self, other):
        raise ValueError("Matrix multiplication is too ambiguous with NamedArrays. Use dot instead.")

    def __rtruediv__(self, other) -> "NamedArray":
        return haliax.true_divide(other, self)

    def __rfloordiv__(self, other) -> "NamedArray":
        return haliax.floor_divide(other, self)

    def __rmod__(self, other) -> "NamedArray":
        return haliax.mod(other, self)

    def __rdivmod__(self, other) -> "NamedArray":
        return haliax.divmod(other, self)

    def __rpow__(self, other) -> "NamedArray":
        return haliax.power(other, self)

    def __rlshift__(self, other) -> "NamedArray":
        return haliax.left_shift(other, self)

    def __rrshift__(self, other) -> "NamedArray":
        return haliax.right_shift(other, self)

    def __rand__(self, other) -> "NamedArray":
        return haliax.bitwise_and(other, self)

    def __rxor__(self, other) -> "NamedArray":
        return haliax.bitwise_xor(other, self)

    def __ror__(self, other) -> "NamedArray":
        return haliax.bitwise_or(other, self)

    def __bool__(self) -> bool:
        return bool(self.array)

    def __complex__(self) -> complex:
        return complex(self.array)

    def __int__(self) -> int:
        return int(self.array)

    def __float__(self) -> float:
        return float(self.array)


def take(array: NamedArray, axis: AxisSelector, index: Union[int, NamedArray]) -> NamedArray:
    """
    Selects elements from an array along an axis, by an index or by another named array

    if index is a NamedArray, then those axes are added to the output array
    """
    axis_index = array._lookup_indices(axis)
    if axis_index is None:
        raise ValueError(f"axis {axis} not found in {array}")
    if isinstance(index, int):
        # just drop the axis
        new_array = jnp.take(array.array, index, axis=axis_index)
        new_axes = array.axes[:axis_index] + array.axes[axis_index + 1 :]
    else:
        new_array = jnp.take(array.array, index.array, axis=axis_index)
        new_axes = array.axes[:axis_index] + index.axes + array.axes[axis_index + 1 :]
    # new axes come from splicing the old axis with
    return NamedArray(new_array, new_axes)


def slice(array: NamedArray, axis: AxisSelector, new_axis: Axis, start: int = 0) -> NamedArray:
    """
    Selects elements from an array along an axis, by an index or by another named array
    This is somewhat better than take if you want a contiguous slice of an array
    """
    axis_index = array._lookup_indices(axis)
    if axis_index is None:
        raise ValueError(f"axis {axis} not found in {array}")

    sliced = jax.lax.dynamic_slice_in_dim(array.array, start, new_axis.size, axis=axis_index)
    new_axes = array.axes[:axis_index] + (new_axis,) + array.axes[axis_index + 1 :]
    # new axes come from splicing the old axis with
    return NamedArray(sliced, new_axes)


def slice_nd(
    array: NamedArray, slices: Mapping[AxisSelector, Union[int, slice_t, NamedArray]]
) -> Union[NamedArray, jnp.ndarray]:
    """
    Selects elements from an array along an axis, by an index or by another named array.
    Typically, you would call this via `array[...]` syntax. For example, you might call
    `array[{"batch": slice(0, 10)}]` to select the first 10 elements of the batch axis.
    :param array:
    :param slices:
    :return: a scalar jnp.ndarray is all axes are sliced with ints, otherwise a NamedArray
    """
    # indices where we have array args
    array_slice_indices = []
    ordered_slices: list = [py_slice(None, None, None)] * len(array.axes)  # type: ignore
    kept_axes = [True] * len(array.axes)
    for axis, slice_ in slices.items():
        axis_index = array._lookup_indices(axis)
        if axis_index is None:
            raise ValueError(f"axis {axis} not found in {array}")
        ordered_slices[axis_index] = slice_
        kept_axes[axis_index] = isinstance(slice_, py_slice)
        if isinstance(slice_, NamedArray):
            array_slice_indices.append(axis_index)

    # advanced indexing
    if len(array_slice_indices) > 0:
        # this requires broadcasting
        broadcasted_arrays, broadcasted_axes = broadcast_arrays_and_return_axes(
            *[ordered_slices[i] for i in array_slice_indices], require_subset=False, ensure_order=True
        )
        # this is tricky. NumPy distinguishes two cases when mixing advanced and basic indexing:
        # https://numpy.org/doc/stable/user/basics.indexing.html#combining-advanced-and-basic-indexing
        # The first is when the advanced indices are all contiguous, and the second is when they are not.
        # (NB that integers count as advanced indices, so this is a bit more complicated than it seems.)
        # When contiguous, the new axes go in the same place as the advanced indices, and the old axes surround them.
        # When not contiguous, the new axes go to the *front* of the array, and the (other) old axes go after them.
        # To tell what case we're in, we check if the advanced indices are contiguous. We can figure out by looking
        # at the "kept_axes": the Falses are the advanced indices.

        # check to make sure we're not accidentally duplicating axes
        for axis_index in range(len(array.axes)):
            if kept_axes[axis_index]:
                if selects_axis(broadcasted_axes, array.axes[axis_index].name):
                    raise ValueError(f"Array Axis {array.axes[axis_index]} is present in slice {slices}")

        for axis_index, selector_array in zip(array_slice_indices, broadcasted_arrays):
            ordered_slices[axis_index] = selector_array.array

        is_advanced_contiguous = True
        first_advanced_index = index_where(lambda x: not x, kept_axes)
        last_advanced_index = first_advanced_index
        true_found = False
        for i in range(first_advanced_index, len(kept_axes)):
            # now find the first True. If any False comes after it, we're not contiguous
            if true_found:
                if not kept_axes[i]:
                    is_advanced_contiguous = False
                    break
            elif kept_axes[i]:
                true_found = True
                last_advanced_index = i - 1

        if not true_found:
            last_advanced_index = len(kept_axes) - 1

        if is_advanced_contiguous:
            # the advanced indices are contiguous, so we can just insert the new axes in the same place
            # as the advanced indices
            new_axes = array.axes[:first_advanced_index] + broadcasted_axes + array.axes[last_advanced_index + 1 :]
        else:
            # the advanced indices are not contiguous, so we need to insert the new axes at the front
            new_axes = broadcasted_axes + tuple(ax for i, ax in enumerate(array.axes) if kept_axes[i])
    else:
        new_axes = tuple(axis.name for axis, keep in zip(array.axes, kept_axes) if keep)

    sliced = array.array[tuple(ordered_slices)]

    if len(new_axes) == 0:
        # this is a scalar
        return sliced

    return haliax.named(sliced, new_axes)


def dot(axis: AxisSelection, *arrays: NamedArray, precision: PrecisionLike = None) -> NamedArray:
    """Returns the tensor product of two NamedArrays. The axes `axis` are contracted over,
    and any other axes that are shared between the arrays are batched over. Non-contracted Axes in one
    that are not in the other are preserved.
    """
    axis = ensure_tuple(axis)

    array_specs = []

    next_index = 0
    axis_mappings: Dict[Axis, int] = {}

    for a in arrays:
        spec = ""
        for ax in a.axes:
            if ax in axis_mappings:
                spec += f"{axis_mappings[ax]} "
            else:
                axis_mappings[ax] = next_index
                spec += f"{next_index} "
                next_index += 1

        array_specs.append(spec)

    # now compute the output axes:
    output_axes = tuple(ax for ax in axis_mappings.keys() if not selects_axis(ax, axis))
    output_spec = " ".join(str(axis_mappings[ax]) for ax in output_axes)

    output = jnp.einsum(
        ", ".join(array_specs) + "-> " + output_spec,
        *[a.array for a in arrays],
        precision=precision,
    )

    return NamedArray(output, output_axes)


def split(a: NamedArray, axis: AxisSelector, new_axes: Sequence[Axis]) -> Sequence[NamedArray]:
    # check the lengths of the new axes
    index = a._lookup_indices(axis)
    if index is None:
        raise ValueError(f"Axis {axis} not found in {a.axes}")

    total_len = sum(x.size for x in new_axes)
    if isinstance(axis, Axis):
        if total_len != axis.size:
            raise ValueError(
                f"The total length of the new axes {total_len} does not match the length of the axis {axis}"
            )

    # now we can split the array
    offsets = np.cumsum([0] + [x.size for x in new_axes])[1:-1]

    new_arrays = np.split(a.array, indices_or_sections=offsets, axis=index)
    ret_axes = [tuple(ax2 if not selects_axis(axis, ax2) else new_axis for ax2 in a.axes) for new_axis in new_axes]

    return [NamedArray(x, ax) for x, ax in zip(new_arrays, ret_axes)]


# TODO: can we add einops-style combined split/merge here?
# e.g. we'd like something like rearrange(array, (..., new_axis), merge_axes={new_axis: (old_axis1, old_axis2)})
# or rearrange(array, (new_axis1, ..., new_axis2), split_axes={old_axis: (new_axis1, new_axis2)})
# or even rearrange(array, (x, ..., b, a), map_axes={old_axis: (a, b), x: (old1, old2)})
def rearrange(array: NamedArray, axes: Sequence[Union[AxisSelector, EllipsisType]]) -> NamedArray:
    """
    Rearrange an array so that its underlying storage conforms to axes.
    axes may include up to 1 ellipsis, indicating that the remaining axes should be
    permuted in the same order as the array's axes.
    """
    if len(axes) == 0 and len(array.axes) != 0:
        raise ValueError("No axes specified")

    # various fast paths
    if len(axes) == 1 and axes[0] is Ellipsis:
        return array

    if axes == array.axes:
        return array

    if axes[-1] is Ellipsis and array.axes[0 : len(axes) - 1] == axes[0 : len(axes) - 1]:
        return array

    if axes[0] is Ellipsis and array.axes[len(axes) - 1 :] == axes[1:]:
        return array

    if axes.count(Ellipsis) > 1:
        raise ValueError("Only one ellipsis allowed")

    used_indices = [False] * len(array.axes)
    permute_spec: List[Union[int, EllipsisType]] = []
    ellipsis_pos = None
    for ax in axes:
        if ax is Ellipsis:
            permute_spec.append(Ellipsis)  # will revisit
            ellipsis_pos = len(permute_spec) - 1
        else:
            assert isinstance(ax, Axis) or isinstance(ax, str)  # please mypy
            index = array._lookup_indices(ax)
            if index is None:
                raise ValueError(f"Axis {ax} not found in {array}")
            if used_indices[index]:
                raise ValueError(f"Axis {ax} specified more than once")
            used_indices[index] = True
            permute_spec.append(index)

    if not all(used_indices):
        # find the ellipsis position, replace it with all the unused indices
        if ellipsis_pos is None:
            missing_axes = [ax for i, ax in enumerate(array.axes) if not used_indices[i]]
            raise ValueError(f"Axes {missing_axes} not found and no ... specified. Array axes: {array.axes}") from None

        permute_spec[ellipsis_pos : ellipsis_pos + 1] = tuple(i for i in range(len(array.axes)) if not used_indices[i])
    elif ellipsis_pos is not None:
        permute_spec.remove(Ellipsis)

    out_axes = tuple(array.axes[i] for i in cast(List[int], permute_spec))
    return NamedArray(jnp.transpose(array.array, permute_spec), out_axes)


def unbind(array: NamedArray, axis: AxisSelector) -> List[NamedArray]:
    """
    Unbind an array along an axis, returning a list of NamedArrays. Analogous to torch.unbind or np.rollaxis
    """
    axis_index = array._lookup_indices(axis)
    if axis_index is None:
        raise ValueError(f"axis {axis} not found in {array}")
    new_axes = array.axes[:axis_index] + array.axes[axis_index + 1 :]
    # this implementation maybe triggers an all-gather in pjit so no good
    # arrays = jnp.rollaxis(array.array, axis=axis_index, start=0)
    # instead we just loop over the axes pulling one out at a time
    axis_size = array.axes[axis_index].size
    arrays = [jnp.take(array.array, i, axis=axis_index) for i in range(axis_size)]
    from haliax.partitioning import auto_sharded

    return [auto_sharded(NamedArray(a, new_axes)) for a in arrays]


def roll(array: NamedArray, shift: Union[int, Tuple[int, ...]], axis: AxisSelection) -> NamedArray:
    """
    Roll an array along an axis or axes. Analogous to np.roll
    """
    axis_indices = array._lookup_indices(axis)
    if axis_indices is None:
        raise ValueError(f"axis {axis} not found in {array}")
    return NamedArray(jnp.roll(array.array, shift, axis_indices), array.axes)


def rename(array: NamedArray, renames: Mapping[AxisSelector, AxisSelector]) -> NamedArray:
    for old, new in renames.items():
        if isinstance(old, Axis) and isinstance(new, Axis) and old.size != new.size:
            raise ValueError(f"Cannot rename axis {old} to {new}: size mismatch")

    def _rename(ax: AxisSelector) -> Axis:
        new_axis = renames.get(ax, None)
        if new_axis is None and isinstance(ax, Axis):
            new_axis_name = renames.get(ax.name, None)
            if isinstance(new_axis_name, str):
                new_axis = Axis(new_axis_name, ax.size)
                return new_axis
            elif isinstance(new_axis_name, Axis):
                if new_axis_name.size != ax.size:
                    raise ValueError(f"Cannot rename axis {ax} to {new_axis_name}: size mismatch")
                return new_axis_name
            else:
                return ax
        elif isinstance(new_axis, Axis):
            return new_axis
        else:
            assert isinstance(new_axis, str)
            ax_size = array.axis_size(ax)
            return Axis(new_axis, ax_size)

    new_axes = tuple(_rename(ax) for ax in array.axes)
    return NamedArray(array.array, new_axes)


def flatten_axes(array: NamedArray, old_axes: AxisSelection, new_axis: AxisSelector) -> NamedArray:
    """
    Merge a sequence of axes into a single axis. The new axis must have the same size as the product of the old axes.
    For now the new axis will always be the last axis
    """
    old_axes = ensure_tuple(old_axes)
    old_axes = array.resolve_axis(old_axes)

    if len(old_axes) == 0:
        raise ValueError("Must specify at least one axis to merge")

    if isinstance(new_axis, Axis):
        if new_axis.size != prod(array.axis_size(ax) for ax in old_axes):
            raise ValueError(f"Cannot merge {old_axes} into {new_axis}: size mismatch")
    else:
        assert isinstance(new_axis, str)
        new_axis = Axis(new_axis, prod(ax.size for ax in old_axes))

    # ensure that the old_axes are contiguous
    # we basically ensure that the old_axes occur after the index of the first old_axis
    intermediate_axes: List[Axis] = []
    new_axes: List[Axis] = []
    index_of_first_old_axis = None
    for i, ax in enumerate(array.axes):
        if ax in old_axes:
            if index_of_first_old_axis is None:
                index_of_first_old_axis = i
                intermediate_axes.extend(old_axes)
                new_axes.append(new_axis)
            else:
                continue
        else:
            intermediate_axes.append(ax)
            new_axes.append(ax)

    array = array.rearrange(intermediate_axes)
    raw_array = array.array.reshape([ax.size for ax in new_axes])
    return NamedArray(raw_array, tuple(new_axes))


def unflatten_axis(array: NamedArray, axis: AxisSelector, new_axes: AxisSpec) -> NamedArray:
    """
    Split an axis into a sequence of axes. The old axis must have the same size as the product of the new axes.
    """
    old_index = array._lookup_indices(axis)
    if old_index is None:
        raise ValueError(f"Axis {axis} not found in {array}")

    axis_size = array.axes[old_index].size

    new_axes = ensure_tuple(new_axes)

    if len(new_axes) == 0:
        if axis_size == 1:
            # just remove the old axis, akin to squeeze
            new_array = jnp.squeeze(array.array, axis=old_index)
            resolved_new_axes = array.axes[:old_index] + array.axes[old_index + 1 :]
            return NamedArray(new_array, resolved_new_axes)
        else:
            raise ValueError("Must specify at least one axis to split")

    if axis_size != prod(ax.size for ax in new_axes):
        raise ValueError(f"Cannot split {axis} into {new_axes}: size mismatch")

    resolved_new_axes = array.axes[:old_index] + tuple(new_axes) + array.axes[old_index + 1 :]
    new_array = jnp.reshape(array.array, [ax.size for ax in resolved_new_axes])
    return NamedArray(new_array, resolved_new_axes)


def named(a: jnp.ndarray, axis: AxisSelection) -> NamedArray:
    """Creates a NamedArray from a numpy array and a list of axes"""
    axes = check_shape(a.shape, axis)
    return NamedArray(a, axes)


@overload
def concat_axis_specs(a1: AxisSpec, a2: AxisSpec) -> Sequence[Axis]:
    pass


@overload
def concat_axis_specs(a1: AxisSelection, a2: AxisSelection) -> Sequence[Union[Axis, str]]:
    pass


def concat_axis_specs(a1: AxisSelection, a2: AxisSelection) -> AxisSelection:
    """Concatenates two AxisSpec. Raises ValueError if any axis is present in both specs"""

    def _ax_name(ax: AxisSelector) -> str:
        if isinstance(ax, Axis):
            return ax.name
        else:
            return ax

    if isinstance(a1, Axis) and isinstance(a2, Axis):
        if _ax_name(a1) == _ax_name(a2):
            raise ValueError(f"Axis {a1} specified twice")
        return (a1, a2)
    else:
        a1 = ensure_tuple(a1)
        a2 = ensure_tuple(a2)

        a1_names = [_ax_name(ax) for ax in a1]
        a2_names = [_ax_name(ax) for ax in a2]

        if len(set(a1_names) & set(a2_names)) > 0:
            overlap = [ax for ax in a1_names if ax in a2_names]
            raise ValueError(f"AxisSpecs overlap! {' '.join(str(x) for x in overlap)}")
        return a1 + a2


# Broadcasting Support
def _broadcast_order(a: NamedArray, b: NamedArray, require_subset: bool = True) -> Tuple[Axis, ...]:
    """
    Returns an ordering of axes for broadcasting a and b.

    If require_subset is True, then one of the array's axes must be a subset of the other's. This requirement is
    a bit stricter than a straightforward generalization of numpy's broadcasting rules, but I've been bitten by
    numpy's rules too many times.
    """
    broadcasted = _broadcast_axes(a.axes, b.axes, require_subset)
    if broadcasted is None:
        # TODO: decide under which conditions we want to allow broadcasting both arrays
        # maybe just add a context manager to allow it?
        raise ValueError(
            f"Cannot broadcast {a} and {b}: no subset relationship. "
            "If you want to broadcast anyway, use the broadcast_axis function to explicitly add axes"
        )
    return broadcasted


def _broadcast_axes(
    a_axes: Tuple[Axis, ...], b_axes: Tuple[Axis, ...], require_subset: bool = True
) -> Optional[Tuple[Axis, ...]]:
    if a_axes == b_axes:
        return a_axes
    if len(a_axes) == 0:
        return b_axes
    if len(b_axes) == 0:
        return a_axes

    if require_subset:
        # check if one is a subset of the other
        if set(a_axes).issubset(set(b_axes)):
            return b_axes
        elif set(b_axes).issubset(set(a_axes)):
            return a_axes
        else:
            return None

    a_size = prod(ax.size for ax in a_axes)
    b_size = prod(ax.size for ax in b_axes)
    if a_size < b_size:
        a_axes, b_axes = b_axes, a_axes

    # we want to order the axes in such a way that we minimize movement, or at least allow
    # large blocks to be memcpy'd when possible.
    # In particular, we'd like to avoid the case of reordering [Y, Z] + [X, Y, Z] -> [Y, Z, X] or other major reshuffles

    # here's what we do: we try to preserve the order of axes in the bigger array, and then stick the axes from the
    # other array on the front (because everything is row major)
    # this ensures we only have to move one array around

    return tuple(x for x in b_axes if x not in a_axes) + a_axes


def broadcast_to(
    a: NamedOrNumeric, axes: AxisSpec, ensure_order: bool = True, enforce_no_extra_axes: bool = True
) -> NamedArray:
    """
    Broadcasts a so that it has the given axes.
     If ensure_order is True (default), then the returned array will have the same axes in the same order as the given
     axes. Otherwise, the axes may not be moved if they are already in the array. The axes may not be contiguous however

    If enforce_no_extra_axes is True and the array has axes that are not in axes, then a ValueError is raised.
    """
    axes = ensure_tuple(axes)

    if not isinstance(a, NamedArray):
        a = named(jnp.asarray(a), ())

    assert isinstance(a, NamedArray)  # mypy gets confused

    if a.axes == axes:
        return a

    to_add = tuple(ax for ax in axes if ax not in a.axes)

    all_axes = to_add + a.axes

    if enforce_no_extra_axes and len(all_axes) != len(axes):
        raise ValueError(f"Cannot broadcast {a} to {axes}: extra axes present")

    extra_axes = tuple(ax for ax in a.axes if ax not in axes)

    # broadcast whatever we need to the front and reorder
    a_array = jnp.broadcast_to(a.array, [ax.size for ax in all_axes])
    a = NamedArray(a_array, all_axes)

    # if the new axes are already in the right order, then we're done
    if ensure_order and not _is_subsequence(axes, all_axes):
        a = a.rearrange(axes + extra_axes)

    return typing.cast(NamedArray, a)


def _is_subsequence(needle, haystack):
    needle_i = 0
    haystack_j = 0
    while needle_i < len(needle) and haystack_j < len(haystack):
        if needle[needle_i] == haystack[haystack_j]:
            needle_i += 1
        haystack_j += 1

    if needle_i < len(needle):
        return False
    return True


@overload
def broadcast_arrays(
    *arrays: NamedArray, require_subset: bool = True, ensure_order: bool = True
) -> Tuple[NamedArray, ...]:
    ...


@overload
def broadcast_arrays(
    *arrays: Optional[NamedOrNumeric], require_subset: bool = True, ensure_order: bool = True
) -> Tuple[Optional[NamedOrNumeric], ...]:
    ...


def broadcast_arrays(
    *arrays: Optional[NamedOrNumeric],
    require_subset: bool = True,
    ensure_order: bool = True,
) -> Tuple[Optional[NamedOrNumeric], ...]:
    return broadcast_arrays_and_return_axes(*arrays, require_subset=require_subset, ensure_order=ensure_order)[0]


@overload
def broadcast_arrays_and_return_axes(
    *arrays: NamedArray, require_subset: bool = True, ensure_order: bool = True
) -> Tuple[Tuple[NamedArray, ...], Tuple[Axis, ...]]:
    ...


@overload
def broadcast_arrays_and_return_axes(
    *arrays: NamedOrNumeric, require_subset: bool = True, ensure_order: bool = True
) -> Tuple[Tuple[NamedOrNumeric, ...], Tuple[Axis, ...]]:
    ...


@overload
def broadcast_arrays_and_return_axes(
    *arrays: Optional[NamedOrNumeric], require_subset: bool = True, ensure_order: bool = True
) -> Tuple[Tuple[Optional[NamedOrNumeric], ...], Tuple[Axis, ...]]:
    ...


def broadcast_arrays_and_return_axes(
    *arrays: Optional[NamedOrNumeric],
    require_subset: bool = True,
    ensure_order: bool = True,
) -> Tuple[Tuple[Optional[NamedOrNumeric], ...], Tuple[Axis, ...]]:
    """
    Broadcasts a sequence of arrays to a common set of axes.

     Parameters
    ----------
    arrays: NamedArray
        The arrays to broadcast
    require_subset: bool
        If True, then one of the arrays must be a subset of the other. This is a bit stricter than numpy's broadcasting
        rules, but I've been bitten by numpy's rules too many times. False is looser than numpy's rules, and allows
        broadcasting any pair of arrays (so long as the axes don't overtly conflict with different sizes for the same
        name.)
    ensure_order: bool
        If True, then the returned arrays will have the same axes in the same order as the given axes. Otherwise, the
        axes may not be moved.
    """
    if len(arrays) == 0:
        return ((), ())

    # sort the arrays by size, so that we use the biggest ones to broadcast the others
    # need to hold on to the order so we can return the arrays in the same order
    actual_arrays = [x for x in arrays if isinstance(x, NamedArray)]
    size_order = sorted(range(len(actual_arrays)), key=lambda i: actual_arrays[i].size, reverse=True)
    all_axes = [actual_arrays[i].axes for i in size_order]
    full_axes = ft.reduce(lambda a, b: _broadcast_axes(a, b, require_subset) if a is not None else None, all_axes)  # type: ignore
    if full_axes is None:
        raise ValueError(f"Cannot broadcast arrays {arrays}: no subset relationship")

    arrays = tuple(
        broadcast_to(a, full_axes, ensure_order=ensure_order) if isinstance(a, NamedArray) else a for a in arrays
    )

    return arrays, full_axes


# TODO: convert to AxisSelection?
def broadcast_axis(a: NamedArray, axis: AxisSpec) -> NamedArray:
    """
    Broadcasts a, ensuring that it has all the axes in axis.
     broadcast_axis is an alias for broadcast_to(a, axis, enforce_no_extra_axes=False, ensure_order=True)
    """
    if isinstance(axis, Axis) and axis in a.axes:
        return a

    return broadcast_to(a, axis, enforce_no_extra_axes=False, ensure_order=True)


def check_shape(jnp_shape: Sequence[int], hax_axes: AxisSelection) -> Tuple[Axis, ...]:
    """Check that the shape of a jax array matches the axes of a NamedArray"""
    axes: Tuple[AxisSelector, ...] = ensure_tuple(hax_axes)
    if len(jnp_shape) != len(axes):
        raise ValueError(f"Shape mismatch: jnp_shape={jnp_shape} hax_axes={hax_axes}")
    result_axes: List[haliax.Axis] = []
    for i in range(len(axes)):
        ax = axes[i]
        if isinstance(ax, Axis):
            if ax.size != jnp_shape[i]:
                raise ValueError(f"Shape mismatch: jnp_shape={jnp_shape} hax_axes={hax_axes}")
            result_axes.append(ax)  # type: ignore
        elif isinstance(ax, str):
            result_axes.append(Axis(ax, jnp_shape[i]))
        else:
            raise ValueError(f"Invalid axis spec: {ax}")

    return tuple(result_axes)


class _Sentinel:
    ...


def is_axis_compatible(ax1: AxisSelector, ax2: AxisSelector):
    if isinstance(ax1, str):
        if isinstance(ax2, str):
            return ax1 == ax2
        return ax1 == ax2.name
    if isinstance(ax2, str):
        return ax1.name == ax2
    return ax1.name == ax2.name


def selects_axis(selector: AxisSelection, selected: AxisSelection) -> bool:
    """Returns true if the selector has every axis in selected and, if dims are given, that they match"""
    if isinstance(selector, Axis) or isinstance(selector, str):
        selected = ensure_tuple(selected)
        try:
            index = index_where(lambda ax: is_axis_compatible(ax, selector), selected)  # type: ignore
            return index >= 0
        except ValueError:
            return False

    selector_dict = _spec_to_dict(selector)

    selected_tuple = ensure_tuple(selected)  # type: ignore
    for ax in selected_tuple:
        if isinstance(ax, Axis):
            selector_size = selector_dict.get(ax.name, _Sentinel)
            if selector_size is not None and selector_size != ax.size:
                return False
        elif isinstance(ax, str):
            if ax not in selector_dict:
                return False
        else:
            raise ValueError(f"Invalid axis spec: {ax}")

    return True


@overload
def _spec_to_dict(axis_spec: AxisSpec) -> Mapping[str, int]:
    ...


@overload
def _spec_to_dict(axis_spec: AxisSelection) -> Mapping[str, Optional[int]]:
    ...


def _spec_to_dict(axis_spec: AxisSelection) -> Mapping[str, Optional[int]]:
    spec = ensure_tuple(axis_spec)  # type: ignore
    shape_dict: Dict[str, Optional[int]] = {}
    for ax in spec:
        if isinstance(ax, Axis):
            shape_dict[ax.name] = ax.size
        elif isinstance(ax, str):
            shape_dict[ax] = None
        else:
            raise ValueError(f"Invalid axis spec: {ax}")

    return shape_dict


__all__ = [
    "NamedArray",
    "concat_axis_specs",
    "dot",
    "named",
    "rearrange",
    "slice",
    "slice_nd",
    "take",
    "split",
    "flatten_axes",
    "unflatten_axis",
    "unbind",
    "roll",
    "_broadcast_order",
    "broadcast_to",
    "broadcast_axis",
    "broadcast_arrays",
    "enable_shape_checks",
    "are_shape_checks_enabled",
    "check_shape",
]
