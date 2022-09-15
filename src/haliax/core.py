import functools as ft
from dataclasses import dataclass
from math import prod
from types import EllipsisType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, cast

import jax
import jax.numpy as jnp
import numpy as np
from jaxlib.xla_extension import DeviceArray

import haliax
from haliax.util import ensure_tuple


@dataclass(frozen=True)
class Axis:
    name: str
    size: int

    def alias(self, new_name: str):
        return Axis(new_name, self.size)


AxisSpec = Union[Axis, Sequence[Axis]]

Scalar = Union[float, int]
NamedNumeric = Union[Scalar, "NamedArray"]


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

        if isinstance(self.array, jax.core.Tracer) or isinstance(self.array, DeviceArray):
            s = jnp.shape(self.array)
            if s != tuple(a.size for a in self.axes):
                raise ValueError(f"Shape of underlying array {s} does not match shape of axes {self.axes}")

    def __array__(self):
        return self.array.__array__()

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

    def _lookup_indices(self, axis: AxisSpec):
        """
        For a single axis, returns an int corresponding to the index of the axis.
        For multiple axes, returns a tuple of ints corresponding to the indices of the axes.

        If the axis is not present, returns None for that position
        """
        if isinstance(axis, Axis):
            try:
                return self.axes.index(axis)
            except ValueError:
                return None
        else:
            return tuple(self._lookup_indices(a) for a in axis)

    # Axis rearrangement
    def rearrange(self, axis: Sequence[Union[Axis, EllipsisType]]) -> "NamedArray":
        return rearrange(self, axis)

    def broadcast_to(self, axes: AxisSpec) -> "NamedArray":
        axes = ensure_tuple(axes)
        return haliax.broadcast_to(self, axes=axes)

    def broadcast_axis(self, axis: AxisSpec) -> "NamedArray":
        return haliax.broadcast_axis(self, axis=axis)

    def split(self, axis: Axis, new_axes: Sequence[Axis]) -> Sequence["NamedArray"]:
        return haliax.split(self, axis=axis, new_axes=new_axes)

    def flatten_axes(self, old_axes: Sequence[Axis], new_axis: Axis) -> "NamedArray":
        return haliax.flatten_axes(self, old_axes=old_axes, new_axis=new_axis)

    def unflatten_axis(self, axis: Axis, new_axes: Sequence[Axis]) -> "NamedArray":
        return haliax.unflatten_axis(self, axis=axis, new_axes=new_axes)

    def unbind(self, axis: Axis) -> Sequence["NamedArray"]:
        return haliax.unbind(self, axis=axis)

    def rename(self, renames: Mapping[Axis, Axis]) -> "NamedArray":
        return haliax.rename(self, renames=renames)

    def take(self, axis: Axis, index: Union[int, "NamedArray"]) -> "NamedArray":
        return haliax.take(self, axis=axis, index=index)

    # np.ndarray methods:
    def all(self, axis: Optional[AxisSpec] = None) -> "NamedArray":
        return haliax.all(self, axis=axis)

    def any(self, axis: Optional[AxisSpec] = None) -> "NamedArray":
        return haliax.any(self, axis=axis)

    # TODO: test
    def argmax(self, axis: Axis) -> "NamedArray":
        return haliax.argmax(self, axis=axis)

    def argmin(self, axis: Axis) -> "NamedArray":
        return haliax.argmin(self, axis=axis)

    def argsort(self, axis: Axis) -> "NamedArray":
        return haliax.argsort(self, axis=axis)

    def astype(self, dtype) -> "NamedArray":
        return NamedArray(self.array.astype(dtype), self.axes)

    # TODO
    # def clip(self, a_min=None, a_max=None) -> Any:
    #     ...

    # def compress(self, condition, axis: Optional[int] = None) -> Any:
    #     ...

    def conj(self) -> "NamedArray":
        return NamedArray(self.array.conj(), self.axes)

    def conjugate(self) -> "NamedArray":
        return NamedArray(self.array.conjugate(), self.axes)

    def copy(self) -> "NamedArray":
        return NamedArray(self.array.copy(), self.axes)

    def cumprod(self, axis: Optional[AxisSpec] = None, *, dtype=None) -> "NamedArray":
        return haliax.cumprod(self, axis=axis, dtype=dtype)

    def cumsum(self, axis: Optional[AxisSpec] = None, *, dtype=None) -> "NamedArray":
        return haliax.cumsum(self, axis=axis, dtype=dtype)

    # def diagonal(self, offset=0, axis1: int = 0, axis2: int = 1) -> Any:
    #     ...

    def dot(self, axis: AxisSpec, b, *, precision=None) -> "NamedArray":
        return dot(axis, self, b, precision=precision)

    @property
    def imag(self) -> "NamedArray":
        return NamedArray(self.array.imag, self.axes)

    def max(
        self,
        axis: Optional[AxisSpec] = None,
        *,
        initial=None,
        where=None,
    ) -> "NamedArray":
        return haliax.max(self, axis=axis, initial=initial, where=where)

    def mean(
        self,
        axis: Optional[AxisSpec] = None,
        *,
        dtype=None,
        where=None,
    ) -> "NamedArray":
        return haliax.mean(self, axis=axis, dtype=dtype, where=where)

    def min(
        self,
        axis: Optional[AxisSpec] = None,
        *,
        initial=None,
        where=None,
    ) -> "NamedArray":
        return haliax.min(self, axis=axis, initial=initial, where=where)

    # TODO
    # def nonzero(self, *, size=None, fill_value=None) -> Any:
    #     ...

    def prod(
        self,
        axis: Optional[AxisSpec] = None,
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

    def ptp(self, axis: AxisSpec = None) -> "NamedArray":
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

    # def searchsorted(self, v, side='left', sorter=None) -> Any:
    #     ...

    def sort(self, axis: Axis, kind="quicksort") -> Any:
        return haliax.sort(self, axis=axis, kind=kind)

    def std(
        self,
        axis: Optional[AxisSpec] = None,
        *,
        dtype=None,
        ddof=0,
        where=None,
    ) -> "NamedArray":
        return haliax.std(
            self,
            axis=axis,
            dtype=dtype,
            ddof=ddof,
            where=where,
        )

    def sum(
        self,
        axis: Optional[AxisSpec] = None,
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

    def trace(self, axis1: Axis, axis2: Axis, offset=0, dtype=None) -> "NamedArray":
        return haliax.trace(self, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype)

    def var(
        self,
        axis: Optional[AxisSpec] = None,
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


def take(array: NamedArray, axis: Axis, index: Union[int, NamedArray]) -> NamedArray:
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


def dot(axis: AxisSpec, *arrays: NamedArray, precision=None) -> NamedArray:
    """Returns the tensor product of two NamedArrays. The axes `axis` are contracted over,
    and any other axes that are shared between the arrays are batched over. Non-contracted Axes in one
    that are not in the other are preserved.
    """
    if isinstance(axis, Axis):
        axis = [axis]

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
    output_axes = tuple(ax for ax in axis_mappings.keys() if ax not in axis)
    output_spec = " ".join(str(axis_mappings[ax]) for ax in output_axes)

    output = jnp.einsum(
        ", ".join(array_specs) + "-> " + output_spec,
        *[a.array for a in arrays],
        precision=precision,
    )

    return NamedArray(output, output_axes)


def split(a: NamedArray, axis: Axis, new_axes: Sequence[Axis]) -> Sequence[NamedArray]:
    # check the lengths of the new axes
    if axis not in a.axes:
        raise ValueError(f"Axis {axis} not found in {a.axes}")

    total_len = sum(x.size for x in new_axes)
    if total_len != axis.size:
        raise ValueError(f"The total length of the new axes {total_len} does not match the length of the axis {axis}")

    index = a._lookup_indices(axis)

    # now we can split the array
    offsets = np.cumsum([0] + [x.size for x in new_axes])[1:-1]

    new_arrays = np.split(a.array, indices_or_sections=offsets, axis=index)
    ret_axes = [tuple(ax2 if ax2 is not axis else new_axis for ax2 in a.axes) for new_axis in new_axes]

    return [NamedArray(x, ax) for x, ax in zip(new_arrays, ret_axes)]


# TODO: can we add einops-style combined split/merge here?
# e.g. we'd like something like rearrange(array, (..., new_axis), merge_axes={new_axis: (old_axis1, old_axis2)})
# or rearrange(array, (new_axis1, ..., new_axis2), split_axes={old_axis: (new_axis1, new_axis2)})
# or even rearrange(array, (x, ..., b, a), map_axes={old_axis: (a, b), x: (old1, old2)})
def rearrange(array: NamedArray, axes: Sequence[Union[Axis, EllipsisType]]):
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
            assert isinstance(ax, Axis)  # please mypy
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


def unbind(array: NamedArray, axis: Axis) -> List[NamedArray]:
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
    arrays = [jnp.take(array.array, i, axis=axis_index) for i in range(axis.size)]
    return [NamedArray(a, new_axes) for a in arrays]


def rename(array: NamedArray, renames: Mapping[Axis, Axis]) -> NamedArray:
    for old, new in renames.items():
        if old.size != new.size:
            raise ValueError(f"Cannot rename axis {old} to {new}: size mismatch")
    new_axes = tuple(renames.get(ax, ax) for ax in array.axes)
    return NamedArray(array.array, new_axes)


def flatten_axes(array: NamedArray, old_axes: Sequence[Axis], new_axis: Axis) -> NamedArray:
    """
    Merge a sequence of axes into a single axis. The new axis must have the same size as the product of the old axes.
    For now the new axis will always be the last axis
    """
    if len(old_axes) == 0:
        raise ValueError("Must specify at least one axis to merge")

    if new_axis.size != prod(ax.size for ax in old_axes):
        raise ValueError(f"Cannot merge {old_axes} into {new_axis}: size mismatch")

    # TODO: might want to do something more clever here when the old_axes aren't at the end
    array = rearrange(array, (...,) + tuple(old_axes))
    new_axes = array.axes[: -len(old_axes)] + (new_axis,)
    raw_array = array.array.reshape(array.array.shape[: -len(old_axes)] + (new_axis.size,))
    return NamedArray(raw_array, new_axes)


def unflatten_axis(array: NamedArray, axis: Axis, new_axes: Sequence[Axis]) -> NamedArray:
    """
    Split an axis into a sequence of axes. The old axis must have the same size as the product of the new axes.
    """
    old_index = array._lookup_indices(axis)
    if old_index is None:
        raise ValueError(f"Axis {axis} not found in {array}")

    if len(new_axes) == 0:
        if axis.size == 1:
            # just remove the old axis, akin to squeeze
            new_array = jnp.squeeze(array.array, axis=old_index)
            new_axes = array.axes[:old_index] + array.axes[old_index + 1 :]
            return NamedArray(new_array, new_axes)
        else:
            raise ValueError("Must specify at least one axis to split")

    if axis.size != prod(ax.size for ax in new_axes):
        raise ValueError(f"Cannot split {axis} into {new_axes}: size mismatch")

    new_axes = array.axes[:old_index] + tuple(new_axes) + array.axes[old_index + 1 :]
    new_array = jnp.reshape(array.array, [ax.size for ax in new_axes])
    return NamedArray(new_array, new_axes)


def named(a: jnp.ndarray, axis: AxisSpec) -> NamedArray:
    """Creates a NamedArray from a numpy array and a list of axes"""
    if isinstance(axis, Axis):
        if jnp.shape(a) != (axis.size,):
            raise ValueError(f"Shape of array {jnp.shape(a)} does not match size of axis {axis.size}")
        return NamedArray(a, (axis,))
    else:
        shape: Tuple[Axis, ...] = ensure_tuple(axis)
        # verify the shape is correct
        if jnp.shape(a) != tuple(x.size for x in shape):
            raise ValueError(f"Shape of array {jnp.shape(a)} does not match shape of axes {axis}")

        return NamedArray(a, shape)


def concat_axis_specs(a1: AxisSpec, a2: AxisSpec) -> AxisSpec:
    """Concatenates two AxisSpec. Raises ValueError if any axis is present in both specs"""
    if isinstance(a1, Axis) and isinstance(a2, Axis):
        if a1 == a2:
            raise ValueError(f"Axis {a1} specified twice")
        return (a1, a2)
    else:
        a1 = ensure_tuple(a1)
        a2 = ensure_tuple(a2)
        if any(x in a2 for x in a1) or any(x in a1 for x in a2):
            overlap = set(a1).intersection(set(a2))
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


def broadcast_to(a: NamedArray, axes: Tuple[Axis, ...], ensure_order: bool = True) -> NamedArray:
    """
    Broadcasts a to the given axes. If ensure_order is True (default), then the returned array will have the same axes
    in the same order as the given axes. Otherwise, the axes may not be moved
    """
    if a.axes == axes:
        return a

    to_add = tuple(ax for ax in axes if ax not in a.axes)

    # broadcast whatever we need to the front and reorder
    a_array = jnp.broadcast_to(a.array, [ax.size for ax in to_add] + [ax.size for ax in a.axes])
    a = NamedArray(a_array, to_add + a.axes)

    if ensure_order:
        a = rearrange(a, axes)

    return a


def broadcast_arrays(
    *arrays: NamedArray,
    require_subset: bool = True,
    ensure_order: bool = True,
) -> Tuple[NamedArray, ...]:
    """
    Broadcasts a sequence of arrays to a common set of axes.

     Parameters
    ----------
    arrays: NamedArray
        The arrays to broadcast
    require_subset: bool
        If True, then one of the arrays must be a subset of the other. This is a bit stricter than numpy's broadcasting
        rules, but I've been bitten by numpy's rules too many times. If False is looser than numpy's rules, and allows
        broadcasting any pair of arrays (so long as the axes don't overtly conflict with different sizes for the same
        name.)
    ensure_order: bool
        If True, then the returned arrays will have the same axes in the same order as the given axes. Otherwise, the
        axes may not be moved.
    """
    if len(arrays) == 0:
        return ()

    # sort the arrays by size, so that we use the biggest ones to broadcast the others
    # need to hold on to the order so we can return the arrays in the same order
    size_order = sorted(range(len(arrays)), key=lambda i: arrays[i].size, reverse=True)
    all_axes = [arrays[i].axes for i in size_order]
    full_axes = ft.reduce(lambda a, b: _broadcast_axes(a, b, require_subset) if a is not None else None, all_axes)  # type: ignore
    if full_axes is None:
        raise ValueError(f"Cannot broadcast arrays {arrays}: no subset relationship")

    return tuple(broadcast_to(a, full_axes, ensure_order=ensure_order) for a in arrays)


def broadcast_axis(a: NamedArray, axis: AxisSpec) -> NamedArray:
    """
    Broadcasts a, ensuring that it has all the axes in axis
    """
    if isinstance(axis, Axis) and axis in a.axes:
        return a

    axis = ensure_tuple(axis)
    new_axes = tuple(ax for ax in axis if ax not in a.axes)
    return broadcast_to(a, new_axes + a.axes)


__all__ = [
    "Axis",
    "AxisSpec",
    "NamedArray",
    "concat_axis_specs",
    "dot",
    "named",
    "rearrange",
    "take",
    "split",
    "flatten_axes",
    "unflatten_axis",
    "unbind",
    "_broadcast_order",
    "broadcast_to",
    "broadcast_axis",
    "broadcast_arrays",
]
