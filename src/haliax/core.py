from dataclasses import dataclass
from math import prod
from types import EllipsisType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, TypeVar, Union, cast

import jax
import jax.numpy as jnp
import numpy as np
from jaxlib.xla_extension import DeviceArray

import haliax


@dataclass(frozen=True)
class Axis:
    name: str
    size: int

    def alias(self, new_name: str):
        return Axis(new_name, self.size)


AxisSpec = Union[Axis, Sequence[Axis]]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class NamedArray:
    array: jnp.ndarray
    axes: Tuple[Axis, ...]

    def __post_init__(self):
        if not isinstance(self.axes, tuple):
            object.__setattr__(self, "axes", tuple(self.axes))
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

    def lookup_indices(self, axis: AxisSpec):
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
            return tuple(self.lookup_indices(a) for a in axis)

    def rearrange(self, axis: Sequence[Union[Axis, EllipsisType]]):
        return rearrange(self, axis)

    # np.ndarray methods:
    def all(self, axis: Optional[AxisSpec] = None, out=None, keepdims=None) -> Any:
        return haliax.all(self, axis=axis, out=out, keepdims=keepdims)

    def any(self, axis: Optional[AxisSpec] = None, out=None, keepdims=None) -> Any:
        return haliax.any(self, axis=axis, out=out, keepdims=keepdims)

    # def select(self, axis: Axis, index: Union[int, 'NamedArray', jnp.ndarray]) -> Any:
    #     if isinstance(index, NamedArray):
    #         index = index.array

    # TODO
    # def argmax(self, axis: Optional[int] = None, out=None, keepdims=None) -> Any:
    #     ...
    #
    # def argmin(self, axis: Optional[int] = None, out=None, keepdims=None) -> Any:
    #     ...
    #
    # def argpartition(self, kth, axis=-1, kind='introselect', order=None) -> Any:
    #     ...
    #
    # def argsort(self, axis: Optional[int] = -1, kind='quicksort', order=None) -> Any:
    #     ...

    def astype(self, dtype) -> Any:
        return NamedArray(self.array.astype(dtype), self.axes)

    # TODO
    # def clip(self, a_min=None, a_max=None, out=None) -> Any:
    #     ...

    # def compress(self, condition, axis: Optional[int] = None, out=None) -> Any:
    #     ...

    def conj(self) -> Any:
        return NamedArray(self.array.conj(), self.axes)

    def conjugate(self) -> Any:
        return NamedArray(self.array.conjugate(), self.axes)

    def copy(self) -> Any:
        return NamedArray(self.array.copy(), self.axes)

    def cumprod(self, axis: Optional[AxisSpec] = None, dtype=None, out=None) -> Any:
        return haliax.cumprod(self, axis=axis, dtype=dtype, out=out)

    def cumsum(self, axis: Optional[AxisSpec] = None, dtype=None, out=None) -> Any:
        return haliax.cumsum(self, axis=axis, dtype=dtype, out=out)

    # def diagonal(self, offset=0, axis1: int = 0, axis2: int = 1) -> Any:
    #     ...

    def dot(self, axis: AxisSpec, b, *, precision=None) -> Any:
        return dot(axis, self, b, precision=precision)

    @property
    def imag(self) -> Any:
        return NamedArray(self.array.imag, self.axes)

    # TODO:
    # def item(self, *args) -> Any:
    #

    def max(
        self,
        axis: Optional[AxisSpec] = None,
        out=None,
        keepdims=None,
        initial=None,
        where=None,
    ) -> Any:
        return haliax.max(self, axis=axis, out=out, keepdims=keepdims, initial=initial, where=where)

    def mean(
        self,
        axis: Optional[AxisSpec] = None,
        dtype=None,
        out=None,
        keepdims=False,
        *,
        where=None,
    ) -> Any:
        return haliax.mean(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where)

    def min(
        self,
        axis: Optional[AxisSpec] = None,
        out=None,
        keepdims=None,
        initial=None,
        where=None,
    ) -> Any:
        return haliax.min(self, axis=axis, out=out, keepdims=keepdims, initial=initial, where=where)

    # TODO
    # def nonzero(self, *, size=None, fill_value=None) -> Any:
    #     ...

    def prod(
        self,
        axis: Optional[AxisSpec] = None,
        dtype=None,
        out=None,
        keepdims=None,
        initial=None,
        where=None,
    ):
        return haliax.prod(
            self,
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    # def ptp(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, out=None,
    #         keepdims=False, ) -> Any:
    #     ...

    # def ravel(self, order='C') -> Any:
    #     ...

    @property
    def real(self) -> Any:
        return NamedArray(self.array.real, self.axes)

    # def repeat(self, repeats, axis: Optional[int] = None, *,
    #            total_repeat_length=None) -> Any:
    #     ...

    # def reshape(self, *args, order='C') -> Any:
    #     ...

    def round(self, decimals=0, out=None) -> Any:
        return haliax.round(self, decimals=decimals, out=out)

    # def searchsorted(self, v, side='left', sorter=None) -> Any:
    #     ...

    # def sort(self, axis: Optional[int] = -1, kind='quicksort', order=None) -> Any:
    #     ...

    def split(self, axis: Axis, new_axes: Sequence[Axis]) -> Sequence["NamedArray"]:
        return haliax.split(self, axis=axis, new_axes=new_axes)

    def flatten_axes(self, old_axes: Sequence[Axis], new_axis: Axis) -> "NamedArray":
        return haliax.flatten_axes(self, old_axes=old_axes, new_axis=new_axis)

    def unflatten_axis(self, axis: Axis, new_axes: Sequence[Axis]) -> "NamedArray":
        return haliax.unflatten_axis(self, axis=axis, new_axes=new_axes)

    def unbind(self, axis: Axis) -> Sequence["NamedArray"]:
        return haliax.unbind(self, axis=axis)

    def rename(self, renames: Mapping[Axis, Axis]):
        return haliax.rename(self, renames=renames)

    # def squeeze(self, axis: Optional[AxisSpec] = None) -> Any:
    #     return haliax.squeeze(self, axis=axis)

    def std(
        self,
        axis: Optional[AxisSpec] = None,
        dtype=None,
        out=None,
        ddof=0,
        keepdims=False,
        *,
        where=None,
    ) -> Any:
        return haliax.std(
            self,
            axis=axis,
            dtype=dtype,
            out=out,
            ddof=ddof,
            keepdims=keepdims,
            where=where,
        )

    def sum(
        self,
        axis: Optional[AxisSpec] = None,
        dtype=None,
        out=None,
        keepdims=None,
        initial=None,
        where=None,
    ) -> Any:
        return haliax.sum(
            self,
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    def take(self, axis: Axis, index: Union[int, "NamedArray"]) -> Any:
        return haliax.take(self, axis=axis, index=index)

    def tobytes(self, order="C") -> Any:
        return self.array.tobytes(order=order)

    def tolist(self) -> Any:
        return self.array.tolist()

    # def trace(self, offset=0, axis1: int = 0, axis2: int = 1, dtype=None,
    #           out=None) -> Any:

    def var(
        self,
        axis: Optional[AxisSpec] = None,
        dtype=None,
        out=None,
        ddof=0,
        keepdims=False,
        *,
        where=None,
    ) -> Any:
        return haliax.var(
            self,
            axis=axis,
            dtype=dtype,
            out=out,
            ddof=ddof,
            keepdims=keepdims,
            where=where,
        )

    # operators
    def __add__(self, other) -> Any:
        # TODO: check shape and broadcast
        raise NotImplementedError

    def __mul__(self, other):
        if jnp.isscalar(other):
            return NamedArray(self.array * other, self.axes)

        raise NotImplementedError

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if jnp.isscalar(other):
            return NamedArray(self.array / other, self.axes)

        raise NotImplementedError

    def __rtruediv__(self, other):
        if jnp.isscalar(other):
            return NamedArray(other / self.array, self.axes)

        raise NotImplementedError


def take(array: NamedArray, axis: Axis, index: Union[int, NamedArray]) -> NamedArray:
    """
    Selects elements from an array along an axis, by an index or by another named array
    """
    axis_index = array.lookup_indices(axis)
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

    index = a.lookup_indices(axis)

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
            index = array.lookup_indices(ax)
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
    axis_index = array.lookup_indices(axis)
    if axis_index is None:
        raise ValueError(f"axis {axis} not found in {array}")
    arrays = jnp.rollaxis(array.array, axis=axis_index, start=0)
    new_axes = array.axes[:axis_index] + array.axes[axis_index + 1 :]
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
    old_index = array.lookup_indices(axis)
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


T = TypeVar("T")


def _ensure_tuple(x: Union[Sequence[T], T]) -> Tuple[T, ...]:
    if isinstance(x, Sequence):
        return tuple(x)
    return (x,)


def named(a: jnp.ndarray, axis: AxisSpec) -> NamedArray:
    """Creates a NamedArray from a numpy array and a list of axes"""
    if isinstance(axis, Axis):
        if jnp.shape(a) != axis.size:
            raise ValueError(f"Shape of array {jnp.shape(a)} does not match size of axis {axis.size}")
        return NamedArray(a, (axis,))
    else:
        shape: Tuple[Axis, ...] = _ensure_tuple(axis)
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
        a1 = _ensure_tuple(a1)
        a2 = _ensure_tuple(a2)
        if any(x in a2 for x in a1) or any(x in a1 for x in a2):
            overlap = set(a1).intersection(set(a2))
            raise ValueError(f"AxisSpecs overlap! {' '.join(str(x) for x in overlap)}")
        return a1 + a2


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
]
