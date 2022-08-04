from dataclasses import dataclass
from typing import Sequence, Union, Any, Dict, TypeVar, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.interpreters.partial_eval import DynamicJaxprTracer
from jaxlib.xla_extension import DeviceArray

import haliax


@dataclass(frozen=True)
class Axis:
    name: str
    size: int


AxisSpec = Union[Axis, Sequence[Axis]]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class NamedArray:
    array: jnp.ndarray
    axes: Sequence[Axis]

    def __post_init__(self):
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
        return cls(*tree, axes=aux)

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

    def max(self, axis: Optional[AxisSpec] = None, out=None, keepdims=None, initial=None, where=None) -> Any:
        return haliax.max(self, axis=axis, out=out, keepdims=keepdims, initial=initial, where=where)

    def mean(self, axis: Optional[AxisSpec] = None, dtype=None, out=None, keepdims=False, *, where=None, ) -> Any:
        return haliax.mean(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where)

    def min(self, axis: Optional[AxisSpec] = None, out=None, keepdims=None, initial=None, where=None) -> Any:
        return haliax.min(self, axis=axis, out=out, keepdims=keepdims, initial=initial, where=where)

    # TODO
    # def nonzero(self, *, size=None, fill_value=None) -> Any:
    #     ...

    def prod(self, axis: Optional[AxisSpec] = None, dtype=None, out=None, keepdims=None, initial=None, where=None) -> Any:
        return haliax.prod(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims, initial=initial, where=where)

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

    # def squeeze(self, axis: Optional[AxisSpec] = None) -> Any:
    #     return haliax.squeeze(self, axis=axis)

    def std(self, axis: Optional[AxisSpec] = None, dtype=None, out=None, ddof=0, keepdims=False, *, where=None) -> Any:
        return haliax.std(self, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims, where=where)

    def sum(self, axis: Optional[AxisSpec] = None, dtype=None, out=None, keepdims=None,
            initial=None, where=None) -> Any:
        return haliax.sum(self, axis=axis, )

    # def take(self, indices, axis: Optional[int] = None, out=None,
    #          mode=None) -> Any:
    #     ...

    def tobytes(self, order='C') -> Any:
        return self.array.tobytes(order=order)

    def tolist(self) -> Any:
        return self.array.tolist()

    # def trace(self, offset=0, axis1: int = 0, axis2: int = 1, dtype=None,
    #           out=None) -> Any:

    def var(self, axis: Optional[AxisSpec] = None,dtype=None, out=None, ddof=0, keepdims=False, *, where=None) -> Any:
        return haliax.var(self, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims, where=where)


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
        spec = ''
        for ax in a.axes:
            if ax in axis_mappings:
                spec += f'{axis_mappings[ax]} '
            else:
                axis_mappings[ax] = next_index
                spec += f'{next_index} '
                next_index += 1

        array_specs.append(spec)

    # now compute the output axes:
    output_axes = [ax for ax in axis_mappings.keys() if ax not in axis]
    output_spec = ' '.join(str(axis_mappings[ax]) for ax in output_axes)

    output = jnp.einsum(', '.join(array_specs) + "-> " + output_spec, *[a.array for a in arrays], precision=precision)

    return NamedArray(output, output_axes)


T = TypeVar('T')


def _ensure_sequence(x: Union[T, Sequence[T]]) -> Sequence[T]:
    if isinstance(x, Sequence):
        return x
    return (x,)


def named(a: jnp.ndarray, axis: AxisSpec) -> NamedArray:
    """Creates a NamedArray from a numpy array and a list of axes"""
    shape = _ensure_sequence(axis)
    # verify the shape is correct
    if jnp.shape(a) != tuple(x.size for x in shape):
        raise ValueError(f"Shape of array {jnp.shape(a)} does not match shape of axes {axis}")

    return NamedArray(a, shape)


