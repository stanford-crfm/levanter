from dataclasses import dataclass
from typing import Sequence, Tuple, Union

from jax.lax import Precision


@dataclass(frozen=True)
class Axis:
    """Axis is a dataclass that represents an axis of an NamedArray. It has a name and a size."""

    name: str
    size: int

    def alias(self, new_name: str):
        return Axis(new_name, self.size)

    def resize(self, size):
        return Axis(self.name, size)


AxisSelector = Union[Axis, str]
"""AxisSelector is a type that can be used to select a single axis from an array. str or Axis"""
AxisSelection = Union[AxisSelector, Sequence[AxisSelector]]
"""AxisSelection is a type that can be used to select multiple axes from an array. str, Axis, or sequence of mixed
str and Axis"""
AxisSpec = Union[Axis, Sequence[Axis]]
"""AxisSpec is a type that can be used to specify the axes of an array, usually for creation or adding a new axis
 whose size can't be determined another way. Axis or sequence of Axis"""
Scalar = Union[float, int]

PrecisionLike = Union[None, str, Precision, Tuple[str, str], Tuple[Precision, Precision]]
