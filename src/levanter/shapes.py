from dataclasses import dataclass
from math import prod
from typing import Optional, Tuple, Type, TypeAlias, Union

import numpy as np
from jax import ShapeDtypeStruct

from haliax import Axis


DType = Union[np.dtype, Type[int], Type[float], Type[bool]]

ShapeSpec: TypeAlias = ShapeDtypeStruct


@dataclass(frozen=True)
class NamedShapeSpec:
    shape: Optional[Tuple[Axis, ...]]
    dtype: Optional[DType]

    size = property(lambda self: prod(ax.size for ax in self.shape))
    ndim = property(lambda self: len(self.shape))


def to_raw_shape(shape: Union[ShapeSpec, NamedShapeSpec]) -> Optional[Tuple[int, ...]]:
    if isinstance(shape, ShapeDtypeStruct):
        return shape.shape
    else:
        raw = shape.shape
        if raw is None:
            return None
        return tuple(ax.size for ax in raw)
