import enum
import typing
from functools import partial

import jax

import haliax as hax
import haliax.nn as hnn


_A = typing.TypeVar("_A", hax.Scalar, hax.NamedArray, jax.Array)
ActivationFunction = typing.Callable[[_A], _A]


class ActivationFunctionEnum(enum.Enum):
    RELU = enum.auto()
    SILU = enum.auto()
    SWISH = enum.auto()
    GELU = enum.auto()
    GELU_NEW = enum.auto()
    QUICK_GELU = enum.auto()

    def to_fn(self) -> ActivationFunction:
        return TO_FN[self]


TO_FN: dict[ActivationFunctionEnum, ActivationFunction] = {
    ActivationFunctionEnum.RELU: hnn.relu,
    ActivationFunctionEnum.SILU: hnn.silu,
    ActivationFunctionEnum.SWISH: hnn.swish,
    ActivationFunctionEnum.GELU: partial(hnn.gelu, approximate=False),
    ActivationFunctionEnum.GELU_NEW: partial(hnn.gelu, approximate=True),
    ActivationFunctionEnum.QUICK_GELU: hnn.quick_gelu,
}
