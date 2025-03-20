import enum
import typing
from functools import partial

import jax

import haliax as hax
import haliax.nn as hnn


_A = typing.TypeVar("_A", hax.Scalar, hax.NamedArray, jax.Array)
ActivationFunction = typing.Callable[[_A], _A]


class ActivationFunctionEnum(enum.Enum):
    relu = enum.auto()
    silu = enum.auto()
    swish = enum.auto()
    gelu = enum.auto()
    gelu_new = enum.auto()
    quick_gelu = enum.auto()

    def to_fn(self) -> ActivationFunction:
        return TO_FN[self]


TO_FN: dict[ActivationFunctionEnum, ActivationFunction] = {
    ActivationFunctionEnum.relu: hnn.relu,
    ActivationFunctionEnum.silu: hnn.silu,
    ActivationFunctionEnum.swish: hnn.swish,
    ActivationFunctionEnum.gelu: partial(hnn.gelu, approximate=False),
    ActivationFunctionEnum.gelu_new: partial(hnn.gelu, approximate=True),
    ActivationFunctionEnum.quick_gelu: hnn.quick_gelu,
}
