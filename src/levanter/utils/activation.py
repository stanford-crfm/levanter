import enum
import typing
from functools import partial

import jax

import haliax as hax
import haliax.nn as hnn


_A = typing.TypeVar("_A", hax.Scalar, hax.NamedArray, jax.Array)
ActivationFunction = typing.Callable[[_A], _A]

TO_FN: dict[str, ActivationFunction] = {
    "relu": hnn.relu,
    "silu": hnn.silu,
    "swish": hnn.swish,
    "gelu": partial(hnn.gelu, approximate=False),
    "gelu_new": partial(hnn.gelu, approximate=True),
    "quick_gelu": hnn.quick_gelu,
}


class ActivationFunctionName(enum.StrEnum):
    RELU = enum.auto()
    SILU = enum.auto()
    SWISH = enum.auto()
    GELU = enum.auto()
    GELU_NEW = enum.auto()
    QUICK_GELU = enum.auto()

    def to_fn(self) -> ActivationFunction:
        return TO_FN[self.value]
