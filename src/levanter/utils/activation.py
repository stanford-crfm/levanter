import enum
import typing
from functools import partial

import jax
import jax.numpy as jnp

import haliax as hax
import haliax.nn as hnn


_A = typing.TypeVar("_A", hax.Scalar, hax.NamedArray, jax.Array)
ActivationFunction = typing.Callable[[_A], _A]


def polynorm(
    x: _A,
    weights: typing.Sequence[float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    bias: float = 0.0,
    eps: float = 1e-6,
) -> _A:
    """PolyNorm activation function.

    The implementation follows the definition from
    https://arxiv.org/html/2411.03884v1, normalizing each polynomial term by
    its root-mean-square (RMS) before applying learned weights and bias.

    Args:
        x: Input array, either a :class:`haliax.NamedArray` or a regular JAX array.
        weights: Sequence of polynomial weights ``[w_1, ..., w_r]`` applied to
            the terms ``x**r, ..., x``. Defaults to a uniform third-degree
            polynomial.
        bias: Bias term added after the weighted sum.
        eps: Small constant added for numerical stability.

    Returns:
        Activated output with each polynomial term RMS-normalized.
    """

    def _rms_norm(y: _A) -> _A:
        if isinstance(y, hax.NamedArray):
            axis = y.axes[-1]
            mean_sq = (y * y).mean(axis)
            denom = hax.sqrt(mean_sq + eps).broadcast_axis(axis)
            return y / denom
        mean_sq = jnp.mean(y * y, axis=-1, keepdims=True)
        return y / jnp.sqrt(mean_sq + eps)

    result: _A
    if isinstance(x, hax.NamedArray):
        result = hax.full_like(x, bias)
    else:
        result = jnp.full_like(x, bias)

    degree = len(weights)
    for w, power in zip(weights, range(degree, 0, -1)):
        term = _rms_norm(x**power)
        result = result + w * term

    return result


class ActivationFunctionEnum(str, enum.Enum):
    relu = "relu"
    silu = "silu"
    swish = "swish"
    gelu = "gelu"
    gelu_new = "gelu_new"
    quick_gelu = "quick_gelu"
    tanh = "tanh"
    polynorm = "polynorm"

    def to_fn(self) -> ActivationFunction:
        return TO_FN[self]


# type: ignore
TO_FN: dict[ActivationFunctionEnum, ActivationFunction] = {
    ActivationFunctionEnum.relu: hnn.relu,
    ActivationFunctionEnum.silu: hnn.silu,
    ActivationFunctionEnum.swish: hnn.swish,
    ActivationFunctionEnum.gelu: partial(hnn.gelu, approximate=False),
    ActivationFunctionEnum.gelu_new: partial(hnn.gelu, approximate=True),
    ActivationFunctionEnum.quick_gelu: hnn.quick_gelu,
    ActivationFunctionEnum.tanh: hax.tanh,
    ActivationFunctionEnum.polynorm: polynorm,
}
