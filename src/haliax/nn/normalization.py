from typing import Optional

import equinox as eqx

import haliax as hax

from ..core import NamedArray
from ..types import AxisSpec


class LayerNorm(eqx.Module):
    """
    Normalises the input along the specified axis (or axes), using the mean and variance of the
    input along that axis.
    """
    axis: AxisSpec = eqx.static_field()
    eps: float = eqx.static_field()

    weight: Optional[NamedArray]
    bias: Optional[NamedArray]

    def __init__(self, axis: AxisSpec, eps: float = 1e-5, learn_scale: bool = True, learn_bias: bool = True):
        self.axis = axis
        self.eps = eps

        if learn_scale:
            self.weight = hax.ones(axis)
        else:
            self.weight = None

        if learn_bias:
            self.bias = hax.zeros(axis)
        else:
            self.bias = None

    def __call__(self, x: NamedArray) -> NamedArray:
        mean = x.mean(self.axis)
        var = x.var(self.axis)
        inv = hax.rsqrt(var + self.eps)
        out = (x - mean) * inv

        if self.weight is not None:
            out = self.weight * out

        if self.bias is not None:
            out = out + self.bias

        return out
