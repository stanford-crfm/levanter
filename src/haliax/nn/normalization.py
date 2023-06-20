from typing import Optional

import equinox as eqx

import haliax as hax

from ..core import NamedArray
from ..types import AxisSpec


class LayerNorm(eqx.Module):
    r"""
    Normalises the input along the specified axis (or axes), using the mean and variance of the
    input along that axis.
    """
    axis: AxisSpec = eqx.static_field()
    weight: Optional[NamedArray]
    bias: Optional[NamedArray]

    eps: float = eqx.static_field(default=1e-5)

    @staticmethod
    def init(axis: AxisSpec, eps: float = 1e-5, use_weight: bool = True, use_bias: bool = True):
        if use_weight:
            weight = hax.ones(axis)
        else:
            weight = None
        if use_bias:
            bias = hax.zeros(axis)
        else:
            bias = None

        return LayerNorm(axis, weight, bias, eps)

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
