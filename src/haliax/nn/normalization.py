from typing import Optional

import equinox as eqx

import haliax as hax

from ..core import AxisSpec, NamedArray


class LayerNorm(eqx.Module):
    r"""
    Normalises the input along the specified axis (or axes), using the mean and variance of the
    input along that axis.
    """
    axis: AxisSpec = eqx.static_field()
    eps: float = eqx.static_field()

    weight: Optional[NamedArray]
    bias: Optional[NamedArray]

    def __init__(self, axis: AxisSpec, eps: float = 1e-5, elementwise_affine: bool = True):
        self.axis = axis
        self.eps = eps

        if elementwise_affine:
            self.weight = hax.ones(axis)
            self.bias = hax.zeros(axis)

    def __call__(self, x: NamedArray) -> NamedArray:
        mean = x.mean(self.axis)
        var = x.var(self.axis)
        inv = hax.rsqrt(var + self.eps)
        out = (x - mean) * inv

        if self.weight is not None:
            out = self.weight * out + self.bias
        return out
