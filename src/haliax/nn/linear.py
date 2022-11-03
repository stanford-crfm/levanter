from typing import Optional

import equinox as eqx
import jax

import haliax as hax

from ..core import NamedArray
from ..types import AxisSpec


class Linear(eqx.Module):
    """A named Linear layer. This module allows you to specify multiple named axes for both input
    and output, which is occasionally useful."""

    weight: NamedArray
    bias: Optional[NamedArray]

    In: AxisSpec = eqx.static_field()
    Out: AxisSpec = eqx.static_field()

    def __init__(self, In: AxisSpec, Out: AxisSpec, *, key, include_bias=True):
        joint_spec = hax.concat_axis_specs(In, Out)
        self.weight = hax.random.normal(key, joint_spec) * 0.02
        self.bias = hax.zeros(Out) if include_bias else None

        self.In = In
        self.Out = Out

    @jax.named_scope(name="linear")
    def __call__(self, inputs):
        q = inputs.dot(self.In, self.weight)

        if self.bias is not None:
            q = q + self.bias

        return q
