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

    @staticmethod
    def init(In: AxisSpec, Out: AxisSpec, *, key, use_bias=True) -> "Linear":
        joint_spec = hax.concat_axis_specs(In, Out)
        weight = hax.random.normal(key, joint_spec) * 0.02
        bias = hax.zeros(Out) if use_bias else None
        return Linear(weight, bias, In, Out)

    @jax.named_scope(name="linear")
    def __call__(self, inputs):
        q = inputs.dot(self.In, self.weight)
        q = hax.auto_sharded(q)

        if self.bias is not None:
            q = q + self.bias
            q = hax.auto_sharded(q)

        return q
