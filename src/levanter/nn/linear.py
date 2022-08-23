from typing import List, Optional

import equinox as eqx
import jmp

import haliax as hax
from haliax import Axis, NamedArray
from levanter.modeling_utils import named_call


class NamedLinear(eqx.Module):
    weight: NamedArray
    bias: Optional[NamedArray]

    in_axis: Axis = eqx.static_field()
    out_axis: Axis = eqx.static_field()
    mp: jmp.Policy = eqx.static_field()

    def __init__(self, in_axis: Axis, out_axis: Axis, *, key, mp: jmp.Policy, include_bias=True):
        self.weight = hax.random.normal(key, (in_axis, out_axis), dtype=mp.param_dtype) * 0.02
        if include_bias:
            self.bias = hax.zeros(out_axis, dtype=mp.param_dtype)
        else:
            self.bias = None

        self.in_axis = in_axis
        self.out_axis = out_axis
        self.mp = mp

    @named_call(name="linear")
    def __call__(self, inputs):
        # TODO: actually take and output named arrays
        # out = inputs.dot(self.in_axis, self.weight)

        kernel = self.weight.array
        q = inputs @ kernel
        q = self.mp.cast_to_compute(q)

        if self.bias is not None:
            q = q + self.bias.array
            q = self.mp.cast_to_compute(q)

        return q

    def torch_key_leaves(self, prefix: Optional[str] = None):
        return _apply_prefix(prefix, ["weight", "bias"] if self.bias is not None else ["weight"])


def _apply_prefix(prefix: Optional[str], leaves: List[Optional[str]]) -> List[Optional[str]]:
    if prefix is None:
        return leaves
    else:
        return [prefix + "." + leaf if leaf else prefix for leaf in leaves]
