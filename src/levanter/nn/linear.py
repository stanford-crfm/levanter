from typing import List, Optional, Sequence, Tuple, TypeVar, Union

import equinox as eqx
import jmp

import haliax as hax
from haliax import AxisSpec, NamedArray
from levanter.jax_utils import named_call


class NamedLinear(eqx.Module):
    weight: NamedArray
    bias: Optional[NamedArray]

    In: AxisSpec = eqx.static_field()
    Out: AxisSpec = eqx.static_field()
    mp: jmp.Policy = eqx.static_field()

    def __init__(self, In: AxisSpec, Out: AxisSpec, *, key, mp: jmp.Policy, include_bias=True):
        joint_spec = hax.concat_axis_specs(In, Out)
        self.weight = hax.random.normal(key, joint_spec, dtype=mp.param_dtype) * 0.02
        self.bias = hax.zeros(Out, dtype=mp.param_dtype) if include_bias else None

        self.In = In
        self.Out = Out
        self.mp = mp

    @named_call(name="linear")
    def __call__(self, inputs):
        q: NamedArray = inputs.dot(self.In, self.weight)
        q = self.mp.cast_to_compute(q)
        if self.bias is not None:
            # TODO: add support for binary ops like addition to named axes and broadcasting
            out_axis = _ensure_tuple(self.Out)
            arr = q.rearrange((...,) + out_axis).array + self.bias.array
            arr = self.mp.cast_to_output(arr)
            q = NamedArray(arr, q.axes)

        return q

    def torch_key_leaves(self, prefix: Optional[str] = None):
        return _apply_prefix(prefix, ["weight", "bias"] if self.bias is not None else ["weight"])


def _apply_prefix(prefix: Optional[str], leaves: List[Optional[str]]) -> List[Optional[str]]:
    if prefix is None:
        return leaves
    else:
        return [prefix + "." + leaf if leaf else prefix for leaf in leaves]


T = TypeVar("T")


def _ensure_tuple(x: Union[Sequence[T], T]) -> Tuple[T, ...]:
    if isinstance(x, Sequence):
        return tuple(x)
    return (x,)
