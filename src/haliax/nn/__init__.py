import functools

import jax.nn as jnn
import jax.numpy as jnp

from ..core import Axis, NamedArray
from ..wrap import wrap_axiswise_call, wrap_elemwise_unary, wrap_reduction_call
from .dropout import Dropout
from .linear import Linear
from .normalization import LayerNorm


relu = wrap_elemwise_unary(jnn.relu)
relu6 = wrap_elemwise_unary(jnn.relu6)
sigmoid = wrap_elemwise_unary(jnn.sigmoid)
softplus = wrap_elemwise_unary(jnn.softplus)
soft_sign = wrap_elemwise_unary(jnn.soft_sign)
silu = wrap_elemwise_unary(jnn.silu)
swish = wrap_elemwise_unary(jnn.swish)
log_sigmoid = wrap_elemwise_unary(jnn.log_sigmoid)
leaky_relu = wrap_elemwise_unary(jnn.leaky_relu)
hard_sigmoid = wrap_elemwise_unary(jnn.hard_sigmoid)
hard_silu = wrap_elemwise_unary(jnn.hard_silu)
hard_swish = wrap_elemwise_unary(jnn.hard_swish)
hard_tanh = wrap_elemwise_unary(jnn.hard_tanh)
elu = wrap_elemwise_unary(jnn.elu)
celu = wrap_elemwise_unary(jnn.celu)
selu = wrap_elemwise_unary(jnn.selu)
gelu = wrap_elemwise_unary(jnn.gelu)
# TODO: glu = wrap_elemwise_unary(jnn.gelu)

logsumexp = wrap_reduction_call(jnn.logsumexp)

softmax = wrap_axiswise_call(jnn.softmax, False)
# TODO: standardize has optional "mean" and "variance" arguments we need to support
# standardize = wrap_normalization_call(jnn.standardize, False)
log_softmax = wrap_axiswise_call(jnn.log_softmax, False)


@functools.wraps(jnn.one_hot)
def one_hot(x: NamedArray, class_axis: Axis, *, dtype=jnp.float_) -> NamedArray:
    array = jnn.one_hot(x.array, num_classes=class_axis.size, dtype=dtype)
    return NamedArray(array, x.axes + (class_axis,))


__all__ = [
    "relu",
    "relu6",
    "sigmoid",
    "softplus",
    "soft_sign",
    "silu",
    "swish",
    "log_sigmoid",
    "leaky_relu",
    "hard_sigmoid",
    "hard_silu",
    "hard_swish",
    "hard_tanh",
    "elu",
    "celu",
    "selu",
    "gelu",
    "logsumexp",
    "softmax",
    "log_softmax",
    "one_hot",
    "Dropout",
    "LayerNorm",
    "Linear",
]
