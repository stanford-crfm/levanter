import functools
import math
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional, TypeVar

import jax
import jaxtyping
import optax
from jax import numpy as jnp

import levanter.tracker
from levanter.optim.config import AdamConfig, OptimizerConfig


M = TypeVar("M")
Ex = TypeVar("Ex")


class ScaleByAdamState(NamedTuple):
    """State for Sophia and similar."""

    count: jaxtyping.Array  # shape=(), dtype=jnp.int32.
    weight_sum: jaxtyping.Array  # shape=(), dtype=jnp.float32
    z: optax.Params  # primary iterates in the sf algorithm. this replaces mu
    nu: optax.Params  # EMA of square


@OptimizerConfig.register_subclass("sf-adam")
@dataclass
class ScheduleFreeAdamConfig(AdamConfig):
    weight_lr_power: float = 2.0

    def build(self, num_train_steps: int):

        return _adam_gradient_transform(
            b1=self.beta1,
            b2=self.beta2,
            eps=self.epsilon,
            weight_decay=self.weight_decay,
            learning_rate=self.learning_rate,
            weight_lr_power=self.weight_lr_power,
            warmup_steps=self._convert_warmup(num_train_steps),
        )


def _adam_gradient_transform(
    b1: float,
    b2: float,
    eps: float,
    weight_decay: float,
    learning_rate: float,
    weight_lr_power: float,
    warmup_steps: int,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    *,
    nesterov: bool = False,
) -> optax.GradientTransformation:
    """Rescale updates according to the Adam algorithm.

    References:
      Kingma et al, `Adam: A Method for Stochastic Optimization
      <https://arxiv.org/abs/1412.6980>`_, 2014

      Dozat, `Incorporating Nesterov Momentum into Adam
      <https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ>`_ 2016

    .. warning::
      PyTorch and optax's adam follow Algorithm 1 of the Kingma
      and Ba's Adam paper, if reproducing old results note that TensorFlow
      used instead the formulation just before Section 2.1 of the paper.
      See https://github.com/deepmind/optax/issues/571 for more detail.

    Args:
      b1: Decay rate for the exponentially weighted average of grads.
      b2: Decay rate for the exponentially weighted average of squared grads.
      eps: Term added to the denominator to improve numerical stability.
      eps_root: Term added to the denominator inside the square-root to improve
        numerical stability when backpropagating gradients through the rescaling.
      mu_dtype: Optional `dtype` to be used for the first order accumulator; if
        `None` then the `dtype` is inferred from `params` and `updates`.
      nesterov: Whether to use Nesterov momentum. The variant of Adam with
        Nesterov momentum is described in [Dozat 2016]

    Returns:
      A `GradientTransformation` object.
    """

    mu_dtype = jax.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        z = jax.tree_util.tree_map(jnp.copy, params)  # schedule-free z
        nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
        return ScaleByAdamState(
            count=jnp.zeros([], jnp.int32),
            weight_sum=jnp.zeros([], jnp.float32),
            z=z,
            nu=nu,
        )

    def update_fn(updates, state, params=None):
        z = state.z
        t = state.count
        weight_sum = state.weight_sum

        sched = jnp.minimum((t + 1) / warmup_steps, 1.0)

        bias_correction2 = 1 - b2 ** (t + 1)

        lr = learning_rate * sched * math.sqrt(bias_correction2)
        weight = lr**weight_lr_power
        new_weight_sum = weight_sum + weight

        try:
            ckp1 = weight / new_weight_sum  # converges to 1/t if no warmup
        except ZeroDivisionError:
            ckp1 = 0

        stats: dict[str, Any] = {
            "optim/param_norm": jnp.sqrt(sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params))),
            "optim/learning_rate": lr,
        }
        levanter.tracker.jit_log_metrics(stats, step=t)

        updates = jax.tree_utils.tree_map(lambda grad, y: grad + weight_decay * y, updates, params)
        nu = optax.tree_utils.tree_update_moment_per_elem_norm(updates, state.nu, b2, 2)
        # # this is from optax https://github.com/google-deepmind/optax/blob/main/optax/_src/transform.py#L317
        # if nesterov:
        #   mu_hat = jtu.tree_map(
        #       lambda m, g: b1 * m + (1 - b1) * g,
        #       otu.tree_bias_correction(
        #           mu, b1, numerics.safe_int32_increment(count_inc)),
        #       otu.tree_bias_correction(updates, b1, count_inc))
        # else:
        #   mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
        # # Dozat 2016 https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ
        # # Algorithm 2 further multiplies Adam's standard nu_hat by b2. It is
        # # unclear why. Other Nadam implementations also omit the extra b2 factor.
        # nu_hat = otu.tree_bias_correction(nu, b2, count_inc)

        denom = jax.tree_utils.tree_map(lambda nu: jnp.sqrt(nu) + eps, nu)
        grad_normalized = jax.tree_utils.tree_map(lambda grad, denom, y: grad / denom, updates, denom, params)

        if mu_dtype is not None:
            z = jax.tree_util.tree_util.tree_map(lambda t: t.astype(mu_dtype), z)

        # update y
        new_y = jax.tree_util.tree_map(
            lambda y, z, u: (1 - ckp1) * y + ckp1 * z + lr * (b1 * (1 - ckp1) - 1) * u,
            params,
            z,
            grad_normalized,
        )
        # update z
        new_z = jax.tree_util.tree_map(lambda z, u: z - lr * u, z, grad_normalized)
        # get actual updates for y
        updates = jax.tree_util.tree_map(lambda new_y, y: new_y - y, new_y, params)

        return updates, ScaleByAdamState(count=t + 1, weight_sum=grad_normalized, z=new_z, nu=nu)

    return optax.GradientTransformation(init_fn, update_fn)


@functools.partial(jax.jit, inline=True)
def bias_correction(moment, decay, count):
    """Performs bias correction. It becomes a no-op as count goes to infinity."""
    # The conversion to the data type of the moment ensures that bfloat16 remains
    # bfloat16 in the optimizer state. This conversion has to be done after
    # `bias_correction_` is calculated as calculating `decay**count` in low
    # precision can result in it being rounded to 1 and subsequently a
    # "division by zero" error.
    bias_correction_ = 1 - decay**count

    # Perform division in the original precision.
    return jax.tree_util.tree_map(lambda t: t / bias_correction_.astype(t.dtype), moment)
