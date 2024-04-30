import functools
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional, TypeVar

import equinox as eqx
import jax
import jaxtyping
import optax
from jax import numpy as jnp
from jax.random import PRNGKey
from jaxtyping import PRNGKeyArray

import levanter.tracker
from levanter.optim.config import OptimizerConfig
from levanter.optim.sophia import SophiaHConfig
from levanter.optim.util import hvp, tree_gaussian_like, tree_rademacher_like
from levanter.utils.jax_utils import parameter_count, tree_filter_like


M = TypeVar("M")
Ex = TypeVar("Ex")

GAMMA_SOPHIA_G = 0.05
GAMMA_SOPHIA_H = 0.01


class ScaleBySophiaState(NamedTuple):
    """State for Sophia and similar."""

    count: jaxtyping.Array  # shape=(), dtype=jnp.int32.
    weight_sum: jaxtyping.Array  # shape=(), dtype=jnp.float32
    hessian_count: jaxtyping.Array  # shape=(), dtype=jnp.int32.
    z: optax.Params  # primary iterates in the sf algorithm. this replaces mu
    h: optax.Updates  # EMA of hessian diagonal
    hess_key: PRNGKey


# @OptimizerConfig.register_subclass("sophia-g")
# @dataclass
# class SophiaGConfig(BaseSophiaConfig):
#     gamma: float = GAMMA_SOPHIA_G
#
#     def compute_hessian(self, fn, model, *batch, hess_key: PRNGKey, **batch_kwargs):
#         return stochastic_diag_gauss_newton(fn, model, *batch, **batch_kwargs, hess_key=hess_key)
#


@OptimizerConfig.register_subclass("sf-sophia-h")
@dataclass
class ScheduleFreeSophiaHConfig(SophiaHConfig):
    weight_lr_power: float = 2.0

    def build(self, num_train_steps: int):
        def _optimizer(gamma) -> optax.GradientTransformation:
            key = jax.random.PRNGKey(self.rng_seed)

            return _sophia_gradient_transform(
                sophia_hess_fn=self.compute_hessian,
                update_interval=self.update_interval,
                b1=self.beta1,
                b2=self.beta2,
                eps=self.epsilon,
                gamma=gamma,
                weight_decay=self.weight_decay,
                learning_rate=self.learning_rate,
                weight_lr_power=self.weight_lr_power,
                warmup_steps=self._convert_warmup(num_train_steps),
                initial_key=key,
                clip_threshold=self.clip_threshold,
            )

        # Hong suggested using cosine decay for gamma
        # gamma_decay_schedule = optax.cosine_decay_schedule(self.gamma, num_train_steps // 2, 0)  # type: ignore
        constant_gamma_schedule = optax.constant_schedule(self.gamma)  # type: ignore
        # gamma_schedule = optax.join_schedules([constant_gamma_schedule, gamma_decay_schedule], [num_train_steps // 2])

        # In schedule-free, learning rate and weight decay are applied internally
        # ignore self.lr_scheduler because it's schedule-free :)
        return optax.inject_hyperparams(_optimizer)(gamma=constant_gamma_schedule)


# def sophia_h(
#     lr: float = 0.85e-3,
#     *,
#     b1: float = 0.965,
#     b2: float = 0.99,
#     eps: float = 1e-8,
#     gamma: float = GAMMA_SOPHIA_H,
#     weight_decay: float = 0.0,
#     clip_threshold: Optional[float] = 1.0,
#     update_interval: int = 10,
#     key: PRNGKey,
# ) -> optax.GradientTransformation:
#     """Sophia-H: https://arxiv.org/pdf/2305.14342.pdf Algorithm 1&3"""
#     components = []

#     components.append(scale_by_sophia_h(b1, b2, eps, gamma, clip_threshold, update_interval, key=key))

#     if weight_decay > 0:
#         components.append(optax.add_decayed_weights(weight_decay))

#     components.append(optax.scale(-lr))

#     return optax.chain(*components)


# def scale_by_sophia_h(
#     b1=0.965,
#     b2=0.99,
#     eps=1e-8,
#     gamma=GAMMA_SOPHIA_H,
#     clip_threshold: Optional[float] = 1.0,
#     update_interval=10,
#     *,
#     key: PRNGKey,
# ):

#     return _sophia_gradient_transform(
#         sophia_hess_fn=stochastic_hessian_diagonal,
#         update_interval=update_interval,
#         b1=b1,
#         b2=b2,
#         eps=eps,
#         gamma=gamma,
#         clip_threshold=clip_threshold,
#         initial_key=key,
#     )


# def sophia_g(
#     lr: float = 1e-3,
#     *,
#     b1: float = 0.99,
#     b2: float = 0.99,
#     eps: float = 1e-8,
#     gamma: float = GAMMA_SOPHIA_G,
#     weight_decay: float = 0.0,
#     clip_threshold: Optional[float] = 1.0,
#     update_interval: int = 10,
#     key: PRNGKey,
# ) -> optax.GradientTransformation:
#     """Sophia-G: https://arxiv.org/pdf/2305.14342.pdf Algorithm 2&3"""
#     components = []

#     components.append(scale_by_sophia_g(b1, b2, eps, gamma, clip_threshold, update_interval, key=key))

#     if weight_decay > 0:
#         components.append(optax.add_decayed_weights(weight_decay))

#     components.append(optax.scale(-lr))

#     return optax.chain(*components)


# def scale_by_sophia_g(
#     b1: float = 0.99,
#     b2: float = 0.99,
#     eps: float = 1e-8,
#     gamma: float = GAMMA_SOPHIA_G,
#     clip_threshold: Optional[float] = 1.0,
#     update_interval=10,
#     *,
#     key: PRNGKeyArray,
# ):

#     return _sophia_gradient_transform(
#         sophia_hess_fn=stochastic_diag_gauss_newton,
#         update_interval=update_interval,
#         b1=b1,
#         b2=b2,
#         eps=eps,
#         gamma=gamma,
#         clip_threshold=clip_threshold,
#         initial_key=key,
#     )


def _sophia_gradient_transform(
    sophia_hess_fn,
    update_interval: int,
    b1: float,
    b2: float,
    eps: float,
    gamma: float,
    weight_decay: float,
    learning_rate: float,
    weight_lr_power: float,
    warmup_steps: int,
    clip_threshold: Optional[float],
    initial_key: PRNGKeyArray,
    mu_dtype: Optional[Any] = None,
) -> optax.GradientTransformation:
    mu_dtype = jax.canonicalize_dtype(mu_dtype) if mu_dtype is not None else None

    def init_fn(params):
        z = jax.tree_util.tree_map(jnp.copy, params)  # schedule-free z
        h = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
        return ScaleBySophiaState(
            count=jnp.zeros([], jnp.int32),
            weight_sum=jnp.zeros([], jnp.float32),
            hessian_count=jnp.zeros([], jnp.int32),
            z=z,
            h=h,
            hess_key=initial_key,
        )

    def update_fn(updates, state, params=None, *, obj_fn, **kwargs):
        # mu = update_moment(updates, state.mu, b1, 1)
        # nu = update_moment_per_elem_norm(updates, state.nu, b2, 2)
        # mu_hat = bias_correction(mu, b1, state.count + 1)

        z = state.z
        t = state.count
        weight_sum = state.weight_sum

        sched = jnp.minimum((t + 1) / warmup_steps, 1.0)

        lr = learning_rate * sched
        weight = lr**weight_lr_power
        new_weight_sum = weight_sum + weight

        try:
            ckp1 = weight / new_weight_sum  # converges to 1/t if no warmup
        except ZeroDivisionError:
            ckp1 = 0

        # bias_correction2 = 1 - b2 ** (t + 1)
        # learning_rate = learning_rate * bias_correction2**0.5

        h_hat = state.h
        # track how often hessian is used
        # mu_leaves = jax.tree_util.tree_leaves(mu_hat)
        h_leaves = jax.tree_util.tree_leaves(h_hat)

        stats: dict[str, Any] = {
            "optim/param_norm": jnp.sqrt(sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params))),
            # "optim/momentum_norm": jnp.sqrt(sum(jnp.sum(m**2) for m in mu_leaves)),
            "optim/hessian_norm": jnp.sqrt(sum(jnp.sum(h**2) for h in h_leaves)),
            "optim/learning_rate": lr,
        }

        # with sophia-g the max(h, 0) is not needed but no harm
        updates = jax.tree_util.tree_map(
            lambda g, h: g / jnp.maximum(gamma * h, eps),
            updates,
            h_hat,
        )

        if clip_threshold is not None:
            # setting to float32 for overflow
            unclipped_count = sum(
                jnp.sum(jnp.abs(u) < clip_threshold).astype(jnp.float32) for u in jax.tree_util.tree_leaves(updates)
            )
            updates = jax.tree_util.tree_map(
                lambda u, y: jnp.clip(u, -clip_threshold, clip_threshold) + weight_decay * y, updates, params
            )
            stats["optim/unclipped_fraction"] = unclipped_count * 1.0 / float(parameter_count(updates))

        levanter.tracker.jit_log_metrics(stats, step=t)

        if mu_dtype is not None:
            z = jax.tree_util.tree_util.tree_map(lambda t: t.astype(mu_dtype), z)

        # update y
        new_y = jax.tree_util.tree_map(
            lambda y, z, u: (1 - ckp1) * y + ckp1 * z + lr * (b1 * (1 - ckp1) - 1) * u,
            params,
            z,
            updates,
        )
        # update z
        new_z = jax.tree_util.tree_map(lambda z, u: z - lr * u, z, updates)
        # get actual updates for y
        updates = jax.tree_map(lambda new_y, y: new_y - y, new_y, params)

        state = ScaleBySophiaState(
            count=t + 1,
            weight_sum=new_weight_sum,
            hessian_count=state.hessian_count,
            z=new_z,
            h=h_hat,
            hess_key=state.hess_key,
        )
        state = update_hessian(state, params, obj_fn=obj_fn, **kwargs)
        return updates, state

    def update_hessian(state, params, *, obj_fn, **kwargs):
        def _do_update():
            key, next_key = jax.random.split(state.hess_key)
            new_hess = sophia_hess_fn(obj_fn, params, hess_key=key, **kwargs)

            new_hess = tree_filter_like(state.h, new_hess)

            # EMAs of hessian
            nu = update_moment(new_hess, state.h, b2, 1)
            return ScaleBySophiaState(
                count=state.count,
                weight_sum=state.weight_sum,
                hessian_count=state.hessian_count + 1,
                z=state.z,
                h=nu,
                hess_key=next_key,
            )

        def _dont_update():
            return state

        return jax.lax.cond(
            jnp.equal(state.count % update_interval, 0),
            lambda _: _do_update(),
            lambda _: _dont_update(),
            state.count,
        )

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


# use this for Sophia-G
def stochastic_diag_gauss_newton(fn, model, *args, hess_key: PRNGKey, **kwargs):
    """

    Approximate the diagonal of the Hessian using an approximation to the Gauss Newton matrix.
    This is Algorithm 2 of https://arxiv.org/pdf/2305.14342.pdf

    Args:
        fn (SophiaGObjective): objective function
        model: model whose Hessian to compute
        hess_key: key for sampling
        *args, **kwargs: passed to fn's logits
    """
    raise NotImplementedError("This is not implemented yet")
    # if not isinstance(fn, SophiaGObjective):
    #     raise ValueError("objective must be a SophiaGObjective")

    # Step 3
    logits, model_backward = eqx.filter_vjp(lambda model: fn.logits(model, *args, **kwargs), model)

    # Step 4
    y_hat = fn.sample(logits, key=hess_key)

    # Step 5
    grad_loss_logits = eqx.filter_grad(fn.loss)(logits, y_hat)
    pseudo_g = model_backward(grad_loss_logits)[0]

    # Step 6
    bs = fn.num_data_points()
    h = jax.tree_util.tree_map(lambda x: x**2 * bs, pseudo_g)

    return h


# Use this for Sophia-H
def stochastic_hessian_diagonal(fn, model, *args, hess_key: PRNGKey, rand: str, **kwargs):
    """Compute the diagonal of the Hessian of a function using a normal distribution.

    https://arxiv.org/pdf/2305.14342.pdf Algorithm 1

    Args:
        fn: function to compute the Hessian of
        model: model to compute the Hessian of
        hess_key: key for the normal distribution
    """
    # cf https://arxiv.org/pdf/2006.00719.pdf eqn 9
    # https://www-users.cse.umn.edu/~saad/PDF/umsi-2005-082.pdf
    # https://arxiv.org/pdf/2208.03268.pdf
    if rand == "rademacher":
        g = tree_rademacher_like(hess_key, model)
    else:
        g = tree_gaussian_like(hess_key, model)
    # TODO: consider allowing for n > 1 gaussians?
    product = hvp(lambda m: fn(m, *args, **kwargs), model, g)
    hessian = jax.tree_util.tree_map(lambda grad, gaussian: grad * gaussian, product, g)

    return hessian


# Cribbed from optax._src.transform
def update_moment(updates, moments, decay, order):
    """Compute the exponential moving average of the `order`-th moment."""
    return jax.tree_util.tree_map(lambda g, t: (1 - decay) * (g**order) + decay * t, updates, moments)


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
