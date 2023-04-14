# implements the prototype hero optimizer
from typing import Any, List, NamedTuple, Optional, Union

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import typing_extensions
from jax.random import PRNGKey
from optax._src import numerics
from optax._src.transform import bias_correction, update_moment


class ScaleByHeroState(NamedTuple):
    """State for the Adam algorithm."""

    count: chex.Array  # shape=(), dtype=jnp.int32.
    hessian_count: chex.Array  # shape=(), dtype=jnp.int32.
    mu: optax.Updates  # momentum
    h: optax.Updates  # EMA of hessian diagonal


class HessianUpdateFn(typing_extensions.Protocol):
    """A callable type for the `update` step of a `GradientTransformation`.

    The `update` step takes a tree of candidate parameter `updates` (e.g. their
    gradient with respect to some loss), an arbitrary structured `state`, and the
    current `params` of the model being optimised. The `params` argument is
    optional, it must however be provided when using transformations that require
    access to the current values of the parameters.
    """

    def __call__(
        self,
        hessian: optax.Updates,
        state: optax.OptState,
    ) -> optax.OptState:
        """Returns the updated `state` given the `hessian` and `state`."""
        pass


class SecondOrderTransformation(NamedTuple):
    """A pair of pure functions implementing a second order gradient transformation."""

    init: optax.TransformInitFn
    update: optax.TransformUpdateFn
    hessian_update: HessianUpdateFn


AnySecondOrderTransformation = Union[SecondOrderTransformation, optax.GradientTransformation]


# cf https://arxiv.org/pdf/2006.00719.pdf eqn 9
# https://www-users.cse.umn.edu/~saad/PDF/umsi-2005-082.pdf
# https://arxiv.org/pdf/2208.03268.pdf
def stochastic_hessian_diagonal(fn, model, *args, g_key: PRNGKey, **kwargs):
    """Compute the diagonal of the Hessian of a function using a normal distribution.

    Args:
        fn: function to compute the Hessian of
        model: model to compute the Hessian of
        g_key: key for the normal distribution
    """
    g = tree_gaussian(g_key, model)

    grad_fn = eqx.filter_grad(fn)

    # todo:use def hvp(f, x, v):
    #   return jvp(grad(f), (x,), (v,))[1]
    # TODO: consider allowing for n > 1 gaussians

    def gaussian_something(model, *args, **kwargs):
        grads = grad_fn(model, *args, **kwargs)
        # dot grads with gaussian
        leaves = jax.tree_util.tree_leaves(grads)
        g_leaves = jax.tree_util.tree_leaves(g)
        grad_dot = sum(
            [jnp.sum(jnp.multiply(g_leaf, leaf)) for g_leaf, leaf in zip(g_leaves, leaves)],
            jnp.zeros([], dtype=jnp.float32),
        )
        return grad_dot

    grad_g = eqx.filter_grad(gaussian_something)(model, *args, **kwargs)
    hessian = jax.tree_util.tree_map(lambda grad, gaussian: grad * gaussian, grad_g, g)

    return hessian


def tree_gaussian(key, tree):
    leaves, structure = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(key, len(leaves))
    g = jax.tree_util.tree_map(lambda x, key: jax.random.normal(key, x.shape), leaves, list(keys))
    g = jax.tree_util.tree_unflatten(structure, g)

    return g


def chain_second_order(*args: AnySecondOrderTransformation) -> SecondOrderTransformation:
    """Applies a list of chainable update transformations. Analogous to optax.chain,
    but for second order transformations.
    """

    init_fns = []
    update_fns = []
    hessian_update_fns: List[Optional[HessianUpdateFn]] = []

    for arg in args:
        if isinstance(arg, SecondOrderTransformation):
            init_fns.append(arg.init)
            update_fns.append(arg.update)
            hessian_update_fns.append(arg.hessian_update)
        else:
            init_fns.append(arg.init)
            update_fns.append(arg.update)
            hessian_update_fns.append(None)

    def init_fn(params):
        return tuple(fn(params) for fn in init_fns)

    def update_fn(updates, state, params=None):
        if len(update_fns) != len(state):
            raise ValueError(
                "The number of updates and states has to be the same in chain! Make sure you have called init first!"
            )

        new_state = []
        for s, fn in zip(state, update_fns):
            updates, new_s = fn(updates, s, params)
            new_state.append(new_s)
        return updates, tuple(new_state)

    def hessian_update_fn(new_hessian, state, params=None):
        if len(hessian_update_fns) != len(state):
            raise ValueError(
                "The number of updates and states has to be the same in chain! Make sure you have called init first!"
            )

        new_state = []
        for s, fn in zip(state, hessian_update_fns):
            if fn is None:
                new_state.append(s)
            else:
                new_s = fn(new_hessian, s, params)
                new_state.append(new_s)
        return tuple(new_state)

    return SecondOrderTransformation(init_fn, update_fn, hessian_update_fn)


def scale_by_hero(
    b1: float = 0.95,
    b2: float = 0.99,
    eps: float = 1e-12,
    gamma: float = 0.1,
    mu_dtype: Optional[Any] = None,
) -> SecondOrderTransformation:
    mu_dtype = jax.canonicalize_dtype(mu_dtype) if mu_dtype is not None else None

    def init_fn(params):
        mu = jax.tree_util.tree_map(lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)  # First moment
        h = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
        return ScaleByHeroState(count=jnp.zeros([], jnp.int32), hessian_count=jnp.zeros([], jnp.int32), mu=mu, h=h)

    def update_fn(updates, state, params=None):
        del params
        mu = update_moment(updates, state.mu, b1, 1)
        # nu = update_moment_per_elem_norm(updates, state.nu, b2, 2)
        count_inc = numerics.safe_int32_increment(state.count)
        mu_hat = bias_correction(mu, b1, count_inc)
        h_hat = state.h
        # TODO: Use slightly lower learning rate for adam, e.g. 0.85 * adam_lr
        # TODO: monitor param norm and momentum norm and trace(hessian) (aka sum of h_hat)
        # TODO: also track how often hessian is used (per coordinate)
        # TODO: track sum( jnp.abs(m) > gamma * jnp.max(jnp.abs(m)) for m in mu_hat), we expect this to be ~70% later in training
        # TODO: track time for hessian computation
        # TODO: 10% update hessian
        updates = jax.tree_util.tree_map(
            lambda m, v: m / jnp.maximum(jnp.maximum(jnp.abs(m), gamma * jnp.maximum(v, 0)), eps), mu_hat, h_hat
        )
        if mu_dtype is not None:
            mu = jax.tree_util.tree_map(lambda t: t.astype(mu_dtype), mu)
        return updates, ScaleByHeroState(count=count_inc, hessian_count=state.hessian_count, mu=mu, h=h_hat)

    def update_hessian(hessian, state, params=None):
        del params
        hessian_count_inc = numerics.safe_int32_increment(state.hessian_count)
        nu = update_moment(hessian, state.h, b2, 1)
        # h_hat = bias_correction(nu, b2, hessian_count_inc)
        return ScaleByHeroState(count=state.count, hessian_count=hessian_count_inc, mu=state.mu, h=nu)

    return SecondOrderTransformation(init_fn, update_fn, update_hessian)


def hero(
    lr: float,
    b1: float = 0.95,
    b2: float = 0.99,
    eps: float = 1e-12,
    gamma: float = 0.1,
    mu_dtype: Optional[Any] = None,
) -> SecondOrderTransformation:
    return chain_second_order(scale_by_hero(b1, b2, eps, gamma, mu_dtype), optax.scale(lr))
