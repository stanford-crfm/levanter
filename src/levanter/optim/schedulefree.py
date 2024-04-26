"""
Mostly ported from https://github.com/evanatyourservice/sophia-schedulefree-jax/tree/main
"""

from typing import Any, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

import optax
from optax._src import (
    utils,
)


class ScheduleFreeState(NamedTuple):
    x: optax.Params
    z: optax.Params
    t: jax.Array
    base_optimizer_state: optax.OptState # ?


def schedule_free(
    base_optimizer: optax.GradientTransformation,
    beta: float,
    mu_dtype: Optional[Any] = None,
):
    mu_dtype = jax.canonicalize_dtype(mu_dtype) if mu_dtype is not None else None

    def init_fn(params: optax.Params) -> ScheduleFreeState:
        return ScheduleFreeState(
            x=jax.tree_map(lambda x: x.astype(mu_dtype), params),
            z=jax.tree_map(lambda x: x.astype(mu_dtype), params),
            t=jnp.zeros([], jnp.int32),
            base_optimizer_state=base_optimizer.init(params),
        )

    def update_fn(
        updates: optax.Updates,
        opt_state: ScheduleFreeState,
        params: optax.Params,
        *args,
        **kwargs,
    ) -> Tuple[optax.Updates, ScheduleFreeState]:
        x_curr = opt_state.x
        z_curr = opt_state.z
        t = opt_state.t + 1

        z_updates, base_optimizer_state = base_optimizer.update(
            updates, opt_state.base_optimizer_state, params, *args, **kwargs
        )
        z_next = optax.apply_updates(z_curr, z_updates)
        x_next = jax.tree_map(  # he wrote jax.tree.map here
            lambda x, z: x * (1 - 1 / t) + z * (1 / t), x_curr, z_next
        )
        y_next = jax.tree_map(
            lambda x, z: x * beta + z * (1 - beta), x_next, z_next)
        )
        updates = jax.tree_map(
            lambda y_dash, y: y_dash - y, y_next, params
        )

        opt_state = opt_state._replace(
            x=x_next, z=z_next, t=t, base_optimizer_state=base_optimizer_state
        )
        return updates, opt_state

    return optax.GradientTransformation(init_fn, update_fn)