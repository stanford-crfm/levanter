import equinox as eqx
import jax
import optax.tree_utils
from optax import GradientTransformation, GradientTransformationExtraArgs
from optax._src.base import init_empty_state

import haliax as hax
from haliax.tree_util import scan_aware_tree_map

import levanter.tracker
from levanter.utils.jax_utils import is_inexact_arrayish


def hvp(f, x, v):
    """Compute the Hessian-vector product of a function."""
    return eqx.filter_jvp(eqx.filter_grad(f), (x,), (v,))[1]


def tree_gaussian_like(key, tree):
    """
    Samples a tree of gaussian noise with the same structure as `tree`, except for leaves which are not inexact arrays,
    for which it returns None
    """
    leaves, structure = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(key, len(leaves))
    rand_n = lambda x, key: jax.random.normal(key, x.shape) if is_inexact_arrayish(x) else None
    g = jax.tree_util.tree_map(rand_n, leaves, list(keys))
    g = jax.tree_util.tree_unflatten(structure, g)

    return g


def log_norm_passthrough(desc: str) -> GradientTransformation:
    """
    Creates a gradient transformation that logs the L2 norm of the updates
    and returns the updates unchanged.
    """

    def init_fn(params):
        return None

    def update_fn(updates, state, params, **extra_args):
        levanter.tracker.jit_log({desc: optax.tree_utils.tree_l2_norm(updates)})
        return updates, None

    return GradientTransformationExtraArgs(init_fn, update_fn)


def scan_aware_clip_by_block_rms(threshold: float) -> GradientTransformation:
    """
    Version of `optax.clip_by_block_rms` that is aware of scan layers
    """

    def update_fn(updates, state, params=None, **extra_args):
        del params

        def _clip_fn(u):
            clip_denom = hax.maximum(1.0, hax.sqrt(hax.mean(u * u)) / threshold)
            return u / clip_denom

        updates = scan_aware_tree_map(_clip_fn, updates)
        return updates, state

    return GradientTransformation(init_empty_state, update_fn)
