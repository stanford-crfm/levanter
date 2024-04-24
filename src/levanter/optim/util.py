import equinox as eqx
import jax
import jax.numpy as jnp

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

def tree_rademacher_like(key, tree):
    leaves, structure = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(key, len(leaves))
    # paper uses normal but we use rademacher
    # see https://www.ethanepperly.com/index.php/2024/01/28/dont-use-gaussians-in-stochastic-trace-estimation/
    g = jax.tree_util.tree_map(
        lambda key, x: jax.random.rademacher(key, x.shape, dtype=jnp.float32),
        list(keys),
        leaves,
    )
    g = jax.tree_util.tree_unflatten(structure, g)
    return g
