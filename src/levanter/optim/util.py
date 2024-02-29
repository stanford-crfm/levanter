import equinox as eqx
import jax

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
