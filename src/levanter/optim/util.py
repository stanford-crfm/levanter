import equinox as eqx
import jax


# TODO: filter_jvp?
def hvp(f, x, v):
    """Compute the Hessian-vector product of a function."""
    return jax.jvp(eqx.filter_grad(f), (x,), (v,))[1]


def tree_gaussian(key, tree):
    """Samples a tree of gaussian noise with the same structure as `tree`."""
    leaves, structure = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(key, len(leaves))
    g = jax.tree_util.tree_map(lambda x, key: jax.random.normal(key, x.shape), leaves, list(keys))
    g = jax.tree_util.tree_unflatten(structure, g)

    return g
