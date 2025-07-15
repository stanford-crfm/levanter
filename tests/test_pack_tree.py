import jax
import jax.numpy as jnp
import numpy as np

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "tree_utils", Path(__file__).resolve().parents[1] / "src" / "levanter" / "utils" / "tree_utils.py"
)
assert spec is not None
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
tree_utils = module
pack_pytree = tree_utils.pack_pytree
unpack_pytree = tree_utils.unpack_pytree


def test_pack_and_unpack_simple():
    tree = {"a": np.arange(3, dtype=np.float32), "b": np.arange(4, dtype=np.float32).reshape(2, 2)}
    offsets, packed = pack_pytree(tree, dtype=jnp.float32)
    rebuilt = unpack_pytree(offsets, packed)
    for orig, new in zip(jax.tree_util.tree_leaves(tree), jax.tree_util.tree_leaves(rebuilt)):
        np.testing.assert_array_equal(np.asarray(orig, dtype=np.float32), np.array(new))


def test_pack_empty_tree():
    tree = {}
    offsets, packed = pack_pytree(tree, dtype=jnp.float32)
    assert packed.size == 0
    rebuilt = unpack_pytree(offsets, packed)
    assert rebuilt == tree
