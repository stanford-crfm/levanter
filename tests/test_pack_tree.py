import importlib.util
from importlib.machinery import ModuleSpec
import jax.numpy as jnp
import jax
import numpy as np
from pathlib import Path

TREE_UTILS_PATH = Path(__file__).resolve().parents[1] / "src" / "levanter" / "utils" / "tree_utils.py"
spec: ModuleSpec | None = importlib.util.spec_from_file_location("tree_utils", str(TREE_UTILS_PATH))
assert spec is not None and spec.loader is not None
tree_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tree_utils)
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
