import json
import logging
import pathlib
from datetime import datetime
from typing import Any, Callable, Optional, Union

import fsspec
import jax
from equinox.serialisation import _is_index, default_deserialise_filter_spec, default_serialise_filter_spec
from fsspec import AbstractFileSystem
from furl import furl
from jaxtyping import PyTree

from levanter.tensorstore_serialization import tree_deserialize_leaves_tensorstore, tree_serialize_leaves_tensorstore


logger = logging.getLogger(__name__)


def save_checkpoint(model, training_state, step: int, checkpoint_path, *, exist_ok=False):
    """
    Save a checkpoint to a given path using TensorStore. If exist_ok is True, the checkpoint
    will be saved even if a checkpoint already exists at the given path.

    If the path does not exist, it will be created.

    If training_state is None, no training state will be saved.

    This method is GlobalDeviceArray-aware, and will save shards in a way that can be restored
    """
    logger.info(f"Saving checkpoint to {checkpoint_path} for step {step}")

    fs: AbstractFileSystem
    fs, _, _ = fsspec.get_fs_token_paths(checkpoint_path)
    fs.makedirs(checkpoint_path, exist_ok=exist_ok)
    tree_serialize_leaves_tensorstore(f"{checkpoint_path}/model", model)
    if training_state is not None:
        tree_serialize_leaves_tensorstore(f"{checkpoint_path}/training_state", training_state)

    metadata = {
        "step": step,
        "timestamp": datetime.now().isoformat(),
    }

    if jax.process_index() == 0:
        with fs.open(f"{checkpoint_path}/metadata.json", "w") as json_out:
            json.dump(metadata, json_out)

    logger.info(f"Saved checkpoint for step {step}")

    return checkpoint_path


def load_checkpoint(model, training_state, checkpoint_path, *, discover_latest=True):
    """
    Load a checkpoint from a given path.

    Returns the loaded model state, training state, and step. If discover_latest is True,
    the latest checkpoint in the given path will be loaded. Otherwise, the checkpoint at
    the given path will be loaded. If no checkpoint is found, returns None

    If training_state is None, no training state will be loaded.
    """
    fs: AbstractFileSystem
    fs, _, _ = fsspec.get_fs_token_paths(str(checkpoint_path))

    if discover_latest:
        checkpoint_path = discover_latest_checkpoint(checkpoint_path)

    if checkpoint_path is None:
        return None

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    with fs.open(f"{checkpoint_path}/metadata.json") as metadata_in:
        metadata = json.load(metadata_in)

    model = tree_deserialize_leaves_tensorstore(f"{checkpoint_path}/model", model)

    if training_state is None:
        training_state = None
    else:
        training_state = tree_deserialize_leaves_tensorstore(f"{checkpoint_path}/training_state", training_state)

    return model, training_state, metadata["step"]


def discover_latest_checkpoint(checkpoint_path) -> Optional[str]:
    """
    Discover the latest checkpoint in a given path.
    """
    # need to use fsspec for this, as glob.glob doesn't work on gs://
    fs: AbstractFileSystem
    checkpoint_path = str(checkpoint_path)
    fs, _, _ = fsspec.get_fs_token_paths(checkpoint_path)
    ckpt_dirs = [d for d in fs.glob(f"{checkpoint_path}/*") if fs.isdir(d)] + [checkpoint_path]
    ckpt_dirs = [d[:-1] if d.endswith("/") else d for d in ckpt_dirs]
    ckpt_dirs = [d for d in ckpt_dirs if fs.exists(f"{d}/metadata.json")]

    def checkpoint_timestamp(ckpt_dir):
        metadata = json.load(fs.open(f"{ckpt_dir}/metadata.json"))
        return datetime.fromisoformat(metadata["timestamp"])

    if len(ckpt_dirs) > 0:
        out = max(ckpt_dirs, key=checkpoint_timestamp)
        logger.info(f"Discovered latest checkpoint from {checkpoint_path} at {out}")
        return out
    else:
        logger.warning(f"No checkpoints found in {checkpoint_path}")
        return None


def tree_serialise_leaves(
    path: Union[str, furl, pathlib.Path],
    pytree: PyTree,
    filter_spec=default_serialise_filter_spec,
    is_leaf: Callable[[Any], bool] = _is_index,
) -> None:
    """Analog to `equinox.tree_serialise_leaves`, but saves the leaves of a PyTree using fsspec."""

    with fsspec.open(str(path), "wb") as f:
        logger.info(f"Serializing to {path}")

        def _serialise(spec, x):
            def __serialise(y):
                spec(f, y)
                return y

            jax.tree_map(__serialise, x, is_leaf=is_leaf)

        jax.tree_map(_serialise, filter_spec, pytree)


def tree_deserialise_leaves(
    path: Union[str, furl, pathlib.Path],
    like: PyTree,
    filter_spec=default_deserialise_filter_spec,
    is_leaf: Callable[[Any], bool] = _is_index,
    fs=None,
) -> PyTree:
    """
    Analog to `equinox.tree_deserialise_leaves`, but loads the leaves of a PyTree using fsspec.
    """

    path = str(path)

    if fs is None:
        fs, _, (path_to_open,) = fsspec.get_fs_token_paths(path)
    else:
        path_to_open = path

    with fs.open(path_to_open, "rb") as f:

        def _deserialise(spec, x):
            def __deserialise(y):
                return spec(f, y)

            return jax.tree_util.tree_map(__deserialise, x, is_leaf=is_leaf)

        out = jax.tree_util.tree_map(_deserialise, filter_spec, like)
    jax.tree_util.tree_map(_assert_same, out, like, is_leaf=is_leaf)
    return out


# similar to eqx but it's a bit more permissive: it just wants things that have shapes and dtypes to be the same
def _assert_same(new, old):
    if hasattr(new, "shape") and hasattr(old, "shape"):
        assert new.shape == old.shape, f"Shapes don't match: {new.shape} vs {old.shape}"
    if hasattr(new, "dtype") and hasattr(old, "dtype"):
        assert new.dtype == old.dtype, f"Dtypes don't match: {new.dtype} vs {old.dtype}"

    # now get mad if one has a shape and the other doesn't
    if hasattr(new, "shape") != hasattr(old, "shape"):
        raise ValueError(f"One has a shape and the other doesn't: {new} vs {old}")
    if hasattr(new, "dtype") != hasattr(old, "dtype"):
        raise ValueError(f"One has a dtype and the other doesn't: {new} vs {old}")
