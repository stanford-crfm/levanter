import json
import pathlib
from datetime import datetime
from typing import Any, Callable, Optional, Union

import fsspec
import jax
from equinox.custom_types import PyTree
from equinox.serialisation import (
    _assert_same,
    _is_index,
    default_deserialise_filter_spec,
    default_serialise_filter_spec,
)
from fsspec import AbstractFileSystem


def save_checkpoint(model, training_state, step: int, checkpoint_path, *, exist_ok=False):
    """
    Save a checkpoint to a given path.

    If the path does not exist, it will be created.

    """
    if jax.process_index() != 0:
        return checkpoint_path

    fs: AbstractFileSystem
    fs, _, _ = fsspec.get_fs_token_paths(checkpoint_path)
    fs.makedirs(checkpoint_path, exist_ok=exist_ok)
    tree_serialise_leaves(f"{checkpoint_path}/model.eqx", model)
    tree_serialise_leaves(f"{checkpoint_path}/training_state.eqx", training_state)
    metadata = {
        "step": step,
        "timestamp": datetime.now().isoformat(),
    }
    with fs.open(f"{checkpoint_path}/metadata.json", "w") as json_out:
        json.dump(metadata, json_out)

    return checkpoint_path


def load_checkpoint(model_state, training_state, checkpoint_path, *, discover_latest=True):
    """
    Load a checkpoint from a given path.

    Returns the loaded model state, training state, and step. If discover_latest is True,
    the latest checkpoint in the given path will be loaded. Otherwise, the checkpoint at
    the given path will be loaded. If no checkpoint is found, returns None

    If training_state is None, the loaded training state will be returned as None.
    """
    if discover_latest:
        checkpoint_path = discover_latest_checkpoint(checkpoint_path)

    if checkpoint_path is None:
        return None

    model_state = tree_deserialise_leaves(f"{checkpoint_path}/model.eqx", model_state)
    fs: AbstractFileSystem
    fs, _, _ = fsspec.get_fs_token_paths(checkpoint_path)

    with fs.open(f"{checkpoint_path}/metadata.json") as metadata_in:
        metadata = json.load(metadata_in)

    if training_state is None:
        training_state = None
    else:
        training_state = tree_deserialise_leaves(f"{checkpoint_path}/training_state.eqx", training_state)

    return model_state, training_state, metadata["step"]


def discover_latest_checkpoint(checkpoint_path) -> Optional[str]:
    """
    Discover the latest checkpoint in a given path.
    """
    # need to use fsspec for this, as glob.glob doesn't work on gs://
    fs: AbstractFileSystem
    fs, _, _ = fsspec.get_fs_token_paths(checkpoint_path)
    ckpt_dirs = [d for d in fs.glob(f"{checkpoint_path}/*") if fs.isdir(d)] + [checkpoint_path]
    ckpt_dirs = [d[:-1] if d.endswith("/") else d for d in ckpt_dirs]
    ckpt_dirs = [d for d in ckpt_dirs if fs.exists(f"{d}/metadata.json")]

    def checkpoint_timestamp(ckpt_dir):
        metadata = json.load(fs.open(f"{ckpt_dir}/metadata.json"))
        return datetime.fromisoformat(metadata["timestamp"])

    if len(ckpt_dirs) > 0:
        return max(ckpt_dirs, key=checkpoint_timestamp)
    else:
        return None


def tree_serialise_leaves(
    path: Union[str, pathlib.Path],
    pytree: PyTree,
    filter_spec=default_serialise_filter_spec,
    is_leaf: Callable[[Any], bool] = _is_index,
) -> None:
    """Analog to `equinox.tree_deserialise_leaves`, but saves the leaves of a PyTree using fsspec."""

    with fsspec.open(path, "wb") as f:

        def _serialise(spec, x):
            def __serialise(y):
                spec(f, y)
                return y

            jax.tree_map(__serialise, x, is_leaf=is_leaf)

        jax.tree_map(_serialise, filter_spec, pytree)


def tree_deserialise_leaves(
    path: Union[str, pathlib.Path],
    like: PyTree,
    filter_spec=default_deserialise_filter_spec,
    is_leaf: Callable[[Any], bool] = _is_index,
) -> PyTree:
    """
    Analog to `equinox.tree_serialise_leaves`, but loads the leaves of a PyTree using fsspec.
    """

    with fsspec.open(path, "rb") as f:

        def _deserialise(spec, x):
            def __deserialise(y):
                return spec(f, y)

            return jax.tree_util.tree_map(__deserialise, x, is_leaf=is_leaf)

        out = jax.tree_util.tree_map(_deserialise, filter_spec, like)
    jax.tree_util.tree_map(_assert_same, out, like, is_leaf=is_leaf)
    return out
