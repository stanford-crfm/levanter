import glob
import json
import os
from datetime import datetime
from typing import Optional

import equinox as eqx

def save_checkpoint(model, training_state, step:int, checkpoint_path, *, exist_ok=False):
    """
    Save a checkpoint to a given path.

    If the path does not exist, it will be created.

    """
    os.makedirs(checkpoint_path, exist_ok=exist_ok)
    eqx.tree_serialise_leaves(f"{checkpoint_path}/model.eqx", model)
    eqx.tree_serialise_leaves(f"{checkpoint_path}/training_state.eqx", training_state)
    metadata = {
        "step": step,
        "timestamp": datetime.now().isoformat(),
    }
    json.dump(metadata, open(f"{checkpoint_path}/metadata.json", "w"))

    return checkpoint_path


def load_checkpoint(model_state, training_state, checkpoint_path, *, discover_latest=True):
    """
    Load a checkpoint from a given path.

    Returns the loaded model state, training state, and step. If discover_latest is True,
    the latest checkpoint in the given path will be loaded. Otherwise, the checkpoint at
    the given path will be loaded. If no checkpoint is found, returns None
    """
    if discover_latest:
        checkpoint_path = discover_latest_checkpoint(checkpoint_path)

    if checkpoint_path is None:
        return None

    model_state = eqx.tree_deserialise_leaves(f"{checkpoint_path}/model.eqx", model_state)
    metadata = json.load(open(f"{checkpoint_path}/metadata.json"))
    training_state = eqx.tree_deserialise_leaves(f"{checkpoint_path}/training_state.eqx", training_state)
    return model_state, training_state, metadata["step"]


def discover_latest_checkpoint(checkpoint_path) -> Optional[str]:
    """
    Discover the latest checkpoint in a given path.
    """
    ckpt_dirs = [d for d in glob.glob(f"{checkpoint_path}/*") if os.path.isdir(d)] + [checkpoint_path]
    ckpt_dirs = [d for d in ckpt_dirs if os.path.exists(f"{d}/metadata.json")]

    def checkpoint_timestamp(ckpt_dir):
        metadata = json.load(open(f"{ckpt_dir}/metadata.json"))
        return datetime.fromisoformat(metadata["timestamp"])
    if len(ckpt_dirs) > 0:
        return max(ckpt_dirs, key=checkpoint_timestamp)
    else:
        return None

