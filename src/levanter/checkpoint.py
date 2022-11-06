import dataclasses
import datetime
import json
import logging
import pathlib
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Union

import fsspec
import jax
from equinox.serialisation import _is_index, default_deserialise_filter_spec, default_serialise_filter_spec
from fsspec import AbstractFileSystem
from furl import furl
from jaxtyping import PyTree

from levanter.tensorstore_serialization import tree_deserialize_leaves_tensorstore, tree_serialize_leaves_tensorstore


logger = logging.getLogger(__name__)

PathLike = Union[str, furl, pathlib.Path]


@dataclass(frozen=True)
class CheckpointInterval:
    every: int  # how often to checkpoint
    until: Optional[int] = None  # until what step to save checkpoints with this policy, None means forever


class Checkpointer:
    """
    A checkpointer class that saves checkpoints with two different, but overlapping policies: time and step.

    Note that this class is stateful: it keeps track of the last time a checkpoint was saved, and the last step
    a checkpoint was saved at.
    """

    base_path: furl
    save_interval: Optional[datetime.timedelta]  # we save at least this frequently
    step_policies: Sequence[CheckpointInterval] = dataclasses.field(
        default_factory=lambda: [CheckpointInterval(every=1000)]
    )

    _last_temporary_checkpoint: Optional[furl] = None

    def __init__(
        self,
        base_path: furl,
        save_interval: Optional[datetime.timedelta],
        step_policies: Sequence[CheckpointInterval],
        dt_now_injection: Optional[Callable[[], datetime.datetime]] = None,
    ):
        """dt_now_injection is used for testing"""
        self.base_path = base_path
        self.save_interval = save_interval
        self.step_policies = list(step_policies)
        self._dt_now_injection = dt_now_injection or datetime.datetime.now
        self._last_save_time = self._dt_now_injection()
        self._last_save_step = 0
        self._last_temporary_checkpoint = None

        # ensure that the step_policies are sorted. We could sort, but instead we'll just insist that they are sorted
        # since it's probably a typo if they aren't
        for i in range(1, len(step_policies)):
            # factor these out so mypy can figure it out
            prev_until = step_policies[i - 1].until
            until = step_policies[i].until
            if prev_until is None:
                raise ValueError("Only the last step policy can have an 'until' value of None")
            if until is None:
                continue
            if prev_until >= until:
                raise ValueError("Step policies must be sorted by 'until' value")

    def load_checkpoint(self, model, training_state, path: Optional[PathLike] = None, *, discover_latest: bool = True):
        if path is None:
            path = self.base_path
        return load_checkpoint(model, training_state, path, discover_latest=discover_latest)

    def load_model(self, model, path: Optional[str] = None, *, discover_latest: bool = True):
        if path is None:
            path = self.base_path
        return load_checkpoint(model, None, path, discover_latest=discover_latest)

    def on_step(self, info, force: bool = False):
        step = info.step

        if step == 0:
            self._last_save_time = self._dt_now_injection()
            if not force:
                return  # don't save checkpoint at step 0 unless forced

        # two reasons we can save: time or step
        # they have different behaviors:
        # * if we save by time, we save the latest checkpoint and delete the previous one
        # * if we save by step, we save the latest checkpoint and keep the previous one
        should_save = force
        next_checkpoint_is_permanent = False

        current_every = self._get_current_step_save_interval(step)
        if current_every is not None and step % current_every == 0:
            should_save = True
            next_checkpoint_is_permanent = True
        elif self.save_interval and self._dt_now_injection() - self._last_save_time >= self.save_interval:
            should_save = True
            next_checkpoint_is_permanent = False

        if should_save:
            last_checkpoint = self._last_temporary_checkpoint
            destination = f"step-{step}"

            self.save_checkpoint(info, destination)

            if not next_checkpoint_is_permanent:
                self._last_temporary_checkpoint = destination
            else:
                self._last_temporary_checkpoint = None

            # TODO: we should consider writing to disk whether it's a temporary checkpoint or not
            # so that we can delete it properly if we recover
            if last_checkpoint is not None:
                self._rm_checkpoint(last_checkpoint)

    def _get_current_step_save_interval(self, step):
        # binary search for the correct interval
        # we assume that the intervals are sorted by until
        current_policy = next(filter(lambda p: p.until is None or p.until >= step, self.step_policies), None)
        if current_policy is None:
            return None
        return current_policy.every

    def _rm_checkpoint(self, checkpoint):
        fs, plain_path = _get_fs_and_plain_path(self.base_path)
        # have to strip protocol from path because fsspec filesystems don't like them
        try:
            fs.rm(str(self.base_path.add(path=checkpoint).path), recursive=True)
        # don't let this take down a run
        except Exception:  # pylint: disable=broad-except
            logger.exception("Failed to delete checkpoint")

    def save_checkpoint(self, info, destination):
        path = self.base_path / destination
        logger.info(f"Saving checkpoint at step {info.step} to {path}")
        save_checkpoint(
            model=info.model,
            training_state=(info.opt_state, info.next_key),
            step=info.step,
            checkpoint_path=path,
        )
        self._last_save_step = info.step
        self._last_save_time = self._dt_now_injection()


def save_checkpoint(model, training_state, step: int, checkpoint_path: PathLike, *, exist_ok=False):
    """
    Save a checkpoint to a given path using TensorStore. If exist_ok is True, the checkpoint
    will be saved even if a checkpoint already exists at the given path.

    If the path does not exist, it will be created.

    If training_state is None, no training state will be saved.

    This method is GlobalDeviceArray-aware, and will save shards in a way that can be restored
    """
    logger.info(f"Saving checkpoint to {checkpoint_path} for step {step}")

    fs: AbstractFileSystem
    fs, _ = _get_fs_and_plain_path(checkpoint_path)
    tree_serialize_leaves_tensorstore(f"{checkpoint_path}/model", model)
    if training_state is not None:
        tree_serialize_leaves_tensorstore(f"{checkpoint_path}/training_state", training_state)

    save_metadata(checkpoint_path, fs, step)

    logger.info(f"Saved checkpoint for step {step}")

    return checkpoint_path


def save_metadata(checkpoint_path, fs, step):
    metadata = {
        "step": step,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    if jax.process_index() == 0:
        with fs.open(f"{checkpoint_path}/metadata.json", "w") as json_out:
            json.dump(metadata, json_out)


def load_checkpoint(model, training_state, checkpoint_path: PathLike, *, discover_latest=True):
    """
    Load a checkpoint from a given path.

    Returns the loaded model state, training state, and step. If discover_latest is True,
    the latest checkpoint in the given path will be loaded. Otherwise, the checkpoint at
    the given path will be loaded. If no checkpoint is found, returns None

    If training_state is None, no training state will be loaded.
    """
    fs: AbstractFileSystem
    fs, _ = _get_fs_and_plain_path(checkpoint_path)

    if discover_latest:
        checkpoint_path = discover_latest_checkpoint(checkpoint_path)

    if checkpoint_path is None:
        return None

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    metadata = load_metadata(checkpoint_path, fs)

    model = tree_deserialize_leaves_tensorstore(f"{checkpoint_path}/model", model)

    if training_state is None:
        training_state = None
    else:
        training_state = tree_deserialize_leaves_tensorstore(f"{checkpoint_path}/training_state", training_state)

    return model, training_state, metadata["step"]


def load_metadata(checkpoint_path, fs=None):
    if fs is None:
        fs: AbstractFileSystem
        fs, _, _ = fsspec.get_fs_token_paths(str(checkpoint_path))
    with fs.open(f"{checkpoint_path}/metadata.json") as metadata_in:
        metadata = json.load(metadata_in)
    return metadata


def discover_latest_checkpoint(checkpoint_path) -> Optional[str]:
    """
    Discover the latest checkpoint in a given path.
    """
    # need to use fsspec for this, as glob.glob doesn't work on gs://
    fs: AbstractFileSystem
    fs, plain_path = _get_fs_and_plain_path(checkpoint_path)
    ckpt_dirs = [d for d in fs.glob(f"{plain_path}/*") if fs.isdir(d)] + [plain_path]
    ckpt_dirs = [d[:-1] if d.endswith("/") else d for d in ckpt_dirs]
    ckpt_dirs = [d for d in ckpt_dirs if fs.exists(f"{d}/metadata.json")]

    def checkpoint_timestamp(ckpt_dir):
        metadata = json.load(fs.open(f"{ckpt_dir}/metadata.json"))
        return datetime.datetime.fromisoformat(metadata["timestamp"])

    if len(ckpt_dirs) > 0:
        out = max(ckpt_dirs, key=checkpoint_timestamp)
        logger.info(f"Discovered latest checkpoint from {checkpoint_path} at {out}")
        return out
    else:
        logger.warning(f"No checkpoints found in {checkpoint_path}")
        return None


def tree_serialise_leaves(
    path: PathLike,
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
    path: PathLike,
    like: PyTree,
    filter_spec=default_deserialise_filter_spec,
    is_leaf: Callable[[Any], bool] = _is_index,
    fs=None,
) -> PyTree:
    """
    Analog to `equinox.tree_deserialise_leaves`, but loads the leaves of a PyTree using fsspec.
    """

    fs, path_to_open = _get_fs_and_plain_path(path, fs)

    with fs.open(path_to_open, "rb") as f:

        def _deserialise(spec, x):
            def __deserialise(y):
                return spec(f, y)

            return jax.tree_util.tree_map(__deserialise, x, is_leaf=is_leaf)

        out = jax.tree_util.tree_map(_deserialise, filter_spec, like)
    jax.tree_util.tree_map(_assert_same, out, like, is_leaf=is_leaf)
    return out


def _get_fs_and_plain_path(path, fs=None):
    if fs is None:
        fs, _, (path_to_open,) = fsspec.get_fs_token_paths(str(path))
    else:
        path_to_open = path
    return fs, path_to_open


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
