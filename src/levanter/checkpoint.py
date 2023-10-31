import dataclasses
import datetime
import json
import logging
import os
import pathlib
import urllib.parse
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, List, Optional, Sequence, Tuple, TypeVar, Union

import equinox
import fsspec
import jax
import jax.numpy as jnp
from draccus import field
from equinox import default_deserialise_filter_spec, default_serialise_filter_spec
from fsspec import AbstractFileSystem
from jax.experimental.multihost_utils import broadcast_one_to_all, sync_global_devices
from jaxtyping import PyTree

import haliax.partitioning

from levanter.tensorstore_serialization import tree_deserialize_leaves_tensorstore, tree_serialize_leaves_tensorstore
from levanter.types import FilterSpec


logger = logging.getLogger(__name__)

PathLike = Union[str, pathlib.Path]

M = TypeVar("M")
S = TypeVar("S")


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

    base_path: str
    save_interval: Optional[datetime.timedelta]  # we save at least this frequently
    step_policies: Sequence[CheckpointInterval] = dataclasses.field(
        default_factory=lambda: [CheckpointInterval(every=1000)]
    )

    _last_temporary_checkpoint: Optional[str] = None

    def __init__(
        self,
        base_path: PathLike,
        save_interval: Optional[datetime.timedelta],
        step_policies: Sequence[CheckpointInterval],
        *,
        keep_params: PyTree[FilterSpec] = True,
        dt_now_injection: Optional[Callable[[], datetime.datetime]] = None,
    ):
        """
        Class for managing checkpoints. Saves checkpoints according to two policies: time and step.

        Time policy: we save a checkpoint at least every `save_interval` seconds.
        Step policy: we save a checkpoint every `every` steps, until `until` steps have been reached.

        Time checkpoints are deleted after the next checkpoint is saved. Step checkpoints are never deleted.

        Args:
            base_path: the base path to save checkpoints to. may be gcs, local, or anything that tensorstore supports
            save_interval: the minimum amount of time between checkpoints (for time)
            step_policies: the step policies to use
            keep_params: a PyTree of FilterSpecs that specifies which parameters to keep in the checkpoint
            dt_now_injection: a function that returns the current time. useful for testing
        """
        self.base_path = str(base_path)
        self.save_interval = save_interval
        self.step_policies = list(step_policies)
        self.keep_params = keep_params
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

    def load_checkpoint(
        self,
        model: M,
        training_state: S,
        path: Optional[PathLike] = None,
        *,
        discover_latest: bool = True,
        axis_mapping: Optional[haliax.partitioning.ResourceMapping] = None,
        mesh: Optional[haliax.partitioning.Mesh] = None,
    ) -> Optional[Tuple[M, S, int]]:
        if path is None:
            path = self.base_path
        return load_checkpoint(
            model, training_state, path, discover_latest=discover_latest, axis_mapping=axis_mapping, mesh=mesh
        )

    def load_model(
        self,
        model: M,
        path: Optional[str] = None,
        *,
        discover_latest: bool = True,
        axis_mapping: Optional[haliax.partitioning.ResourceMapping] = None,
        mesh: Optional[haliax.partitioning.Mesh] = None,
    ) -> Optional[Tuple[M, int]]:
        if path is None:
            path = self.base_path
        ckpt = load_checkpoint(
            model, None, path, discover_latest=discover_latest, axis_mapping=axis_mapping, mesh=mesh
        )
        if ckpt is None:
            return None
        model, _, step = ckpt
        return model, step

    def on_step(self, info, force: bool = False):
        step = info.step

        if step == 0:
            self._last_save_time = self._dt_now_injection()
            if not force:
                return  # don't save checkpoint at step 0 unless forced

        if step == self._last_save_step:
            # we've already saved a checkpoint at this step
            return

        # two reasons we can save: time or step
        # they have different behaviors for retention.
        # if the previous checkpoint was a temporary checkpoint (i.e. saved b/c of time), we can delete it

        # there's a potential clock skew issue here: if we save by time, and the clock is skewed across processes,
        # then we could end up with a situation where one process saves a checkpoint, and then another process
        # saves a checkpoint for the next step, etc. This leads to partial checkpoints, no good.
        # we fix by having process 0 make the decision
        my_should_save = force
        my_save_permanent_ckpt = force

        current_every = self._get_current_step_save_interval(step)
        last_save_time = self._dt_now_injection() - self._last_save_time
        if current_every is not None and step % current_every == 0:
            my_should_save = True
            my_save_permanent_ckpt = True
        elif self.save_interval and last_save_time >= self.save_interval:
            my_should_save = True
            my_save_permanent_ckpt = False

        should_save, save_permanent_ckpt = broadcast_one_to_all(
            jnp.array([my_should_save, my_save_permanent_ckpt], dtype=jnp.bool_)
        )

        # log the decision
        if should_save:
            if save_permanent_ckpt:
                logger.info(f"Saving checkpoint at step {step}.")
            else:
                logger.info(f"Saving temporary checkpoint at step {step}.")

        if should_save:
            last_checkpoint = self._last_temporary_checkpoint
            destination = f"step-{step}"

            self.save_checkpoint(info, destination)

            if not save_permanent_ckpt:
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
        if jax.process_index() != 0:
            return

        fs, plain_path = _get_fs_and_plain_path(self.base_path)
        # have to strip protocol from path because fsspec filesystems don't like them
        try:
            cp_path = os.path.join(plain_path, checkpoint)
            logger.info(f"Deleting checkpoint {checkpoint} from {cp_path}")
            fs.rm(cp_path, recursive=True)
        # don't let this take down a run
        except Exception:  # pylint: disable=broad-except
            logger.exception("Failed to delete checkpoint", exc_info=True)

    def save_checkpoint(self, info, destination: str):
        path = os.path.join(self.base_path, destination)
        logger.info(f"Saving checkpoint at step {info.step} to {path}")
        model = equinox.filter(info.model, self.keep_params)
        save_checkpoint(
            model=model,
            training_state=(info.opt_state, info.next_key),
            step=info.step,
            checkpoint_path=path,
        )
        # also write a little sentinel file to indicate that we wrote this checkpoint from this worker
        # just for debugging purposes
        sentinel_path = os.path.join(path, f"worker-{jax.process_index()}.cert")
        with fsspec.open(sentinel_path, "w") as f:
            f.write("worker participated in checkpoint")
        self._last_save_step = info.step
        self._last_save_time = self._dt_now_injection()
        logger.info(f"Saved checkpoint at step {info.step} to {path}. Save time is {self._last_save_time}")


def save_checkpoint(model, training_state, step: int, checkpoint_path: PathLike, *, exist_ok: bool = False):
    """
    Save a checkpoint to a given path using TensorStore. If exist_ok is True, the checkpoint
    will be saved even if a checkpoint already exists at the given path.

    If the path does not exist, it will be created.

    If training_state is None, no training state will be saved.

    This method is jax.Array-aware and will save shards in a way that can be restored
    """
    checkpoint_path = str(checkpoint_path)
    logger.info(f"Saving checkpoint to {checkpoint_path} for step {step}")

    fs: AbstractFileSystem
    fs, plain_path = _get_fs_and_plain_path(checkpoint_path)
    fs.makedirs(plain_path, exist_ok=exist_ok)

    tree_serialize_leaves_tensorstore(os.path.join(checkpoint_path, "model"), model)
    if training_state is not None:
        tree_serialize_leaves_tensorstore(os.path.join(checkpoint_path, "training_state"), training_state)

    save_metadata(checkpoint_path, fs, step)

    logger.info(f"Saved checkpoint for step {step}")

    # make sure that all processes agree on the checkpoint path and also synchronize hosts
    sync_global_devices(checkpoint_path)

    return checkpoint_path


def save_metadata(checkpoint_path, fs, step):
    metadata = {
        "step": step,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    if jax.process_index() == 0:
        with fs.open(os.path.join(checkpoint_path, "metadata.json"), "w") as json_out:
            json.dump(metadata, json_out)


def load_checkpoint(
    model: M,
    training_state: S,
    checkpoint_path: PathLike,
    *,
    discover_latest=True,
    axis_mapping: Optional[haliax.partitioning.ResourceMapping] = None,
    mesh: Optional[jax.sharding.Mesh] = None,
) -> Optional[Tuple[M, S, int]]:
    """
    Load a checkpoint from a given path.

    Returns the loaded model state, training state, and step. If discover_latest is True,
    the latest checkpoint in the given path will be loaded. Otherwise, the checkpoint at
    the given path will be loaded. If no checkpoint is found, returns None

    If training_state is None, no training state will be loaded.
    """
    fs: AbstractFileSystem
    fs, _ = _get_fs_and_plain_path(checkpoint_path)

    checkpoint_path = str(checkpoint_path)

    if discover_latest:
        checkpoint_path = discover_latest_checkpoint(checkpoint_path)  # type: ignore

    if checkpoint_path is None or not fs.exists(checkpoint_path):
        return None

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    metadata = load_metadata(checkpoint_path, fs)

    model = tree_deserialize_leaves_tensorstore(
        os.path.join(checkpoint_path, "model"), model, axis_mapping=axis_mapping, mesh=mesh
    )

    if training_state is None:
        training_state = None
    else:
        training_state = tree_deserialize_leaves_tensorstore(
            os.path.join(checkpoint_path, "training_state"), training_state, axis_mapping=axis_mapping, mesh=mesh
        )

    return model, training_state, metadata["step"]


def load_metadata(checkpoint_path, fs=None):
    if fs is None:
        fs: AbstractFileSystem
        fs, _, _ = fsspec.get_fs_token_paths(str(checkpoint_path))
    with fs.open(os.path.join(checkpoint_path, "metadata.json")) as metadata_in:
        metadata = json.load(metadata_in)
    return metadata


def discover_latest_checkpoint(checkpoint_path: PathLike) -> Optional[str]:
    """
    Discover the latest checkpoint in a given path.
    """
    checkpoint_path = str(checkpoint_path)
    # need to use fsspec for this, as glob.glob doesn't work on gs://
    fs: AbstractFileSystem
    fs, _ = _get_fs_and_plain_path(checkpoint_path)

    def is_checkpoint_dir(path: str):
        return fs.exists(os.path.join(path, "metadata.json"))

    def maybe_unstrip_protocol(path: str):
        base_path_protocol = urllib.parse.urlparse(str(checkpoint_path)).scheme
        if base_path_protocol != "" and not urllib.parse.urlparse(path).scheme != "":
            return f"{base_path_protocol}://{path}"
        return path

    ckpt_dirs = [maybe_unstrip_protocol(d) for d in fs.glob(os.path.join(checkpoint_path, "*")) if fs.isdir(d)]
    ckpt_dirs.append(checkpoint_path)
    ckpt_dirs = [d for d in ckpt_dirs if is_checkpoint_dir(d)]

    def checkpoint_sort_key(ckpt_dir):
        metadata = json.load(fs.open(os.path.join(ckpt_dir, "metadata.json")))
        return (datetime.datetime.fromisoformat(metadata["timestamp"]), metadata["step"])

    if len(ckpt_dirs) > 0:
        out = max(ckpt_dirs, key=checkpoint_sort_key)
        logger.info(f"Discovered latest checkpoint from {checkpoint_path} at {out}")
        return out
    else:
        logger.warning(f"No checkpoints found in {checkpoint_path}")
        return None


def tree_serialise_leaves(
    path: PathLike,
    pytree: PyTree,
    filter_spec=default_serialise_filter_spec,
    is_leaf: Optional[Callable[[Any], bool]] = None,
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
    is_leaf: Optional[Callable[[Any], bool]] = None,
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


@dataclass
class CheckpointerConfig:
    base_path: str = "checkpoints/"
    save_interval: timedelta = timedelta(minutes=15)
    # TODO: I'd like to write this, but it's not supported by draccus
    # keep: List[CheckpointInterval] = field(default_factory=lambda: [CheckpointInterval(every=1000)])
    keep: List[dict] = field(
        default_factory=lambda: [dict(every=10000)]
    )  # list of dicts with two keys: every and until

    def expanded_path(self, run_id):
        return os.path.expanduser(os.path.join(self.base_path, run_id))

    def create(self, run_id, keep_params: PyTree[FilterSpec] = True) -> Checkpointer:
        keeps = [CheckpointInterval(**k) for k in self.keep]
        return Checkpointer(
            base_path=self.expanded_path(run_id),
            save_interval=self.save_interval,
            step_policies=keeps,
            keep_params=keep_params,
        )

    def __post_init__(self):
        self.base_path = os.path.expanduser(self.base_path)

        # validate the checkpoint intervals.
        # we want to make sure that the intervals are monotonic. only the last one can be None
        prev_interval = None
        for interval in self.keep:
            if prev_interval is not None:
                assert prev_interval["until"] is not None, "Only the last checkpoint interval can be None"
                assert (
                    interval["until"] is None or interval["until"] > prev_interval["until"]
                ), "Checkpoint intervals must be monotonic"
            prev_interval = interval
