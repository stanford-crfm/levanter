import dataclasses
import datetime
import json
import logging
import os
import pathlib
import queue
import threading
import time
import urllib.parse
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, List, Optional, ParamSpec, Sequence, TypeVar, Union

import equinox
import fsspec
import jax
import jax.numpy as jnp
from draccus import field
from fsspec import AbstractFileSystem
from jax.experimental.array_serialization.serialization import GlobalAsyncCheckpointManager
from jax.experimental.multihost_utils import broadcast_one_to_all
from jaxtyping import PyTree

import haliax.partitioning
from haliax.jax_utils import is_in_jit, is_jax_array_like

from levanter.tensorstore_serialization import tree_deserialize_leaves_tensorstore, tree_serialize_leaves_tensorstore
from levanter.types import FilterSpec


logger = logging.getLogger(__name__)

PathLike = Union[str, pathlib.Path]

M = TypeVar("M", bound=PyTree)
Sig = ParamSpec("Sig")


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

        # The default of 5 minutes is too short even for modestly sized models for some reason
        self._manager = GlobalAsyncCheckpointManager(timeout_secs=60 * 30)

        if jax.process_index() == 0:
            self._async_checkpoint_remover_queue: queue.Queue[str] = queue.Queue(maxsize=-1)
            self._async_checkpoint_remover_thread = threading.Thread(
                target=self._async_checkpoint_remover, daemon=True
            )
            self._async_checkpoint_remover_thread.start()
            self._checkpoint_being_removed = None

    def load_checkpoint(
        self,
        state: M,
        path: Optional[PathLike] = None,
        *,
        discover_latest: bool = True,
        axis_mapping: Optional[haliax.partitioning.ResourceMapping] = None,
        mesh: Optional[haliax.partitioning.Mesh] = None,
    ) -> Optional[M]:
        if path is None:
            path = self.base_path
        return load_checkpoint(state, path, discover_latest=discover_latest, axis_mapping=axis_mapping, mesh=mesh)

    def load_model(
        self,
        model: M,
        path: Optional[str] = None,
        *,
        discover_latest: bool = True,
        axis_mapping: Optional[haliax.partitioning.ResourceMapping] = None,
        mesh: Optional[haliax.partitioning.Mesh] = None,
    ) -> Optional[M]:
        """
        Convenience method/holdover from  previous API for loading checkpoints.
        Loads just the model assuming the model is in the `model` subdir of the discovered checkpoint.
        """
        ret_dict = self.load_checkpoint(
            {"model": model}, path, discover_latest=discover_latest, axis_mapping=axis_mapping, mesh=mesh
        )
        if ret_dict is None:
            return None
        return ret_dict["model"]

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

            if not save_permanent_ckpt:
                self._last_temporary_checkpoint = destination
            else:
                self._last_temporary_checkpoint = None

            def callback():
                if last_checkpoint is not None:
                    self._rm_checkpoint(last_checkpoint)

            self.save_checkpoint(info, destination, commit_callback=callback)

    def _get_current_step_save_interval(self, step):
        # binary search for the correct interval
        # we assume that the intervals are sorted by until
        current_policy = next(filter(lambda p: p.until is None or p.until >= step, self.step_policies), None)
        if current_policy is None:
            return None
        return current_policy.every

    def wait_until_finished(self):
        self._manager.wait_until_finished()
        if jax.process_index() == 0:
            while self._checkpoint_being_removed is not None or not self._async_checkpoint_remover_queue.empty():
                time.sleep(0.2)

    def _rm_checkpoint(self, checkpoint):
        if jax.process_index() == 0:
            logger.info(f"Removing checkpoint {checkpoint}")
            self._async_checkpoint_remover_queue.put(checkpoint)

    def _do_rm_checkpoint(self, checkpoint):
        fs, plain_path = _get_fs_and_plain_path(self.base_path)
        # have to strip protocol from path because fsspec filesystems don't like them
        try:
            cp_path = os.path.join(plain_path, checkpoint)
            logger.info(f"Deleting old checkpoint {checkpoint} from {cp_path}")
            time_in = time.time()
            fs.rm(cp_path, recursive=True)
            time_out = time.time()
            logger.info(f"Deleted old checkpoint {checkpoint} from {cp_path} in {time_out - time_in:.2f} seconds")
        except Exception:  # pylint: disable=broad-except
            logger.exception(f"Failed to delete checkpoint {checkpoint}", exc_info=True)

    def save_checkpoint(self, info, destination: str, commit_callback: Optional[Callable[[], None]] = None):
        path = os.path.join(self.base_path, destination)
        logger.info(f"Saving checkpoint at step {info.step} to {path}")
        state = info.state.saveable_state

        save_checkpoint(
            state,
            step=info.step,
            checkpoint_path=path,
            manager=self._manager,
            commit_callback=commit_callback,
        )
        self._last_save_step = info.step
        self._last_save_time = self._dt_now_injection()

    def _async_checkpoint_remover(self):
        while True:
            checkpoint = self._async_checkpoint_remover_queue.get(block=True)
            self._checkpoint_being_removed = checkpoint
            self._do_rm_checkpoint(checkpoint)
            self._checkpoint_being_removed = None


def save_checkpoint(
    tree: M,
    step: int,
    checkpoint_path: PathLike,
    manager: Optional[GlobalAsyncCheckpointManager] = None,
    *,
    commit_callback: Optional[Callable[[], None]] = None,
):
    """
    Save a checkpoint to a given path using TensorStore.

    If the path does not exist, it will be created.

    If training_state is None, no training state will be saved.

    This method is jax.Array-aware and will save shards in a way that can be restored
    """
    step = int(step)
    checkpoint_path = str(checkpoint_path)
    logger.info(f"Saving checkpoint to {checkpoint_path} for step {step}")

    fs: AbstractFileSystem
    fs, plain_path = _get_fs_and_plain_path(checkpoint_path)
    fs.makedirs(plain_path, exist_ok=True)

    def my_callback():
        save_metadata(checkpoint_path, fs, step)
        logger.info(f"Saved checkpoint to {checkpoint_path} for step {step}")

        if commit_callback is not None:
            commit_callback()

    tree = equinox.filter(tree, lambda x: is_jax_array_like(x) or isinstance(x, (int, float, bool, complex)))

    tree_serialize_leaves_tensorstore(checkpoint_path, tree, manager, commit_callback=my_callback)

    return checkpoint_path


def save_metadata(checkpoint_path, fs, step):
    metadata = {"step": step, "timestamp": datetime.datetime.now().isoformat()}
    if jax.process_index() == 0:
        with fs.open(os.path.join(checkpoint_path, "metadata.json"), "w") as json_out:
            json.dump(metadata, json_out)


def load_checkpoint(
    tree: M,
    checkpoint_path: PathLike,
    *,
    subpath: Optional[str] = None,
    discover_latest=True,
    axis_mapping: Optional[haliax.partitioning.ResourceMapping] = None,
    mesh: Optional[jax.sharding.Mesh] = None,
) -> M:
    """
    Load a checkpoint from a given path. If discover_latest is True, then the latest checkpoint
    in a subdirectory of the given path will be loaded. If subpath is not None, then the checkpoint
    loads only that subpath of the checkpoint. This is useful for loading, e.g., just the model and not
    the entire training state.

    Args:
        tree: an exemplar of the tree to load. Can be a PyTree[ShapeDTypeStruct] instead of a PyTree[Any]
        checkpoint_path: the path to load the checkpoint from
        subpath: the subpath to load from the checkpoint
        discover_latest: whether to discover the latest checkpoint in the given path
        axis_mapping: the axis mapping to use for loading the checkpoint
        mesh: the mesh to use for loading the checkpoint
    Returns:
        the loaded checkpoint, with the same structure as the exemplar tree

    """
    fs: AbstractFileSystem
    fs, _ = _get_fs_and_plain_path(checkpoint_path)

    checkpoint_path = str(checkpoint_path)

    if is_in_jit():
        logger.warning("Loading checkpoint in jit. This is not recommended and probably won't work.")

    if discover_latest:
        checkpoint_path = discover_latest_checkpoint(checkpoint_path)  # type: ignore

    if checkpoint_path is None or not fs.exists(checkpoint_path):
        raise FileNotFoundError(f"Could not find checkpoint at {checkpoint_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    metadata = load_metadata(checkpoint_path, fs)

    if subpath:
        checkpoint_path = os.path.join(checkpoint_path, subpath)

    ser, non_ser = equinox.partition(tree, is_jax_array_like)
    try:
        tree = tree_deserialize_leaves_tensorstore(checkpoint_path, ser, axis_mapping=axis_mapping, mesh=mesh)
        tree = equinox.combine(tree, non_ser)
        return tree
    except:  # noqa
        from levanter.trainer_state import TrainerState

        if not isinstance(tree, TrainerState):
            raise
        else:
            logger.warning("Attempting to load old-style checkpoint")
            model, training_state = tree.model, (tree.opt_state, tree.training_key)

            model = tree_deserialize_leaves_tensorstore(
                os.path.join(checkpoint_path, "model"), model, axis_mapping=axis_mapping, mesh=mesh
            )

            if training_state is None:
                opt_state = None
                key = None
            else:
                training_state = tree_deserialize_leaves_tensorstore(
                    os.path.join(checkpoint_path, "training_state"),
                    training_state,
                    axis_mapping=axis_mapping,
                    mesh=mesh,
                )
                opt_state, key = training_state

            # TODO: pretty sure this is right, but should verify
            step = metadata["step"]
            new_state = dataclasses.replace(
                tree, step=step + 1, model=model, opt_state=opt_state, training_key=key  # type: ignore
            )
            return new_state


def load_checkpoint_or_initialize(
    init_fn: Callable[Sig, M],
    checkpoint_path: PathLike,
    *,
    subpath: Optional[str] = None,
    discover_latest=True,
    axis_mapping: Optional[haliax.partitioning.ResourceMapping] = None,
    mesh: Optional[jax.sharding.Mesh] = None,
    is_checkpointed: FilterSpec = True,
    donate_args: FilterSpec = True,
    donate_kwargs: Optional[FilterSpec] = None,
    do_load: Optional[bool] = None,
) -> Callable[Sig, M]:
    """
    Load a checkpoint from a given path. If discover_latest is True, then the latest checkpoint
    in a subdirectory of the given path will be loaded. If subpath is not None, then the checkpoint
    loads only that subpath of the checkpoint. This is useful for loading, e.g., just the model and not
    the entire training state.

    This function supports "partial" checkpoint loading, where only a subset of the parameters of the
    state is loaded from the checkpoint. This is useful for initializing just some parameters.
    (Note that you have to declare which parameters you are expecting to load via is_checkpointed.
     Things can't just be missing from the checkpoint.)

    init_fn will be called inside eval_shape and inside jit, so it should be a pure function. In particular,
    it should not do any I/O.

    This function is commonly used for initializing training state from a possibly non-existent checkpoint, but it can be used
    for initializing any state from a checkpoint.

    By default, this function will donate all arguments to init_fn that are not present in the checkpoint.

    Args:
        init_fn: a function to initialize if needed
        checkpoint_path: the path to load the checkpoint from
        subpath: the subpath to load from the checkpoint
        discover_latest: whether to discover the latest checkpoint in the given path
        axis_mapping: the axis mapping to use for loading the checkpoint
        mesh: the mesh to use for loading the checkpoint
        is_checkpointed: a FilterSpec that specifies which parameters are checkpointed
        donate_args: a FilterSpec that specifies which arguments to donate to init_fn if we need to initialize
        donate_kwargs: a FilterSpec that specifies which kwargs to donate to init_fn if we need to initialize
        do_load: if True, always load the checkpoint. If False, always initialize. If None, load if the checkpoint exists, otherwise initialize

    Returns:
        A function that takes the same arguments as init_fn, but loads the checkpoint if it exists and returns the
        loaded state.

    """

    # some state might not be initialized, so we need to initialize it
    # JAX will be smart and only do the compute for things we actually need
    @haliax.named_jit(
        axis_resources=axis_mapping,
        out_axis_resources=axis_mapping,
        donate_args=donate_args,
        donate_kwargs=donate_kwargs,
    )
    def init_and_merge(state, *args, **kwargs):
        init_state = init_fn(*args, **kwargs)
        return equinox.combine(state, init_state)

    def load_or_init(*args, **kwargs):
        # we need to call init_fn to get the shape, dtype, and structure of the state
        # we'll use this to deserialize the checkpoint
        state_shape = equinox.filter_eval_shape(init_fn, *args, **kwargs)

        # we need to filter the state to get the parameters we want to load
        # we'll use this to deserialize the checkpoint
        filtered_state_shape = equinox.filter(state_shape, is_checkpointed)
        # strip out all the shape stuff, leaving only the dtype and structure
        loaded_state = equinox.filter(state_shape, lambda _: False)

        if do_load is not False:
            # now we can load the checkpoint
            try:
                loaded_state = load_checkpoint(
                    filtered_state_shape,
                    checkpoint_path,
                    subpath=subpath,
                    discover_latest=discover_latest,
                    axis_mapping=axis_mapping,
                    mesh=mesh,
                )
            except FileNotFoundError:
                if do_load is True:
                    raise
                logger.info(f"Checkpoint not found at {checkpoint_path}. Initializing from scratch.")

        state = init_and_merge(loaded_state, *args, **kwargs)

        return state

    return load_or_init


def load_metadata(checkpoint_path, fs=None):
    if fs is None:
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


def _get_fs_and_plain_path(path, fs=None):
    if fs is None:
        fs, _, (path_to_open,) = fsspec.get_fs_token_paths(str(path))
    else:
        path_to_open = path
    return fs, path_to_open


@dataclass
class CheckpointerConfig:
    base_path: str = "checkpoints/"
    save_interval: timedelta = timedelta(minutes=15)
    # TODO: I'd like to write this, but it's not supported by draccus
    # keep: List[CheckpointInterval] = field(default_factory=lambda: [CheckpointInterval(every=1000)])
    keep: List[dict] = field(
        default_factory=lambda: [dict(every=10000)]
    )  # list of dicts with two keys: every and until

    append_run_id_to_base_path: bool = True

    def expanded_path(self, run_id) -> str:
        if self.append_run_id_to_base_path:
            return os.path.expanduser(os.path.join(self.base_path, run_id))
        return os.path.expanduser(self.base_path)

    def create(self, run_id) -> Checkpointer:
        keeps = [CheckpointInterval(**k) for k in self.keep]
        return Checkpointer(
            base_path=self.expanded_path(run_id),
            save_interval=self.save_interval,
            step_policies=keeps,
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


# TODO: add partial checkpoint loading
