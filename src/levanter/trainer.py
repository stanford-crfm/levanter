import atexit
import copy
import functools
import logging as pylogging
import os
import sys
import typing
import warnings
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import equinox as eqx
import jax
import jmp
import numpy as np
from draccus import field
from jax.experimental import multihost_utils
from jax.sharding import Mesh
from jaxtyping import PRNGKeyArray, PyTree
from optax import GradientTransformation

import haliax as hax
from haliax import Axis
from haliax.partitioning import ResourceAxis, ResourceMapping, named_jit
from haliax.quantization import Fp8Config
from haliax.types import Scalar

import levanter.checkpoint
import levanter.logging
import levanter.tracker
import levanter.tracker.wandb
from levanter import tracker
from levanter.checkpoint import CheckpointerConfig, load_checkpoint_or_initialize
from levanter.config import JsonAtom
from levanter.data import AsyncDataset, DataLoader
from levanter.distributed import DistributedConfig, RayConfig
from levanter.grad_accum import microbatched
from levanter.tracker import TrackerConfig, capture_time
from levanter.trainer_state import TrainerState, saveable_training_mask
from levanter.types import ComputeLossFunction, FilterSpec
from levanter.utils import cloud_utils, fsspec_utils
from levanter.utils.jax_utils import create_fsdp_mesh
from levanter.utils.tree_utils import inference_mode


logger = pylogging.getLogger(__name__)

M = TypeVar("M")  # Model
X = TypeVar("X")  # Input
S = TypeVar("S", bound=TrainerState)

DEFAULT_JAX_CONFIG: Dict[str, JsonAtom] = {
    "jax_threefry_partitionable": True,
    "jax_softmax_custom_jvp": True,
}


# A note on the semantics of "step" vs "next_step":
# The "step" of a TrainerState is the state after `step` steps have been taken.
# A "StepInfo"'s step is the step that was just completed. If you want the next step, use `next_step`.
@dataclass
class StepInfo(Generic[S]):
    state: S
    loss: float
    step_duration: float

    model = property(lambda self: self.state.model)
    opt_state = property(lambda self: self.state.opt_state)

    step = property(lambda self: int(self.state.step) - 1)
    """
    The step that was just completed. If you want the next step, use `next_step`.
    """
    next_step = property(lambda self: int(self.state.step))


@dataclass
class _Hook:
    fn: Callable[[StepInfo], None]
    every: int


class TrainerHooks:
    hooks: List[_Hook]

    def __init__(self):
        self.hooks = []

    def run_hooks(self, info: StepInfo, force: bool = False):
        for hook in self.hooks:
            if force or info.step % hook.every == 0:
                hook.fn(info)

    def add_hook(self, fn: Optional[Callable[[StepInfo], Any]] = None, *, every: int = 1):
        def decorator(fn: Callable[[StepInfo], None]):
            self.hooks.append(_Hook(fn, every))

        if fn is None:
            return decorator
        else:
            return decorator(fn)


def _unify_model_and_model_init(model: Optional[M], model_init: Optional[Callable[[], M]]) -> Callable[[], M]:
    if model is not None:
        if model_init is not None:
            raise ValueError("only one of model and model_init should be specified")

        if model is not None:
            # we can't just use `lambda: model` because JAX jit can't see captures, but it can see jax partials
            model_init = jax.tree_util.Partial(lambda m: m, model)
    elif model_init is None:
        raise ValueError("one of model and model_init must be specified")

    return model_init


class Trainer:
    config: "TrainerConfig"
    optimizer: GradientTransformation
    hooks: TrainerHooks
    tracker: levanter.tracker.Tracker
    is_trainable_param: PyTree[FilterSpec]
    _raw_loss_function: Callable
    _cmanagers: List[typing.ContextManager] = []

    def __init__(
        self,
        config: "TrainerConfig",
        optimizer: GradientTransformation,
        loss_fn: ComputeLossFunction,
        *,
        add_default_hooks: bool = True,
    ):
        """

        Args:
            config:  the trainer config
            optimizer: the optimizer, e.g. `optax.adam(1e-3)` or produced by [levanter.optim.OptimizerConfig][]
            loss_fn (Callable): the loss function. This should be a function that takes a model and some inputs and returns a
                scalar loss. It should be jit-able and should not have any side effects.
        """
        self.hooks = TrainerHooks()
        self.config = config
        self.optimizer = optimizer
        self._raw_loss_function = loss_fn
        if isinstance(config.tracker, Sequence):
            self.tracker = levanter.tracker.CompositeTracker([c.init(self.run_id) for c in config.tracker])
        else:
            self.tracker = config.tracker.init(self.run_id)

        self._cmanagers = []

        if add_default_hooks:
            self._add_default_hooks()

        self._cmanagers = []

    @cached_property
    def loss_fn(self):
        """
        Wrapped loss function that casts the model to compute precision and sets the context axis mapping to compute
        """

        @functools.wraps(self._raw_loss_function)
        def fn(model, *batch, **batch_kwargs):
            with hax.axis_mapping(self.compute_axis_mapping):
                model = self.mp.cast_to_compute(model)
                return _ensure_scalar(self._raw_loss_function(model, *batch, **batch_kwargs))

        return fn

    @property
    def run_id(self) -> str:
        """Returns the run id"""
        assert self.config.id is not None
        return self.config.id

    @property
    def mp(self) -> jmp.Policy:
        """Returns the mixed precision policy"""
        return self.config.mp

    @property
    def fp8(self) -> Optional[Fp8Config]:
        if self.config.fp8 is True:
            return Fp8Config()
        elif self.config.fp8 is False:
            return None
        else:
            return self.config.fp8

    @property
    def num_train_steps(self) -> int:
        return self.config.num_train_steps

    @typing.overload
    def add_hook(self, fn: Callable[[StepInfo], Any], *, every: int = 1):
        ...

    @typing.overload
    def add_hook(self, *, every: int = 1):
        ...

    def add_hook(self, fn: Optional[Callable[[StepInfo], Any]] = None, *, every: int = 1):
        return self.hooks.add_hook(fn, every=every)

    def run_hooks(self, info: StepInfo, force: bool = False):
        self.hooks.run_hooks(info, force=force)

    @property
    def parameter_axis_mapping(self) -> ResourceMapping:
        return self.config.parameter_axis_mapping

    @property
    def compute_axis_mapping(self) -> ResourceMapping:
        return self.config.compute_axis_mapping

    @property
    def device_mesh(self) -> Mesh:
        return self.config.device_mesh

    @property
    def TrainBatch(self):
        return self.config.TrainBatch

    @property
    def EvalBatch(self):
        return self.config.EvalBatch

    def __enter__(self):
        if len(self._cmanagers) > 0:
            raise RuntimeError("Trainer is already entered")

        self._cmanagers = [
            levanter.current_tracker(self.tracker),
            self.device_mesh,
            hax.axis_mapping(self.parameter_axis_mapping),
        ]

        for cmanager in self._cmanagers:
            cmanager.__enter__()

        return self

    def __exit__(self, *args):
        problems = []
        for cmanager in reversed(self._cmanagers):
            try:
                cmanager.__exit__(*args)
            except Exception as e:
                problems.append(e)

        self._cmanagers = []

        if len(problems) > 0:
            raise RuntimeError("Exception(s) occurred while exiting trainer", problems) from problems[0]

    def initial_state(
        self,
        training_key: PRNGKeyArray,
        model: Optional[M] = None,
        model_init: Optional[Callable[[], M]] = None,
        *,
        is_trainable: PyTree[FilterSpec] = True,
    ) -> TrainerState[M]:
        """
        Either loads a checkpoint or initializes a fresh trainer state. This is the recommended way to initialize
        a trainer state.

        This method is smart enough to handle subclasses of TrainerState. If you want to extend TrainerState, you
        can override _initialize_state_from_scratch

        Args
            is_trainable: optional filter spec for the trainable parameters. This is used to filter out non-trainable
                parameters for the optimizer state and for computing gradients. Non-trainable parameters are also
                not checkpointed. If you don't specify this, all parameters are assumed to be trainable.

        Returns:
            TrainerState: the initial state,
        """
        model_init = _unify_model_and_model_init(model, model_init)

        del model
        assert model_init is not None

        # first try to load a full trainer state checkpoint
        checkpoint_path = self.checkpoint_path

        load_checkpoint = self.config.load_checkpoint
        # we don't save the full trainer state, so we need to filter out the non-trainable parameters
        if load_checkpoint is True and not fsspec_utils.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist")
        elif load_checkpoint is None:
            load_checkpoint = fsspec_utils.exists(checkpoint_path)

        if load_checkpoint is False and self.config.initialize_from is not None:
            # we're not going to load a checkpoint, so see if we can initialize from a model
            logger.info(f"Initializing from {self.config.initialize_from}")
            # todo: we are potentially holding two models in memory at once here, if we pass in a model
            # instead of a model_init and we use initialize_from. We could avoid this by deleting
            # any to-be-loaded parameters from the model before loading, but that's a bit more complicated
            # it also seems unlikely that we'd want to do this, so I'm not going to bother for now
            loaded_model = load_checkpoint_or_initialize(
                model_init,
                self.config.initialize_from,
                axis_mapping=self.parameter_axis_mapping,
                mesh=self.device_mesh,
                subpath="model",
                do_load=True,
            )()
            model_init = jax.tree_util.Partial(lambda m: m, loaded_model)

        def init_state_and_model(model_init, training_key):
            model = model_init()
            # only force trainable params to param precision. Other params are cast to compute precision
            state = TrainerState.init(
                self.optimizer,
                model,
                key=training_key,
                is_trainable=is_trainable,
                mp=self.mp,
                fp8=self.fp8,
            )
            return state

        trainer_state_shape = eqx.filter_eval_shape(init_state_and_model, model_init, training_key)
        saveable_train_state = saveable_training_mask(trainer_state_shape, is_trainable)

        state = load_checkpoint_or_initialize(
            init_state_and_model,
            checkpoint_path,
            axis_mapping=self.parameter_axis_mapping,
            mesh=self.device_mesh,
            is_checkpointed=saveable_train_state,
            do_load=load_checkpoint,
        )(model_init, training_key)

        return state

    @property
    def checkpoint_path(self) -> str:
        checkpoint_path = self.config.load_checkpoint_path
        if checkpoint_path is None:
            checkpoint_path = self.config.checkpointer.expanded_path(self.run_id)
        return checkpoint_path

    def train_step(self, state: S, *batch: X, **batch_kwargs) -> StepInfo[S]:
        """
        Performs a single training step.
        """
        with capture_time() as step_time:
            loss, new_state = self._jit_train_step_fn(state, *batch, **batch_kwargs)
            # force the loss so timing numbers are accurate. laziness isn't going to help here (i think?)
            loss = loss.item()  # type: ignore

        return StepInfo(new_state, loss, step_time())

    def training_steps(self, state: S, train_loader, run_hooks: bool = True) -> typing.Iterator[StepInfo[S]]:
        """
        Generator that yields training steps and runs hooks.
        """
        iter_data = iter(train_loader)

        while int(state.step) < self.num_train_steps:
            with capture_time() as loading_time:
                example = next(iter_data)
            info = self.train_step(state, example)
            state = info.state

            if run_hooks:
                with capture_time() as hook_time:
                    self.run_hooks(info)

                levanter.tracker.log_metrics({"throughput/hook_time": hook_time()}, step=info.step)

            levanter.tracker.log_metrics({"throughput/loading_time": loading_time()}, step=info.step)

            yield info

    def train(self, state: S, train_loader: Iterable[X], run_hooks: bool = True) -> StepInfo[S]:
        """
        Performs training until the number of steps is reached.
        """
        for info in self.training_steps(state, train_loader, run_hooks=run_hooks):
            pass

        if run_hooks:
            # force hooks to run at the end
            self.run_hooks(info, force=True)

        return info

    def _add_default_hooks(self):
        from levanter import callbacks

        self.add_hook(callbacks.pbar_logger(total=self.config.num_train_steps), every=1)
        self.add_hook(callbacks.log_step_info(self.config.num_train_steps), every=1)
        # engine.add_hook(callbacks.log_memory_usage(), every=1)
        checkpointer = self.config.checkpointer.create(self.run_id)
        self.add_hook(checkpointer.on_step, every=1)  # checkpointer manages its own frequency
        if self.config.profiler:
            profile_path = self.config.log_dir / self.run_id / "profiler"
            total_prof_steps = self.config.profiler_num_steps
            if total_prof_steps + self.config.profiler_start_step > self.config.num_train_steps:
                logger.warning(
                    f"Adjusting profiler_total_steps from {total_prof_steps} to"
                    f" {self.config.num_train_steps - self.config.profiler_start_step}"
                )
                total_prof_steps = self.config.num_train_steps - self.config.profiler_start_step
            self.add_hook(
                callbacks.profile(
                    str(profile_path),
                    self.config.profiler_start_step,
                    total_prof_steps,
                    self.config.profiler_perfetto_link,
                ),
                every=1,
            )

    def add_eval_hook(self, eval_dataset, name: Optional[str] = None):
        from levanter import callbacks

        eval_loader = self.data_loader(eval_dataset, self.EvalBatch)

        if eval_loader and (self.config.max_eval_batches is None or self.config.max_eval_batches > 0):

            @eqx.filter_jit
            def eval_loss(model, *batch, **batch_kwargs):
                model = inference_mode(model, True)
                model = self.mp.cast_to_compute(model)
                return self.loss_fn(model, *batch, **batch_kwargs, key=None)

            self.add_hook(
                callbacks.compute_validation_loss(
                    eval_loss,
                    eval_loader,
                    max_batches=self.config.max_eval_batches,
                    name=name,
                ),
                every=self.config.steps_per_eval,
            )

    def data_loader(self, dataset: AsyncDataset[X], batch_axis: Axis) -> DataLoader[X]:
        """Creates a data loader for the given dataset and batch axis.

        Args:
            dataset (AsyncDataset): the dataset to load
            batch_axis (Axis): the batch axis

        Returns:
            DataLoader: the data loader
        """
        return DataLoader(
            batch_axis,
            dataset,
            max_buffered_batches=128,
            mesh=self.device_mesh,
            axis_resources=self.compute_axis_mapping,
            prefetch_size=32,
        )

    @cached_property
    def _jit_train_step_fn(self):
        return named_jit(
            self._train_step,
            axis_resources=self.parameter_axis_mapping,
            out_axis_resources=self.parameter_axis_mapping,
            donate_args=(True,),
        )

    def _train_step(self, state: S, *batch, **batch_kwargs) -> tuple[Scalar, S]:
        key, new_key = jax.random.split(state.training_key)
        model = inference_mode(state.model, False)

        loss, grads = self._compute_gradients_microbatched(self.loss_fn, model, *batch, **batch_kwargs, key=key)

        # Sophia needs to be able to access the loss function in the optimizer
        def obj_fun(trainable_model):
            model = eqx.combine(trainable_model, state.model)
            with hax.axis_mapping(self.compute_axis_mapping):
                model = self.mp.cast_to_compute(model)
                return self._raw_loss_function(model, *batch, **batch_kwargs, key=key).scalar()

        new_state = state.take_step(grads, obj_fun=obj_fun)
        new_state = hax.shard(new_state, self.parameter_axis_mapping)
        return loss, new_state

    def _compute_gradients_microbatched(self, loss_fn, model: M, *batch, **batch_kwargs) -> tuple[Scalar, M]:
        grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=False)
        mbs = self.config.microbatch_size
        grad_fn = microbatched(
            grad_fn,
            self.TrainBatch,
            mbs,
            self.parameter_axis_mapping,
            self.compute_axis_mapping,
        )
        with hax.axis_mapping(self.compute_axis_mapping):
            return grad_fn(model, *batch, **batch_kwargs)


def _initialize_global_tracker(config, run_id):
    if isinstance(config, Sequence):
        tracker = levanter.tracker.CompositeTracker([c.init(run_id) for c in config])
    else:
        tracker = config.init(run_id)

    levanter.tracker.set_global_tracker(tracker)


@dataclass
class TrainerConfig:
    seed: int = 0  # random seed
    mp: jmp.Policy = jmp.get_policy("f32")  # mixed precision policy
    fp8: Optional[bool | Fp8Config] = None

    wandb: Optional[tracker.wandb.WandbConfig] = None
    log_dir: Path = Path("logs/")
    id: Optional[str] = None  # run id. if None, will be set to a random string

    tracker: TrackerConfig | Tuple[TrackerConfig, ...] = field(default_factory=tracker.wandb.WandbConfig)

    # TODO: refactor callbacks
    profiler: bool = False
    profiler_start_step: int = 5
    profiler_num_steps: int = 100
    profiler_perfetto_link: bool = False

    # config related to partitioning

    batch_axis: Optional[str] = "batch"  # Batch axis for data parallel.
    fsdp_axis: Optional[Union[str, List[str]]] = "embed"  # Axis/Axes to use for FSDP
    tensor_parallel_axes: Optional[List[str]] = None  # Axes, if any, to use for tensor parallelism

    axis_resources: Mapping[str, Union[Tuple[str], str]] = field(default_factory=dict)
    """mapping from logical axis to physical axis. batch_axis, fsdp_axis, and tensor_parallel_axes are preferred"""
    parameter_axis_resources: Mapping[str, Union[Tuple[str], str]] = field(
        default_factory=dict
    )  # overrides axis_mapping for parameter
    """logical->physical mapping for parameter/optimizer sharding. fsdp_axis and tensor_parallel_axes are preferred"""

    """Interchip Interconnect (ICI) & Data Center Networking (DCN) shardings https://cloud.google.com/tpu/docs/multislice-introduction"""
    replica_ici_axis_size: int = 1
    model_axis_size: int = 1
    """how many devices within each slice for sharding with DP. Fix TP=1, the rest of the devices is for FSDP."""
    replica_dcn_axis_size: int = 1
    """how many slices in the multislice scheme for sharding with DP and TP. The rest of the devices is for FSDP."""

    # Config related to batch sizes
    train_batch_size: int = 512
    per_device_parallelism: int = -1
    """how many examples to process in parallel on each device. -1 (default) means train_batch_size/num_devices"""

    per_device_eval_parallelism: int = -1
    """how many examples to process in parallel on each device. -1 (default) means same as per_device_parallelism"""

    # Config related to duration
    num_train_steps: int = 400_000  # number of training steps
    steps_per_eval: int = 1_000  # how often to evaluate
    max_eval_batches: Optional[int] = None  # max number of batches to evaluate on. None means all batches

    checkpointer: CheckpointerConfig = field(default_factory=CheckpointerConfig)
    load_checkpoint: Optional[bool] = None
    """if None (default), we'll load a checkpoint if it exists. If true, we must load a checkpoint"""
    load_checkpoint_path: Optional[str] = None
    """can be a parent (to find latest) or a specific checkpoint. if None, will set to checkpointer.base_path."""
    initialize_from: Optional[str] = None  # Levanter trainer checkpoint to initialize from

    jax_config: Mapping[str, JsonAtom] = field(
        default_factory=lambda: copy.deepcopy(DEFAULT_JAX_CONFIG)
    )  # config to pass to jax.config.update

    distributed: DistributedConfig = DistributedConfig()
    ray: RayConfig = field(default_factory=RayConfig)

    # whether or not to require an accelerator (e.g. TPU or GPU).
    # default depends on the platform: on macos False, else True
    require_accelerator: Optional[bool] = None

    # whether or not to shutdown the tpu at exit. If a float, shutdown after that many seconds. True = 5 minutes
    shutdown_at_exit: Union[bool, float] = False

    @property
    def TrainBatch(self):
        return Axis("batch", self.train_batch_size)

    @property
    def EvalBatch(self):
        return Axis("batch", self.eval_batch_size)

    @property
    def microbatch_size(self):
        return self.per_device_parallelism * self.data_axis_size

    def __post_init__(self):
        if self.wandb is not None:
            warnings.warn(
                "wandb is deprecated. use tracker with type wandb instead",
                DeprecationWarning,
            )
            self.tracker = self.wandb

    def initialize(self):
        """Initializes jax, logging, setting the run name/id in the process"""
        self._initialize_jax_config()
        # Can't do full logging setup until we've initialized jax b/c we use jax for rank id
        pylogging.basicConfig(level=pylogging.INFO)
        self.distributed.initialize()
        self._validate_and_set_defaults()

        id = self._maybe_set_id()
        levanter.logging.init_logging(self.log_dir, f"{id}.log")
        _initialize_global_tracker(self.tracker, id)

        self.ray.initialize()

        if self.require_accelerator is None:
            self.require_accelerator = not sys.platform.startswith("darwin")

        if self.require_accelerator:
            if jax.default_backend() == "cpu":
                raise RuntimeError("No accelerator found. Please run on a TPU or GPU.")

        if self.shutdown_at_exit is not False:
            if isinstance(self.shutdown_at_exit, bool):
                self.shutdown_at_exit = 5.0 * 60
            logger.info(f"At end of run, shutting down TPU VM in {self.shutdown_at_exit} seconds")
            atexit.register(cloud_utils.shutdown_tpu_vm, self.shutdown_at_exit)

    @cached_property
    def device_mesh(self) -> Mesh:
        return create_fsdp_mesh(
            self.replica_ici_axis_size,
            self.data_ici_axis_size,
            self.model_axis_size,
            self.replica_dcn_axis_size,
            self.data_dcn_axis_size,
        )

    @property
    def eval_batch_size(self):
        return self.per_device_eval_parallelism * self.data_axis_size

    @cached_property
    def num_slices(self):
        """number of nodes"""
        return max(getattr(device, "slice_index", 0) for device in jax.devices()) + 1

    @property
    def num_devices_per_slice(self):
        """number of devices within a slice"""
        return jax.device_count() // self.num_slices

    @property
    def data_ici_axis_size(self):
        """size of the FSDP axis within slices"""
        assert self.num_devices_per_slice % (self.replica_ici_axis_size * self.model_axis_size) == 0
        return self.num_devices_per_slice // (self.replica_ici_axis_size * self.model_axis_size)

    @property
    def data_dcn_axis_size(self):
        """size of the FSDP axis across slices"""
        assert self.num_slices % self.replica_dcn_axis_size == 0
        return self.num_slices // self.replica_dcn_axis_size

    @property
    def data_axis_size(self):
        """size of the data parallel/batch parallel axis."""
        return (
            self.data_dcn_axis_size * self.data_ici_axis_size * self.replica_dcn_axis_size * self.replica_ici_axis_size
        )

    @property
    def replica_axis_size(self):
        """size of the data parallel/batch parallel axis."""
        return self.replica_dcn_axis_size * self.replica_ici_axis_size

    @cached_property
    def compute_axis_mapping(self) -> ResourceMapping:
        """Mapping from logical axis to physical axis for compute."""
        axes_to_return = dict(self.axis_resources)

        tp_axes = self.tensor_parallel_axes or []
        if tp_axes and len(axes_to_return) > 0:
            logger.warning(f"tensor parallelism axes {tp_axes} will override axis_resources {axes_to_return}")
        for axis in tp_axes:
            axes_to_return[axis] = ResourceAxis.MODEL

        if self.batch_axis is not None:
            axes_to_return[self.batch_axis] = (ResourceAxis.REPLICA, ResourceAxis.DATA)  # type: ignore

        return axes_to_return

    @cached_property
    def parameter_axis_mapping(self) -> ResourceMapping:
        mapping = dict(self.compute_axis_mapping)

        for axis, resource in self.parameter_axis_resources.items():
            mapping[axis] = resource

        if isinstance(self.fsdp_axis, str):
            mapping[self.fsdp_axis] = ResourceAxis.DATA
        elif isinstance(self.fsdp_axis, list):
            for axis in self.fsdp_axis:
                mapping[axis] = ResourceAxis.DATA

        return mapping

    def _initialize_jax_config(self):
        for key, value in self.jax_config.items():
            jax.config.update(key, value)

    def _maybe_set_id(self):
        # always do this so we don't get weird hangs if the id isn't set right
        # for random ids, we want to ensure that all hosts have the same id
        # NB: do NOT use the run seed here. we want the run id to be independent of the seed
        seed = np.random.randint(0, 2**31 - 1)
        seed = multihost_utils.broadcast_one_to_all(jax.numpy.array(seed, dtype=np.int32)).item()

        # RUN ID comes from a few places: the config, the environment, or wandb, or a random string
        if self.id is None:
            # TODO: this doesn't work with wandb sweeps. need to reconcile when we merge
            if "RUN_ID" in os.environ:
                self.id = os.environ["RUN_ID"]
            elif self.wandb is not None and self.wandb.id is not None:
                self.id = self.wandb.id
            else:
                # wandb run ids are 8 characters [a-z0-9], which we'll emulate here
                # we also want to ensure that all hosts have the same run id
                # we do this by syncing a random seed across all hosts and then using that to generate the run id
                gen = np.random.default_rng(seed)
                self.id = "".join(gen.choice(list("abcdefghijklmnopqrstuvwxyz0123456789"), 8))

            logger.info(f"Setting run id to {self.id}")

        return self.id

    # we can't do this in post_init because we don't want to call jax.device_count before calling distributed.initialize
    def _validate_and_set_defaults(self):
        if jax.device_count() % self.model_axis_size != 0:
            raise ValueError(
                f"num_devices ({jax.device_count()}) is not divisible by model_axis_size ({self.model_axis_size})"
            )

        if (
            jax.local_device_count() % self.model_axis_size != 0
            and self.model_axis_size % jax.local_device_count() != 0
        ):
            raise ValueError("either model_axis_size or local_device_count must be divisible by the other")

        assert self.train_batch_size != -1 or self.per_device_parallelism != -1

        if self.per_device_parallelism == -1:
            self.per_device_parallelism = self.train_batch_size // self.data_axis_size

        if self.train_batch_size == -1:
            self.train_batch_size = self.per_device_parallelism * self.data_axis_size

        # validate size of per_device_parallelism
        if self.train_batch_size % (self.per_device_parallelism * self.data_axis_size) != 0:
            raise ValueError(
                f"train_batch_size ({self.train_batch_size}) must be divisible by per_device_parallelism *"
                f" data_axis_size ({self.per_device_parallelism}, {self.data_axis_size})"
            )

        if self.per_device_eval_parallelism == -1:
            self.per_device_eval_parallelism = self.per_device_parallelism

        if self.replica_dcn_axis_size == -1:
            self.replica_dcn_axis_size = self.num_slices
            logger.info(f"Setting replica_dcn_axis_size to {self.replica_dcn_axis_size}")


class AllConfig(Protocol):
    trainer: TrainerConfig


def initialize(config: TrainerConfig | AllConfig):
    """Initializes jax, logging, setting the run name/id in the process. Also initializes tracking and saves config
    as hyperparameters and an artifact"""
    if isinstance(config, TrainerConfig):
        trainer_config = config
    else:
        trainer_config = config.trainer

    trainer_config.initialize()
    levanter.tracker.log_configuration(config)


def _ensure_scalar(x: hax.types.Scalar | hax.NamedArray) -> hax.types.Scalar:
    if isinstance(x, hax.NamedArray):
        return x.scalar()
    else:
        return x
