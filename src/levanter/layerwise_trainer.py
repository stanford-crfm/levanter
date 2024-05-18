"""Trainer for layer-wise distillation.

This accepts a configuration with a teacher and a student model.
The student model is assumed to have the same output dimensions for each layer,
but will typically have a factorized internal structure in order to reduce the
number of parameters.
"""

import atexit
import copy
import dataclasses
import functools
import logging as pylogging
import os
import sys
import typing
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
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
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from draccus import field
from haliax import Axis
from haliax.partitioning import ResourceAxis, ResourceMapping, named_jit
from haliax.quantization import (
    Fp8Config,
    fp8_linear_layers,
)
from haliax.types import IntScalar, Scalar
from jax.experimental import multihost_utils
from jax.sharding import Mesh
from jaxtyping import PRNGKeyArray, PyTree
from optax import GradientTransformation, OptState

import levanter.checkpoint
import levanter.logging
import levanter.tracker
from levanter.tracker.helpers import log_optimizer_hyperparams
import levanter.tracker.wandb
from levanter import tracker
from levanter.checkpoint import (
    CheckpointerConfig,
    discover_latest_checkpoint,
    load_checkpoint,
    load_checkpoint_or_initialize,
)
from levanter.config import JsonAtom
from levanter.data import (
    Dataset,
    ReplicatedBatchLoader,
    ShardableDataset,
    ShardedBatchLoader,
)
from levanter.distributed import DistributedConfig, RayConfig
from levanter.logging import capture_time
from levanter.models.lm_model import LmExample
from levanter.tracker import TrackerConfig
from levanter.trainer_state import (
    _ensure_int_is_array,
    cast_params_by_trainability,
    init_optimizer_for_trainables,
    saveable_training_mask,
    take_train_step,
    trainables_only,
)
from levanter.types import (
    ComputeLossFunction,
    FilterSpec,
    FilterTree,
    ModuleComputeLoss,
)
from levanter.utils import cloud_utils, fsspec_utils
from levanter.utils.tree_utils import inference_mode

logger = pylogging.getLogger(__name__)

M = TypeVar("M")  # Model
X = TypeVar("X")  # Input
S = TypeVar("S")
Student = TypeVar("Student")
Teacher = TypeVar("Teacher")

DEFAULT_JAX_CONFIG = {
    "jax_threefry_partitionable": True,
    "jax_softmax_custom_jvp": True,
}


def _per_layer_loss(inputs, layer, key, mask):
    loss, teacher_x = inputs
    key, layer_k = jax.random.split(key, 2)
    student, teacher = layer
    student_y = student(teacher_x, mask=mask, key=layer_k)
    teacher_y = teacher(teacher_x, mask=mask, key=layer_k)
    loss += hax.mean((teacher_y - student_y) ** 2)
    return (loss, teacher_y)


def _layer_loss(student, teacher, batch, key, compute_axis_mapping):
    with hax.axis_mapping(compute_axis_mapping):
        teacher_x = teacher.embeddings.embed(batch.tokens)
        student_layers = student.transformer.layers.stacked
        teacher_layers = teacher.transformer.layers.stacked
        initial_loss = hax.NamedArray(0.0, axes=())

        block_axis = student.transformer.layers.Block

        def _is_block(leaf):
            if hax.is_named_array(leaf):
                print(leaf.shape, leaf.axes)
            return hax.is_named_array(leaf) and leaf.axes[0] == block_axis

        _loss_fn = hax.filter_checkpoint(_per_layer_loss)

        loss, _ = hax.fold(
            _loss_fn,
            axis=block_axis,
            is_scanned=_is_block,
        )(
            (initial_loss, teacher_x),
            (student_layers, teacher_layers),
            key=key,
            mask=batch.attn_mask,
        )
        return loss.scalar()


class TrainerState(eqx.Module):
    step: IntScalar = eqx.field(converter=_ensure_int_is_array)
    student: Student
    teacher: Teacher
    optimizer: GradientTransformation = eqx.field(static=True)
    opt_state: OptState
    training_key: PRNGKeyArray

    is_trainable: FilterTree = eqx.field(static=True)
    mp: jmp.Policy = eqx.field(static=True)

    @property
    def trainable_model(self) -> M:
        return trainables_only(self.model, self.is_trainable)

    @property
    def saveable_state(self) -> FilterTree:
        return eqx.filter(self, saveable_training_mask(self, self.is_trainable))

    @classmethod
    def init(
        cls,
        optimizer: GradientTransformation,
        student: Student,
        teacher: Teacher,
        *args,
        key: PRNGKeyArray,
        is_trainable: FilterTree = True,
        mp: Optional[jmp.Policy] = None,
        fp8: Fp8Config = None,
        **kwargs,
    ) -> "TrainerState":
        if mp is not None:
            student = cast_params_by_trainability(student, mp, is_trainable)
            teacher = cast_params_by_trainability(teacher, mp, is_trainable)
        else:
            mp = jmp.get_policy("f32")

        if fp8 is not None:
            student = fp8_linear_layers(student, fp8)

        opt_state = init_optimizer_for_trainables(optimizer, student, is_trainable)
        return cls(
            0,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            opt_state=opt_state,
            training_key=key,
            is_trainable=is_trainable,
            mp=mp,
            *args,
            **kwargs,
        )

    def take_step(self: S, grads: PyTree, obj_fun: Optional[Callable[[M], Scalar]] = None) -> S:
        assert isinstance(self, TrainerState)  # make mypy happy
        student, opt_state = take_train_step(
            optimizer=self.optimizer,
            model=self.student,
            opt_state=self.opt_state,
            grads=grads,
            obj_fun=obj_fun,
            is_trainable=self.is_trainable,
        )
        return dataclasses.replace(self, student=student, opt_state=opt_state, step=self.step + 1)


def init_model(
    model_init: Optional[Callable[[], M]],
    checkpoint_path: Path,
    axis_mapping: ResourceMapping,
    device_mesh: Mesh,
):
    if not checkpoint_path or not fsspec_utils.exists(checkpoint_path):
        return model_init()

    checkpoint_path = discover_latest_checkpoint(checkpoint_path)

    if checkpoint_path:
        loaded_model = load_checkpoint_or_initialize(
            model_init,
            checkpoint_path,
            axis_mapping=axis_mapping,
            mesh=device_mesh,
            subpath="model",
            do_load=True,
        )()
        return loaded_model
    else:
        return model_init()


@dataclass
class StepInfo:
    loss: float
    step_duration: float
    step: int
    next_step: int


class Trainer:
    config: "TrainerConfig"
    optimizer: GradientTransformation
    is_trainable_param: PyTree[FilterSpec]
    _raw_loss_function: Callable
    _cmanagers: List[typing.ContextManager] = []

    def __init__(
        self,
        config: "TrainerConfig",
        optimizer: GradientTransformation,
        loss_fn: Optional[ComputeLossFunction] = None,
    ):
        """

        Args:
            config:  the trainer config
            optimizer: the optimizer, e.g. `optax.adam(1e-3)` or produced by [levanter.optim.OptimizerConfig][]
            loss_fn (Callable): the loss function. This should be a function that takes a model and some inputs and returns a
                scalar loss. It should be jit-able and should not have any side effects.
        """
        self.config = config
        self.optimizer = optimizer
        self._raw_loss_function = loss_fn or ModuleComputeLoss()
        if isinstance(config.tracker, Sequence):
            self.tracker = levanter.tracker.CompositeTracker([c.init(self.run_id) for c in config.tracker])
        else:
            self.tracker = config.tracker.init(self.run_id)

        self._raw_loss_function = loss_fn or ModuleComputeLoss()
        if isinstance(config.tracker, Sequence):
            self.tracker = levanter.tracker.CompositeTracker([c.init(self.run_id) for c in config.tracker])
        else:
            self.tracker = config.tracker.init(self.run_id)

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
        student_init,
        teacher_init,
        *,
        is_trainable: PyTree[FilterSpec] = True,
    ) -> TrainerState:
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

        def init_state_and_model(training_key):
            student = init_model(
                model_init=student_init,
                checkpoint_path=self.config.load_checkpoint_path,
                axis_mapping=self.parameter_axis_mapping,
                device_mesh=self.device_mesh,
            )
            teacher = init_model(
                model_init=teacher_init,
                checkpoint_path=self.config.load_checkpoint_path,
                axis_mapping=self.parameter_axis_mapping,
                device_mesh=self.device_mesh,
            )
            # only force trainable params to param precision. Other params are cast to compute precision
            state = TrainerState.init(
                self.optimizer,
                student=student,
                teacher=teacher,
                key=training_key,
                is_trainable=is_trainable,
                mp=self.mp,
                fp8=self.fp8,
            )
            return state

        trainer_state_shape = eqx.filter_eval_shape(init_state_and_model, training_key)
        saveable_train_state = saveable_training_mask(trainer_state_shape, is_trainable)

        state = load_checkpoint_or_initialize(
            init_state_and_model,
            self.checkpoint_path,
            axis_mapping=self.parameter_axis_mapping,
            mesh=self.device_mesh,
            is_checkpointed=saveable_train_state,
            do_load=load_checkpoint,
        )(training_key)

        return state

    @property
    def checkpoint_path(self) -> str:
        checkpoint_path = self.config.load_checkpoint_path
        if checkpoint_path is None:
            checkpoint_path = self.config.checkpointer.expanded_path(self.run_id)
        return checkpoint_path

    def replicated_loader(self, dataset: Dataset[X], batch_axis: Axis) -> ReplicatedBatchLoader[X]:
        """Creates a replicated batch loader for the given dataset. Generally you should use this
        if you either be able to make a single pass over the dataset.

        Args:
            dataset (Dataset): the dataset to load
            batch_axis (Axis): the batch axis

        Returns:
            ReplicatedBatchLoader: the batch loader
        """
        return ReplicatedBatchLoader(dataset, self.device_mesh, batch_axis, self.compute_axis_mapping)

    def sharded_loader(self, dataset: ShardableDataset[X], batch_axis: Axis) -> ShardedBatchLoader[X]:
        """Creates a sharded batch loader for the given dataset. Generally you should use this
        for training and you don't care about epoch boundaries.

        Args:
            dataset (Dataset): the dataset to load
            batch_axis (Axis): the batch axis

        Returns:
            ShardedBatchLoader: the batch loader
        """
        return ShardedBatchLoader(dataset, self.device_mesh, batch_axis, self.compute_axis_mapping)

    def train(self, state: S, loader: ReplicatedBatchLoader[X]) -> S:
        for step in range(self.num_train_steps):
            batch = next(loader)
            state, loss = self.train_step(state, batch)
            print("Training:", step, loss)
        return state

    def train_step(self, state: S, *batch: X, **batch_kwargs):
        """
        Performs a single training step.
        """
        with capture_time() as step_time:
            loss, state = self._jit_train_step_fn(state, *batch, **batch_kwargs)
            # force the loss so timing numbers are accurate. laziness isn't going to help here (i think?)
            loss = loss.item()  # type: ignore

            step = int(state.step) - 1

            levanter.tracker.log_metrics({"train/loss": loss, "global_step": step, "step_time": step_time}, step=step)
            log_optimizer_hyperparams(state.opt_state, step=step, prefix="optim")

        return state, loss

    @cached_property
    def _jit_train_step_fn(self):
        return named_jit(
            self._layerwise_train_step,
            axis_resources=self.parameter_axis_mapping,
            out_axis_resources=self.parameter_axis_mapping,
            donate_args=True,
        )

    def _layerwise_train_step(self, state: S, batch: LmExample) -> tuple[Scalar, S]:
        student = inference_mode(state.student, False)
        teacher = inference_mode(state.teacher, True)

        # tokens: hax.NamedArray
        # loss_mask: hax.NamedArray
        # attn_mask: AttentionMask | NamedArray = AttentionMask.causal()

        # manually thread the teacher and student models
        k_t, key = jax.random.split(state.training_key, 2)
        loss, grad = eqx.filter_value_and_grad(_layer_loss)(student, teacher, batch, key, self.compute_axis_mapping)
        new_state = state.take_step(grad)
        new_state = hax.shard(new_state, self.parameter_axis_mapping)
        return loss, new_state


def _initialize_global_tracker(config: TrackerConfig | Tuple[TrackerConfig, ...], run_id: Optional[str]):
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

    log_dir: Path = Path("logs/")
    run_base_dir: Path = Path("runs/")
    id: Optional[str] = None  # run id. if None, will be set to a random string

    tracker: TrackerConfig | Tuple[TrackerConfig, ...] = field(default_factory=tracker.TrackerConfig)

    # TODO: refactor callbacks
    profiler: bool = False
    profiler_start_step: int = 5
    profiler_num_steps: int = 100
    profiler_perfetto_link: bool = False

    # config related to partitioning

    batch_axis: Optional[str] = "batch"  # Batch axis for data parallel.
    fsdp_axis: Optional[Union[str, List[str]]] = "embed"  # Axis/Axes to use for FSDP
    tensor_parallel_axes: Optional[List[str]] = None  # Axes, if any, to use for tensor parallelism

    # TODO: in theory we can support tuples of physical axis names, but I don't think anyone actually uses that.
    axis_resources: Mapping[str, str] = field(default_factory=dict)
    """mapping from logical axis to physical axis. batch_axis, fsdp_axis, and tensor_parallel_axes are preferred"""
    parameter_axis_resources: Mapping[str, str] = field(default_factory=dict)  # overrides axis_mapping for parameter
    """logical->physical mapping for parameter/optimizer sharding. fsdp_axis and tensor_parallel_axes are preferred"""
    model_axis_size: int = 1  # how many devices to shard each model over. Data axis is the other axis

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

    jax_config: Dict[str, JsonAtom] = field(
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
        devices = jax.devices()
        devices = np.array(devices).reshape(self.data_axis_size, self.model_axis_size)
        return Mesh(devices, (ResourceAxis.DATA, ResourceAxis.MODEL))

    @property
    def eval_batch_size(self):
        return self.per_device_eval_parallelism * self.data_axis_size

    @property
    def data_axis_size(self):
        """size of the data parallel/batch parallel axis."""
        assert jax.device_count() % self.model_axis_size == 0
        return jax.device_count() // self.model_axis_size

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
            axes_to_return[self.batch_axis] = ResourceAxis.DATA

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
            elif self.tracker and self.tracker.id:
                self.id = self.tracker.id
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
