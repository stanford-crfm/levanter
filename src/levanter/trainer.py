import atexit
import copy
import logging as pylogging
import sys
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, TypeVar, Union

import jax
import jmp
import numpy as np
import optax
from draccus import field
from jax._src.interpreters.pxla import Mesh
from jaxtyping import PRNGKeyArray, PyTree

from haliax.partitioning import ResourceAxis, ResourceMapping

import levanter.logging
from levanter.checkpoint import CheckpointerConfig
from levanter.config import JsonAtom
from levanter.distributed import DistributedConfig, RayConfig
from levanter.logging import WandbConfig
from levanter.utils import cloud_utils


logger = pylogging.getLogger(__name__)

M = TypeVar("M")
S = TypeVar("S")
DEFAULT_JAX_CONFIG = {
    "jax_threefry_partitionable": True,
}


@dataclass
class StepInfo:
    step: int
    model: PyTree
    opt_state: Any
    loss: float
    next_key: PRNGKeyArray
    step_duration: float


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


@dataclass
class TrainerConfig:
    seed: int = 0
    mp: jmp.Policy = jmp.get_policy("f32")

    wandb: WandbConfig = field(default_factory=WandbConfig)
    log_dir: Path = Path("logs/")
    run_base_dir: Path = Path("runs/")

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
    num_train_steps: int = 400_000
    steps_per_eval: int = 1_000  # how often to evaluate
    max_eval_batches: Optional[int] = None  # max number of batches to evaluate on. None means all batches

    checkpointer: CheckpointerConfig = field(default_factory=CheckpointerConfig)
    load_checkpoint: Optional[bool] = None
    """if None (default), we'll load a checkpoint if it exists. If true, we must load a checkpoint"""
    load_checkpoint_path: Optional[str] = None
    """can be a parent (to find latest) or a specific checkpoint. if None, will set to checkpointer.base_path."""

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
    def run_name(self) -> str:
        import wandb

        return wandb.run and (wandb.run.name or wandb.run.id) or "unnamed"

    @property
    def run_id(self) -> str:
        import wandb

        return wandb.run and wandb.run.id or "unnamed"

    @property
    def run_dir(self) -> Path:
        return self.run_base_dir / self.run_name

    def initialize(self, all_config):
        """Initializes jax, wandb, logging, setting the run name in the process"""
        self.distributed.initialize()
        self.ray.initialize()
        self._initialize_jax_config()
        self.wandb.init(all_config)
        self._initialize_logging()
        self._validate_and_set_defaults()

        if self.require_accelerator is None:
            self.require_accelerator = not sys.platform.startswith("darwin")

        if self.require_accelerator:
            assert jax.default_backend() != "cpu", "Accelerator required but not found"

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

    def _initialize_logging(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        levanter.logging.init_logger(self.log_dir / f"{self.run_name}.log")

    def maybe_load_checkpoint(
        self, model: M, training_state: S, *, axis_mapping=None, mesh=None
    ) -> Tuple[M, S, Optional[int]]:
        """Loads a checkpoint if one exists and we're supposed to load it,
        otherwise returns the model and training state as is"""
        if self.load_checkpoint is not False:
            checkpointer = self.checkpointer.create(self.run_id)
            assert (
                self.load_checkpoint_path is not None
            ), "load_checkpoint_path should have been set during initialization"
            ckpt = checkpointer.load_checkpoint(
                model, training_state, self.load_checkpoint_path, axis_mapping=axis_mapping, mesh=mesh
            )

            if ckpt is None:
                if self.load_checkpoint is True:
                    raise ValueError(f"Could not load checkpoint from {self.load_checkpoint_path}")
                return (model, training_state, None)
            else:
                model, training_state, step = ckpt
                return (model, training_state, step)
        else:
            return (model, training_state, None)

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

        if self.per_device_parallelism == -1:
            self.per_device_parallelism = self.train_batch_size // jax.device_count()

        # validate size of per_device_parallelism
        if self.train_batch_size % (self.per_device_parallelism * self.data_axis_size) != 0:
            raise ValueError(
                f"train_batch_size ({self.train_batch_size}) must be divisible by per_device_parallelism *"
                f" data_axis_size ({self.per_device_parallelism}, {self.data_axis_size})"
            )

        if self.per_device_eval_parallelism == -1:
            self.per_device_eval_parallelism = self.per_device_parallelism

        if self.load_checkpoint_path is None:
            self.load_checkpoint_path = self.checkpointer.expanded_path(self.run_id)


@dataclass
class OptimizerConfig:
    # Config related to optimizer (always adam for now)
    learning_rate: float = 6e-4
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    max_grad_norm: Optional[float] = 1.0

    min_lr_ratio: float = 0.0
    warmup_ratio: float = 0.01  # fraction of training steps to use as warmup
    lr_schedule: str = "cosine"  # constant, cosine, linear

    def build(self, num_train_steps):
        """Creates the optimizer"""
        # indirection makes it work with optax.inject_hyperparams so we can log the learning rate
        def _optimizer(learning_rate):
            components = []

            if self.max_grad_norm:
                components.append(optax.clip_by_global_norm(self.max_grad_norm))

            components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))

            if self.weight_decay > 0:
                # TODO: add weight decay masking??
                components.append(optax.add_decayed_weights(self.weight_decay))

            # - learning rate for descent
            components.append(optax.scale(-learning_rate))

            optimizer = optax.chain(*components)

            return optimizer

        return optax.inject_hyperparams(_optimizer)(learning_rate=self.lr_scheduler(num_train_steps))

    def lr_scheduler(self, num_train_steps):
        warmup_steps = int(self.warmup_ratio * num_train_steps)
        lr_decay_steps = num_train_steps - warmup_steps
        min_lr = self.learning_rate * self.min_lr_ratio

        if self.lr_schedule == "constant":
            schedule = optax.constant_schedule(self.learning_rate)
        elif self.lr_schedule == "cosine":
            schedule = optax.cosine_decay_schedule(self.learning_rate, lr_decay_steps, self.min_lr_ratio)
        elif self.lr_schedule == "linear":
            schedule = optax.linear_schedule(self.learning_rate, min_lr, lr_decay_steps - warmup_steps)
        else:
            raise ValueError(f"Unknown lr_schedule: {self.lr_schedule}")

        if warmup_steps != 0:
            warmup = optax.linear_schedule(0.0, self.learning_rate, warmup_steps)
            schedule = optax.join_schedules([warmup, schedule], [warmup_steps])

        return schedule
