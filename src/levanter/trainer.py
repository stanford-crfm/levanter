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
from typing import Any, Callable, Dict, Generic, Iterable, List, Mapping, Optional, Tuple, TypeVar, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import wandb
from draccus import field
from jax import ShapeDtypeStruct
from jax.experimental import multihost_utils
from jax.sharding import Mesh
from jaxtyping import PRNGKeyArray, PyTree
from optax import GradientTransformation, OptState

import haliax as hax
from haliax import Axis
from haliax.partitioning import ResourceAxis, ResourceMapping, named_jit

import levanter.logging
from levanter.checkpoint import CheckpointerConfig
from levanter.config import JsonAtom
from levanter.data import Dataset, ReplicatedBatchLoader, ShardableDataset, ShardedBatchLoader
from levanter.distributed import DistributedConfig, RayConfig
from levanter.grad_accum import accumulate_gradients_sharded
from levanter.logging import WandbConfig, capture_time
from levanter.types import FilterSpec
from levanter.utils import cloud_utils
from levanter.utils.jax_utils import is_inexact_arrayish
from levanter.utils.tree_utils import inference_mode


logger = pylogging.getLogger(__name__)

X = TypeVar("X")  # Input
M = TypeVar("M", bound=PyTree)
S = TypeVar("S", bound=PyTree)

DEFAULT_JAX_CONFIG = {
    "jax_threefry_partitionable": True,
    "jax_softmax_custom_jvp": True,
}

# A note on the semantics of "step" vs "next_step":
# The "step" of a TrainerState is the state after `step` steps have been taken.
# A "StepInfo"'s step is the step that was just completed. If you want the next step, use `next_step`.


@dataclass
class TrainerState(Generic[M]):
    step: int
    model: M
    opt_state: OptState
    training_key: PRNGKeyArray


@dataclass
class StepInfo(Generic[M]):
    state: TrainerState[M]
    loss: float
    step_duration: float

    model = property(lambda self: self.state.model)
    opt_state = property(lambda self: self.state.opt_state)
    next_key = property(lambda self: self.state.training_key)

    step = property(lambda self: self.state.step - 1)
    """
    The step that was just completed. If you want the next step, use `next_step`.
    """
    next_step = property(lambda self: self.state.step)


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


class Trainer:
    config: "TrainerConfig"
    optimizer: GradientTransformation
    hooks: TrainerHooks
    is_trainable_param: Optional[PyTree[FilterSpec]]
    _raw_loss_function: Callable

    def __init__(
        self,
        config: "TrainerConfig",
        optimizer: GradientTransformation,
        loss_fn: Callable,
        *,
        is_trainable: PyTree[FilterSpec] = True,
    ):
        """

        Args:
            config:  the trainer config
            optimizer: the optimizer, e.g. `optax.adam(1e-3)` or produced by [levanter.trainer.OptimizerConfig][]
            loss_fn (Callable): the loss function. This should be a function that takes a model and some inputs and returns a
                scalar loss. It should be jit-able and should not have any side effects.
            is_trainable: optional filter spec for the trainable parameters. This is used to filter out non-trainable
                parameters for the optimizer state and for computing gradients. Non-trainable parameters are also
                not checkpointed. If you don't specify this, all parameters are assumed to be trainable.
        """
        self.hooks = TrainerHooks()
        self.config = config
        self._raw_loss_function = loss_fn
        self.optimizer = optimizer
        self.is_trainable_param = is_trainable

    @cached_property
    def loss_fn(self):
        """
        Wrapped loss function that casts the model to compute precision and sets the context axis mapping to compute
        """

        @named_jit(in_axis_resources=self.parameter_axis_mapping, axis_resources=self.compute_axis_mapping)
        @functools.wraps(self._raw_loss_function)
        def fn(model, *batch, **batch_kwargs):
            with hax.axis_mapping(self.compute_axis_mapping):
                model = self.mp.cast_to_compute(model)
                return self._raw_loss_function(model, *batch, **batch_kwargs)

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

    def initial_state(
        self, training_key: PRNGKeyArray, model: Optional[M] = None, model_init: Optional[Callable[[], M]] = None
    ) -> TrainerState:
        """
        Initializes the model, optimizer state, and random key. Also handles loading a checkpoint if needed.

        Returns:
            model, opt_state, key, resume_step
        """

        if model is not None and model_init is not None:
            raise ValueError("only one of model and model_init should be specified")
        elif model is None and model_init is None:
            raise ValueError("one of model and model_init must be specified")

        if model is not None:
            # we can't just use `lambda: model` because JAX jit can't see captures, but it can see partials
            # We can't use plain partials because they aren't pytrees
            model_init = jax.tree_util.Partial(lambda m: m, model)

        model_shape, opt_state_shape = eqx.filter_eval_shape(self._init_model_and_opt_state, model_init)

        # we only checkpoint the trainable parameters, so we need to filter out the non-trainable ones
        trainable_model_shape = self.trainable_params_only(model_shape)

        ckpt = self.maybe_load_checkpoint(
            trainable_model_shape,
            (opt_state_shape, training_key),
            axis_mapping=self.parameter_axis_mapping,
            mesh=self.device_mesh,
        )

        if ckpt is not None:
            trainable_model, (opt_state, training_key), completed_step = ckpt
            if model is not None:
                model = eqx.combine(trainable_model, model)
            elif any(isinstance(leaf, ShapeDtypeStruct) for leaf in jax.tree_leaves(trainable_model)):
                # if we're resuming, we need to re-initialize the non-trainable parameters to their original values
                non_trainable = named_jit(self._init_non_trainable_params, self.parameter_axis_mapping)(model_init)
                model = eqx.combine(trainable_model, non_trainable)
            else:
                model = trainable_model
            step = completed_step + 1
        else:
            model, opt_state = named_jit(self._init_model_and_opt_state, self.parameter_axis_mapping)(model_init)
            step = 0

        return TrainerState(step, model, opt_state, training_key)

    def train_step(self, state: TrainerState[M], *batch: X, **batch_kwargs) -> StepInfo[M]:
        """
        Performs a single training step.
        """
        with capture_time() as step_time:
            key, new_key = jax.random.split(state.training_key)
            loss, new_model, new_optstate = self._train_step_fn(
                state.model, state.opt_state, *batch, **batch_kwargs, key=key
            )
            # force the loss so timing numbers are accurate. laziness isn't going to help here (i think?)
            loss = loss.item()  # type: ignore

        return StepInfo(TrainerState(state.step + 1, new_model, new_optstate, new_key), loss, step_time())

    def training_steps(
        self, state: TrainerState[M], train_loader, run_hooks: bool = True
    ) -> typing.Iterator[StepInfo]:
        """
        Generator that yields training steps and runs hooks.
        """
        iter_data = iter(train_loader)

        while state.step < self.config.num_train_steps:
            with capture_time() as loading_time:
                example = next(iter_data)

            # TODO: refactor logging
            wandb.log({"throughput/loading_time": loading_time()}, step=state.step)

            info = self.train_step(state, example)
            state = info.state

            if run_hooks:
                with capture_time() as hook_time:
                    self.run_hooks(info)

                wandb.log({"throughput/hook_time": hook_time()}, step=state.step)

            yield info

    def train(self, state: TrainerState[M], train_loader: Iterable[X], run_hooks: bool = True) -> StepInfo[M]:
        """
        Performs training until the number of steps is reached.
        """
        for info in self.training_steps(state, train_loader, run_hooks=run_hooks):
            pass

        if run_hooks:
            # force hooks to run at the end
            self.run_hooks(info, force=True)

        return info

    def add_default_hooks(self, eval_loader: Optional[Iterable[X]] = None):
        from levanter import callbacks

        self.add_hook(callbacks.pbar_logger(total=self.config.num_train_steps), every=1)
        self.add_hook(callbacks.log_to_wandb, every=1)
        self.add_eval_hook(eval_loader)
        self.add_hook(callbacks.wandb_xla_logger(self.config.wandb), every=self.config.steps_per_eval)
        # engine.add_hook(callbacks.log_memory_usage(), every=1)
        checkpointer = self.config.checkpointer.create(self.run_id, self.is_trainable_param)
        self.add_hook(checkpointer.on_step, every=1)  # checkpointer manages its own frequency

    def add_eval_hook(self, eval_loader):
        from levanter import callbacks

        if eval_loader and (self.config.max_eval_batches is None or self.config.max_eval_batches > 0):

            @eqx.filter_jit
            def eval_loss(model, *batch, **batch_kwargs):
                model = inference_mode(model, True)
                return self.loss_fn(model, *batch, **batch_kwargs, key=None)

            self.add_hook(
                callbacks.compute_validation_loss(eval_loss, eval_loader, max_batches=self.config.max_eval_batches),
                every=self.config.steps_per_eval,
            )

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

    @cached_property
    def _train_step_fn(self):
        @named_jit(
            axis_resources=self.parameter_axis_mapping,
            out_axis_resources=self.parameter_axis_mapping,
            donate_args=(True, True),
        )
        def train_step(model, opt_state, *batch, **batch_kwargs):
            model = inference_mode(model, False)

            # we do this so that we only take the gradients of the trainable parameters
            trainable_model, rest_model = self.partition_trainable_params(model)

            def split_loss_fn(trainable_model, *batch, **batch_kwargs):
                model = eqx.combine(trainable_model, rest_model)
                return self.loss_fn(model, *batch, **batch_kwargs)

            loss, grads = accumulate_gradients_sharded(
                split_loss_fn, self.TrainBatch, self.config.per_device_parallelism, self.parameter_axis_mapping
            )(trainable_model, *batch, **batch_kwargs)

            updates, opt_state = self.optimizer.update(grads, opt_state, params=trainable_model)
            model = eqx.apply_updates(model, updates)

            return loss, model, opt_state

        return train_step

    def _init_model_and_opt_state(self, model_init):
        model = model_init()
        # only force trainable params to param precision. Other params are cast to compute precision
        trainable, non_trainable = self.partition_trainable_params(model)
        trainable = self.mp.cast_to_param(trainable)
        non_trainable = self.mp.cast_to_compute(non_trainable)
        model = eqx.combine(trainable, non_trainable)
        opt_state = self.optimizer.init(trainable)
        return model, opt_state

    def _init_non_trainable_params(self, model_init):
        model = model_init()
        # only force trainable params to param precision. Other params are cast to compute precision
        trainable, non_trainable = self.partition_trainable_params(model)
        non_trainable = self.mp.cast_to_compute(non_trainable)
        return non_trainable

    def trainable_params_only(self, model: M) -> M:
        """
        Filters out non-trainable parameters from the model. This is used internally to
        for the optimizer state and to compute gradients, but you can also use it to filter out
        params for logging or something.
        """
        return self.partition_trainable_params(model)[0]

    def partition_trainable_params(self, model):
        """
        Partitions the model into trainable and non-trainable parameters. This is used internally
        for the gradient calculation and checkpointing, but you can also use it to filter out params for logging
        or something.

        Returns:
            trainable, non-trainable
        """

        def trainable_and_diffable(pred):
            if callable(pred):
                return lambda x: pred(x) and is_inexact_arrayish(x)
            elif pred is True:
                return is_inexact_arrayish
            else:
                return pred

        combined_mask = jax.tree_util.tree_map(trainable_and_diffable, self.is_trainable_param)
        return eqx.partition(model, combined_mask)

    def maybe_load_checkpoint(
        self, model: M, training_state: S, *, axis_mapping=None, mesh=None
    ) -> Optional[Tuple[M, S, int]]:
        """Loads a checkpoint if one exists and we're supposed to load it,
        otherwise returns the model and training state as is"""
        if self.config.load_checkpoint is not False:
            # TODO: don't remake the checkpointer every time
            checkpointer = self.config.checkpointer.create(self.run_id)
            load_checkpoint_path = self.config.load_checkpoint_path

            if load_checkpoint_path is None:
                load_checkpoint_path = self.config.checkpointer.expanded_path(self.run_id)

            ckpt = checkpointer.load_checkpoint(
                model, training_state, load_checkpoint_path, axis_mapping=axis_mapping, mesh=mesh
            )

            if ckpt is None and self.config.load_checkpoint is True:
                raise ValueError(f"Could not load checkpoint from {load_checkpoint_path}")

            return ckpt
        else:
            return None


@dataclass
class TrainerConfig:
    seed: int = 0  # random seed
    mp: jmp.Policy = jmp.get_policy("f32")  # mixed precision policy

    wandb: WandbConfig = field(default_factory=WandbConfig)
    log_dir: Path = Path("logs/")
    run_base_dir: Path = Path("runs/")
    id: Optional[str] = None  # run id. if None, will be set to a random string

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
        try:
            import wandb

            return wandb.run and (wandb.run.name or wandb.run.id) or "unnamed"
        except ImportError:
            return "unnamed"

    @property
    def TrainBatch(self):
        return Axis("batch", self.train_batch_size)

    @property
    def EvalBatch(self):
        return Axis("batch", self.eval_batch_size)

    def initialize(self, all_config):
        """Initializes jax, wandb, logging, setting the run name/id in the process"""
        self.distributed.initialize()
        self._maybe_set_id()
        self.ray.initialize()
        self._initialize_jax_config()
        self._validate_and_set_defaults()
        self.wandb.init(self.id, all_config)
        self._initialize_logging()

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
        levanter.logging.init_logger(self.log_dir / f"{self.id}.log")

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
            elif self.wandb.id is not None:
                self.id = self.wandb.id
            else:
                # wandb run ids are 8 characters [a-z0-9], which we'll emulate here
                # we also want to ensure that all hosts have the same run id
                # we do this by syncing a random seed across all hosts and then using that to generate the run id
                gen = np.random.default_rng(seed)
                self.id = "".join(gen.choice(list("abcdefghijklmnopqrstuvwxyz0123456789"), 8))

            logger.info(f"Setting run id to {self.id}")

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


@dataclass
class OptimizerConfig:
    # Config related to optimizer (always adam for now)
    learning_rate: float = 6e-4
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    max_grad_norm: Optional[float] = 1.0

    min_lr_ratio: float = 0.1
    warmup_ratio: Optional[float] = None  # Deprecated. fraction of training steps to use as warmup
    warmup: float = 0.01
    """fraction of training steps to use as warmup, or steps to use. 0.0 means no warmup"""
    cooldown: float = 0.0
    """fraction of training steps to use as cooldown, or steps to use. 0.0 means no cooldown"""
    lr_schedule: str = "cosine"  # constant, cosine, linear

    def build(self, num_train_steps: int) -> GradientTransformation:
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
        warmup_steps = self._convert_warmup(num_train_steps)
        cooldown_steps = _convert_ratio_or_steps(self.cooldown, num_train_steps)
        lr_decay_steps = num_train_steps - warmup_steps - cooldown_steps
        min_lr = self.learning_rate * self.min_lr_ratio

        match self.lr_schedule:
            case "constant":
                schedule = optax.constant_schedule(self.learning_rate)
            case "cosine":
                schedule = optax.cosine_decay_schedule(self.learning_rate, lr_decay_steps, self.min_lr_ratio)
            case "linear":
                schedule = optax.linear_schedule(self.learning_rate, min_lr, lr_decay_steps - warmup_steps)
            case "inv_sqrt":
                schedule = _inv_sqrt_decay_schedule(self.learning_rate, min_lr, warmup_steps, 10000)
            case _:
                raise ValueError(f"Unknown lr_schedule: {self.lr_schedule}")

        schedules = []
        boundaries = []

        if warmup_steps != 0:
            warmup = optax.linear_schedule(0.0, self.learning_rate, warmup_steps)
            schedules.append(warmup)
            boundaries.append(warmup_steps)

        schedules.append(schedule)

        if cooldown_steps != 0:
            final_main_lr = schedule(lr_decay_steps)
            cooldown = optax.linear_schedule(final_main_lr, min_lr, cooldown_steps)
            schedules.append(cooldown)
            boundaries.append(num_train_steps - cooldown_steps)

        if len(schedules) > 1:
            schedule = optax.join_schedules(schedules, boundaries)

        return schedule

    def _convert_warmup(self, num_train_steps: int):
        if self.warmup_ratio is not None:
            warnings.warn("warmup_ratio is deprecated. Use warmup instead")
            return int(self.warmup_ratio * num_train_steps)
        else:
            return _convert_ratio_or_steps(self.warmup, num_train_steps)


def _inv_sqrt_decay_schedule(lr: float, min_lr: float, warmup_steps: int, timescale: float = 10000):
    def schedule(count):
        decay = jnp.minimum(1.0, 1.0 / jnp.sqrt(jnp.maximum(count + warmup_steps, 1) / timescale))
        return jnp.maximum(lr * decay, min_lr)

    return schedule


def _params_only(t):
    return eqx.filter(t, is_inexact_arrayish)


def _convert_ratio_or_steps(ratio_or_steps: float, num_train_steps: int):
    if ratio_or_steps < 1.0:
        return int(ratio_or_steps * num_train_steps)
    else:
        return int(ratio_or_steps)
