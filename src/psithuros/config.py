# Various Pyrallis configs
import dataclasses
from dataclasses import dataclass
from functools import cached_property
from typing import Optional, List, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import optax
import pyrallis
from jax.experimental.maps import Mesh
from pyrallis import field

from psithuros.axis_names import ResourceAxis


@dataclass
class WandbConfig:
    """
    Configuration for wandb.
    """
    entity: Optional[str] = None
    project: Optional[str] = None
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    id: Optional[str] = None
    group: Optional[str] = None
    mode: Optional[str] = None

    def init(self, hparams=None, **extra_hparams):
        import wandb
        if hparams is None:
            hparams = {}
        elif dataclasses.is_dataclass(hparams):
            hparams = dataclasses.asdict(hparams)
        else:
            hparams = dict(hparams)

        if extra_hparams:
            hparams.update(extra_hparams)

        # for distributed runs, we only want the primary worker to use wandb, so we disable everyone else
        mode = self.mode
        if jax.process_index() != 0:
            mode = "disabled"

        wandb.init(
            entity=self.entity,
            project=self.project,
            name=self.name,
            tags=self.tags,
            id=self.id,
            group=self.group,
            mode=mode,
            config=hparams,
        )


@dataclass
class MeshInfo:
    mesh: Mesh
    batch_size: Optional[int]
    per_device_parallelism: int  # how many inputs we can execute per device

    process_index: int = jax.process_index()
    process_count: int = jax.process_count()
    devices_per_node: int = jax.local_device_count()

    @property
    def axis_names(self):
        return self.mesh.axis_names

    @cached_property
    def model_axis_size(self):
        return self.mesh.shape[self.model_axis_name]

    @cached_property
    def data_axis_size(self):
        return self.mesh.shape[self.data_axis_name]

    @property
    def data_axis_name(self):
        return ResourceAxis.DATA

    @property
    def model_axis_name(self):
        return ResourceAxis.MODEL

    @property
    def local_data_axis_size(self):
        """number of groups of devices on this node processing distinct examples"""
        local_device_count = jax.local_device_count()
        local_model_size = self.local_model_axis_size
        assert local_device_count % local_model_size == 0
        return local_device_count // local_model_size

    @property
    def local_model_axis_size(self):
        """size of the model axis for devices on this node. This is local_device_count if the model axis size exceeds
        the number of devices on this node. """
        local_device_count = jax.local_device_count()
        if local_device_count <= self.model_axis_size:
            return local_device_count
        else:
            assert local_device_count % self.model_axis_size == 0
            return self.model_axis_size

    @cached_property
    def local_device_grid_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns a tuple of nd arrays, one for each axis, indicating the position of each device on the grid. Analogous to what np.where would return."""
        pi = self.process_index
        # our device indices are [process_index * num_devices_per_node, (process_index + 1) * num_devices_per_node)
        # we could be clever here and do math to figure out where we are in the grid, but it's simpler and less
        # fragile to just search the grid for our devices
        my_device_pos = np.vectorize(lambda dev: dev.process_index == pi)(self.mesh.devices)
        return my_device_pos.nonzero()

    @property
    def process_mesh_position(self) -> Tuple[int, int]:
        """
        If we envision each process as a subgrid of the mesh for its devices, this is the position of the process
        in the coarsened process-level mesh
        """
        upper_left_position = np.array([np.min(axis) for axis in self.local_device_grid_positions])
        local_mesh_size = self.mesh.local_mesh.devices.shape
        pos = upper_left_position / local_mesh_size
        assert len(pos) == 2
        return pos[0], pos[1]

    @property
    def process_mesh_size(self) -> Tuple[int, int]:
        """
        If we envision each process as a subgrid of the mesh for its devices, then there is a process grid that
        is a coarsened version of the mesh. This is the size of the process grid.
        """
        local_mesh_size = self.mesh.local_mesh.devices.shape
        assert self.data_axis_size % local_mesh_size[0] == 0
        assert self.model_axis_size % local_mesh_size[1] == 0
        return self.data_axis_size // local_mesh_size[0], self.model_axis_size // local_mesh_size[1]

    @property
    def microbatch_size(self):
        """number of examples in a microbatch, across all nodes"""
        return self.data_axis_size * self.per_device_parallelism

    @property
    def microbatches_per_step(self):
        if self.batch_size is None:
            return 1

        assert self.batch_size % self.microbatch_size == 0
        return self.batch_size // self.microbatch_size

    @property
    def local_microbatch_size(self):
        """number of examples in a microbatch per process in the training run. typically one process per node"""
        return self.local_data_axis_size * self.per_device_parallelism

    @property
    def local_batch_size(self):
        """number of examples processed by this process for an entire batch. typically one process per node"""
        return self.microbatches_per_step * self.local_microbatch_size


@dataclass
class TrainerConfig:
    seed: int = 0

    # Config related to batch sizes
    model_axis_size: int = 1  # how many devices to shard each model over

    train_batch_size: int = 512
    per_device_train_batch_size: int = -1

    per_device_eval_batch_size: int = -1

    # Config related to duration
    num_train_steps: int = 400_000
    steps_per_eval: int = 10_000

    steps_per_save: int = 20_000
    load_last_checkpoint: bool = True
    load_checkpoint_path: Optional[str] = None

    # Config related to optimizer (always adam for now)
    learning_rate: float = 6e-4
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1E-8
    max_grad_norm: Optional[float] = 1.0

    warmup_ratio: float = 0.01  # fraction of training steps to use as warmup
    lr_schedule: str = "cosine"  # constant, cosine, linear

    @cached_property
    def device_mesh(self):
        devices = jax.devices()
        devices = np.array(devices).reshape(self.data_axis_size, self.model_axis_size)
        return Mesh(devices, (ResourceAxis.DATA, ResourceAxis.MODEL))

    @cached_property
    def train_mesh_info(self):
        return MeshInfo(self.device_mesh, self.train_batch_size, self.per_device_train_batch_size)

    @cached_property
    def eval_mesh_info(self):
        return MeshInfo(self.device_mesh, None, self.per_device_eval_batch_size)

    @property
    def data_axis_size(self):
        """size of the data parallel/batch parallel axis."""
        assert jax.device_count() % self.model_axis_size == 0
        return jax.device_count() // self.model_axis_size

    @property
    def local_eval_batch_size(self):
        """number of examples processed by this process for an entire batch during eval. typically one process per node"""
        return self.eval_mesh_info.local_batch_size

    @property
    def train_total_microbatches(self):
        return self.num_train_steps * self.train_mesh_info.microbatches_per_step

    def optimizer(self):
        """Creates the optimizer"""

        # indirection makes it work with optax.inject_hyperparams so we can can log the learning rate
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

        optimizer = optax.inject_hyperparams(_optimizer)(learning_rate=self.lr_scheduler())

        return optimizer

    def lr_scheduler(self):
        warmup_steps = int(self.warmup_ratio * self.num_train_steps)
        lr_decay_steps = self.num_train_steps - warmup_steps
        if warmup_steps == 0 and self.lr_schedule == "constant":
            schedule = optax.constant_schedule(self.learning_rate)
        else:
            if self.lr_schedule == "constant":
                schedule = optax.constant_schedule(self.learning_rate)
            elif self.lr_schedule == "cosine":
                schedule = optax.cosine_decay_schedule(self.learning_rate, lr_decay_steps - warmup_steps)
            elif self.lr_schedule == "linear":
                schedule = optax.linear_schedule(self.learning_rate, 0.0, lr_decay_steps - warmup_steps)
            else:
                raise ValueError(f"Unknown lr_schedule: {self.lr_schedule}")

            if warmup_steps != 0:
                warmup = optax.linear_schedule(0.0, self.learning_rate, warmup_steps)
                schedule = optax.join_schedules([warmup, schedule], [warmup_steps])
        return schedule

    # post init
    def __post_init__(self):
        if jax.device_count() % self.model_axis_size != 0:
            raise ValueError(
                f"num_devices ({jax.device_count()}) is not divisible by model_axis_size ({self.model_axis_size})"
            )

        if jax.local_device_count() % self.model_axis_size != 0 and self.model_axis_size % jax.local_device_count() != 0:
            raise ValueError(
                "either model_axis_size or local_device_count must be divisible by the other"
            )

        if self.per_device_train_batch_size == -1:
            self.per_device_train_batch_size = self.train_batch_size // jax.device_count()

        # validate size of per_device_train_batch_size
        if self.train_batch_size % (self.per_device_train_batch_size * self.data_axis_size) != 0:
            raise ValueError(
                f"train_batch_size ({self.train_batch_size}) must be divisible by "
                f"per_device_train_batch_size * data_axis_size ({self.per_device_train_batch_size}, {self.data_axis_size})"
            )

        if self.per_device_eval_batch_size == -1:
            self.per_device_eval_batch_size = self.per_device_train_batch_size


def register_codecs():
    pyrallis.encode.register(jnp.dtype, lambda dtype: dtype.name)
    pyrallis.encode.register(type(jnp.float32), lambda meta: meta.dtype.name)
    pyrallis.decode.register(jnp.dtype, lambda dtype_name: jnp.dtype(dtype_name))


register_codecs()
