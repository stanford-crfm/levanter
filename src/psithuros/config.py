# Various Pyrallis configs
import dataclasses
from dataclasses import dataclass
from typing import Optional, List

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

        wandb.init(
            entity=self.entity,
            project=self.project,
            name=self.name,
            tags=self.tags,
            id=self.id,
            group=self.group,
            mode=self.mode,
            config=hparams,
        )


@dataclass
class TrainerConfig:
    seed: int = 0

    # Config related to batch sizes
    num_devices: Optional[int] = None
    model_shards: int = 1  # how many devices to shard each model over

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

    # computed properties
    def devices(self):
        d = jax.devices()
        if self.num_devices is not None:
            if self.num_devices > len(d):
                raise ValueError(
                    f"num_devices ({self.num_devices}) is greater than the number of devices ({len(d)})"
                )
            return d[: self.num_devices]

    def device_mesh(self, data_name: str = ResourceAxis.DATA, model_name: str = ResourceAxis.MODEL):
        devices = self.devices()
        devices = np.array(devices).reshape(self.batch_axis_size, self.model_shards)
        return Mesh(devices, (data_name, model_name))

    @property
    def batch_axis_size(self):
        assert self.num_devices % self.model_shards == 0
        return self.num_devices // self.model_shards

    @property
    def train_microbatches_per_step(self):
        return self.train_batch_size // (self.per_device_train_batch_size * self.batch_axis_size)

    @property
    def train_total_microbatches(self):
        return self.num_train_steps * self.train_microbatches_per_step

    def optimizer(self):
        """Creates the optimizer, which is gradient-accumulation aware"""
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

        if self.num_devices is None:
            self.num_devices = len(jax.devices())

        if self.num_devices % self.model_shards != 0:
            raise ValueError(
                f"num_devices ({self.num_devices}) is not divisible by model_shards ({self.model_shards})"
            )

        if self.per_device_train_batch_size == -1:
            self.per_device_train_batch_size = self.train_batch_size // self.num_devices

        # validate size of per_device_train_batch_size
        if self.train_batch_size % (self.per_device_train_batch_size * self.batch_axis_size) != 0:
            raise ValueError(
                f"train_batch_size ({self.train_batch_size}) must be divisible by "
                f"per_device_train_batch_size * batch_axis_size ({self.per_device_train_batch_size}, {self.batch_axis_size})"
            )

        if self.per_device_eval_batch_size == -1:
            self.per_device_eval_batch_size = self.per_device_train_batch_size


def register_codecs():
    pyrallis.encode.register(jnp.dtype, lambda dtype: dtype.name)
    pyrallis.decode.register(jnp.dtype, lambda dtype_name: jnp.dtype(dtype_name))


register_codecs()
