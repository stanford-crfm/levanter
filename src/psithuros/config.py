# Various Pyrallis configs
import dataclasses
from dataclasses import dataclass
from typing import Optional, List

import jax
import optax
from pyrallis import field


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

    def init(self, hparams=None):
        import wandb
        if hparams is None:
            hparams = {}
        elif dataclasses.is_dataclass(hparams):
            hparams = dataclasses.asdict(hparams)
        else:
            hparams = dict(hparams)

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
    wandb: WandbConfig = field(default_factory=WandbConfig)

    seed: int = 0

    # Config related to batch sizes
    num_devices: Optional[int] = None
    train_batch_size: int = 32
    per_device_train_batch_size: int = -1

    per_device_eval_batch_size: int = -1

    # Config related to duration
    num_train_steps: int = 400_000
    steps_per_eval: int = 10_000

    num_save_steps: int = 20_000

    # Config related to optimizer (always adam for now)
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1E-8
    max_grad_norm: Optional[float] = 1.0

    warmup_ratio: float = 0.01  # fraction of training steps to use as warmup
    lr_schedule: str = "cosine"  # constant, cosine, linear

    # TODO: fp16
    # TODO: checkpoints

    def devices(self):
        d = jax.devices()
        if self.num_devices is not None:
            if self.num_devices > len(d):
                raise ValueError(
                    f"num_devices ({self.num_devices}) is greater than the number of devices ({len(d)})"
                )
            return d[: self.num_devices]


    # computed properties
    @property
    def train_microbatches_per_step(self):
        return self.train_batch_size // self.per_device_train_batch_size

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

            components.append(optax.scale(learning_rate))

            optimizer = optax.chain(*components)


            return optimizer

        optimizer = optax.inject_hyperparams(_optimizer)(learning_rate=self.lr_scheduler())
        if self.train_microbatches_per_step > 1:
            optimizer = optax.MultiSteps(optimizer, self.train_microbatches_per_step)

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

        if self.per_device_train_batch_size == -1:
            self.per_device_train_batch_size = self.train_batch_size // self.num_devices

        # validate size of per_device_train_batch_size
        if self.train_batch_size % (self.per_device_train_batch_size * self.num_devices) != 0:
            raise ValueError(
                f"train_batch_size ({self.train_batch_size}) must be divisible by "
                f"per_device_train_batch_size * num_devices ({self.per_device_train_batch_size}, {self.num_devices})"
            )

        if self.per_device_eval_batch_size == -1:
            self.per_device_eval_batch_size = self.per_device_train_batch_size



