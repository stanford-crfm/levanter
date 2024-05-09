import abc
import re
import warnings
from dataclasses import dataclass
from typing import Optional

import draccus
import equinox as eqx
import jax
import optax
from jax import numpy as jnp

from levanter.utils.jax_utils import leaf_key_paths


@dataclass
class OptimizerConfig(draccus.ChoiceRegistry, abc.ABC):
    learning_rate: float = 6e-4
    weight_decay: float = 0.0

    min_lr_ratio: float = 0.1
    warmup_ratio: Optional[float] = None  # Deprecated. fraction of training steps to use as warmup
    """The lr scheduler operates on 4 stages: [warmup] - [stable] - [decay] - [cooldown]"""
    warmup: float = 0.01
    """fraction of training steps to use as warmup, or steps to use. 0.0 means no warmup"""
    stable: float = 0.00
    """fraction of training steps to use as stable, or steps to use. 0.0 means no stable"""
    cooldown: float = 0.0
    """fraction of training steps to use as cooldown, or steps to use. 0.0 means no cooldown"""
    lr_schedule: str = "cosine"  # constant, cosine, linear
    weight_decay_modules: Optional[list[str] | str] = None
    """A regex or a list of strings to identify where to mask weight.
    For nano-GPT, this field can be set as `r".*attn.*weight|.*mlp.*weight|.*token_embeddings|.*position_embeddings"`"""

    @classmethod
    def default_choice_name(cls) -> Optional[str]:
        return "adam"

    @abc.abstractmethod
    def build(self, num_train_steps: int):
        raise NotImplementedError

    def build_weight_decay_mask(self):
        if self.weight_decay_modules is None:
            return None
        else:
            # mask based on regex or module path
            def _apply_on(x, key_path):
                if isinstance(self.weight_decay_modules, str):
                    compiled_regex = re.compile(self.weight_decay_modules)
                    return compiled_regex.match(key_path) is not None
                else:
                    return any(key_path.__contains__(target) for target in self.weight_decay_modules)

            def mask_fn(model):
                return jax.tree_util.tree_map(
                    _apply_on,
                    model,
                    leaf_key_paths(model, is_leaf=eqx.is_array),
                    is_leaf=eqx.is_array,
                )

            return mask_fn

    def lr_scheduler(self, num_train_steps):
        warmup_steps = self._convert_warmup(num_train_steps)
        stable_steps = _convert_ratio_or_steps(self.stable, num_train_steps)
        cooldown_steps = _convert_ratio_or_steps(self.cooldown, num_train_steps)
        lr_decay_steps = num_train_steps - warmup_steps - stable_steps - cooldown_steps
        min_lr = self.learning_rate * self.min_lr_ratio

        match self.lr_schedule:
            case "constant":
                schedule = optax.constant_schedule(self.learning_rate)
            case "cosine":
                schedule = optax.cosine_decay_schedule(self.learning_rate, lr_decay_steps, self.min_lr_ratio)
            case "linear":
                schedule = optax.linear_schedule(self.learning_rate, min_lr, lr_decay_steps)
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

        if stable_steps != 0:
            stable = optax.constant_schedule(self.learning_rate)
            schedules.append(stable)
            boundaries.append(warmup_steps + stable_steps)

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


def _convert_ratio_or_steps(ratio_or_steps: float, num_train_steps: int):
    if ratio_or_steps < 1.0:
        return int(ratio_or_steps * num_train_steps)
    else:
        return int(ratio_or_steps)


@dataclass
class HessianOptConfig(OptimizerConfig, abc.ABC):
    update_interval: int = 10
    """How often to update the hessian approximation."""


@OptimizerConfig.register_subclass("adam")
@dataclass
class AdamConfig(OptimizerConfig):
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    max_grad_norm: Optional[float] = 1.0

    def build(self, num_train_steps):
        """Creates the optimizer"""
        # indirection makes it work with optax.inject_hyperparams so we can log the learning rate
        def _optimizer(learning_rate):
            components = []

            if self.max_grad_norm:
                components.append(optax.clip_by_global_norm(self.max_grad_norm))

            components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))

            if self.weight_decay > 0:
                components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))

            # - learning rate for descent
            components.append(optax.scale(-learning_rate))

            optimizer = optax.chain(*components)

            return optimizer

        return optax.inject_hyperparams(_optimizer)(learning_rate=self.lr_scheduler(num_train_steps))
