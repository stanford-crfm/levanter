import abc
import re
import warnings
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple, Optional

import draccus
import equinox as eqx
import jax
import numpy as np
import optax
from jax import numpy as jnp

import haliax

import levanter.tracker
from levanter.utils.jax_utils import leaf_key_paths


class SkipStepState(NamedTuple):
    inner_opt_state: optax.OptState
    losses: jax.Array  # Shape: (rolling_interval_length + 1,), dtype: jnp.float32
    grad_norms: jax.Array  # Shape: (rolling_interval_length + 1,), dtype: jnp.float32
    current_idx: jax.Array  # Scalar int
    count: jax.Array  # Scalar int


@dataclass
class OptimizerConfig(draccus.ChoiceRegistry, abc.ABC):
    learning_rate: float = 6e-4
    weight_decay: float = 0.1

    min_lr_ratio: float = 0.1
    """The lr scheduler operates on 4 stages: [warmup] - {[stable] - [decay]} x haps - [cooldown]"""
    warmup: int | float = 0.01
    """fraction of training steps to use as warmup, or steps to use. 0.0 means no warmup"""
    decay: int | float | None = None
    """fraction of training steps to use as decay, or steps to use. None means full decay"""
    rewarmup: int | float = 0.0
    "If using a cycle, how much of the cycle to use as re-warmup. 0.0 means no re-warmup."
    cooldown: Optional[float] = None
    """Deprecated, as its semantics are confusing."""
    cycle_length: int | float | None | list[int] = None
    """ Length of cycle. If <= 1, it is treated as a fraction of the total number of steps. None is equivalent to 1.0."""
    cycles: int | list[int] | None = None
    """Number of cycles or a list of cycle endpoints. Can use at most one of cycle_length, cycles, or haps."""

    lr_schedule: str = "cosine"  # constant, cosine, linear
    haps: Optional[list[int]] = None
    """Deprecated."""
    weight_decay_modules: Optional[list[str] | str] = None
    """A regex or a list of strings to identify where to mask weight.
    For nano-GPT, this field can be set as `r".*attn.*weight|.*mlp.*weight|.*token_embeddings|.*position_embeddings"`"""
    default_weight_decay_mask: Optional[bool] = None
    """Whether to apply a default reasonable weight decay to modules not explicitly masked. None means it will if
    no weight_decay_modules are set. False means it will not. True means it will regardless of weight_decay_modules."""

    skip_step: Optional["SkipStepConfig"] = None
    """If set, defines the configuration for skipping steps when gradients are too large."""

    @classmethod
    def default_choice_name(cls) -> Optional[str]:
        return "adam"

    @abc.abstractmethod
    def build(self, num_train_steps: int):
        raise NotImplementedError

    def build_weight_decay_mask(self):
        def reasonable_default(module, path):
            # TODO: gross
            if "LayerNorm" in path:
                return False
            if "RMSNorm" in path:
                return False
            if "RmsNorm" in path:
                return False
            if "Embedding" in path:
                return False
            if path.endswith("bias"):
                return False
            return None

        if self.weight_decay_modules is None and self.default_weight_decay_mask is False:
            return None
        else:
            should_use_default = self.default_weight_decay_mask is True or (
                self.default_weight_decay_mask is None and self.weight_decay_modules is None
            )

            def is_leaf(x):
                return eqx.is_array(x) or isinstance(x, eqx.Module) or haliax.is_named_array(x)

            # mask based on regex or module path
            def _apply_on(decayed_paths, x, from_root_key_path, from_class_keypath):
                if isinstance(x, eqx.Module):
                    is_leaf_here = lambda y: x is not y and is_leaf(y)  # noqa: E731
                    # we want to support both Linear.weight and transformer.encoder.layers.0.mlp.dense.weight
                    class_name = x.__class__.__name__
                    # recursively apply to submodules.
                    from_root_key_paths = leaf_key_paths(x, is_leaf=is_leaf_here, prefix=from_root_key_path)
                    from_class_key_paths = leaf_key_paths(x, is_leaf=is_leaf_here, prefix=class_name)
                    this_mask = jax.tree_util.tree_map(
                        partial(_apply_on, decayed_paths),
                        x,
                        from_root_key_paths,
                        from_class_key_paths,
                        is_leaf=lambda y: x is not y and is_leaf(y),
                    )
                    return this_mask
                elif not haliax.util.is_jax_or_hax_array_like(x):
                    return x

                should_decay = None
                for key_path in [from_root_key_path, from_class_keypath]:
                    if key_path is None:
                        continue

                    if isinstance(self.weight_decay_modules, str):
                        compiled_regex = re.compile(self.weight_decay_modules)
                        should_decay = should_decay or compiled_regex.match(key_path) is not None
                    elif isinstance(self.weight_decay_modules, list):
                        should_decay = should_decay or any(
                            key_path.__contains__(target) for target in self.weight_decay_modules
                        )

                    if should_use_default and not should_decay:
                        should_decay = reasonable_default(x, key_path)

                    if should_decay:
                        break

                if should_decay is None:
                    if should_use_default:
                        should_decay = True
                    else:
                        should_decay = False

                if should_decay:
                    decayed_paths.append(from_root_key_path)

                return should_decay

            def mask_fn(model):
                decayed_paths = []
                mask = jax.tree_util.tree_map(
                    partial(_apply_on, decayed_paths, from_class_keypath=None),
                    model,
                    leaf_key_paths(model, is_leaf=is_leaf),
                    is_leaf=is_leaf,
                )

                # log all decayed weights
                levanter.tracker.log_hyperparameters({"decayed_weights": sorted(decayed_paths)})

                return mask

            return mask_fn

    def lr_scheduler(self, num_train_steps):
        if self.cooldown is not None:
            warnings.warn("cooldown is deprecated. Just use the normal schedule.", DeprecationWarning)
            cooldown_steps = _convert_frac_or_steps(self.cooldown, num_train_steps)
        else:
            cooldown_steps = 0

        total_main_steps = num_train_steps - cooldown_steps
        cooldown_points = self._get_cycle_minima(total_main_steps)

        min_lr = self.learning_rate * self.min_lr_ratio

        schedules = []
        boundaries = []

        previous_end = 0.0

        for cycle, (start, end) in enumerate(zip(cooldown_points[:-1], cooldown_points[1:])):
            cycle_steps = end - start
            if cycle == 0:  # warmup
                warmup_steps = _convert_frac_or_steps(self.warmup, cycle_steps)
            else:
                warmup_steps = _convert_frac_or_steps(self.rewarmup, cycle_steps)

            if warmup_steps != 0:
                warmup = optax.linear_schedule(previous_end, self.learning_rate, warmup_steps)
                schedules.append(warmup)
                boundaries.append(start + warmup_steps)

            lr_decay_steps = (
                _convert_frac_or_steps(self.decay, cycle_steps)
                if self.decay is not None
                else cycle_steps - warmup_steps
            )
            stable_steps = cycle_steps - warmup_steps - lr_decay_steps

            if stable_steps != 0:
                stable = optax.constant_schedule(self.learning_rate)
                schedules.append(stable)
                boundaries.append(start + warmup_steps + stable_steps)

            match self.lr_schedule:
                case "constant":
                    schedule = optax.constant_schedule(self.learning_rate)
                case "cosine":
                    schedule = optax.cosine_decay_schedule(self.learning_rate, lr_decay_steps, self.min_lr_ratio)
                case "linear":
                    schedule = optax.linear_schedule(self.learning_rate, min_lr, lr_decay_steps)
                case "inv_sqrt":
                    schedule = _inv_sqrt_decay_schedule(self.learning_rate, min_lr, warmup_steps, 10000)
                case "inv":
                    schedule = _inv_decay_schedule(self.learning_rate, min_lr, lr_decay_steps)
                case _:
                    raise ValueError(f"Unknown lr_schedule: {self.lr_schedule}")

            previous_end = schedule(lr_decay_steps)

            schedules.append(schedule)
            boundaries.append(end)

        if cooldown_steps != 0:
            final_main_lr = schedule(lr_decay_steps)
            cooldown = optax.linear_schedule(final_main_lr, min_lr, cooldown_steps)
            schedules.append(cooldown)

        if len(schedules) > 1:
            schedule = optax.join_schedules(schedules, boundaries)
        else:
            schedule = schedules[0]

        return schedule

    def _get_cycle_minima(self, total_main_steps):
        if self.cycle_length is not None:
            if self.cycles is not None:
                raise ValueError("Can't use both cycle_length and cycles.")
            if self.haps is not None:
                warnings.warn("haps is deprecated. Use cycles instead.", DeprecationWarning)
                raise ValueError("Can't use both cycle_length and haps.")

            if isinstance(self.cycle_length, int | float):
                cycle_length = _convert_frac_or_steps(self.cycle_length, total_main_steps)
                cooldown_points = [i * cycle_length for i in range(1, total_main_steps // cycle_length)]
                if total_main_steps % cycle_length != 0:
                    warnings.warn(
                        "Cycle length does not divide total number of steps. The last cycle will be shorter."
                    )

            elif isinstance(self.cycle_length, list):
                lengths = np.array(self.cycle_length)
                steps = np.cumsum(lengths)
                if steps[-1] > total_main_steps:
                    raise ValueError(f"Cycle lengths exceed total number of steps: {steps[-1]} > {total_main_steps}")
                cooldown_points = steps.tolist()
            else:
                raise ValueError("Invalid cycle_length. Must be a fraction, number of steps, or a list of steps.")

        elif self.haps is not None:
            warnings.warn("haps is deprecated. Use cycles instead.", DeprecationWarning)
            cooldown_points = list(self.haps)
        elif isinstance(self.cycles, int):
            # insert a warmup then the rest of the steps
            cooldown_points = [int(total_main_steps / self.cycles * (i + 1)) for i in range(self.cycles - 1)]
        elif isinstance(self.cycles, list):
            cooldown_points = list(self.cycles)
        else:
            cooldown_points = []

        cooldown_points.insert(0, 0)
        if cooldown_points[-1] != total_main_steps:
            cooldown_points.append(total_main_steps)
        return cooldown_points


def _inv_sqrt_decay_schedule(lr: float, min_lr: float, warmup_steps: int, timescale: float = 10000):
    def schedule(count):
        decay = jnp.minimum(1.0, 1.0 / jnp.sqrt(jnp.maximum(count + warmup_steps, 1) / timescale))
        return jnp.maximum(lr * decay, min_lr)

    return schedule


def _inv_decay_schedule(lr: float, min_lr: float, decay_steps: int):
    def schedule(count):
        decay = jnp.minimum(1.0, 1.0 / ((lr / min_lr - 1) * jnp.maximum(count, 1) / decay_steps + 1))
        return jnp.maximum(lr * decay, min_lr)

    return schedule


def _convert_frac_or_steps(frac_or_steps: float | int, num_train_steps: int):
    # if it's greater than 1, it must be a whole number of steps
    if frac_or_steps < 0.0 or (frac_or_steps > 1.0 and frac_or_steps % 1 != 0):
        raise ValueError(f"Invalid fraction {frac_or_steps}. Must be between 0 and 1. You can also use (whole) steps.")
    if frac_or_steps <= 1.0:
        return int(frac_or_steps * num_train_steps)

    return int(frac_or_steps)


@dataclass
class HessianOptConfig(OptimizerConfig, abc.ABC):
    update_interval: int = 10
    """How often to update the hessian approximation."""


@dataclass
class SkipStepConfig:
    rolling_interval_length: int = 128
    sigma_factor: float = 6.0

    def wrap(self, inner_optimizer: optax.GradientTransformation) -> optax.GradientTransformation:
        def init_fn(params):
            inner_opt_state = inner_optimizer.init(params)
            losses = jnp.full((self.rolling_interval_length + 1,), jnp.nan, dtype=jnp.float32)
            grad_norms = jnp.full((self.rolling_interval_length + 1,), jnp.nan, dtype=jnp.float32)
            current_idx = jnp.zeros((), dtype=jnp.int32)
            count = jnp.zeros((), dtype=jnp.int32)
            return SkipStepState(
                inner_opt_state=inner_opt_state,
                losses=losses,
                grad_norms=grad_norms,
                current_idx=current_idx,
                count=count,
            )

        def update_fn(updates, state: SkipStepState, params, *, loss: Optional[jax.Array] = None, **extra_args):
            if loss is None:
                raise ValueError(
                    "Loss must be provided as a keyword argument to the update function when using SkipStepOptimizer."
                )

            global_norm = optax.global_norm(updates)

            # Update rolling statistics (circular buffer logic)
            new_losses = state.losses.at[state.current_idx].set(loss)
            new_grad_norms = state.grad_norms.at[state.current_idx].set(global_norm)
            new_current_idx = (state.current_idx + 1) % (self.rolling_interval_length + 1)
            new_count = jnp.minimum(state.count + 1, self.rolling_interval_length + 1)

            # Decision Logic (step_factor)
            min_data_points = jnp.maximum(2, self.rolling_interval_length // 2)
            can_skip = new_count >= min_data_points

            # Create a mask for valid (non-NaN) entries
            idx_range = jnp.arange(self.rolling_interval_length + 1)
            valid_mask = idx_range < new_count

            # Calculate stats for skipping, ensuring NaNs are handled and std is not zero
            masked_losses = jnp.where(valid_mask, new_losses, jnp.nan)
            loss_mean = jnp.nanmean(masked_losses)
            loss_std = jnp.nanstd(masked_losses)
            loss_std_safe = jnp.maximum(loss_std, 1e-6)

            masked_grad_norms = jnp.where(valid_mask, new_grad_norms, jnp.nan)
            grad_norm_mean = jnp.nanmean(masked_grad_norms)
            grad_norm_std = jnp.nanstd(masked_grad_norms)
            grad_norm_std_safe = jnp.maximum(grad_norm_std, 1e-6)
            
            # if mean or std is nan (e.g. if new_count is 0 or 1), then don't skip
            # this is a bit of a hack, but it's fine because can_skip will be false anyway in early steps
            loss_mean = jnp.nan_to_num(loss_mean, nan=0.0)
            loss_std_safe = jnp.nan_to_num(loss_std_safe, nan=1e-6)
            grad_norm_mean = jnp.nan_to_num(grad_norm_mean, nan=0.0)
            grad_norm_std_safe = jnp.nan_to_num(grad_norm_std_safe, nan=1e-6)


            loss_threshold = loss_mean + self.sigma_factor * loss_std_safe
            grad_norm_threshold = grad_norm_mean + self.sigma_factor * grad_norm_std_safe

            loss_ok = loss <= loss_threshold
            grad_norm_ok = global_norm <= grad_norm_threshold
            
            # we skip if loss OR grad_norm are NOT ok.
            # so should_skip is true if NOT (loss_ok AND grad_norm_ok)
            # which is (NOT loss_ok) OR (NOT grad_norm_ok)
            should_skip = jnp.logical_not(jnp.logical_and(loss_ok, grad_norm_ok))

            step_factor = jnp.where(jnp.logical_and(can_skip, should_skip), 0.0, 1.0)

            # Apply Update based on step_factor
            # We need to define the two branches for jax.lax.cond
            # Signature for cond branches: operand -> result
            # Here, operand is (updates, state.inner_opt_state, params, extra_args)
            # Result is (new_inner_updates, new_inner_opt_state)

            def skip_step_branch(_):
                # updates are zeroed out, inner state is preserved
                zeroed_updates = jax.tree_util.tree_map(jnp.zeros_like, updates)
                return zeroed_updates, state.inner_opt_state

            def proceed_step_branch(_):
                # updates are applied as is, inner state is updated
                return inner_optimizer.update(updates, state.inner_opt_state, params, **extra_args)

            # Use jax.lax.cond to choose branch based on step_factor
            # pred is true if step_factor == 0.0 (skip)
            new_inner_updates, new_inner_opt_state = jax.lax.cond(
                step_factor == 0.0,
                skip_step_branch,
                proceed_step_branch,
                operand=None  # operand not strictly needed as branches capture necessary values
            )

            # Log whether a step was skipped
            # TODO: make this a proper metric when levanter.tracker.log_metrics is a thing
            # For now, we can log it as a hyperparameter, though it's not ideal.
            # Alternatively, print it, but that's noisy.
            # For now, let's skip explicit logging here as it's not requested and might be noisy.
            # A more advanced solution would involve passing a logging callback or similar.

            return new_inner_updates, SkipStepState(
                inner_opt_state=new_inner_opt_state,
                losses=new_losses,
                grad_norms=new_grad_norms,
                current_idx=new_current_idx,
                count=new_count,
            )

        return optax.GradientTransformation(init_fn, update_fn)


@OptimizerConfig.register_subclass("adam")
@dataclass
class AdamConfig(OptimizerConfig):
    beta1: float = 0.9
    # cf https://docs.mosaicml.com/projects/composer/en/latest/api_reference/generated/composer.optim.DecoupledAdamW.html
    # https://x.com/giffmana/status/1692641748445438301
    beta2: float = 0.95
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

        optimizer_instance = optax.inject_hyperparams(_optimizer)(learning_rate=self.lr_scheduler(num_train_steps))

        if self.skip_step is not None:
            optimizer_instance = self.skip_step.wrap(optimizer_instance)

        return optimizer_instance
