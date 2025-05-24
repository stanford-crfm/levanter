from dataclasses import dataclass
from typing import NamedTuple, Optional

import jax
import optax
from jax import numpy as jnp


class SkipStepState(NamedTuple):
    inner_opt_state: optax.OptState
    losses: jax.Array  # Shape: (rolling_interval_length + 1,), dtype: jnp.float32
    grad_norms: jax.Array  # Shape: (rolling_interval_length + 1,), dtype: jnp.float32
    current_idx: jax.Array  # Scalar int
    count: jax.Array  # Scalar int


@dataclass
class SkipStepConfig:
    """
    Configuration for "skip step" logic in an optimizer.

    This optimizer skips steps based on the history of loss and gradient norms.

    If the current loss or gradient norm is significantly higher than the historical mean plus a multiple of the
    standard deviation, the step is skipped. The decision is made based on a rolling window of the last
    `rolling_interval_length` steps.

    IF there are fewer than `rolling_interval_length // 2` steps in the history, the step is always taken.
    """

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

            # Decision Logic (step_factor) based on PAST data
            min_data_points = jnp.maximum(2, self.rolling_interval_length // 2)
            # can_skip_based_on_history uses state.count (count before adding current data point)
            can_skip_based_on_history = state.count >= min_data_points

            # Create a mask for valid (non-NaN) entries in the *current* state.losses/grad_norms
            idx_range = jnp.arange(self.rolling_interval_length + 1)
            # valid_mask_history uses state.count
            valid_mask_history = idx_range < state.count

            # Calculate stats for skipping from history, ensuring NaNs are handled and std is not zero
            masked_losses_history = jnp.where(valid_mask_history, state.losses, jnp.nan)
            loss_mean_history = jnp.nanmean(masked_losses_history)
            loss_std_history = jnp.nanstd(masked_losses_history)
            loss_std_safe_history = jnp.maximum(loss_std_history, 1e-6)

            masked_grad_norms_history = jnp.where(valid_mask_history, state.grad_norms, jnp.nan)
            grad_norm_mean_history = jnp.nanmean(masked_grad_norms_history)
            grad_norm_std_history = jnp.nanstd(masked_grad_norms_history)
            grad_norm_std_safe_history = jnp.maximum(grad_norm_std_history, 1e-6)

            # if mean or std is nan (e.g. if state.count is 0 or 1), set to benign values.
            # can_skip_based_on_history will likely be false anyway in these early stages.
            loss_mean_history = jnp.nan_to_num(loss_mean_history, nan=0.0)
            loss_std_safe_history = jnp.nan_to_num(loss_std_safe_history, nan=1e-6, posinf=1e-6, neginf=1e-6)
            grad_norm_mean_history = jnp.nan_to_num(grad_norm_mean_history, nan=0.0)
            grad_norm_std_safe_history = jnp.nan_to_num(grad_norm_std_safe_history, nan=1e-6, posinf=1e-6, neginf=1e-6)

            loss_threshold = loss_mean_history + self.sigma_factor * loss_std_safe_history
            grad_norm_threshold = grad_norm_mean_history + self.sigma_factor * grad_norm_std_safe_history

            # Compare current loss and grad_norm with thresholds derived from history
            loss_ok = loss <= loss_threshold
            grad_norm_ok = global_norm <= grad_norm_threshold

            # we skip if loss OR grad_norm are NOT ok.
            # so should_skip is true if NOT (loss_ok AND grad_norm_ok)
            # which is (NOT loss_ok) OR (NOT grad_norm_ok)
            should_skip = jnp.logical_not(jnp.logical_and(loss_ok, grad_norm_ok))

            step_factor = jnp.where(jnp.logical_and(can_skip_based_on_history, should_skip), 0.0, 1.0)

            # Update rolling statistics with CURRENT data point (AFTER decision is made)
            new_losses = state.losses.at[state.current_idx].set(loss)
            new_grad_norms = state.grad_norms.at[state.current_idx].set(global_norm)
            new_current_idx = (state.current_idx + 1) % (self.rolling_interval_length + 1)
            new_count = jnp.minimum(state.count + 1, self.rolling_interval_length + 1)

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
                operand=None,  # operand not strictly needed as branches capture necessary values
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
