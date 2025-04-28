import dataclasses
from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple, List, Union

import chex
import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu
from jax.sharding import PartitionSpec

import haliax
from haliax.nn import Linear
from haliax.partitioning import infer_resource_partitions

from levanter.optim.config import OptimizerConfig
from levanter.optim.util import map_flattened_linear_layers
from levanter.utils.jax_utils import leaf_key_paths


@OptimizerConfig.register_subclass("mini")
@dataclass
class MiniConfig(OptimizerConfig):
    """
    Mini optimizer configuration: Momentum Orthogonalized by Newton-Schulz.
    """

    lr: float = 0.02
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    nesterov: bool = False

    def build(self, num_train_steps):
        """
        Creates the optimizer.
        """
        learning_rate_schedule = self.lr_scheduler(num_train_steps)

        def optimizer(learning_rate):

            def mini_transform(mean_axis):
                components = []
                components.append(scale_with_mini(self.beta1, self.beta2, self.epsilon, mean_axis = mean_axis))
                if self.weight_decay > 0:
                    components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))
                components.append(optax.scale(-learning_rate))
                optimizer = optax.chain(*components)
                return optimizer

            def adamw_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                if self.weight_decay > 0:
                    components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))
                components.append(optax.scale(-learning_rate))
                optimizer = optax.chain(*components)
                return optimizer

            transformations = {
                "embedding": mini_transform(mean_axis = (1,)),
                "lm_head": mini_transform(mean_axis = (1,)),
                "query": mini_transform(mean_axis = (2, 3, 4)),
                "key": mini_transform(mean_axis = (2, 3)),
                "value": mini_transform(mean_axis = (3,)),
                "output": mini_transform(mean_axis = (2, 3)),
                "linear": mini_transform(mean_axis = (2,)),
                "adamw": adamw_transform(),
            }

            return optax.multi_transform(transformations, self.create_mask)

        return optax.inject_hyperparams(optimizer)(learning_rate=learning_rate_schedule)

    def create_mask(self, params):
        """
        Creates a mask that labels parameters as 'mini' or 'adamw' based on their
        dimensionality and module path, using AdamW for Embedding and lm_head parameters.
        """
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            if "embedding" in path_str:
                print(f"shape: {param.shape}")
                return "embedding"
            elif "q_proj" in path_str:
                return "query"
            elif "k_proj" in path_str:
                return "key"
            elif "v_proj" in path_str:
                return "value"
            elif "o_proj" in path_str:
                return "output"
            elif "lm_head" in path_str:
                return "lm_head"
            elif isinstance(param, Linear):
                # mini for linear layers
                return dataclasses.replace(param, weight="linear", bias="adamw" if param.bias is not None else None)
            else:
                return "adamw"

        return jax.tree_util.tree_map(mask_fn, params, paths, is_leaf=lambda x: isinstance(x, Linear))


class ScaleByMiniState(NamedTuple):
    """State for the Mars algorithm."""
    count: chex.Array  # shape=(), dtype=jnp.int32.
    momentum_buffer: optax.Updates
    second_moment_buffer: optax.Updates


def scale_with_mini(beta1, beta2, epsilon, mean_axis=(1,)):
    """
    Implementation of the Mini algorithm: Momentum optimizer with second moment scaling.
    
    Args:
        beta1: momentum decay rate
        beta2: second moment decay rate
        epsilon: small constant for numerical stability
        mean_axis: axes over which to compute the mean for the second moment
    
    Returns:
        An optax GradientTransformation
    """
    def init_fn(params):
        momentum_buffer = otu.tree_zeros_like(params)  # First moment
        second_moment_buffer = otu.tree_zeros_like(params)  # Second moment
        count = jnp.zeros([], jnp.int32)
        
        # Get sharding specs if using haliax partitioned arrays
        # param_specs = None
        # if any(isinstance(x, haliax.NamedArray) for x in jax.tree_util.tree_leaves(params)):
        #     param_specs = infer_resource_partitions(params)
        
        # # Initialize the second moment buffer by taking the mean over the specified axes
        # def init_second_moment(param, spec=None):
        #     # Create a buffer with appropriate shape after taking mean over specified axes
        #     reduced_shape = list(param.shape)
        #     for axis in sorted(mean_axis):
        #         if axis < len(reduced_shape):
        #             reduced_shape[axis] = 1
            
        #     # Adjust the sharding spec for the reduced dimensions
        #     adjusted_spec = None
        #     # if spec is not None:
        #     #     # Extract the PartitionSpec from spec
        #     #     if hasattr(spec, 'spec'):
        #     #         spec = spec.spec
                
        #     #     # Create a new spec with None for the reduced dimensions
        #     #     if isinstance(spec, PartitionSpec):
        #     #         spec_list = list(spec)
        #     #         for axis in sorted(mean_axis):
        #     #             if axis < len(spec_list):
        #     #                 # Replace the sharding with None for reduced dimensions
        #     #                 spec_list[axis] = None
        #     #         adjusted_spec = PartitionSpec(*spec_list)
            
        #     return jnp.zeros(reduced_shape, dtype=param.dtype), adjusted_spec
        
        # if param_specs is not None:
        #     # If we have sharding info, use it to create properly sharded buffers
        #     outputs = jax.tree_util.tree_map(init_second_moment, params, param_specs)
        #     second_moment_buffer = jax.tree_util.tree_map(lambda x: x[0], outputs)
        #     second_moment_specs = jax.tree_util.tree_map(lambda x: x[1], outputs)
            
        #     # Apply the sharding to the second moment buffer
        #     from haliax.partitioning import with_sharding_constraint
        #     second_moment_buffer = jax.tree_util.tree_map(
        #         lambda b, s: with_sharding_constraint(b, s) if s is not None else b,
        #         second_moment_buffer, second_moment_specs
        #     )
        # else:
        #     # If no sharding info, just initialize the buffers without sharding
        #     second_moment_buffer = jax.tree_util.tree_map(
        #         lambda p: init_second_moment(p)[0], params
        #     )
        
        return ScaleByMiniState(count=count, momentum_buffer=momentum_buffer, 
                               second_moment_buffer=second_moment_buffer)

    def update_fn(updates, state, params=None):
        # Update momentum buffer (step 6 in algorithm)
        momentum_buffer = otu.tree_update_moment(updates, state.momentum_buffer, beta1, 1)
        
        # Calculate second moment along specified axes (step 8 in algorithm)
        def calc_and_update_second_moment(update, prev_moment):
            # Calculate squared and take mean over specified axes
            squared = update * update
            for axis in sorted(mean_axis, reverse=True):
                if axis < len(update.shape):
                    squared = jnp.mean(squared, axis=axis, keepdims=True)
            # Update second moment buffer
            for axis in sorted(mean_axis, reverse=True):
                if axis < len(update.shape):
                    squared = squared.repeat(update.shape[axis], axis=axis)
            return prev_moment * beta2 + squared * (1 - beta2)

        second_moment_buffer = jax.tree_util.tree_map(
            calc_and_update_second_moment,
            updates,
            state.second_moment_buffer)
        
        # Bias correction (steps 7 and 9 in algorithm)
        count_inc = optax.safe_increment(state.count)
        momentum_hat = otu.tree_bias_correction(momentum_buffer, beta1, count_inc)
        second_moment_hat = otu.tree_bias_correction(second_moment_buffer, beta2, count_inc)
        
        # Calculate updates (step 10 in algorithm)
        def apply_update(m_hat, v_hat):
            return m_hat / (jnp.sqrt(v_hat) + epsilon)
        
        # Apply the update using the momentum and second moment
        updates = jax.tree_util.tree_map(
            lambda m, v: apply_update(m, v), 
            momentum_hat, 
            second_moment_hat
        )
        
        return updates, ScaleByMiniState(
            count=count_inc, 
            momentum_buffer=momentum_buffer, 
            second_moment_buffer=second_moment_buffer
        )
    
    return optax.GradientTransformation(init_fn, update_fn)