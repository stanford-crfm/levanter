import dataclasses
from dataclasses import dataclass
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu

import haliax
from haliax.nn import Linear

from levanter.optim.config import OptimizerConfig
from levanter.optim.util import map_flattened_linear_layers
from levanter.utils.jax_utils import leaf_key_paths


@OptimizerConfig.register_subclass("muon")
@dataclass
class MuonConfig(OptimizerConfig):
    """
    Muon optimizer configuration: Momentum Orthogonalized by Newton-Schulz.
    """

    lr: float = 0.02
    muon_to_adam_lr: float = 0.18  # Scaling factor between AdamW and Muon learning rates
    momentum: float = 0.95
    nesterov: bool = True
    backend_steps: int = 10  # Number of steps for Newton-Schulz orthogonalization
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    # adam_modules: Optional[list[str] | str] = None
    # """A regex or a list of strings to identify where to mask weight.
    # For nano-GPT, this field can be set as `r".*attn.*weight|.*mlp.*weight|.*token_embeddings|.*position_embeddings"`"""
    # default_adam_mask: Optional[bool] = None
    # """Whether to apply a default reasonable weight decay to modules not explicitly masked. None means it will if
    # no weight_decay_modules are set. False means it will not. True means it will regardless of weight_decay_modules."""

    def build(self, num_train_steps):
        """
        Creates the optimizer.
        """
        learning_rate_schedule = self.lr_scheduler(num_train_steps)

        def optimizer(learning_rate):
            adam_lr = learning_rate * self.muon_to_adam_lr

            def muon_transform():
                components = []
                # Muon seems incompatible with gradient clipping, need to investigate
                # if self.max_grad_norm:
                #     components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_with_muon(self.momentum, self.nesterov, self.backend_steps))
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
                components.append(optax.scale(-adam_lr))
                optimizer = optax.chain(*components)
                return optimizer

            transformations = {
                "muon": muon_transform(),
                "adamw": adamw_transform(),
            }

            return optax.multi_transform(transformations, self.create_mask)

        return optax.inject_hyperparams(optimizer)(learning_rate=learning_rate_schedule)

    def create_mask(self, params):
        """
        Creates a mask that labels parameters as 'muon' or 'adamw' based on their
        dimensionality and module path, using AdamW for Embedding and lm_head parameters.
        """
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            if "Embedding" in path_str or "lm_head" in path_str:
                return "adamw"
            elif isinstance(param, Linear):
                # muon for linear layers
                return dataclasses.replace(param, weight="muon", bias="adamw" if param.bias is not None else None)
            else:
                return "adamw"

        return jax.tree_util.tree_map(mask_fn, params, paths, is_leaf=lambda x: isinstance(x, Linear))


class ScaleByMuonState(NamedTuple):
    """State for the Mars algorithm."""

    momentum_buffer: optax.Updates


def scale_with_muon(momentum=0.95, nesterov=True, steps=5):
    def init_fn(params):
        momentum_buffer = otu.tree_zeros_like(params)  # First moment
        return ScaleByMuonState(momentum_buffer=momentum_buffer)

    def update_fn(updates, state, params=None):
        buf = state.momentum_buffer
        buf = jax.tree.map(
            lambda m, g: None if g is None else momentum * m + g,
            buf,
            updates,
            is_leaf=lambda x: x is None,
        )
        if nesterov:
            updates = jax.tree.map(
                lambda m, g: None if g is None else momentum * m + g,
                buf,
                updates,
                is_leaf=lambda x: x is None,
            )
        else:
            updates = buf

        def transform_linear_layer(layer: haliax.nn.Linear):
            assert layer.weight.ndim == 2

            updated_weight_array = zeropower_via_newtonschulz5(layer.weight.array, steps=steps)

            scale = jnp.sqrt(jnp.maximum(1, updated_weight_array.shape[0] / updated_weight_array.shape[1]))
            updated_weight_array *= scale

            updated_weight = dataclasses.replace(layer.weight, array=updated_weight_array)

            return dataclasses.replace(layer, weight=updated_weight)  # type: ignore

        updates = map_flattened_linear_layers(transform_linear_layer, updates)

        return updates, ScaleByMuonState(momentum_buffer=buf)

    return optax.GradientTransformation(init_fn, update_fn)


def zeropower_via_newtonschulz5(X, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
    chex.assert_rank(X, 2)
    a, b, c = (3.4445, -4.7750, 2.0315)
    X /= jnp.linalg.norm(X) + eps  # Ensure top singular value <= 1
    transpose = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transpose = True
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X
