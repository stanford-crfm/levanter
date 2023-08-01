"""
Implements LoRA https://arxiv.org/abs/2106.09685 transforms on Levanter models
"""
import dataclasses
import functools
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TypeVar, Union

import equinox as eqx
import jax
from jaxtyping import PyTree

import haliax as hax
import haliax.nn as hnn
from haliax import Axis
from haliax.jax_utils import shaped_rng_split

from levanter.compat.torch_serialization import (
    StateDict,
    StateDictSerializationMixin,
    save_state_dict,
    to_numpy_state_dict,
)
from levanter.utils.jax_utils import join_key, key_iterator, leaf_key_paths


M = TypeVar("M", bound=PyTree)

# Tasks
# - bias
# - dropout
# - registry of targets for different models
# - better filtering of parameters (make our own old-style filter_grad)
# - replicate alpaca-lora functionality
# - document alpaca-lora functionality


LORA_R = "LORA_R"


@dataclass(frozen=True)
class LoraConfig:
    target_modules: Union[List[str], str]
    r: int = 8  # rank of LoRA transform
    alpha: float = 8.0  # scaling factor for LoRA transform
    # TODO: bias
    # TODO: dropout


def loraize(model: M, config: LoraConfig, key: jax.random.PRNGKey) -> M:
    """
    Applies LoRA transform to the given model by replacing Linear layers that match the given pattern with LoraLinear layers.
    """
    return _loraize(model, config, key, "", batch_dims=())


class LowRankLinear(eqx.Module):
    lora_A: hnn.Linear
    lora_B: hnn.Linear
    scale: float = eqx.field(static=True)

    def __call__(self, x):
        z = self.lora_A(x)
        z = self.lora_B(z)
        return z * self.scale

    @staticmethod
    def init(In: hax.Axis, Out: Axis, r: int, alpha: float, *, key):
        """
        Initializes a LoraLinear module.
        """
        _R = hax.Axis(LORA_R, r)
        key_A, key_B = jax.random.split(key)
        # Peft always uses out_first=True (i.e. normal Torch convention) for linear, even for gpt2-style Conv1d
        lora_A = hnn.Linear.init(In, _R, key=key_A, use_bias=False, out_first=True)
        lora_B = hnn.Linear.init(_R, Out, key=key_B, use_bias=False, out_first=True)

        return LowRankLinear(lora_A, lora_B, alpha / r)

    def merge(self) -> hax.NamedArray:
        return hax.dot(LORA_R, self.lora_A.weight, self.lora_B.weight) * self.scale


class LoraLinear(eqx.Module, StateDictSerializationMixin):
    """
    Linear layer with LoRA transform.
    """

    wrapped: hnn.Linear
    lora: LowRankLinear

    def __call__(self, x):
        return self.lora(x) + self.wrapped(x)

    def merge(self):
        weight = self.lora.merge() + self.wrapped.weight
        return dataclasses.replace(self.wrapped, weight=weight)

    @staticmethod
    def init(wrapped: hnn.Linear, r: int, alpha: float, *, key):
        """
        Initializes a LoraLinear module.
        """
        lora = LowRankLinear.init(wrapped.In, wrapped.Out, r, alpha, key=key)
        return LoraLinear(wrapped, lora)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"wrapped": None, "lora": None}


def _is_lora_compatible_module(module):
    # TODO: more modules
    return isinstance(module, hnn.Linear)


def filter_lora_params(params: M) -> M:
    """
    Filters LoRA parameters from the given parameter tree.
    """

    return eqx.filter(params, is_lora_param, is_leaf=is_lora_param)


def partition_lora_params(params: M) -> Tuple[M, M]:
    """
    Partitions the given parameter tree into base/non-LoRA parameters and non-LoRA parameters.
    """
    partitioned = eqx.partition(params, is_lora_param, is_leaf=is_lora_param)
    return partitioned[1], partitioned[0]


def combine_lora_params(params: M, lora_params: M) -> M:
    """
    Combines the given LoRA parameters with the given parameter tree.
    """
    return eqx.combine(params, lora_params, is_leaf=is_lora_param)


def is_lora_param(node):
    return isinstance(node, LowRankLinear)


def _loraize(model: M, config: LoraConfig, key: jax.random.PRNGKey, prefix: str, batch_dims: Tuple[Axis, ...]) -> M:
    """

    This implementation is mostly straightforward, with one major wrinkle: scan layers like Stacked, which
    add an extra batch dimension and thus require vmap, and thus require a vmap'ed LoRA transform.

    As an example, the GPT-2 Model has a Stacked[Gpt2Block] member, which means that all members of the Gpt2Block
    have an extra Layers dimension. E.g. the c_attn will have shape weight=NamedArray(['layers', 'embed', 'qkv', 'heads', 'head_size']
    even though it's defined as just Linear(In=Embed, Out=(Qkv, Heads, HeadSize)). The Stacked adds the initial Layers dimension.

    There are two ways we can approach scan layers: one is to ask implementors of lora layers to handle
    this themselves, and the other is to handle it here. The former is more flexible, but the latter is
    more convenient, even if it runs the risk of being a leaky abstraction. We choose the latter.
    """
    key_iter = key_iterator(key)

    def _is_special_module(module):
        return _is_lora_compatible_module(module) or isinstance(module, hnn.Stacked)

    if isinstance(config.target_modules, str):
        compiled = re.compile(config.target_modules)
        matches_target = lambda key_path: compiled.match(key_path) is not None  # noqa
    else:
        matches_target = lambda key_path: any(key_path.endswith(target) for target in config.target_modules)  # noqa

    def _batchify_ctor(ctor):
        # this is gross but it basically just vmaps the ctor over each batch dimension
        return functools.reduce(lambda ctor, batch_axis: hax.vmap(ctor, batch_axis), reversed(batch_dims), ctor)

    def _loraize_module(module, key_path):
        # TODO: turn into a registry
        if isinstance(module, hnn.Stacked):
            new_inner = _loraize(
                module.stacked,
                config,
                next(key_iter),
                prefix=join_key(key_path, "stacked"),
                batch_dims=batch_dims + (module.Block,),
            )
            return dataclasses.replace(module, stacked=new_inner)
        elif matches_target(key_path) and _is_lora_compatible_module(module):
            my_key = next(key_iter)
            batched_key = shaped_rng_split(my_key, [axis.size for axis in batch_dims])
            return _batchify_ctor(LoraLinear.init)(module, config.r, config.alpha, key=batched_key)
        else:
            return module

    return jax.tree_util.tree_map(
        _loraize_module,
        model,
        leaf_key_paths(model, is_leaf=_is_special_module, prefix=prefix),
        is_leaf=_is_special_module,
    )


def merge_lora_modules(module: M) -> M:
    """
    Merges LoRA modules into their wrapped modules. That is, it adds the LoRA parameters to the wrapped weights,
    producing a modified base model with no LoRA parameters.
    """

    def _merge_lora_modules(module):
        if isinstance(module, LoraLinear):
            return module.merge()
        else:
            return module

    return jax.tree_util.tree_map(_merge_lora_modules, module, is_leaf=lambda node: isinstance(node, LoraLinear))


SAFETENSORS_WEIGHTS_NAME = "adapter_model.safetensors"
CONFIG_NAME = "adapter_config.json"
DEFAULT_DICT_PREFIX = "base_model.model.transformer"


def save_peft_pretrained(
    lora_model: M, config: LoraConfig, base_model_name_or_path, path: str, prefix: Optional[str] = DEFAULT_DICT_PREFIX
):
    """
    Saves a LoRA model as a HuggingFace checkpoint, compatible with Peft.
    """
    os.makedirs(path, exist_ok=True)
    hf_config = to_hf_config(config, base_model_name_or_path=base_model_name_or_path)
    state_dict = lora_state_dict(lora_model, prefix=prefix)

    save_state_dict(state_dict, f"{path}/{SAFETENSORS_WEIGHTS_NAME}")

    with open(f"{path}/{CONFIG_NAME}", "w") as f:
        json.dump(hf_config, f)


def to_hf_config(config: LoraConfig, base_model_name_or_path: Optional[str] = None, **kwargs) -> dict:
    """
    Converts a LoraConfig to a HuggingFace config.
    """
    # Example:
    # {
    #   "base_model_name_or_path": "decapoda-research/llama-7b-hf",
    #   "bias": "none",
    #   "enable_lora": null,
    #   "fan_in_fan_out": false,
    #   "inference_mode": true,
    #   "lora_alpha": 16,
    #   "lora_dropout": 0.05,
    #   "merge_weights": false,
    #   "modules_to_save": null,
    #   "peft_type": "LORA",
    #   "r": 16,
    #   "target_modules": [
    #     "q_proj",
    #     "k_proj",
    #     "v_proj",
    #     "o_proj"
    #   ],
    #   "task_type": "CAUSAL_LM"
    # }
    return {
        "base_model_name_or_path": base_model_name_or_path,
        "bias": "none",  # TODO: support bias
        "enable_lora": None,
        "fan_in_fan_out": False,  # TODO: support fan_in_fan_out
        "inference_mode": True,  # TODO: support inference_mode
        "lora_alpha": config.alpha,
        "lora_dropout": 0.00,  # TODO: support dropout
        "merge_weights": False,
        "modules_to_save": None,  # TODO: support modules_to_save?
        "peft_type": "LORA",
        "r": config.r,
        "target_modules": config.target_modules,
        "task_type": "CAUSAL_LM",  # TODO: support task_type
        **kwargs,
    }


def lora_state_dict(model: M, prefix: Optional[str] = DEFAULT_DICT_PREFIX) -> StateDict:
    """
    Returns a state dict of the LoRA parameters of the given model without other parameters.
    This method attempts to return a state dict compatible with PEFT's import method.
    """
    state_dict = to_numpy_state_dict(filter_lora_params(model), prefix=prefix)
    return {k: v for k, v in state_dict.items() if v is not None}
