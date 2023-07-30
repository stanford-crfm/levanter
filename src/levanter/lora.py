"""
Implements LoRA https://arxiv.org/abs/2106.09685 transforms on Levanter models
"""
import dataclasses
import functools
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TypeVar, Union

import equinox as eqx
import jax
from jaxtyping import PyTree

import haliax
import haliax.nn as hnn
from haliax import Axis
from haliax.jax_utils import shaped_rng_split

from levanter.compat.torch_serialization import StateDict, StateDictSerializationMixin, jax_tree_to_state_dict
from levanter.utils.jax_utils import join_key, key_iterator, leaf_key_paths


M = TypeVar("M", bound=PyTree)

# Tasks
# - filter parameters for training
# - training script
# - Peft export
# - Peft import
# - bias
# - dropout
# - registry of targets for different models

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


DEFAULT_DICT_PREFIX = "base_model.model"


def lora_state_dict(model: M, prefix: Optional[str] = DEFAULT_DICT_PREFIX) -> StateDict:
    """
    Returns a state dict of the LoRA parameters of the given model without other parameters.
    This method attempts to return a state dict compatible with PEFT's import method.
    """
    state_dict = jax_tree_to_state_dict(filter_lora_params(model), prefix=prefix)
    return {k: v for k, v in state_dict.items() if v is not None}


class LoraLinear(eqx.Module, StateDictSerializationMixin):
    """
    Linear layer with LoRA transform.
    """

    wrapped: hnn.Linear
    lora_A: hnn.Linear
    lora_B: hnn.Linear
    alpha: float = eqx.field(static=True)

    def __call__(self, x):
        y = self.wrapped(x)
        z = self.lora_A(x)
        z = self.lora_B(z)

        z = z * (self.alpha / self.r)

        return z + y

    @property
    def r(self) -> int:
        return self.lora_A.Out.size  # type: ignore

    @staticmethod
    def init(wrapped: hnn.Linear, r: int, alpha: float, *, key):
        """
        Initializes a LoraLinear module.
        """
        _R = haliax.Axis(LORA_R, r)
        key_A, key_B = jax.random.split(key)
        # Peft always uses out_first=True (i.e. normal Torch convention) for linear, even for gpt2-style Conv1d
        lora_A = hnn.Linear.init(wrapped.In, _R, key=key_A, use_bias=False, out_first=True)
        lora_B = hnn.Linear.init(_R, wrapped.Out, key=key_B, use_bias=False, out_first=True)

        return LoraLinear(wrapped, lora_A, lora_B, alpha)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {
            "wrapped": None,
        }


def _is_lora_compatible_module(module):
    # TODO: more modules
    return isinstance(module, hnn.Linear)


def filter_lora_params(params: M) -> M:
    """
    Filters LoRA parameters from the given parameter tree.
    """

    def _keep_only_lora_params(node):
        if not isinstance(node, LoraLinear):
            return None
        return dataclasses.replace(node, wrapped=None)

    return jax.tree_util.tree_map(_keep_only_lora_params, params, is_leaf=lambda node: isinstance(node, LoraLinear))


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

    :param model:
    :param config:
    :param key:
    :param prefix: prefix of the current key path
    :param batch_dims: batch dimensions of the current key path
    :return:
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
        return functools.reduce(lambda ctor, batch_axis: haliax.vmap(ctor, batch_axis), reversed(batch_dims), ctor)

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
