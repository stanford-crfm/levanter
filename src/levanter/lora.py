"""
Implements LoRA https://arxiv.org/abs/2106.09685 transforms on Levanter models
"""
import dataclasses
import functools
import re
from dataclasses import dataclass
from typing import List, Tuple, TypeVar, Union

import equinox as eqx
import jax
from jaxtyping import PyTree

import haliax
import haliax.nn as hnn
from haliax import Axis
from haliax.jax_utils import shaped_rng_split

from levanter.utils.jax_utils import join_key, key_iterator, leaf_key_paths


M = TypeVar("M", bound=PyTree)

# Tasks
# - Match layers to LoRA transform
# - Define LoRA module
# - Do the module surgery
# - Peft export
# - Peft import
# - bias
# - dropout
# - registry of targets for different models


@dataclass(frozen=True)
class LoraConfig:
    target_modules: Union[List[str], str]
    r: int = 8  # rank of LoRA transform
    # TODO: bias
    # TODO: dropout


def _is_lora_compatible_module(module):
    # TODO: more modules
    return isinstance(module, hnn.Linear)


class LoraLinear(eqx.Module):
    """
    Linear layer with LoRA transform.
    """

    # TODO: is making this static a good idea?
    wrapped: hnn.Linear
    lora_A: hnn.Linear
    lora_B: hnn.Linear

    def __call__(self, x):
        y = self.wrapped(x)
        z = self.lora_A(x)
        z = self.lora_B(z)

        return z + y

    @staticmethod
    def init(wrapped: hnn.Linear, r: int, *, key):
        """
        Initializes a LoraLinear module.
        """
        _R = haliax.Axis("LORA_R", r)
        key_A, key_B = jax.random.split(key)
        lora_A = hnn.Linear.init(wrapped.In, _R, key=key_A)
        lora_B = hnn.Linear.init(_R, wrapped.Out, key=key_B)

        return LoraLinear(wrapped, lora_A, lora_B)


def loraize(model: M, config: LoraConfig, key: jax.random.PRNGKey) -> M:
    """
    Applies LoRA transform to the given model by replacing Linear layers that match the given pattern with LoraLinear layers.
    """
    return _loraize(model, config, key, "", batch_dims=())


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

        def matches_target(key_path: str):
            return compiled.match(key_path) is not None

    else:

        def matches_target(key_path: str):
            return any(key_path.endswith(target) for target in config.target_modules)

    def _batchify_ctor(ctor):
        # this is gross but it basically just vmaps the ctor over each batch dimension
        return functools.reduce(lambda ctor, batch_axis: haliax.vmap(ctor, batch_axis), reversed(batch_dims), ctor)

    def _loraize_module(module, key_path):
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
            return _batchify_ctor(LoraLinear.init)(module, config.r, key=batched_key)
        else:
            return module

    return jax.tree_util.tree_map(
        _loraize_module,
        model,
        leaf_key_paths(model, is_leaf=_is_special_module, prefix=prefix),
        is_leaf=_is_special_module,
    )
