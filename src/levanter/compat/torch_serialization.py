from typing import Any, Dict, Optional, Tuple, TypeVar, cast

import equinox as eqx
import jax.numpy as jnp
import numpy as onp
from jaxtyping import PyTree

from haliax import NamedArray


# TODO: figure out how to make mypy happy
# if typing.TYPE_CHECKING:
#     try:
#         import torch
#
#         StateDict = Dict[str, torch.Tensor]
#         TorchTensor = torch.Tensor
#     except ImportError:
#         StateDict: typing.Type[Dict[str, Any]] = Dict[str, Any]
#         TorchTensor: typing.Type[Any] = Any
# else:
StateDict = Dict[str, Any]
TorchTensor = Any


def apply_prefix(prefix: Optional[str], leaf: Optional[str]) -> Optional[str]:
    if prefix is None:
        return leaf
    elif leaf is None:
        return prefix
    else:
        return f"{prefix}.{leaf}"


Mod = TypeVar("Mod", bound=eqx.Module)


class TorchSerializationMixin:
    """An eqx.Module that can be serialized to a torch-style state dict."""

    def to_torch_dict(self, prefix: Optional[str] = None) -> StateDict:
        return jax_tree_to_torch_state_dict(self, prefix)

    def from_torch_dict(self: Mod, torch_dict: StateDict, prefix: Optional[str] = None) -> Mod:
        return default_eqx_module_from_torch_dict(self, torch_dict, prefix)

    def update_torch_dict(self, torch_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        return default_update_torch_dict_with_eqx_module(torch_dict, self, prefix)

    def _torch_key_map(self) -> Optional[Dict[str, Optional[str]]]:
        """Returns a dict mapping eqx.Module keys to torch keys that need to be renamed for serialization"""
        return None


def jax_to_torch(t: Optional[jnp.ndarray]) -> Optional[TorchTensor]:
    import torch

    if t is None:
        return None
    return torch.from_numpy(onp.array(t))


def torch_to_jax(t: Optional[TorchTensor]) -> Optional[jnp.ndarray]:
    if t is None:
        return None
    return jnp.array(t.cpu().numpy())


def jax_tree_from_torch_state_dict(tree: PyTree, torch_dict: StateDict, prefix: Optional[str] = None) -> PyTree:
    # TODO: assert compatibility of old and new values (type, shape, etc.)
    if isinstance(tree, eqx.Module):
        if hasattr(tree, "from_torch_dict"):
            return tree.from_torch_dict(torch_dict, prefix)
        else:
            return default_eqx_module_from_torch_dict(tree, torch_dict, prefix)
    elif isinstance(tree, list):
        return [
            jax_tree_from_torch_state_dict(item, torch_dict, apply_prefix(prefix, str(i)))
            for i, item in enumerate(tree)
        ]
    elif isinstance(tree, dict):
        return {
            k: jax_tree_from_torch_state_dict(v, torch_dict, prefix=apply_prefix(prefix, k)) for k, v in tree.items()
        }
    elif isinstance(tree, NamedArray):
        # TODO: where's the best place to put this logic for NamedArrays
        if prefix is None:
            raise ValueError("Cannot extract a leaf value from a torch dict without a prefix")
        return NamedArray(torch_to_jax(torch_dict[prefix]), axes=tree.axes)
    elif tree is None:
        if prefix is None:
            return None
        return torch_dict.get(prefix, None)
    else:
        if prefix is None:
            raise ValueError("Cannot extract a leaf value from a torch dict without a prefix")
        # TODO: add "strict" flag so we can return None in cases where it's just missing
        return torch_to_jax(torch_dict[prefix])


def update_torch_dict_with_jax_tree(tree: PyTree, torch_dict: StateDict, prefix: Optional[str] = None) -> None:
    if isinstance(tree, eqx.Module):
        if hasattr(tree, "update_torch_dict"):
            tree.update_torch_dict(torch_dict, prefix)
        else:
            default_update_torch_dict_with_eqx_module(torch_dict, tree, prefix)
    elif isinstance(tree, list):
        for i, item in enumerate(tree):
            update_torch_dict_with_jax_tree(item, torch_dict, prefix=apply_prefix(prefix, str(i)))
    elif isinstance(tree, dict):
        for k, v in tree.items():
            update_torch_dict_with_jax_tree(v, torch_dict, prefix=apply_prefix(prefix, k))
    elif isinstance(tree, NamedArray):
        # TODO: where's the best place to put this logic for NamedArrays
        assert prefix is not None
        torch_dict[prefix] = jax_to_torch(tree.array)
    else:
        if prefix is not None:
            torch_dict[prefix] = jax_to_torch(tree)
        else:
            raise ValueError("Cannot update torch dict with a leaf value.")


def jax_tree_to_torch_state_dict(tree: PyTree, prefix: Optional[str] = None) -> StateDict:
    state_dict: StateDict = {}
    update_torch_dict_with_jax_tree(tree, state_dict, prefix)
    return state_dict


def default_eqx_module_from_torch_dict(mod: Mod, torch_dict: StateDict, prefix: Optional[str] = None) -> Mod:
    key_map = None
    if hasattr(mod, "_torch_key_map"):
        key_map = mod._torch_key_map()

    old_values, mod_state = mod.tree_flatten()
    dyn_keys = mod_state[0]

    new_values = []
    for k, old in zip(dyn_keys, old_values):
        if key_map is not None and k in key_map:
            k = key_map[k]
        # TODO: might want to add a flag that allows missing keys?
        new_values.append(jax_tree_from_torch_state_dict(old, torch_dict, apply_prefix(prefix, k)))

    return mod.tree_unflatten(mod_state, new_values)


def default_eqx_module_to_torch_dict(mod: Mod, prefix: Optional[str] = None) -> StateDict:
    state_dict: StateDict = {}
    default_update_torch_dict_with_eqx_module(state_dict, mod, prefix)

    return state_dict


def default_update_torch_dict_with_eqx_module(
    state_dict: StateDict, mod: Mod, prefix: Optional[str] = None
) -> StateDict:
    key_map = None
    if hasattr(mod, "_torch_key_map"):
        key_map = mod._torch_key_map()

    values, mod_state = mod.tree_flatten()
    dyn_keys = mod_state[0]
    for k, v in zip(dyn_keys, values):
        if key_map is not None and k in key_map:
            k = key_map[k]

        update_torch_dict_with_jax_tree(v, state_dict, apply_prefix(prefix, k))

    return state_dict


def reshape_linear_layer(
    in_dict: StateDict, prefix: Optional[str], in_shape: Tuple[int, ...], out_shape: Tuple[int, ...]
) -> StateDict:
    """Reshape the weights and bias for a linear layer in a torch dict to a new shape."""
    new_dict: StateDict = {}
    weight_key = cast(str, apply_prefix(prefix, "weight"))
    bias_key = cast(str, apply_prefix(prefix, "bias"))
    weight = in_dict[weight_key]
    bias = in_dict[bias_key]
    weight = weight.reshape((-1,) + in_shape + out_shape)
    bias = bias.reshape((-1,) + out_shape)
    new_dict[weight_key] = weight
    new_dict[bias_key] = bias

    return new_dict
