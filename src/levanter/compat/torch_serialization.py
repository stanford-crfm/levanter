from typing import Any, Dict, Optional, TypeVar

import equinox as eqx
import jax.numpy as jnp
import numpy as onp
from equinox.custom_types import PyTree

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
        return _default_eqx_module_to_dict(self, prefix, key_map=self._torch_key_map())

    def from_torch_dict(self: Mod, torch_dict: StateDict, prefix: Optional[str] = None) -> Mod:
        return _default_eqx_module_from_dict(self, torch_dict, prefix, key_map=self._torch_key_map())

    def update_torch_dict(self, torch_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        return _default_update_dict_with_eqx_module(self, prefix, torch_dict, key_map=self._torch_key_map())

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
        return eqx_module_from_torch_dict(tree, torch_dict, prefix)
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
        update_torch_dict_with_eqx_module(torch_dict, tree, prefix)
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


def eqx_module_from_torch_dict(mod: Mod, torch_dict: StateDict, prefix: Optional[str] = None) -> Mod:
    if hasattr(mod, "from_torch_dict"):
        return mod.from_torch_dict(torch_dict, prefix)

    return _default_eqx_module_from_dict(mod, torch_dict, prefix)


def _default_eqx_module_from_dict(mod, torch_dict, prefix, key_map=None):
    old_values, mod_state = mod.tree_flatten()
    dyn_keys = mod_state[0]

    new_values = []
    for k, old in zip(dyn_keys, old_values):
        if key_map is not None and k in key_map:
            k = key_map[k]
        # TODO: might want to add a flag that allows missing keys?
        new_values.append(jax_tree_from_torch_state_dict(old, torch_dict, apply_prefix(prefix, k)))

    return mod.tree_unflatten(mod_state, new_values)


def eqx_module_to_torch_dict(mod: Mod, prefix: Optional[str] = None) -> StateDict:
    if hasattr(mod, "to_torch_dict"):
        return mod.to_torch_dict(prefix)

    return _default_eqx_module_to_dict(mod, prefix)


def _default_eqx_module_to_dict(mod, prefix, key_map=None):
    state_dict = {}
    update_torch_dict_with_eqx_module(state_dict, mod, prefix)

    if key_map is not None:
        state_dict = {key_map.get(k, k): v for k, v in state_dict.items()}

    return state_dict


def update_torch_dict_with_eqx_module(state_dict: StateDict, mod: Mod, prefix: Optional[str] = None) -> StateDict:
    if hasattr(mod, "update_torch_dict"):
        mod.update_torch_dict(state_dict, prefix)
        return state_dict

    return _default_update_dict_with_eqx_module(mod, prefix, state_dict)


def _default_update_dict_with_eqx_module(mod: eqx.Module, prefix, state_dict, key_map=None):
    values, mod_state = mod.tree_flatten()
    dyn_keys = mod_state[0]
    for k, v in zip(dyn_keys, values):
        if key_map is not None and k in key_map:
            k = key_map[k]

        update_torch_dict_with_jax_tree(v, state_dict, apply_prefix(prefix, k))

    return state_dict
