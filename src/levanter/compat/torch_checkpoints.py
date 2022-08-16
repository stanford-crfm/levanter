import json
import os
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as onp
import torch
from huggingface_hub import cached_download, hf_hub_url
from jax.random import PRNGKey
from transformers import GPT2Config as HfGpt2Config

from haliax import Axis
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel


def load_hf_model_checkpoint(location_or_id, model_file="pytorch_model.bin", map_location=None, revision=None):
    """
    Loads a PyTorch model checkpoint.
    """
    if map_location is None:
        map_location = torch.device('cpu')

    if os.path.exists(f"{location_or_id}/{model_file}"):
        config = json.load(open(f"{location_or_id}/config.json"))
        checkpoint = torch.load(f"{location_or_id}/{model_file}", map_location=map_location)
    else:
        url = hf_hub_url(location_or_id, model_file, revision=revision)
        model_path = cached_download(url)
        checkpoint = torch.load(model_path, map_location=map_location)

        config_url = hf_hub_url(location_or_id, "config.json", revision=revision)
        config_path = cached_download(config_url)
        config = json.load(open(config_path))

    return config, checkpoint


def hf_gpt2_config_to_levanter(config: HfGpt2Config) -> Gpt2Config:
    # various things we don't support in our gpt2
    assert not config.scale_attn_by_inverse_layer_idx

    levanter_config = Gpt2Config(
        seq_len=config.n_positions,
        # vocab_size=config.vocab_size,
        num_layers=config.n_layer,
        num_heads=config.n_head,
        hidden_dim=config.n_embd,
        initializer_range=config.initializer_range,
        attn_pdrop=config.attn_pdrop,
        embed_pdrop=config.embd_pdrop,
        layer_norm_epsilon=config.layer_norm_epsilon,
        activation_function=config.activation_function,
    )

    return levanter_config


def gpt2_config_to_hf(vocab_size: int, config: Gpt2Config) -> HfGpt2Config:
    # various things we don't support in our gpt2

    hf_config = HfGpt2Config(
        vocab_size=vocab_size,
        n_positions=config.seq_len,
        n_layer=config.num_layers,
        n_head=config.num_heads,
        n_embd=config.hidden_dim,
        initializer_range=config.initializer_range,
        attn_pdrop=config.attn_pdrop,
        embd_pdrop=config.embed_pdrop,
        layer_norm_epsilon=config.layer_norm_epsilon,
        activation_function=config.activation_function,
    )

    return hf_config


def load_hf_gpt2_checkpoint(location_or_id, map_location=None, revision=None):
    config, checkpoint = load_hf_model_checkpoint(location_or_id, map_location=map_location, revision=revision)

    config = HfGpt2Config.from_dict(config)

    vocab = Axis("vocab", config.vocab_size)
    lev_config = hf_gpt2_config_to_levanter(config)
    key = PRNGKey(0)
    model = Gpt2LMHeadModel(vocab, lev_config, key=key)

    model = use_torch_weights(model, checkpoint)

    return model


def save_hf_gpt2_checkpoint(path, model: Gpt2LMHeadModel):
    config = gpt2_config_to_hf(model.vocab_size, model.config)
    torch_dict = make_torch_model_dict(model)
    os.makedirs(path, exist_ok=True)
    torch.save(torch_dict, f"{path}/pytorch_model.bin")
    with open(f"{path}/config.json", "w") as f:
        json.dump(config.to_dict(), f)


def use_torch_weights(model: eqx.Module, checkpoint: dict) -> eqx.Module:
    """Given an equinox Module that implements torch_key_leaves (as per our gpt2 implementation),
    return a modified version of the module with the weights initialized to those values"""

    # TODO: make a class for torch_key_leaves that we can use here
    torch_keys = model.torch_key_leaves()

    def to_jax(t: Optional[torch.Tensor]) -> Optional[jnp.ndarray]:
        if t is None:
            return None
        return jnp.array(t.cpu().numpy())

    tensors = [to_jax(checkpoint[k]) if k else None for k in torch_keys]
    leaves, structure = jax.tree_flatten(model)
    # verify shapes match
    for old, new in zip(leaves, tensors):
        if old is None:
            assert new is None
        else:
            assert old is not None  # make flow typing happy
            assert new is not None  # make flow typing happy
            assert old.shape == new.shape, f"{old.shape} != {new.shape}"

    model = jax.tree_unflatten(structure, tensors)

    return model


def make_torch_model_dict(model: eqx.Module) -> dict:
    """Given an equinox Module that implements torch_key_leaves (as per our gpt2 implementation),
    return a dictionary of the weights compatible with a torch state dict"""

    torch_keys = model.torch_key_leaves()

    def to_torch(t: Optional[jnp.ndarray]) -> Optional[torch.Tensor]:
        if t is None:
            return None
        return torch.from_numpy(onp.array(t))

    leaves, structure = jax.tree_flatten(model)

    assert len(leaves) == len(torch_keys)

    return {k: to_torch(v) for k, v in zip(torch_keys, leaves) if k is not None}
