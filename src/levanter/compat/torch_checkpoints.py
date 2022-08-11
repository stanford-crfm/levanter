# At some point we should maybe figure out a way to reflect over equinox trees for handling checkpoints. However,
# that is a bit tricky to pull off.
import json
import os
from typing import Optional

import numpy as np
import torch

import equinox as eqx
from huggingface_hub import hf_hub_url, cached_download
import jax
from jax.random import PRNGKey
import jax.numpy as jnp
import jax.random as jrandom
import numpy as onp
from transformers import GPT2Config as HfGpt2Config, AutoModelForCausalLM

from haliax import Axis
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel


def load_pytorch_model_checkpoint(location_or_id, model_file="pytorch_model.bin", map_location=None):
    """
    Loads a PyTorch model checkpoint.
    """
    if os.path.exists(f"{location_or_id}/{model_file}"):
        config = json.load(open(f"{location_or_id}/config.json"))
        checkpoint = torch.load(f"{location_or_id}/{model_file}", map_location=map_location)
    else:
        url = hf_hub_url(location_or_id, model_file)
        model_path = cached_download(url)
        checkpoint = torch.load(model_path, map_location=map_location)

        config_url = hf_hub_url(location_or_id, "config.json")
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
    # layer_norm_epsilon=config.layer_norm_epsilon,
        activation_function=config.activation_function
    )

    return levanter_config


def load_torch_gpt2_checkpoint(location_or_id, map_location=None):
    config, checkpoint = load_pytorch_model_checkpoint(location_or_id, map_location=map_location)

    config = HfGpt2Config.from_dict(config)

    vocab = Axis("vocab", config.vocab_size)
    lev_config = hf_gpt2_config_to_levanter(config)
    key = PRNGKey(0)
    model = Gpt2LMHeadModel(vocab, lev_config, key=key)

    model = use_torch_weights(model, checkpoint)

    return model


def use_torch_weights(model: eqx.Module, checkpoint: dict)-> eqx.Module:
    """Given an equinox Module that implements torch_key_leaves (as per our gpt2 implementation),
    return a modified version of the module with the weights initialized to those values"""

    # TODO: make a class for torch_key_leaves that we can use here
    torch_keys = model.torch_key_leaves()

    def to_jax(t: Optional[torch.Tensor])->Optional[jnp.ndarray]:
        if t is None:
            return None
        return jnp.array(t.cpu().numpy())

    tensors = [to_jax(checkpoint[k]) if k else None for k in torch_keys]
    leaves, structure = jax.tree_flatten(model)
    # verify shapes match
    for old, new in zip(leaves, tensors):
        assert old.shape == new.shape, f"{old.shape} != {new.shape}"

    model = jax.tree_unflatten(structure, tensors)

    return model


if __name__ == '__main__':
    config, data = load_pytorch_model_checkpoint("gpt2")
    config = HfGpt2Config.from_dict(config)

    model = load_torch_gpt2_checkpoint("gpt2")

    del data

    # Sanity check that the model behaves similarly
    torch_model = AutoModelForCausalLM.from_pretrained("gpt2")
    torch_model.eval()

    def rand_input(key: PRNGKey, num: int,  seq_len: int) -> jnp.ndarray:
        return jrandom.randint(key, (num, seq_len,), 0, config.vocab_size)

    input = rand_input(PRNGKey(0), 1, config.n_positions)

    torch_out = torch_model(torch.from_numpy(onp.array(input)).to(torch.int32))
    torch_out = torch_out.logits[0].detach().cpu().numpy()

    jax_out = model(input[0], key=None)

    assert torch_out.shape == jax_out.shape, f"{torch_out.shape} != {jax_out.shape}"
    assert np.isclose(torch_out, onp.array(jax_out)).all(), f"{torch_out} != {jax_out}"

    print(config)
