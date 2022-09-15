import json
import os

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
        map_location = torch.device("cpu")

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
        scale_attn_by_inverse_layer_idx=config.scale_attn_by_inverse_layer_idx,
        upcast_attn=config.reorder_and_upcast_attn,
    )

    return levanter_config


def gpt2_config_to_hf(vocab_size: int, config: Gpt2Config) -> HfGpt2Config:
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
        scale_attn_by_inverse_layer_idx=config.scale_attn_by_inverse_layer_idx,
        reorder_and_upcast_attn=config.upcast_attn,
    )

    return hf_config


def load_hf_gpt2_checkpoint(location_or_id, map_location=None, revision=None):
    config, checkpoint = load_hf_model_checkpoint(location_or_id, map_location=map_location, revision=revision)

    config = HfGpt2Config.from_dict(config)

    Vocab = Axis("vocab", config.vocab_size)
    lev_config = hf_gpt2_config_to_levanter(config)
    key = PRNGKey(0)
    model = Gpt2LMHeadModel(Vocab, lev_config, key=key)

    has_transformer_prefix = False
    for k in checkpoint.keys():
        if k.startswith("transformer."):
            has_transformer_prefix = True
            break
        elif k.startswith(".h"):
            break

    if has_transformer_prefix:
        model = model.from_torch_dict(checkpoint, prefix="transformer")
    else:
        model = model.from_torch_dict(checkpoint)

    return model


def save_hf_gpt2_checkpoint(path, model: Gpt2LMHeadModel):
    config = gpt2_config_to_hf(model.vocab_size, model.config)
    torch_dict = model.to_torch_dict()
    os.makedirs(path, exist_ok=True)
    torch.save(torch_dict, f"{path}/pytorch_model.bin")
    with open(f"{path}/config.json", "w") as f:
        json.dump(config.to_dict(), f)
