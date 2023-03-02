import json
import os

import safetensors
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from jax.random import PRNGKey
from transformers import GPT2Config as HfGpt2Config

from haliax import Axis
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel


PYTORCH_MODEL = "pytorch_model.bin"
SAFE_TENSORS_MODEL = "model.safetensors"


def load_hf_model_checkpoint(location_or_id, device=None, revision=None):
    """
    Loads a safetensors or PyTorch model checkpoint.
    If model_file is None, this function attempts to load via safetensors first, then PyTorch.
    """
    if os.path.exists(f"{location_or_id}/config.json"):
        config = json.load(open(f"{location_or_id}/config.json"))
        if os.path.exists(f"{location_or_id}/{SAFE_TENSORS_MODEL}"):
            checkpoint = safetensors.torch.load_file(f"{location_or_id}/{SAFE_TENSORS_MODEL}", device=device)
        elif os.path.exists(f"{location_or_id}/{PYTORCH_MODEL}"):
            import torch

            checkpoint = torch.load(f"{location_or_id}/{PYTORCH_MODEL}", map_location=device)
        else:
            raise ValueError(f"Could not find model file for {location_or_id}")
    else:
        config_path = hf_hub_download(location_or_id, "config.json", revision=revision)
        config = json.load(open(config_path))

        try:
            model_path = hf_hub_download(location_or_id, SAFE_TENSORS_MODEL, revision=revision)
            checkpoint = safetensors.torch.load_file(model_path, device=device)
        except EntryNotFoundError:  # noqa: E722
            model_path = hf_hub_download(location_or_id, PYTORCH_MODEL, revision=revision)
            import torch

            if isinstance(device, str):
                device = torch.device(device)
            checkpoint = torch.load(model_path, map_location=device)

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


def load_hf_gpt2_checkpoint(location_or_id, device=None, revision=None):
    config, checkpoint = load_hf_model_checkpoint(location_or_id, device=device, revision=revision)

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
    import torch

    config = gpt2_config_to_hf(model.vocab_size, model.config)
    torch_dict = model.to_torch_dict()
    os.makedirs(path, exist_ok=True)
    torch.save(torch_dict, f"{path}/pytorch_model.bin")
    with open(f"{path}/config.json", "w") as f:
        json.dump(config.to_dict(), f)
