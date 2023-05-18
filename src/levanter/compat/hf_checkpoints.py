import json
import logging
import os
import shutil
import tempfile
import urllib.parse
from typing import Optional, Union, cast

import fsspec
import huggingface_hub
import jax
import jax.numpy as jnp
import numpy as np
import safetensors
import safetensors.numpy
from fsspec import AbstractFileSystem
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from jax.experimental import multihost_utils
from jax.experimental.multihost_utils import sync_global_devices
from jax.random import PRNGKey
from transformers import GPT2Config as HfGpt2Config

from haliax import Axis
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel
from levanter.trainer_hooks import StepInfo


logger = logging.getLogger(__name__)


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
            checkpoint = safetensors.numpy.load_file(f"{location_or_id}/{SAFE_TENSORS_MODEL}")
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
            checkpoint = safetensors.numpy.load_file(model_path)
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
    model = Gpt2LMHeadModel.init(Vocab, lev_config, key=key)

    has_transformer_prefix = False
    for k in checkpoint.keys():
        if k.startswith("transformer."):
            has_transformer_prefix = True
            break
        elif k.startswith(".h"):
            break

    if has_transformer_prefix:
        model = model.from_state_dict(checkpoint, prefix="transformer")
    else:
        model = model.from_state_dict(checkpoint)

    return model


def _save_hf_gpt2_checkpoint_local(model: Gpt2LMHeadModel, path):
    os.makedirs(path, exist_ok=True)
    config = gpt2_config_to_hf(model.vocab_size, model.config)
    with open(f"{path}/config.json", "w") as f:
        json.dump(config.to_dict(), f)

    def get_to_cpu(arr: Union[jnp.ndarray, np.ndarray]):
        if isinstance(arr, np.ndarray):
            return arr
        elif "cpu" in arr.device().device_kind:
            return np.array(arr)
        elif arr.is_fully_addressable:
            r = np.array(arr)
            return r
        else:
            return np.array(jax.device_get(multihost_utils.process_allgather(arr, tiled=True)))

    # need to make sure the model is on *this machine* and *this machine's CPU* before saving
    model = jax.tree_map(lambda arr: get_to_cpu(arr), model)

    # TODO: it's be nice if safetensors supported an iterator or something so we could do the allgather one at a time
    state_dict = model.to_state_dict()

    # now that we've moved the model to the CPU, we don't need to do this on all processes
    if jax.process_index() == 0:
        # the "pt" is a lie but it doesn't seem to actually matter and HF demands it
        safetensors.numpy.save_file(state_dict, f"{path}/{SAFE_TENSORS_MODEL}", metadata={"format": "pt"})


def _is_url_like(path):
    return urllib.parse.urlparse(path).scheme != ""


def save_hf_gpt2_checkpoint(model: Gpt2LMHeadModel, path, hf_repo: Optional[str] = None, **hf_upload_kwargs):
    """
    If hf_repo is provided, this will upload the checkpoint to the huggingface hub, passing
    any additional kwargs to the huggingface_hub.upload_folder function.

    :param path: the path to save the checkpoint to. path may be a GCS bucket path, in which case the checkpoint will be
    uploaded to GCS after being written to a tmp
    :param model: the model to save
    :param hf_repo:
    :param hf_upload_kwargs: any additional kwargs to pass to huggingface_hub.upload_folder
    :return:
    """
    tmpdir: Optional[str] = None
    if _is_url_like(path):
        tmpdir = tempfile.mkdtemp()
        local_path = tmpdir
    else:
        local_path = path

    try:
        logger.info(f"Saving HF-compatible checkpoint to {local_path}")
        _save_hf_gpt2_checkpoint_local(cast(Gpt2LMHeadModel, model), local_path)
        logger.debug(f"Finished saving HF-compatible checkpoint to {local_path}")

        sync_global_devices(path)

        if jax.process_index() == 0:
            if tmpdir is not None:  # we're uploading to GCS or similar
                logger.info(f"Copying HF-compatible checkpoint to {path}")
                fs: AbstractFileSystem
                fs = fsspec.core.get_fs_token_paths(path, mode="wb")[0]
                fs.put(os.path.join(local_path, "*"), path, recursive=True)
                logger.debug(f"Finished copying HF-compatible checkpoint to {path}")

            if hf_repo is not None:
                logger.info(f"Uploading HF-compatible checkpoint to {hf_repo}")
                huggingface_hub.upload_folder(local_path, hf_repo, **hf_upload_kwargs)
                logger.debug(f"Finished uploading HF-compatible checkpoint to {hf_repo}")

        sync_global_devices(path + " done")
    finally:
        if tmpdir is not None:
            shutil.rmtree(tmpdir)


def save_hf_gpt2_checkpoint_callback(base_path, hf_repo: Optional[str] = None, **hf_upload_kwargs):
    """
    If hf_repo is provided, this will upload the checkpoint to the huggingface hub, passing
    any additional kwargs to the huggingface_hub.upload_folder function.

    :param base_path: the base path to save the checkpoint to. `/step-<step>` will be appended to this. base_path
    may be a GCS bucket path, in which case the checkpoint will be uploaded to GCS after being written to a tmp
    :param hf_repo:
    :param hf_upload_kwargs:
    :return:
    """

    def cb(step: StepInfo):
        nonlocal hf_upload_kwargs
        if step.step == 0:
            return
        if hf_repo is not None and "commit_message" not in hf_upload_kwargs:
            my_upload_kwargs = hf_upload_kwargs.copy()
            my_upload_kwargs["commit_message"] = f"Upload for step {step.step} from Levanter"
        else:
            my_upload_kwargs = hf_upload_kwargs
        save_hf_gpt2_checkpoint(
            cast(Gpt2LMHeadModel, step.model), f"{base_path}/step-{step.step}", hf_repo=hf_repo, **my_upload_kwargs
        )

    return cb
