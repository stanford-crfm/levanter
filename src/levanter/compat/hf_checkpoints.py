import abc
import json
import logging
import os
import shutil
import tempfile
import urllib.parse
from dataclasses import dataclass
from functools import cached_property
from typing import Generic, Optional, Tuple, Type, TypeVar, Union, cast

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
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2Config as HfGpt2Config
from transformers import PretrainedConfig as HfConfig
from transformers import PreTrainedTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.auto.auto_factory import _get_model_class

import haliax
from haliax import Axis
from haliax.jax_utils import filter_eval_shape
from haliax.partitioning import ResourceMapping
from levanter.trainer_hooks import StepInfo


logger = logging.getLogger(__name__)


PYTORCH_MODEL = "pytorch_model.bin"
SAFE_TENSORS_MODEL = "model.safetensors"


@dataclass
class RemoteRef:
    model_name_or_path: str
    revision: Optional[str] = None

    @staticmethod
    def from_string(s: str) -> "RemoteRef":
        """syntax is <model_name_or_path>@<revision>"""
        if "@" not in s:
            return RemoteRef(s)
        model_name_or_path, revision = s.split("@")
        return RemoteRef(model_name_or_path, revision)

    def __str__(self) -> str:
        return f"{self.model_name_or_path}@{self.revision}"

    def __repr__(self) -> str:
        return f"RemoteRev({self.model_name_or_path!r}, {self.revision!r})"


class ConfigWithHFSer(abc.ABC):
    @abc.abstractmethod
    def to_hf_config(self, vocab_size: int, config_overrides: Optional[dict] = None) -> HfConfig:
        pass

    @classmethod
    @abc.abstractmethod
    def from_hf_config(cls, hf_config: HfConfig):
        pass


MConfig = TypeVar("MConfig", bound=ConfigWithHFSer)


class LmWithHFSer(abc.ABC, Generic[MConfig]):
    config: MConfig

    def get_hf_config(self):
        return self.config.to_hf_config(self.Vocab.size)

    @property
    @abc.abstractmethod
    def Vocab(self) -> Axis:
        pass

    @classmethod
    @abc.abstractmethod
    def init(cls, Vocab: Axis, config: MConfig, *, key: PRNGKey) -> "LmWithHFSer":
        pass


def _coerce_to_rr(s: Union[str, RemoteRef]) -> RemoteRef:
    if isinstance(s, RemoteRef):
        return s
    else:
        return RemoteRef.from_string(s)


@dataclass
class HFAutoMapConfig:
    """
    To create a custom AutoModel class, in your model's config.json,
    you will need to add a field like this:
    "auto_map": {
        "AutoConfig": "backpack_config.BackpackGPT2Config",
        "AutoModelForCausalLM": "backpack_model.BackpackGPT2LMHeadModel"
    },
    """

    AutoConfig: Optional[str] = None  # path of the AutoConfig class
    AutoModelForCausalLM: Optional[str] = None  # path of the AutoModel class

    def to_dict(self) -> dict:
        """A helper function to convert class to dict"""
        return {k: v for k, v in self.__dict__.items() if v is not None}


LevConfig = TypeVar("LevConfig", bound=ConfigWithHFSer)


@dataclass
class HFCheckpointConverter(abc.ABC, Generic[LevConfig]):
    LevConfigClass: Type[LevConfig]
    reference_checkpoint: Optional[Union[str, RemoteRef]] = None
    "A reference HF Hub checkpoint to extract non-parameter files (like model code an config from)"

    hf_config_class: Optional[Union[str, Type]] = None
    "The HFConfig class to use. If None, will be inferred from the reference_checkpoint"

    config_overrides: Optional[dict] = None
    "A dictionary of config overrides to apply to the HFConfig when saving. typically used for auto_map"

    trust_remote_code: bool = False
    "If True, will trust the remote code and not download it locally."

    ignore_prefix: Optional[str] = None
    """A prefix to optionally ignore when loading checkpoints. Typically this is 'transformer' to deal with the
    fact that some models are saved as XXXPreTrainedModel and others are saved as XXXLMHeadModel"""

    def __post_init__(self):
        self.reference_checkpoint = _coerce_to_rr(self.reference_checkpoint)

    @cached_property
    def HFConfigClass(self) -> Type:
        if self.hf_config_class is None:
            path, rev = self._get_ref(None)
            config = AutoConfig.from_pretrained(
                path,
                revision=rev,
                trust_remote_code=self.trust_remote_code,
            )
            return type(config)
        elif isinstance(self.hf_config_class, str):
            path, rev = self._get_ref(None)
            HFConfig = get_class_from_dynamic_module(
                self.hf_config_class,
                path,
                revision=rev,
                local_files_only=not self.trust_remote_code,
            )
            return HFConfig
        else:
            return self.hf_config_class

    @cached_property
    def default_hf_config(self) -> HfConfig:
        return self.hf_config_from_hf_checkpoint(None)

    def HFAutoModelClass(self, auto_class: Type[AutoModel] = AutoModelForCausalLM) -> Type[AutoModel]:
        # figure out the
        config = self.hf_config_from_hf_checkpoint()
        cls_name = auto_class.__name__
        if hasattr(config, "auto_map") and cls_name in config.auto_map:
            class_ref = config.auto_map[cls_name]
            path, rev = self._get_ref(None)
            model_class = get_class_from_dynamic_module(
                class_ref,
                path,
                revision=rev,
                local_files_only=not self.trust_remote_code,
            )
            return model_class  # type: ignore
        elif type(config) in auto_class._model_mapping.keys():
            model_class = _get_model_class(config, auto_class._model_mapping)
            return model_class

        raise ValueError(f"Could not find model class {auto_class} for {config}")

    def load_tokenizer(self, ref: Optional[Union[str, RemoteRef]] = None) -> PreTrainedTokenizer:
        path, rev = self._get_ref(ref)
        tokenizer = AutoTokenizer.from_pretrained(path, revision=rev, trust_remote_code=self.trust_remote_code)
        return tokenizer

    @cached_property
    def default_Vocab(self) -> Axis:
        tokenizer = self.load_tokenizer()
        return Axis("vocab", len(tokenizer))

    def config_from_hf_config(self, hf_config) -> LevConfig:
        return self.LevConfigClass.from_hf_config(hf_config)

    def hf_config_from_config(self, config: LevConfig, vocab_size: Optional[int] = None) -> HfConfig:
        if vocab_size is None:
            vocab_size = len(self.load_tokenizer())
        return config.to_hf_config(vocab_size=vocab_size, config_overrides=self.config_overrides)

    def config_from_hf_checkpoint(self, ref: Optional[Union[str, RemoteRef]] = None) -> LevConfig:
        config = self.hf_config_from_hf_checkpoint(ref)
        return self.config_from_hf_config(config)

    def hf_config_from_hf_checkpoint(self, ref: Optional[Union[str, RemoteRef]] = None) -> HfConfig:
        path, rev = self._get_ref(ref)
        config = AutoConfig.from_pretrained(path, revision=rev, trust_remote_code=self.trust_remote_code)
        return config

    def _get_ref(self, ref) -> Tuple[str, Optional[str]]:
        if ref is None:
            if self.reference_checkpoint is None:
                raise ValueError("Must provide a reference checkpoint to load HFConfig from")
            ref = self.reference_checkpoint
        ref = _coerce_to_rr(ref)
        return ref.model_name_or_path, ref.revision

    def load_state_dict(self, ref: Optional[Union[str, RemoteRef]] = None):
        if ref is None:
            ref = self.reference_checkpoint
        if ref is None:
            raise ValueError("Must provide a checkpoint to load from")

        if os.path.exists(f"{ref}/{SAFE_TENSORS_MODEL}"):
            state_dict = safetensors.numpy.load_file(f"{ref}/{SAFE_TENSORS_MODEL}")
        elif os.path.exists(f"{ref}/{PYTORCH_MODEL}"):
            import torch

            device = torch.device("cpu")
            state_dict = torch.load(f"{ref}/{PYTORCH_MODEL}", map_location=device)
            state_dict = {k: v.cpu().numpy() for k, v in state_dict.items()}
        else:
            id, rev = self._get_ref(ref)
            try:
                model_path = hf_hub_download(id, SAFE_TENSORS_MODEL, revision=rev)
                state_dict = safetensors.numpy.load_file(model_path)
            except EntryNotFoundError:  # noqa: E722
                model_path = hf_hub_download(id, PYTORCH_MODEL, revision=rev)
                import torch

                device = torch.device("cpu")
                state_dict = torch.load(model_path, map_location=device)
                state_dict = {k: v.cpu().numpy() for k, v in state_dict.items()}

        return state_dict

    def load_lm_model(
        self,
        lm_model_cls: Type[LmWithHFSer],
        ref: Optional[Union[str, RemoteRef]] = None,
        Vocab: Optional[Axis] = None,
        axis_mapping: Optional[ResourceMapping] = None,
    ) -> LmWithHFSer:
        state_dict = self.load_state_dict(ref)
        config = self.config_from_hf_checkpoint(ref)

        Vocab = Vocab or self.default_Vocab

        ignore_prefix: Optional[str] = None
        if self.ignore_prefix:
            for k in state_dict.keys():
                if k.startswith(f"{self.ignore_prefix}."):
                    ignore_prefix = self.ignore_prefix
                    break

        # TODO: i still think this isn't the best way to do this
        with jax.default_device(jax.devices("cpu")[0]):
            lev_model = filter_eval_shape(lm_model_cls.init, Vocab, config, key=PRNGKey(0))
            lev_model = lev_model.from_state_dict(state_dict, prefix=ignore_prefix)

        if axis_mapping is not None:
            lev_model = haliax.shard_with_axis_mapping(lev_model, axis_mapping)

        return lev_model


def load_hf_model_checkpoint(location_or_id, device=None, revision=None):
    """
    Loads a safetensors or PyTorch model checkpoint.
    If model_file is None, this function attempts to load via safetensors first, then PyTorch.
    """
    local = os.path.exists(f"{location_or_id}/config.json")
    if local:
        config = json.load(open(f"{location_or_id}/config.json"))
    else:
        config_path = hf_hub_download(location_or_id, "config.json", revision=revision)
        config = json.load(open(config_path))

    if local:
        if os.path.exists(f"{location_or_id}/{SAFE_TENSORS_MODEL}"):
            checkpoint = safetensors.numpy.load_file(f"{location_or_id}/{SAFE_TENSORS_MODEL}")
        elif os.path.exists(f"{location_or_id}/{PYTORCH_MODEL}"):
            import torch

            if isinstance(device, str):
                device = torch.device(device)
            checkpoint = torch.load(f"{location_or_id}/{PYTORCH_MODEL}", map_location=device)
            checkpoint = {k: v.cpu().numpy() for k, v in checkpoint.items()}
        else:
            raise ValueError(f"Could not find model file for {location_or_id}")
    else:
        try:
            model_path = hf_hub_download(location_or_id, SAFE_TENSORS_MODEL, revision=revision)
            checkpoint = safetensors.numpy.load_file(model_path)
        except EntryNotFoundError:  # noqa: E722
            model_path = hf_hub_download(location_or_id, PYTORCH_MODEL, revision=revision)
            import torch

            if isinstance(device, str):
                device = torch.device(device)
            checkpoint = torch.load(model_path, map_location=device)
            checkpoint = {k: v.cpu().numpy() for k, v in checkpoint.items()}

    return config, checkpoint


def backpack_config_to_hf(vocab_size: int, config, auto_map_config: Optional[HFAutoMapConfig] = None) -> HfConfig:
    config = HfConfig(
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
        num_senses=config.num_senses,
        sense_intermediate_scale=config.sense_intermediate_scale,
    )
    if auto_map_config is not None:
        config.auto_map = auto_map_config.to_dict()
    return config


def hf_backpack_config_to_levanter(config: HfConfig):
    from levanter.models.backpack import BackpackConfig

    return BackpackConfig(
        seq_len=config.n_positions,
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
        num_senses=config.num_senses,
        sense_intermediate_scale=config.sense_intermediate_scale,
    )


def load_hf_gpt2_checkpoint(location_or_id, device=None, revision=None):
    config, checkpoint = load_hf_model_checkpoint(location_or_id, device=device, revision=revision)

    config = HfGpt2Config.from_dict(config)

    Vocab = Axis("vocab", config.vocab_size)
    from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel

    lev_config = Gpt2Config.from_hf_config(config)
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


def _save_hf_gpt2_checkpoint_local(model, path):
    os.makedirs(path, exist_ok=True)
    config = model.config.to_hf_config(model.vocab_size)
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


def _save_backpack_hf_checkpoint_local(
    model,
    path: str,
    model_type: Optional[str] = None,
    auto_map_config: Optional[HFAutoMapConfig] = None,
):
    # Extract and save the model configuration
    to_hf_config_func = backpack_config_to_hf
    os.makedirs(path, exist_ok=True)
    config = to_hf_config_func(model.vocab_size, model.config, auto_map_config)
    config = config.to_dict()
    if model_type is not None:
        config["model_type"] = model_type

    with open(f"{path}/config.json", "w") as f:
        json.dump(config, f)

    # need to make sure the model is on *this machine* and *this machine's CPU* before saving
    model = jax.tree_map(
        lambda arr: np.array(jax.device_get(multihost_utils.process_allgather(arr, tiled=True))), model
    )

    # TODO: it's be nice if safetensors supported an iterator or something so we could do the allgather one at a time
    state_dict = model.to_state_dict()

    # now that we've moved the model to the CPU, we don't need to do this on all processes
    if jax.process_index() != 0:
        return

    # the "pt" is a lie but it doesn't seem to actually matter and HF demands it
    safetensors.numpy.save_file(state_dict, f"{path}/{SAFE_TENSORS_MODEL}", metadata={"format": "pt"})
    print(f"Saved checkpoint to {path}")


def _is_url_like(path):
    return urllib.parse.urlparse(path).scheme != ""


def save_hf_gpt2_checkpoint(model, path, hf_repo: Optional[str] = None, **hf_upload_kwargs):
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
        from levanter.models.gpt2 import Gpt2LMHeadModel

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
        from levanter.models.gpt2 import Gpt2LMHeadModel

        save_hf_gpt2_checkpoint(
            cast(Gpt2LMHeadModel, step.model), f"{base_path}/step-{step.step}", hf_repo=hf_repo, **my_upload_kwargs
        )

    return cb
