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
import pyrallis
import safetensors
import safetensors.numpy
from fsspec import AbstractFileSystem
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from jax.experimental import multihost_utils
from jax.experimental.multihost_utils import sync_global_devices
from jax.random import PRNGKey
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers import PretrainedConfig as HfConfig
from transformers import PreTrainedTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.auto.auto_factory import _get_model_class

import haliax
from haliax import Axis
from haliax.jax_utils import filter_eval_shape
from haliax.partitioning import ResourceMapping
from levanter.compat.torch_serialization import StateDictSerializationMixin
from levanter.trainer_hooks import StepInfo


logger = logging.getLogger(__name__)


PYTORCH_MODEL = "pytorch_model.bin"
SAFE_TENSORS_MODEL = "model.safetensors"


@dataclass
class RepoRef:
    """Represents a reference to a model (or similar) in a remote repository or local file system, and
    optionally a revision. This lets you load particular revisions or branches of a model.

    The string syntax is <model_name_or_path>@<revision>.

    """

    model_name_or_path: str
    revision: Optional[str] = None

    @staticmethod
    def from_string(s: str) -> "RepoRef":
        if "@" not in s:
            return RepoRef(s)
        model_name_or_path, revision = s.split("@")
        return RepoRef(model_name_or_path, revision)

    def __str__(self) -> str:
        return f"{self.model_name_or_path}@{self.revision}"

    def __repr__(self) -> str:
        return f"RemoteRev({self.model_name_or_path!r}, {self.revision!r})"


# register pyrallis parsing
pyrallis.decode.register(RepoRef, RepoRef.from_string)
pyrallis.encode.register(RepoRef, str)


class ConfigWithHFSer(abc.ABC):
    @abc.abstractmethod
    def to_hf_config(self, vocab_size: int, config_overrides: Optional[dict] = None) -> HfConfig:
        pass

    @classmethod
    @abc.abstractmethod
    def from_hf_config(cls, hf_config: HfConfig):
        pass


MConfig = TypeVar("MConfig", bound=ConfigWithHFSer)


class LmWithHFSer(abc.ABC, Generic[MConfig], StateDictSerializationMixin):
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


def _coerce_to_rr(s: Union[str, RepoRef]) -> RepoRef:
    if isinstance(s, RepoRef):
        return s
    else:
        return RepoRef.from_string(s)


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

# just for generating unique ids
_GLOBAL_SAVE_COUNT = 0


@dataclass
class HFCheckpointConverter(Generic[LevConfig]):
    LevConfigClass: Type[LevConfig]
    reference_checkpoint: Optional[Union[str, RepoRef]] = None
    "A reference HF Hub checkpoint to extract non-parameter files (like model code an config from)"

    hf_config_class: Optional[Union[str, Type]] = None
    "The HFConfig class to use. If None, will be inferred from the reference_checkpoint"

    config_overrides: Optional[dict] = None
    "A dictionary of config overrides to apply to the HFConfig when saving. typically used for auto_map"

    trust_remote_code: bool = False
    "If True, will trust the remote code and not download it locally."

    ignore_prefix: Optional[str] = None
    """A prefix to optionally ignore when loading checkpoints. For "gpt2" this is 'transformer' to deal with the
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

    def load_tokenizer(self, ref: Optional[Union[str, RepoRef]] = None) -> PreTrainedTokenizer:
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

    def config_from_hf_checkpoint(self, ref: Optional[Union[str, RepoRef]] = None) -> LevConfig:
        config = self.hf_config_from_hf_checkpoint(ref)
        return self.config_from_hf_config(config)

    def hf_config_from_hf_checkpoint(self, ref: Optional[Union[str, RepoRef]] = None) -> HfConfig:
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

    def load_state_dict(self, ref: Optional[Union[str, RepoRef]] = None):
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
        ref: Optional[Union[str, RepoRef]] = None,
        axis_mapping: Optional[ResourceMapping] = None,
    ) -> LmWithHFSer:
        """
        Loads a levanter model from a huggingface checkpoint.

        Args:
            lm_model_cls: The model class to load
            ref: The reference to load from. If None, will use the reference_checkpoint
            axis_mapping: The axis mapping to use for sharding. If None, will use the context axis mapping
        """
        state_dict = self.load_state_dict(ref)
        hf_config = self.hf_config_from_hf_checkpoint(ref)
        config = self.config_from_hf_config(hf_config)

        vocab_size = hf_config.vocab_size

        Vocab = self.default_Vocab.resize(vocab_size)

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
        else:
            lev_model = haliax.auto_sharded(lev_model)

        return lev_model

    def save_model_local(self, model: LmWithHFSer, path: str):
        logger.info(f"Saving HF-compatible checkpoint to {path}")
        os.makedirs(path, exist_ok=True)
        config = model.config.to_hf_config(model.Vocab.size)
        dict_config = config.to_dict()

        if self.config_overrides:
            dict_config.update(self.config_overrides)

        with open(f"{path}/config.json", "w") as f:
            json.dump(dict_config, f)

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

        global _GLOBAL_SAVE_COUNT
        sync_global_devices(f"local {_GLOBAL_SAVE_COUNT}")
        _GLOBAL_SAVE_COUNT += 1
        logger.info(f"Finished saving HF-compatible checkpoint to {path}")

    def save_model(
        self, model: LmWithHFSer, path, upload_to_hf: Union[bool, str, RepoRef] = False, **hf_upload_kwargs
    ):
        """
        If hf_repo is provided, this will upload the checkpoint to the huggingface hub, passing
        any additional kwargs to the huggingface_hub.upload_folder function.

        :param path: the path to save the checkpoint to. path may be a GCS bucket path or other fsspec path,
        in which case the checkpoint will be uploaded to GCS after being written to a tmpdir
        :param model: the model to save
        :param hf_repo: if provided, the checkpoint will be uploaded to the huggingface hub. If True, will use
        the reference_checkpoint as the repo name
        :param hf_upload_kwargs: any additional kwargs to pass to huggingface_hub.upload_folder
        :return:
        """
        tmpdir: Optional[str] = None
        if _is_url_like(path):
            tmpdir = tempfile.mkdtemp()
            logger.info(f"Saving model to {path} via temp path {tmpdir}")
            local_path = tmpdir
        else:
            local_path = path

        if isinstance(upload_to_hf, (str, RepoRef)):
            hf_repo, hf_branch = self._get_ref(upload_to_hf)
        elif upload_to_hf is True:
            hf_repo, hf_branch = self._get_ref(self.reference_checkpoint)
        else:
            hf_repo = None

        self.save_model_local(model, local_path)

        try:
            if jax.process_index() == 0:
                if tmpdir is not None:  # we're uploading to GCS or similar
                    logger.info(f"Copying HF-compatible checkpoint to {path}")
                    fs: AbstractFileSystem = fsspec.core.get_fs_token_paths(path, mode="wb")[0]
                    fs.put(os.path.join(local_path, "*"), path, recursive=True)
                    logger.info(f"Finished copying HF-compatible checkpoint to {path}")

                if hf_repo is not None:
                    logger.info(f"Uploading HF-compatible checkpoint to {hf_repo}")
                    huggingface_hub.upload_folder(local_path, hf_repo, **hf_upload_kwargs)
                    logger.info(f"Finished uploading HF-compatible checkpoint to {hf_repo}")
            else:
                logger.info(f"Waiting for process 0 to finish saving checkpoint to {path}")

            global _GLOBAL_SAVE_COUNT
            sync_global_devices(f"upload? {path}{_GLOBAL_SAVE_COUNT}")
            _GLOBAL_SAVE_COUNT += 1
        finally:
            if tmpdir is not None:
                shutil.rmtree(tmpdir)


def _save_backpack_hf_checkpoint_local(
    model,
    path: str,
    model_type: Optional[str] = None,
    auto_map_config: Optional[HFAutoMapConfig] = None,
):
    # Extract and save the model configuration
    os.makedirs(path, exist_ok=True)
    from levanter.models.backpack import BackpackConfig

    to_hf_config_func = BackpackConfig.to_hf_config
    config = to_hf_config_func(model.vocab_size, model.config, {"auto_map": auto_map_config})
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


def save_hf_checkpoint_callback(
    base_path,
    converter: HFCheckpointConverter,
    upload_to_hf: Union[bool, str, RepoRef] = False,
    **hf_upload_kwargs,
):
    """
    If hf_repo is provided, this will upload the checkpoint to the huggingface hub, passing
    any additional kwargs to the huggingface_hub.upload_folder function.

    :param base_path: the base path to save the checkpoint to. `/step-<step>` will be appended to this. base_path
    may be a GCS bucket path, in which case the checkpoint will be uploaded to GCS after being written to a tmp
    :param upload_to_hf:
    :param hf_upload_kwargs:
    :return:
    """

    def cb(step: StepInfo):
        nonlocal hf_upload_kwargs
        if step.step == 0:
            return
        if upload_to_hf is not None and "commit_message" not in hf_upload_kwargs:
            my_upload_kwargs = hf_upload_kwargs.copy()
            my_upload_kwargs["commit_message"] = f"Upload for step {step.step} from Levanter"
        else:
            my_upload_kwargs = hf_upload_kwargs
        converter.save_model(
            cast(LmWithHFSer, step.model),
            f"{base_path}/step-{step.step}",
            upload_to_hf=upload_to_hf,
            **my_upload_kwargs,
        )

    return cb
