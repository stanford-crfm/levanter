import abc
import dataclasses
import gc
import json
import logging
import os
import shutil
import tempfile
import urllib.parse
import warnings
from dataclasses import dataclass
from functools import cached_property
from typing import Generic, Optional, Tuple, Type, TypeVar, Union, cast
from urllib.parse import urlparse

import draccus
import equinox as eqx
import fsspec
import huggingface_hub
import jax
import mergedeep
import safetensors
import safetensors.numpy
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import EntryNotFoundError, GatedRepoError, HFValidationError
from jax.experimental.multihost_utils import sync_global_devices
from jax.random import PRNGKey
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers import PretrainedConfig as HfConfig
from transformers import PreTrainedTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.auto.auto_factory import _get_model_class

import haliax
from haliax import Axis
from haliax.partitioning import ResourceMapping

from levanter.compat.torch_serialization import StateDictSerializationMixin, save_state_dict, to_numpy_state_dict
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.trainer import StepInfo
from levanter.utils.cloud_utils import temp_dir_before_upload
from levanter.utils.jax_utils import use_cpu_device
from levanter.utils.py_utils import classproperty, dataclass_with_default_init


logger = logging.getLogger(__name__)


PYTORCH_MODEL = "pytorch_model.bin"
SAFE_TENSORS_MODEL = "model.safetensors"
PYTORCH_WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
SAFE_TENSORS_INDEX_NAME = "model.safetensors.index.json"


@dataclass(frozen=True)
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
        if self.revision is None:
            return self.model_name_or_path
        return f"{self.model_name_or_path}@{self.revision}"

    def __repr__(self) -> str:
        return f"RemoteRev({self.model_name_or_path!r}, {self.revision!r})"


# register draccus parsing
draccus.decode.register(RepoRef, RepoRef.from_string)
draccus.encode.register(RepoRef, str)


class HFCompatConfig(LmConfig["LmWithHfSerializationMixin"]):
    @abc.abstractmethod
    def to_hf_config(self, vocab_size: int, config_overrides: Optional[dict] = None) -> HfConfig:
        pass

    @classmethod
    @abc.abstractmethod
    def from_hf_config(cls, hf_config: HfConfig):
        pass

    @classproperty
    @abc.abstractmethod
    def default_hf_checkpoint_converter(cls) -> "HFCheckpointConverter":
        """The default HFCheckpointConverter to use for this config class. We recommend that you
        define this as a @cached_property on your config class."""
        pass


MConfig = TypeVar("MConfig", bound=HFCompatConfig)


class LmWithHfSerializationMixin(LmHeadModel, Generic[MConfig], StateDictSerializationMixin):
    def get_hf_config(self):
        return self.config.to_hf_config(self.Vocab.size)

    @property
    @abc.abstractmethod
    def Vocab(self) -> Axis:
        pass

    @classmethod
    @abc.abstractmethod
    def init(cls, Vocab: Axis, config: MConfig, *, key: PRNGKey) -> "LmWithHfSerializationMixin":
        pass


def _coerce_to_rr(s: Union[str, RepoRef]) -> RepoRef:
    if isinstance(s, RepoRef):
        return s
    else:
        return RepoRef.from_string(s)


LevConfig = TypeVar("LevConfig", bound=HFCompatConfig)

# just for generating unique ids

KEYS_TO_COPY_FROM_BASE_CONFIG = {
    "architectures",
    "auto_map",
}


@dataclass_with_default_init(frozen=True)
class HFCheckpointConverter(Generic[LevConfig]):
    """
    A class to convert between Levanter and HF models. This class establishes a bidirectional mapping
    between Levanter and HF models, and provides methods to convert between configs and loading/saving HF checkpoints.
    It also handles the bundled tokenizer and code, as applicable.

    This mapping supports:
    * translating between Levanter and HF configs
    * loading a Levanter model from an HF checkpoint
    * saving a HF checkpoint from a Levanter model

    HF checkpoints can be saved with params and config, and optionally with the tokenizer and code.
    """

    LevConfigClass: Type[LevConfig]
    reference_checkpoint: Optional[RepoRef]
    "A reference HF Hub checkpoint to extract non-parameter files (like model code an config from)"

    HfConfigClass: Type
    "The HFConfig class to use. If None is provided, will be inferred from the reference_checkpoint"

    tokenizer: PreTrainedTokenizerFast | PreTrainedTokenizer
    "The tokenizer to use. If None, will be inferred from the reference_checkpoint"

    config_overrides: Optional[dict] = None
    "A dictionary of config overrides to apply to the HFConfig when saving. typically used for auto_map"

    trust_remote_code: bool = False
    "If True, will trust the remote code and not download it locally."

    ignore_prefix: Optional[str] = None
    """A prefix to optionally ignore when loading checkpoints. For "gpt2" this is 'transformer' to deal with the
    fact that some models are saved as XXXPreTrainedModel and others are saved as XXXLMHeadModel"""

    def __init__(
        self,
        LevConfigClass: Type[LevConfig],
        reference_checkpoint: Optional[Union[RepoRef, str]] = None,
        HfConfigClass: Optional[Union[str, Type]] = None,
        tokenizer: Optional[Union[str, PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
        config_overrides: Optional[dict] = None,
        trust_remote_code: bool = False,
        ignore_prefix: Optional[str] = None,
    ):
        # stupid python won't let you have a custom constructor with a frozen dataclass

        ref = _coerce_to_rr(reference_checkpoint) if reference_checkpoint is not None else None
        HfConfigClass = HFCheckpointConverter._infer_config_class(HfConfigClass, ref, trust_remote_code)
        tokenizer = HFCheckpointConverter._infer_tokenizer(tokenizer, ref, trust_remote_code)

        self.__default_init__(  # type: ignore
            LevConfigClass=LevConfigClass,
            reference_checkpoint=ref,
            HfConfigClass=HfConfigClass,
            tokenizer=tokenizer,
            config_overrides=config_overrides,
            trust_remote_code=trust_remote_code,
            ignore_prefix=ignore_prefix,
        )

    @staticmethod
    def from_hf(model_name_or_path: Union[RepoRef, str], trust_remote_code: bool = False) -> "HFCheckpointConverter":
        ref = _coerce_to_rr(model_name_or_path)
        config_class = HFCheckpointConverter._infer_config_class(None, ref, trust_remote_code)
        tokenizer = HFCheckpointConverter._infer_tokenizer(None, ref, trust_remote_code)

        # TODO: this is very hacky, we should add another registry or something
        # attempt to find the Levanter config class by checking the registry
        # TODO: hacky hacky
        for k, v in LmConfig.get_known_choices().items():
            if issubclass(v, HFCompatConfig):
                if v.default_hf_checkpoint_converter.HfConfigClass.__name__ == config_class.__name__:
                    LevConfigClass = v
                    break
        else:
            raise ValueError(f"No Levanter config found for {config_class}")

        return HFCheckpointConverter(
            LevConfigClass=LevConfigClass,
            reference_checkpoint=ref,
            HfConfigClass=config_class,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
        )

    def replaced(
        self,
        reference_checkpoint: Optional[Union[RepoRef, str]] = None,
        tokenizer: Optional[Union[str, PreTrainedTokenizerBase]] = None,
        trust_remote_code: Optional[bool] = None,
    ) -> "HFCheckpointConverter":
        replacements: dict = {}
        if reference_checkpoint is not None:
            replacements["reference_checkpoint"] = reference_checkpoint
        if tokenizer is not None:
            replacements["tokenizer"] = tokenizer
        if trust_remote_code is not None:
            replacements["trust_remote_code"] = trust_remote_code

        return dataclasses.replace(self, **replacements)  # type: ignore

    def with_config_overrides(self, config_overrides: dict, merge: bool = True) -> "HFCheckpointConverter":
        if self.config_overrides is not None and merge:
            config_overrides = mergedeep.merge({}, self.config_overrides, config_overrides)
        return dataclasses.replace(self, config_overrides=config_overrides)  # type: ignore

    @staticmethod
    def _infer_config_class(hf_config_class, ref, trust_remote_code):
        if hf_config_class is None:
            if ref is None:
                raise ValueError("Must provide either config class or reference_checkpoint")
            path, rev = ref.model_name_or_path, ref.revision
            config = AutoConfig.from_pretrained(
                path,
                revision=rev,
                trust_remote_code=trust_remote_code,
            )
            clss = type(config)
        elif isinstance(hf_config_class, str):
            if ref is None:
                raise ValueError("Must provide either config class or reference_checkpoint")
            path, rev = ref.model_name_or_path, ref.revision
            HFConfig = get_class_from_dynamic_module(
                hf_config_class,
                path,
                revision=rev,
                local_files_only=not trust_remote_code,
            )
            clss = HFConfig
        else:
            clss = hf_config_class
        return clss

    @staticmethod
    def _infer_tokenizer(
        tokenizer, ref, trust_remote_code: bool = False
    ) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        if tokenizer is None:
            if ref is None:
                raise ValueError("Must provide either tokenizer or reference_checkpoint")
            tokenizer = ref

        if isinstance(tokenizer, (str, RepoRef)):
            ref = _coerce_to_rr(tokenizer)
            path, rev = ref.model_name_or_path, ref.revision
            tokenizer = load_tokenizer(
                path,
                revision=rev,
                trust_remote_code=trust_remote_code,
            )
        else:
            pass

        assert isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast))

        return tokenizer

    @cached_property
    def default_hf_config(self) -> HfConfig:
        return self.hf_config_from_hf_checkpoint(None)

    @cached_property
    def default_config(self) -> LevConfig:
        return self.config_from_hf_config(self.default_hf_config)

    def HFAutoModelClass(self, auto_class: Type[AutoModel] = AutoModelForCausalLM) -> Type[AutoModel]:
        # first, see if it's a built-in model
        try:
            return auto_class._model_mapping[self.HfConfigClass]
        except KeyError:
            pass

        config = self.default_hf_config
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

    @cached_property
    def Vocab(self) -> Axis:
        return Axis("vocab", len(self.tokenizer))

    def config_from_hf_config(self, hf_config, overrides: Optional[dict] = None) -> LevConfig:
        config = self.LevConfigClass.from_hf_config(hf_config)
        if overrides is not None:
            config = dataclasses.replace(config, **overrides)  # type: ignore
        return config

    def hf_config_from_config(self, config: LevConfig, vocab_size: Optional[int] = None) -> HfConfig:
        if vocab_size is None:
            vocab_size = self.Vocab.size
        return config.to_hf_config(vocab_size=vocab_size)

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

        id, rev = self._get_ref(ref)

        for index_file in [SAFE_TENSORS_INDEX_NAME, PYTORCH_WEIGHTS_INDEX_NAME]:
            try:
                return self._load_shards(id, index_file, rev)
            except EntryNotFoundError:
                pass
            except HFValidationError:
                pass

        if os.path.exists(os.path.join(id, SAFE_TENSORS_MODEL)):
            state_dict = safetensors.numpy.load_file(os.path.join(id, SAFE_TENSORS_MODEL))
        elif os.path.exists(os.path.join(id, PYTORCH_MODEL)):
            import torch

            device = torch.device("cpu")
            state_dict = torch.load(os.path.join(id, PYTORCH_MODEL), map_location=device)
            state_dict = {k: _convert_to_jnp(v) for k, v in state_dict.items()}
        else:
            try:
                model_path = hf_hub_download(id, SAFE_TENSORS_MODEL, revision=rev)
                state_dict = safetensors.numpy.load_file(model_path)
            except (EntryNotFoundError, HFValidationError):
                model_path = hf_hub_download(id, PYTORCH_MODEL, revision=rev)
                import torch

                device = torch.device("cpu")
                state_dict = torch.load(model_path, map_location=device)
                return {k: _convert_to_jnp(v) for k, v in state_dict.items()}

        return state_dict

    def _load_shards(self, id: str, index_file: str, rev: Optional[str]) -> dict:
        """Load model from sharded files based on the provided index."""
        index_path = os.path.join(id, index_file)
        if not os.path.exists(index_path):
            # Download the index file if not found locally
            index_path = hf_hub_download(id, index_file, revision=rev)

        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        shard_files = list(set(index["weight_map"].values()))
        final_state_dict = {}

        if "safetensors" in index_file:
            import safetensors

            loader = safetensors.numpy.load_file
        else:
            import torch

            def loader(path):
                device = torch.device("cpu")
                state_dict = torch.load(path, map_location=device)
                return {k: _convert_to_jnp(v) for k, v in state_dict.items()}

        for shard_file in shard_files:
            shard_path = os.path.join(id, shard_file)
            if not os.path.exists(shard_path):
                # Download the shard if not found locally
                shard_path = hf_hub_download(id, shard_file, revision=rev)

            state_dict = loader(shard_path)
            final_state_dict.update(state_dict)

            del state_dict

        return final_state_dict

    def load_pretrained(
        self,
        lm_model_cls: Union[Type[LmWithHfSerializationMixin], LevConfig],
        ref: Optional[Union[str, RepoRef]] = None,
        axis_mapping: Optional[ResourceMapping] = None,
        resize_vocab_to_match_tokenizer: bool = True,
    ) -> LmWithHfSerializationMixin:
        """
        Loads a levanter model from a huggingface checkpoint.

        Args:
            lm_model_cls: The model class to load or the config to use to load the model class
            ref: The reference to load from. If None, will use the reference_checkpoint
            axis_mapping: The axis mapping to use for sharding. If None, will use the context axis mapping
        """
        from contextlib import nullcontext

        if axis_mapping is None:
            axis_mapping_cm = nullcontext()
        else:
            axis_mapping_cm = haliax.axis_mapping(axis_mapping)
        with use_cpu_device(), axis_mapping_cm:
            # TODO: in an ideal world, we would only load the part of the array we needed, but
            # AFAICT neither torch state dicts nor safetensors support this.
            state_dict = self.load_state_dict(ref)

            hf_config = self.hf_config_from_hf_checkpoint(ref)

            if isinstance(lm_model_cls, type(self.default_config)):
                config = lm_model_cls
                lm_model_cls = config.model_type
            else:
                config = self.config_from_hf_config(hf_config)

            # Vocab: first we have to resize the vocab as loaded from the checkpoint
            tokenizer_Vocab = self.Vocab
            Vocab = tokenizer_Vocab.resize(hf_config.vocab_size)

            ignore_prefix: Optional[str] = None
            if self.ignore_prefix:
                for k in state_dict.keys():
                    if k.startswith(f"{self.ignore_prefix}."):
                        ignore_prefix = self.ignore_prefix
                        break

            # TODO: this could be simpler if we just started using a "persistent" or "buffer" thing
            # TODO: the strategy is a bit too clever here.
            # we first evaluate the shape of our model, then use from_state_dict to actually populate the model
            # with the arrays.
            lev_model = eqx.filter_eval_shape(lm_model_cls.init, Vocab, config, key=PRNGKey(0))
            lev_model = lev_model.from_state_dict(state_dict, prefix=ignore_prefix)
            del state_dict
            gc.collect()  # sometimes takes a while to free buffers otherwise

            # However, this might miss some buffers that don't get persisted in the state dict
            # (e.g. pytorch buffers with persistent=false), so we have to reinitialize them. We then init the model
            # again, this time keeping only the (missing) buffers, and then combine the two models.
            lev_model = _patch_missing_buffers_for_deser(
                lev_model, lm_model_cls, Vocab, config, PRNGKey(0), axis_mapping
            )

            if Vocab.size != tokenizer_Vocab.size:
                if resize_vocab_to_match_tokenizer:
                    logger.info(
                        f"Resizing model from {Vocab.size} to {tokenizer_Vocab.size} to match tokenizer vocab size"
                    )
                    # run in jit b/c we're manipulating sharded tensors
                    lev_model = haliax.named_jit(
                        lambda m: m.resize_vocab(tokenizer_Vocab.size), axis_mapping, donate_args=(True,)
                    )(lev_model)
                else:
                    logger.warning(
                        f"Model vocab size ({Vocab.size}) does not match tokenizer vocab size ({tokenizer_Vocab.size})"
                    )

        if axis_mapping is not None:
            lev_model = haliax.shard_with_axis_mapping(lev_model, axis_mapping)
        else:
            lev_model = haliax.auto_sharded(lev_model)

        # once more for good measure
        gc.collect()

        return lev_model

    def _save_pretrained_local(
        self,
        model: LmWithHfSerializationMixin,
        path: str,
        save_tokenizer: bool = True,
        save_reference_code: Optional[bool] = None,
    ):
        """
        Saves a HF-compatible checkpoint to a local path.
        :param model: The model to convert and save
        :param path: The path to save the output to
        :param save_tokenizer: Save the tokenizer to the checkpoint
        :param save_reference_code: Save any code from the reference checkpoint
        :return:
        """
        logger.info(f"Saving HF-compatible checkpoint to {path}")
        os.makedirs(path, exist_ok=True)

        # if save_reference_code is None, we save code for models that aren't in the HF repo.
        if save_reference_code is None:
            #  the way we determine this is if the config class is in the HF package or not
            save_reference_code = not self.HfConfigClass.__module__.startswith("transformers.")

        # save code first because we'll likely be overwriting it
        if save_reference_code:
            logger.info(f"Copying reference code from {self.reference_checkpoint}")
            self._save_code_local(path)

        if save_tokenizer:
            logger.info("Saving tokenizer")
            self.tokenizer.save_pretrained(path)

        # Config
        config = model.config.to_hf_config(model.Vocab.size)
        dict_config = config.to_dict()

        # copy over the default keys
        try:
            for k in KEYS_TO_COPY_FROM_BASE_CONFIG:
                attr = getattr(self.default_hf_config, k, None)
                if attr is not None:
                    dict_config[k] = attr
        # except GatedRepoError:
        #     warnings.warn("Could not copy keys from base config because the repo is gated. Making assumptions.")
        except Exception as e:  # noqa
            if isinstance(e, GatedRepoError) or isinstance(e.__cause__, GatedRepoError):
                warnings.warn("Could not copy keys from base config because the repo is gated. Making assumptions.")

                # this is probably llama, but in general we just need to set the auto_map and architectures
                dict_config["auto_map"] = {
                    "AutoModelForCausalLM": self.HFAutoModelClass(AutoModelForCausalLM).__qualname__,
                    "AutoConfig": self.HfConfigClass.__qualname__,
                }

                dict_config["architectures"] = [self.HFAutoModelClass(AutoModelForCausalLM).__name__]
            else:
                raise

        if self.config_overrides:
            dict_config = mergedeep.merge({}, dict_config, self.config_overrides)

        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(dict_config, f)

        # Model
        state_dict = to_numpy_state_dict(model)
        save_state_dict(state_dict, os.path.join(path, SAFE_TENSORS_MODEL))
        logger.info(f"Finished saving HF-compatible checkpoint to {path}")

    def save_pretrained(
        self,
        model: LmWithHfSerializationMixin,
        path,
        upload_to_hf: Union[bool, str, RepoRef] = False,
        save_reference_code: Optional[bool] = None,
        save_tokenizer: bool = True,
        **hf_upload_kwargs,
    ):
        """
        Saves a Levanter model to a huggingface "pretrained model" checkpoint.

        If hf_repo is provided, this will upload the checkpoint to the huggingface hub, passing
        any additional kwargs to the huggingface_hub.upload_folder function.

        :param path: the path to save the checkpoint to. path may be a GCS bucket path or other fsspec path,
        in which case the checkpoint will be uploaded to GCS after being written to a tmpdir
        :param model: the model to save
        :param hf_repo: if provided, the checkpoint will be uploaded to the huggingface hub. If True, will use
        the reference_checkpoint as the repo name
        :param hf_upload_kwargs: any additional kwargs to pass to huggingface_hub.upload_folder
        :param save_reference_code: if True, will save the reference code (from reference_checkpoint) to the checkpoint.
        This is useful when using custom architectures, as it will allow the model to be loaded without the custom
        architecture code being present (using trust_remote_code=True). "Code" here means anything not stored in LFS.
        If None, will save code for models that aren't in the HF repo.
        """
        with temp_dir_before_upload(path) as local_path:
            if path != local_path:
                logger.info(f"Saving model to {path} via temp path {local_path}")

            self._save_pretrained_local(
                model, local_path, save_reference_code=save_reference_code, save_tokenizer=save_tokenizer
            )

            if upload_to_hf is True:
                if self.reference_checkpoint is None:
                    raise ValueError("No reference checkpoint provided, so no repo name to upload to")
                upload_to_hf = self.reference_checkpoint
            if not isinstance(upload_to_hf, bool):
                assert isinstance(upload_to_hf, (str, RepoRef))
                upload_to_hub(local_path, upload_to_hf, **hf_upload_kwargs)

    def _save_code_local(self, path):
        if self.reference_checkpoint is None:
            warnings.warn("No reference checkpoint provided, so no code will be saved")
            return

        repo, revision = self._get_ref(self.reference_checkpoint)

        # first we're going to decide what code to save
        # as a heuristic, we'll use .gitattributes to decide what to save: anything not in LFS will be saved
        # need to also save the .gitattributes file itself
        # TODO: .gitignore too? it's not used a lot with the hub
        if os.path.exists(repo):
            # local path
            if revision is not None:
                warnings.warn("Ignoring revision because this is a local path. We don't handle this case well yet")
            attributes_path = os.path.join(repo, ".gitattributes")
            if not os.path.exists(attributes_path):
                attributes_path = None
        else:
            # check hub
            try:
                attributes_path = hf_hub_download(repo_id=repo, filename=".gitattributes", revision=revision)
            except EntryNotFoundError:
                attributes_path = None
            except GatedRepoError:
                attributes_path = None

        if attributes_path is None:
            warnings.warn("HF Export - No .gitattributes file found, using a heuristic to decide what to save")
            ignore_files = [
                ".git",
                "*.bin.*",
                "*.lfs.*",
                "*.bin",
                "*.h5",
                "*.tflite",
                "*.tar.gz",
                "*.ot",
                "*.onnx",
                "*.msgpack",
                "model.safetensors",
            ]
        else:
            # read the attributes file and get the globs
            with open(attributes_path) as f:
                attributes = f.read()
            ignore_files = [".git"]
            for line in attributes.split("\n"):
                line = line.strip()
                if line.startswith("#") or line == "":
                    continue
                # NB: this is not a full implementation of .gitattributes, but it's good enough for our purposes
                if "filter=lfs" in line:
                    ignore_files.append(line.split()[0])

        if os.path.exists(repo):
            local_code_path = repo
        else:
            local_code_path = snapshot_download(repo, revision=revision, ignore_patterns=ignore_files)

        # now we'll save the code
        os.makedirs(path, exist_ok=True)

        shutil_ignore = shutil.ignore_patterns(*ignore_files)
        shutil.copytree(local_code_path, path, ignore=shutil_ignore, dirs_exist_ok=True)

        logger.debug(f"Saved code to {path}")


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
        converter.save_pretrained(
            cast(LmWithHfSerializationMixin, step.model),
            os.path.join(base_path, f"step-{step.step}"),
            upload_to_hf=upload_to_hf,
            **my_upload_kwargs,
        )

    return cb


def load_tokenizer(model_name_or_path, revision=None, local_cache_dir=None, trust_remote_code=True):
    """Like AutoTokenizer.from_pretrained, but works with gs:// paths or anything on fsspec"""
    is_url_like = urlparse(model_name_or_path).scheme != ""
    if is_url_like:
        if revision is not None:
            raise ValueError("revision is not supported for URLs")
        # tokenizers are directories, so we have to copy them locally
        if local_cache_dir is None:
            local_cache_dir = tempfile.mkdtemp()

        fs, path = fsspec.core.url_to_fs(model_name_or_path)
        fs.get(path, local_cache_dir, recursive=True)
        base_path = os.path.basename(path)
        return AutoTokenizer.from_pretrained(
            os.path.join(local_cache_dir, base_path), trust_remote_code=trust_remote_code
        )
    else:
        return AutoTokenizer.from_pretrained(
            model_name_or_path, revision=revision, trust_remote_code=trust_remote_code
        )


_sync_count = 0


def upload_to_hub(local_path: str, repo_ref: Union[str, RepoRef], **hf_upload_kwargs):
    ref = _coerce_to_rr(repo_ref)

    if jax.process_index() == 0:
        logger.info(f"Uploading HF-compatible checkpoint to {ref.model_name_or_path}")
        huggingface_hub.upload_folder(
            folder_path=local_path, repo_id=(ref.model_name_or_path), revision=(ref.revision), **hf_upload_kwargs
        )
        logger.info(f"Finished uploading HF-compatible checkpoint to {ref.model_name_or_path}")
    else:
        logger.info(f"Finished waiting for rank 0 to upload checkpoint to {ref.model_name_or_path}")

    global _sync_count
    sync_global_devices(f"upload? {ref.model_name_or_path}{ref.revision} {_sync_count}")
    _sync_count += 1


def _convert_to_jnp(v):
    import torch

    # we'd rather not convert to float32 to conserve memory, so we convert direct to jax.numpy
    # if v.dtype == torch.bfloat16:
    #     v = v.to(torch.float32)
    if v is None:
        return None
    elif v.dtype == torch.bfloat16:
        return jax.numpy.array(v.cpu().view(torch.float16).numpy()).view(jax.numpy.bfloat16)
    else:
        return jax.numpy.array(v.cpu().numpy())


def _patch_missing_buffers_for_deser(lev_model, lm_model_cls, Vocab, config, key, axis_mapping):
    """
    State dict serialization doesn't always save buffers, so when we initialize a model from a state dict, we might
    need to re-initialize the buffers. We do this by rerunning the constructor, but only save the buffers.
    JAX jit will avoid actually doing the computation.
    """
    dtype_structs, real_arrays = eqx.partition(lev_model, lambda x: isinstance(x, jax.ShapeDtypeStruct))

    buffer_leaves = jax.tree_util.tree_leaves(dtype_structs)
    if len(buffer_leaves) == 0:
        return lev_model

    # ok, we now want to re-initialize the buffers by running the constructor for real, getting only the missing
    # arrays. We're relying on jit to do all the work in eliminating the unnecessary computation/memory
    @haliax.named_jit(axis_resources=axis_mapping)
    def _init_buffers():
        new_model = lm_model_cls.init(Vocab, config, key=key)

        def select_if_missing(missing_leaf, new_value):
            if isinstance(missing_leaf, jax.ShapeDtypeStruct):
                return new_value
            else:
                return None

        return jax.tree_map(select_if_missing, dtype_structs, new_model, is_leaf=lambda x: x is None)

    new_buffers = _init_buffers()

    result = eqx.combine(real_arrays, new_buffers)
    return result
