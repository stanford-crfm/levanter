import dataclasses
from dataclasses import dataclass
from typing import Callable, Optional, Type, Union

import equinox as eqx
from jaxtyping import PRNGKeyArray

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import maybe_rng_split

from levanter.compat.hf_checkpoints import HFCheckpointConverter, HFCompatConfig, ModelWithHfSerializationMixin
from levanter.compat.torch_serialization import StateDictSerializationMixin
from levanter.logging import silence_transformer_nag
from levanter.models.asr_model import ASRConfig, ASRMixin
from levanter.models.attention import AttentionMask
from levanter.models.llama import LlamaConfig, LlamaLMHeadModel
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.models.mistral import MistralLMHeadModel
from levanter.models.whisper import ACT2FN, WhisperConfig, WhisperEncoder
from levanter.utils.py_utils import cached_classproperty


silence_transformer_nag()
from transformers import PretrainedConfig as HfConfig  # noqa: E402


class ViaConnector(eqx.Module, StateDictSerializationMixin):
    conv1: hnn.Conv
    act: Callable = eqx.static_field()

    @classmethod
    def init(cls, config: ViaConfig, *, key) -> "ViaConnector":
        k_conv1 = maybe_rng_split(key, 3)

        Len = hax.Axis("position", size=config.SourcePos.size * 2)
        Mid = hax.Axis("mid", config.Embed.size)
        conv1 = hnn.Conv.init(Len, config.Mels, Mid, kernel_size=3, padding=1, key=k_conv1)
        if isinstance(config.activation_function, str):
            activation_fn = ACT2FN[config.activation_function]
        act = activation_fn  # type: ignore

        return ViaConnector(conv1, act)

    def __call__(self, spec: NamedArray, *, key=None) -> NamedArray:
        k_conv1 = maybe_rng_split(key, 3)
        spec = spec.astype(self.conv1.weight.dtype)
        x = self.act(self.conv1(spec, key=k_conv1))
        return x


@LmConfig.register_subclass("via")
@dataclass(frozen=True)
class ViaConfig(HFCompatConfig, ASRConfig):
    # SubConfigs
    enc_config: WhisperConfig
    dec_config: LlamaConfig

    @property
    def model_type(self) -> Type["ViaModel"]:
        return ViaModel

    @property
    def asr_model_type(self) -> Type["ViaASRModel"]:
        return ViaASRModel

    def to_hf_config(self, vocab_size: int, config_overrides: Optional[dict] = None) -> HfConfig:
        hf_enc_config = self.enc_config.to_hf_config(vocab_size, config_overrides)
        hf_dec_config = self.dec_config.to_hf_config(vocab_size, config_overrides)
        merged_config = {"encoder": hf_enc_config.to_dict(), "decoder": hf_dec_config.to_dict()}
        return HfConfig.from_dict(merged_config)

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig):
        hf_enc_config = HfConfig.from_dict(hf_config["encoder"])
        hf_dec_config = HfConfig.from_dict(hf_config["decoder"])
        enc_config = WhisperConfig(hf_enc_config)
        dec_config = LlamaConfig(hf_dec_config)
        return ViaConfig(enc_config, dec_config)

    @cached_classproperty
    def default_hf_checkpoint_converter(cls) -> HFCheckpointConverter["ViaModel"]:  # type: ignore
        return HFCheckpointConverter(cls, "openai/whisper-base", ignore_prefix="model")


class ViaModel(eqx.Module, ModelWithHfSerializationMixin[ViaConfig]):
    encoder: WhisperEncoder
    connector: ViaConnector
    decoder: Union[LlamaLMHeadModel | MistralLMHeadModel]

    @property
    def config(self):
        return self.encoder.config

    @property
    def Pos(self) -> Axis:
        return self.config.Pos

    @property
    def Vocab(self) -> Axis:
        return self.decoder.embeddings.Vocab

    def resize_vocab(self, new_size: int, key: Optional[PRNGKeyArray] = None) -> "ViaModel":
        new_decoder = self.decoder.resize_vocab(new_size, key)
        return dataclasses.replace(self, decoder=new_decoder)

    @classmethod
    def init(
        cls,
        Vocab: Axis,
        config: ViaConfig,
        *,
        key,
        dec_cls: Type["LmHeadModel"] = MistralLMHeadModel,
    ) -> "ViaModel":
        k_enc, k_connector, k_dec = maybe_rng_split(key, 3)
        encoder = WhisperEncoder.init(config.enc_config, key=k_enc)
        connector = ViaConnector.init(config, key=k_connector)
        decoder = dec_cls.init(Vocab, config.dec_config, key=k_dec)

        return cls(encoder, connector, decoder)

    def __call__(
        self,
        mel: NamedArray,
        input_ids: NamedArray,
        attn_mask: Optional[AttentionMask | NamedArray] = None,
        *,
        key=None,
    ) -> NamedArray:
        if attn_mask is not None and not isinstance(attn_mask, AttentionMask):
            attn_mask = AttentionMask.explicit(attn_mask)
        k_encoder, k_connector, k_decoder, k_head = maybe_rng_split(key, 4)
        audio_features = self.encoder(mel, key=k_encoder)
        virtual_tokens = self.connector(audio_features, key=k_connector)
        embedded_tokens = self.decoder.embeddings.embed(input_ids)
        tokens_and_targets = hax.concatenate([virtual_tokens, embedded_tokens], axis=self.Pos)
        x = self.decoder.transformer(tokens_and_targets, attn_mask=attn_mask, key=k_decoder)
        lm_logits = self.decoder.lm_head(x, key=k_head)
        return lm_logits


class ViaASRModel(ViaModel, ASRMixin):
    pass
