import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Type, Union

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import maybe_rng_split

from levanter.compat.hf_checkpoints import HFCheckpointConverter, HFCompatConfig, ModelWithHfSerializationMixin
from levanter.compat.torch_serialization import (
    StateDict,
    StateDictSerializationMixin,
    apply_prefix,
    flatten_linear_layers,
    unflatten_linear_layers,
)
from levanter.logging import silence_transformer_nag
from levanter.models.asr_model import ASRConfig, ASRMixin
from levanter.models.attention import AttentionMask
from levanter.models.llama import LlamaConfig, LlamaLMHeadModel
from levanter.models.lm_model import LmHeadModel
from levanter.models.mistral import MistralLMHeadModel
from levanter.models.whisper import ACT2FN, WhisperConfig, WhisperEncoder
from levanter.utils.py_utils import cached_classproperty


silence_transformer_nag()
from transformers import PretrainedConfig as HfConfig  # noqa: E402


@dataclass(frozen=True)
class ViaConfig(HFCompatConfig, ASRConfig):
    # SubConfigs
    enc_config: WhisperConfig = field(default_factory=WhisperConfig)
    dec_config: LlamaConfig = field(default_factory=LlamaConfig)

    # Connector Config
    time_dialation: int = 25
    pre_audio_prompt: Sequence[int] = field(default_factory=lambda: [1, 518, 25580, 29962, 376])
    pre_text_prompt: Sequence[int] = field(
        default_factory=lambda: [376, 13, 830, 11666, 393, 1250, 304, 592, 29889, 518, 29914, 25580, 29962]
    )

    prefix = property(lambda self: hax.named(self.pre_audio_prompt, axis="position"))
    suffix = property(lambda self: hax.named(self.pre_text_prompt, axis="position"))
    Pos = property(lambda self: Axis(name="position", size=self.dec_config.seq_len))
    AudioPos = property(lambda self: [self.enc_config.Mels, self.enc_config.MelPos])
    KeyPos = property(lambda self: self.Pos.alias("key_position"))
    TimeGroup = property(
        lambda self: Axis(name="position", size=(self.enc_config.SourcePos.size // self.time_dialation))
    )
    GroupEmbed = property(
        lambda self: Axis(name="group_embed", size=(self.enc_config.Embed.size * self.time_dialation))
    )

    @property
    def model_type(self) -> Type["ViaModel"]:
        return ViaModel

    @property
    def asr_model_type(self) -> Type["ViaASRModel"]:
        return ViaASRModel

    def to_hf_config(self, vocab_size: int, config_overrides: Optional[dict] = None) -> HfConfig:
        hf_enc_config = self.enc_config.to_hf_config(vocab_size, config_overrides)
        hf_dec_config = self.dec_config.to_hf_config(vocab_size, config_overrides)
        merged_config = {
            "encoder": hf_enc_config.to_dict(),
            "decoder": hf_dec_config.to_dict(),
            "time_dialation": self.time_dialation,
        }
        return HfConfig.from_dict(merged_config)

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig):
        config_dict = hf_config.to_dict()
        hf_enc_config = HfConfig.from_dict(config_dict["encoder"])
        hf_dec_config = HfConfig.from_dict(config_dict["decoder"])
        enc_config = WhisperConfig.from_hf_config(hf_enc_config)
        dec_config = LlamaConfig.from_hf_config(hf_dec_config)
        return ViaConfig(enc_config, dec_config, config_dict["time_dialation"])

    @cached_classproperty
    def default_hf_checkpoint_converter(cls) -> HFCheckpointConverter["ViaModel"]:  # type: ignore
        return HFCheckpointConverter(cls, "WillHeld/via-llama")


def connector_only(model):
    frozen_tree = jax.tree_util.tree_map(lambda _: False, model)
    return eqx.tree_at(
        lambda tree: (tree.connector.dialator.weight, tree.connector.dialator.bias), frozen_tree, (True, True)
    )


class ViaConnector(eqx.Module, StateDictSerializationMixin):
    Grouping: Sequence[Axis]
    dialator: hnn.Linear
    act: Callable = eqx.static_field()
    config: ViaConfig = eqx.static_field()

    @classmethod
    def init(cls, config: ViaConfig, *, key) -> "ViaConnector":
        dialator = hnn.Linear.init(In=config.GroupEmbed, Out=config.dec_config.Embed, key=key)

        if isinstance(config.enc_config.activation_function, str):
            activation_fn = ACT2FN[config.enc_config.activation_function]
        act = activation_fn  # type: ignore

        Grouping = (config.TimeGroup, config.GroupEmbed)

        return ViaConnector(
            Grouping,
            dialator,
            act,
            config,
        )

    def __call__(self, encoder_outputs: NamedArray, *, key=None) -> NamedArray:
        flat_encoder_outputs = hax.flatten_axes(encoder_outputs, ("position", "embed_dim"), "flat_embed")
        grouped_encoder_outputs = hax.unflatten_axis(flat_encoder_outputs, "flat_embed", self.Grouping)
        x = self.act(self.dialator(grouped_encoder_outputs, key=key))
        return x

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> "ViaConnector":
        # convert to Haliax's nice multiple dim input linear syntax
        d = {}
        d.update(unflatten_linear_layers(apply_prefix(prefix, "dialator"), state_dict, self.dialator, None))

        return super().from_state_dict(d, prefix)

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        # need to undo the reshape we did in from_state_dict
        my_dict: StateDict = {}
        super().update_state_dict(my_dict, prefix)

        my_dict.update(flatten_linear_layers(apply_prefix(prefix, "dialator"), self.dialator, None))

        state_dict.update(my_dict)
        return state_dict


class ViaModel(eqx.Module, ModelWithHfSerializationMixin[ViaConfig]):
    encoder: WhisperEncoder
    connector: ViaConnector
    decoder: Union[LlamaLMHeadModel | MistralLMHeadModel]

    @property
    def config(self):
        return self.connector.config

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

        return cls(
            encoder,
            connector,
            decoder,
        )

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
        prefix = self.decoder.embeddings.embed(self.config.prefix.broadcast_axis(input_ids.resolve_axis("batch")))
        embedded_tokens = self.decoder.embeddings.embed(input_ids)
        suffix = self.decoder.embeddings.embed(self.config.suffix.broadcast_axis(input_ids.resolve_axis("batch")))
        tokens_and_targets = hax.concatenate("position", [prefix, virtual_tokens, suffix, embedded_tokens])
        x = self.decoder.transformer(tokens_and_targets, attn_mask=attn_mask, key=k_decoder)
        lm_logits = self.decoder.lm_head(x, key=k_head)
        prompt_length = lm_logits.resolve_axis("position").size - input_ids.resolve_axis("position").size
        return lm_logits["position", prompt_length:]


class ViaASRModel(ViaModel, ASRMixin):
    pass
