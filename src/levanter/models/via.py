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
from levanter.models.whisper import ACT2FN, WhisperConfig, WhisperEncoder, WhisperTransformer
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
    dialation_factor: int = 4
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
    DialationEmbed = property(
        lambda self: Axis(
            name="dialation_embed", size=((self.enc_config.Embed.size * self.time_dialation) // self.dialation_factor)
        )
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
        lambda tree: (tree.query_tokens, tree.projection.weight, tree.projection.bias), frozen_tree, (True, True, True)
    )


# def connector_only(model):
#    frozen_tree = jax.tree_util.tree_map(lambda _: False, model)
#    return eqx.tree_at(
#        lambda tree: (tree.connector.dialator.weight, tree.connector.dialator.bias), frozen_tree, #(True, True)
#    )


class ViaConnector(eqx.Module, StateDictSerializationMixin):
    Grouping: Sequence[Axis]
    dialator: hnn.Linear
    compressor: hnn.Linear
    act: Callable = eqx.static_field()
    config: ViaConfig = eqx.static_field()

    @classmethod
    def init(cls, config: ViaConfig, *, key) -> "ViaConnector":
        dialator = hnn.Linear.init(In=config.GroupEmbed, Out=config.DialationEmbed, key=key)
        compressor = hnn.Linear.init(In=config.DialationEmbed, Out=config.dec_config.Embed, key=key)

        if isinstance(config.enc_config.activation_function, str):
            activation_fn = ACT2FN[config.enc_config.activation_function]
        act = activation_fn  # type: ignore

        Grouping = (config.TimeGroup, config.GroupEmbed)

        return ViaConnector(
            Grouping,
            dialator,
            compressor,
            act,
            config,
        )

    def __call__(self, encoder_outputs: NamedArray, *, key=None) -> NamedArray:
        flat_encoder_outputs = hax.flatten_axes(encoder_outputs, ("position", "embed_dim"), "flat_embed")
        grouped_encoder_outputs = hax.unflatten_axis(flat_encoder_outputs, "flat_embed", self.Grouping)
        x = self.act(self.dialator(grouped_encoder_outputs, key=key))
        x = self.compressor(x)
        return x

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> "ViaConnector":
        # convert to Haliax's nice multiple dim input linear syntax
        d = {}
        d.update(unflatten_linear_layers(apply_prefix(prefix, "dialator"), state_dict, self.dialator, None))

        d.update(unflatten_linear_layers(apply_prefix(prefix, "compressor"), state_dict, self.compressor, None))

        return super().from_state_dict(d, prefix)

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        # need to undo the reshape we did in from_state_dict
        my_dict: StateDict = {}
        super().update_state_dict(my_dict, prefix)

        my_dict.update(flatten_linear_layers(apply_prefix(prefix, "dialator"), self.dialator, None))

        my_dict.update(flatten_linear_layers(apply_prefix(prefix, "compressor"), self.compressor, None))

        state_dict.update(my_dict)
        return state_dict


class ViaModel(eqx.Module, ModelWithHfSerializationMixin[ViaConfig]):
    query_tokens: NamedArray
    projection: hnn.Linear
    encoder: WhisperEncoder
    connector: WhisperTransformer
    decoder: Union[LlamaLMHeadModel | MistralLMHeadModel]
    _config: ViaConfig = eqx.static_field()

    @property
    def config(self):
        return self._config

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
        dec_cls: Type["LmHeadModel"] = LlamaLMHeadModel,
    ) -> "ViaModel":
        k_query, k_projection, k_enc, k_connector, k_dec = maybe_rng_split(key, 5)
        encoder = WhisperEncoder.init(config.enc_config, key=k_enc)
        connector = WhisperTransformer.init(
            config.enc_config.DecoderLayer,
            config.enc_config.DecoderHeads,
            config.enc_config.DecoderHeadSize,
            config.enc_config.DecoderMlp,
            config.enc_config,
            has_cross=True,
            key=k_connector,
        )
        query_tokens = hax.random.normal(k_query, (config.TimeGroup, config.enc_config.Embed)) * 0.02
        projection = hnn.Linear.init(In=config.enc_config.Embed, Out=config.dec_config.Embed, key=key)
        decoder = dec_cls.init(Vocab, config.dec_config, key=k_dec)

        return cls(query_tokens, projection, encoder, connector, decoder, config)

    def __call__(
        self,
        mel: NamedArray,
        input_ids: NamedArray,
        attn_mask: Optional[AttentionMask | NamedArray] = None,
        *,
        key=None,
    ) -> NamedArray:
        # Setup
        InputPosition = input_ids.resolve_axis("position")
        OtherAxes = hax.axis.eliminate_axes(input_ids.axes, "position")
        causal_mask = AttentionMask.causal()
        if attn_mask is not None:
            causal_mask = causal_mask & attn_mask
        k_encoder, k_connector, k_decoder, k_head = maybe_rng_split(key, 4)

        # Encode Audio With Whisper Encoder
        audio_features = self.encoder(mel, key=k_encoder)

        # Convert to Virtual LLM Tokens
        position_embeds = self.orig_position_embeddings.embed(hax.arange(self.TimeGroup))

        virt_whisper_tokens = self.connector(
            (self.query_tokens + position_embeds).broadcast_axis(OtherAxes),
            audio_features,
            causal_mask,
            key=k_connector,
        )
        virtual_tokens = self.projection(virt_whisper_tokens)

        # Embed Real LLM Tokens
        prefix = self.decoder.embeddings.embed(self.config.prefix)
        suffix = self.decoder.embeddings.embed(self.config.suffix)
        embedded_tokens = self.decoder.embeddings.embed(input_ids)

        # Create Mixed Virtual and Real Input
        in_tokens = hax.concatenate(
            "position",
            [
                prefix.broadcast_axis(OtherAxes),
                virtual_tokens,
                suffix.broadcast_axis(OtherAxes),
            ],
        )
        tokens_and_targets = hax.concatenate("position", [in_tokens, embedded_tokens])
        llm_input = tokens_and_targets["position", : self.Pos.size]

        # Create LLM Response
        in_tokens_size = in_tokens.resolve_axis("position").size
        x = self.decoder.transformer(llm_input, attn_mask=causal_mask, key=k_decoder)
        target_x = x["position", in_tokens_size:]
        target_logits = self.decoder.lm_head(target_x, key=k_head)

        # Reconstruct Padded Output Prediction with Input Predictions Removed
        diff = InputPosition.size - target_logits.resolve_axis("position").size
        OtherAxes = hax.axis.eliminate_axes(target_logits.axes, "position")
        return hax.concatenate(
            "position", [target_logits, hax.zeros(InputPosition.resize(diff)).broadcast_axis(OtherAxes)]
        )


class ViaASRModel(ViaModel, ASRMixin):
    pass
