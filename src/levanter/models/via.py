import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Sequence, Type, Union

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import maybe_rng_split

from levanter.compat.hf_checkpoints import HFCheckpointConverter, HFCompatConfig, ModelWithHfSerializationMixin
from levanter.logging import silence_transformer_nag
from levanter.models.asr_model import ASRConfig, ASRMixin, AudioTextExample
from levanter.models.attention import AttentionMask
from levanter.models.llama import LlamaConfig, LlamaLMHeadModel
from levanter.models.lm_model import LmHeadModel
from levanter.models.mistral import MistralLMHeadModel
from levanter.models.whisper import WhisperConfig, WhisperDecoder, WhisperEncoder
from levanter.utils.py_utils import cached_classproperty


silence_transformer_nag()
from transformers import PretrainedConfig as HfConfig  # noqa: E402


@dataclass(frozen=True)
class ViaConfig(HFCompatConfig, ASRConfig):
    # SubConfigs
    enc_config: WhisperConfig = field(default_factory=WhisperConfig)
    dec_config: LlamaConfig = field(default_factory=LlamaConfig)

    # Connector Config
    pre_audio_prompt: Sequence[int] = field(default_factory=lambda: [128000, 128006, 882, 128007, 271])
    # pre_audio_prompt: Sequence[int] = field(default_factory=lambda: [128000])
    pre_text_prompt: Sequence[int] = field(default_factory=lambda: [128009, 128006, 78191, 128007, 271])

    prefix = property(lambda self: hax.named(self.pre_audio_prompt, axis="position"))
    suffix = property(lambda self: hax.named(self.pre_text_prompt, axis="position"))
    Pos = property(lambda self: Axis(name="position", size=448))
    AudioPos = property(lambda self: [self.enc_config.Mels, self.enc_config.MelPos])
    KeyPos = property(lambda self: self.Pos.alias("key_position"))
    QueryPos = property(lambda self: Axis(name="position", size=448))

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
        }
        return HfConfig.from_dict(merged_config)

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig):
        config_dict = hf_config.to_dict()
        hf_enc_config = HfConfig.from_dict(config_dict["encoder"])
        hf_dec_config = HfConfig.from_dict(config_dict["decoder"])
        enc_config = WhisperConfig.from_hf_config(hf_enc_config)
        dec_config = LlamaConfig.from_hf_config(hf_dec_config)
        return ViaConfig(enc_config, dec_config)

    @cached_classproperty
    def default_hf_checkpoint_converter(cls) -> HFCheckpointConverter["ViaModel"]:  # type: ignore
        return HFCheckpointConverter(cls, "WillHeld/via-llama")


def connector_only(model):
    frozen_tree = jax.tree_util.tree_map(lambda _: False, model)
    return eqx.tree_at(
        lambda tree: (tree.query_tokens, tree.projection.weight, tree.projection.bias, tree.connector),
        frozen_tree,
        (True, True, True, True),
    )


class ViaModel(eqx.Module, ModelWithHfSerializationMixin[ViaConfig]):
    query_tokens: NamedArray
    projection: hnn.Linear
    encoder: WhisperEncoder
    connector: WhisperDecoder
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
        connector = WhisperDecoder.init(config.enc_config, key=k_connector)
        query_tokens = hax.random.normal(k_query, (config.QueryPos, config.enc_config.Embed)) * 0.02
        projection = hnn.Linear.init(
            In=config.enc_config.Embed.alias("whisp_embed"), Out=config.dec_config.Embed, key=key
        )
        decoder = dec_cls.init(Vocab, config.dec_config, key=k_dec)

        return cls(query_tokens, projection, encoder, connector, decoder, config)

    @property
    def query_position_embeds(self) -> NamedArray:
        return self.connector.embeddings.position_embeddings.embed(hax.arange(self.config.QueryPos))

    def __call__(
        self,
        mel: NamedArray,
        input_ids: NamedArray,
        attn_mask: Optional[AttentionMask | NamedArray] = None,
        pad_token_id: int = 128002,
        *,
        key=None,
    ) -> NamedArray:
        # Setup
        Batch = input_ids.resolve_axis("batch")
        OtherAxes = hax.axis.eliminate_axes(input_ids.axes, "position")
        causal_mask = AttentionMask.causal()
        if attn_mask is not None:
            causal_mask = causal_mask & attn_mask
        k_encoder, k_connector, k_decoder, k_head = maybe_rng_split(key, 4)

        # Encode Audio With Whisper Encoder
        audio_features = self.encoder(mel, key=k_encoder)

        # Convert to Virtual LLM Tokens
        virt_whisper_tokens = self.connector.transformer(
            (self.query_tokens + self.query_position_embeds).broadcast_axis(OtherAxes),
            audio_features,
            causal_mask,
            key=k_connector,
        )

        virtual_tokens = self.projection(virt_whisper_tokens.rename({"embed": "whisp_embed"}))

        # Embed Real LLM Tokens
        prefix = self.decoder.embeddings.embed(self.config.prefix)
        suffix = self.decoder.embeddings.embed(self.config.suffix)

        # Create Mixed Virtual and Real Input
        audio_embeds = hax.concatenate(
            "position",
            [
                prefix.broadcast_axis(OtherAxes),
                virtual_tokens,
                suffix.broadcast_axis(OtherAxes),
            ],
        )

        text_tokens = hax.concatenate(
            "position",
            [
                self.config.prefix.broadcast_axis(OtherAxes),
                input_ids,
                self.config.suffix.broadcast_axis(OtherAxes),
            ],
        )
        push_back_padding = hax.argsort(text_tokens == pad_token_id, "position")

        text_tokens_left_pad = text_tokens[{"batch": hax.arange(Batch), "position": push_back_padding}]

        text_embeds = self.decoder.embeddings.embed(text_tokens_left_pad)
        # Create LLM Response
        audio = self.decoder.transformer(audio_embeds, attn_mask=causal_mask, key=k_decoder)
        text = self.decoder.transformer(text_embeds, attn_mask=causal_mask, key=k_decoder)

        push_forward_padding = hax.argsort(input_ids != pad_token_id, "position")
        input_ids_right_pad = text_tokens[{"batch": hax.arange(Batch), "position": push_forward_padding}]
        return (
            audio["position", -1],
            text[
                {
                    "batch": hax.arange(Batch),
                    "position": (hax.sum(text_tokens == pad_token_id, "position") * -1) - 1,
                }
            ],
            virtual_tokens,
            self.decoder.embeddings.embed(input_ids_right_pad),
        )


class ViaASRModel(ViaModel, ASRMixin):
    def compute_loss(
        self,
        example: AudioTextExample,
        *,
        key=None,
        reduction: Optional[hax.ReductionFunction] = hax.mean,
        reduction_axis: Optional[hax.AxisSelection] = None,
    ) -> NamedArray:
        LocalPos = example.tokens.resolve_axis("position")
        # Compute Distillation Contrastive Loss
        # Since the weight matrix is frozen, equivalent but faster than KL Div
        audio_pred, text_pred, virtual_embeds, real_embeds = self(
            example.audio, example.tokens, example.attn_mask, key=key
        )
        diff_distill = audio_pred - text_pred
        kl_proxy_loss = hax.dot(diff_distill, diff_distill, axis="embed") ** 0.5

        # Compute Contrastive Loss on Input
        # Correct for Normal Autoregressive Loss Mask
        corrected_loss_mask = hax.roll(example.loss_mask, 1, LocalPos) + hax.nn.one_hot(
            0, LocalPos, dtype=jax.numpy.float32
        )
        # Mask Final Tokens So That Initial Tokens can be used for extra computation
        reversed_loss_mask = corrected_loss_mask["position", -1::-1]
        diff_contrast = virtual_embeds - real_embeds
        loss2 = hax.dot(diff_contrast, diff_contrast, axis="embed") ** 0.5

        if reduction is None:
            return kl_proxy_loss
        else:
            # return reduction(kl_proxy_loss, axis=reduction_axis) + reduction(
            #    loss2, axis=reduction_axis, where=reversed_loss_mask
            # )
            return reduction(loss2, axis=reduction_axis, where=reversed_loss_mask)
