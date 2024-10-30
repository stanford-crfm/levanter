import dataclasses
from dataclasses import dataclass
from typing import Optional, Type, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import maybe_rng_split
from haliax.partitioning import ResourceMapping

from levanter.compat.hf_checkpoints import (
    HFCheckpointConverter,
    HFCompatConfig,
    ModelWithHfSerializationMixin,
    RepoRef,
)
from levanter.logging import silence_transformer_nag
from levanter.models.asr_model import ASRConfig, ASRMixin, AudioTextExample
from levanter.models.attention import AttentionMask
from levanter.models.gemma import GemmaConfig, GemmaLMHeadModel
from levanter.models.llama import LlamaConfig, LlamaLMHeadModel
from levanter.models.lm_model import LmConfig
from levanter.models.mistral import MistralConfig, MistralLMHeadModel
from levanter.models.whisper import WhisperConfig, WhisperDecoder, WhisperEncoder, WhisperModel


silence_transformer_nag()
from transformers import AutoTokenizer  # noqa: E402
from transformers import PretrainedConfig as HfConfig  # noqa: E402


class DivaHFCheckpointer(HFCheckpointConverter["DivaModel"]):
    def load_pretrained(
        self,
        lm_model_cls: Type[ModelWithHfSerializationMixin],
        ref: Optional[Union[str, RepoRef]] = None,
        config: Optional[HFCompatConfig] = None,
        axis_mapping: Optional[ResourceMapping] = None,
        resize_vocab_to_match_tokenizer: bool = True,
        dtype: Optional[jnp.dtype] = None,
    ) -> ModelWithHfSerializationMixin:
        lev_model: DivaModel = super().load_pretrained(
            DivaModel, ref, config, axis_mapping, resize_vocab_to_match_tokenizer, dtype
        )  # type: ignore[assignment]
        llm: Union[LlamaLMHeadModel | MistralLMHeadModel | GemmaLMHeadModel] = HFCheckpointConverter(
            type(lev_model.config.dec_config),
            lev_model.config.reference_decoder,
        ).load_pretrained(
            lev_model.config.dec_config.model_type,
            lev_model.config.reference_decoder,
            lev_model.config.dec_config,
            axis_mapping,
            resize_vocab_to_match_tokenizer,
            dtype,
        )  # type: ignore[assignment]
        whisper: WhisperModel = HFCheckpointConverter(
            WhisperConfig, lev_model.config.reference_encoder
        ).load_pretrained(
            WhisperModel,
            lev_model.config.reference_encoder,
            lev_model.config.enc_config,
            axis_mapping,
            resize_vocab_to_match_tokenizer,
            dtype,
        )  # type: ignore[assignment]
        lev_model.encoder = whisper.encoder
        lev_model.decoder = llm

        return lev_model


def load_correct_config(reference_decoder):
    model_id = reference_decoder.lower()
    hf_config = HfConfig.from_pretrained(reference_decoder)
    if "llama" in model_id:
        config = LlamaConfig.from_hf_config(hf_config)
    elif "gemma" in model_id:
        config = GemmaConfig.from_hf_config(hf_config)
    elif "mistral" in model_id:
        config = MistralConfig.from_hf_config(hf_config)
    return config


def get_prefix(tokenizer_ref):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref)
    prefix, suffix = tokenizer.apply_chat_template(
        [{"role": "user", "content": "PLACEHOLDER"}], tokenize=False, add_generation_prompt=True
    ).split("PLACEHOLDER")
    prefix_tok = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tok = tokenizer.encode(suffix, add_special_tokens=False)
    return prefix_tok, suffix_tok


@LmConfig.register_subclass("diva")
@dataclass(frozen=True)
class DivaConfig(HFCompatConfig, ASRConfig):
    # Reference Models
    reference_encoder: str = "openai/whisper-large-v3-turbo"
    reference_decoder: str = "meta-llama/Llama-3.1-8B-Instruct"
    reference_checkpoint: str = "WillHeld/DiVA-llama-3-v0-8b"
    init_from_submodel: bool = True

    # Connector Config
    pre_audio_prompt = property(lambda self: get_prefix(self.reference_decoder)[0])
    pre_text_prompt = property(lambda self: get_prefix(self.reference_decoder)[1])

    # SubConfigs
    enc_config = property(
        lambda self: WhisperConfig.from_hf_config(HfConfig.from_pretrained(self.reference_encoder)),
    )
    dec_config = property(
        lambda self: load_correct_config(self.reference_decoder),
    )
    prefix = property(lambda self: hax.named(self.pre_audio_prompt, axis="position"))
    suffix = property(lambda self: hax.named(self.pre_text_prompt, axis="position"))
    Pos = property(lambda self: Axis(name="position", size=448))
    AudioPos = property(lambda self: self.enc_config.AudioPos)
    KeyPos = property(lambda self: self.Pos.alias("key_position"))

    @property
    def model_type(self) -> Type["DivaModel"]:
        return DivaModel

    @property
    def asr_model_type(self) -> Type["DivaASRModel"]:
        return DivaASRModel

    def to_hf_config(self, vocab_size: int, config_overrides: Optional[dict] = None) -> HfConfig:
        merged_config = {
            "model_type": "diva",
            "architectures": ["DiVAModel"],
            "auto_map": {"AutoConfig": "configuring_diva.DiVAConfig", "AutoModel": "modeling_diva.DiVAModel"},
            "vocab_size": vocab_size,
            "reference_encoder": self.reference_encoder,
            "reference_decoder": self.reference_decoder,
        }
        return HfConfig.from_dict(merged_config)

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig):
        config_dict = hf_config.to_dict()
        reference_encoder = config_dict["encoder_reference"]
        reference_decoder = config_dict["decoder_reference"]
        return DivaConfig(reference_encoder, reference_decoder)

    def hf_checkpoint_converter(cls) -> HFCheckpointConverter["DivaModel"]:  # type: ignore
        return DivaHFCheckpointer(cls, cls.reference_checkpoint, trust_remote_code=True)


def diva_connector_only(model):
    frozen_tree = jax.tree_util.tree_map(lambda _: False, model)
    return eqx.tree_at(
        lambda tree: (tree.query_tokens, tree.projection.weight, tree.projection.bias, tree.connector),
        frozen_tree,
        (True, True, True, True),
    )


class DivaModel(eqx.Module, ModelWithHfSerializationMixin[DivaConfig]):
    query_tokens: NamedArray
    projection: hnn.Linear
    encoder: WhisperEncoder
    connector: WhisperDecoder
    decoder: Union[LlamaLMHeadModel | MistralLMHeadModel | GemmaLMHeadModel]
    _config: DivaConfig = eqx.static_field()

    @property
    def config(self):
        return self._config

    @property
    def Pos(self) -> Axis:
        return self.config.Pos

    @property
    def Vocab(self) -> Axis:
        return self.decoder.embeddings.Vocab

    def resize_vocab(self, new_size: int, key: Optional[PRNGKeyArray] = None) -> "DivaModel":
        new_decoder = self.decoder.resize_vocab(new_size, key)
        return dataclasses.replace(self, decoder=new_decoder)

    @classmethod
    def init(
        cls,
        Vocab: Axis,
        config: DivaConfig,
        *,
        key,
        init_from_submodels: bool = False,
    ) -> "DivaModel":
        k_query, k_projection, k_enc, k_connector, k_dec = maybe_rng_split(key, 5)

        query_tokens = hax.random.normal(k_query, (config.Pos, config.enc_config.Embed)) * 0.02
        projection = hnn.Linear.init(
            In=config.enc_config.Embed.alias("whisp_embed"), Out=config.dec_config.Embed, key=key
        )

        if init_from_submodels:
            llm: Union[LlamaLMHeadModel | MistralLMHeadModel | GemmaLMHeadModel] = HFCheckpointConverter(
                type(config.dec_config), config.reference_decoder
            ).load_pretrained(
                config.dec_config.model_type,
                config.reference_decoder,
                config.dec_config,
            )  # type: ignore[assignment]
            whisper: WhisperModel = HFCheckpointConverter(
                WhisperConfig, config.reference_encoder, ignore_prefix="model"
            ).load_pretrained(
                WhisperModel,
                config.reference_encoder,
                config.enc_config,
            )  # type: ignore[assignment]
            encoder = whisper.encoder
            connector = whisper.decoder
            decoder = llm
        else:
            encoder = WhisperEncoder.init(config.enc_config, key=k_enc)
            connector = WhisperDecoder.init(config.enc_config, key=k_connector)
            decoder = config.dec_config.model_type.init(Vocab, config.dec_config, key=k_dec)

        return cls(query_tokens, projection, encoder, connector, decoder, config)

    @property
    def query_position_embeds(self) -> NamedArray:
        return self.connector.embeddings.position_embeddings.embed(hax.arange(self.config.Pos))

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

        virtual_tokens = self.projection(virt_whisper_tokens.rename({"embed_dim": "whisp_embed"}))

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


class DivaASRModel(DivaModel, ASRMixin):
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
            return reduction(kl_proxy_loss, axis=reduction_axis) + reduction(
                loss2, axis=reduction_axis, where=reversed_loss_mask
            )
