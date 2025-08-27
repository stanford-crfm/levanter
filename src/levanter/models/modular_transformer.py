from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional, Type

import equinox as eqx
import jax.random as jrandom
from jaxtyping import PRNGKeyArray

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, AxisSpec, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call, shaped_rng_split
from haliax.nn.scan import ScanCheckpointPolicy, Stacked
from haliax.state_dict import ModuleWithStateDictSerialization

from levanter.layers import LayerNormConfigBase, RmsNormConfig
from levanter.layers.attention import Attention, AttentionBackend, AttentionConfig, AttentionMask
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig, RotaryEmbeddingsConfig
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.models.llama import LlamaMlp
from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.types import BlockFoldable

import logging

logger = logging.getLogger(__name__)


class ModularDecoderLayer(eqx.Module):
    """A transformer decoder layer that exposes its components for easy swapping."""

    config: "ModularConfig" = eqx.field(static=True)
    self_attn: eqx.Module
    mlp: eqx.Module
    input_layernorm: hnn.RmsNorm
    post_attention_layernorm: hnn.RmsNorm

    @staticmethod
    def init(config: "ModularConfig", *, key) -> "ModularDecoderLayer":
        k_attn, k_mlp = jrandom.split(key, 2)

        attn = config.attention_cls.init(config.attention_config(), key=k_attn)
        mlp = config.mlp_cls.init(
            config.Embed,
            config.Mlp,
            config.activation_function,
            key=k_mlp,
            use_bias=config.use_bias,
        )
        ln_1 = config.mk_LayerNorm(config.Embed)
        ln_2 = config.mk_LayerNorm(config.Embed)
        return ModularDecoderLayer(config, attn, mlp, ln_1, ln_2)

    @named_call
    def __call__(
        self, x: NamedArray, mask: Optional[NamedArray | AttentionMask], *, key=None, pos_ids: NamedArray | None = None
    ) -> NamedArray:
        k_attn, k_mlp = maybe_rng_split(key, 2)

        residual = x
        x = self.input_layernorm(x)
        attn_output = self.self_attn(x=x, mask=mask, key=k_attn, pos_ids=pos_ids)
        x = residual + attn_output

        residual = x
        x = self.post_attention_layernorm(x)
        mlp_output = self.mlp(x, key=k_mlp)
        output = residual + mlp_output
        return output


@LmConfig.register_subclass("modular")
@dataclass(frozen=True)
class ModularConfig(LmConfig):
    """Config for :class:`ModularLMHeadModel`.

    This config exposes hooks for swapping out attention, MLP, and even the
    decoder layer class.  By default it uses the same building blocks as the
    Llama model but any component can be replaced.
    """

    seq_len: int = 2048
    hidden_dim: int = 4096
    intermediate_dim: int = 11008
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int | None = None
    num_kv_heads: int = 32
    activation_function: ActivationFunctionEnum = ActivationFunctionEnum.silu
    initializer_range: float = 0.02
    layer_norm_epsilon: float = 1e-5
    tie_word_embeddings: bool = False
    input_embedding_norm: bool = False
    gradient_checkpointing: bool | ScanCheckpointPolicy | str = True
    scan_layers: bool = True
    use_bias: bool = False
    use_layer_norm_weight: bool = True

    # Attention-related config
    upcast_attn: bool = False
    attn_backend: Optional[AttentionBackend] = None
    flash_attention_block_size: Optional[int] = None
    rope: RotaryEmbeddingsConfig = dataclasses.field(default_factory=DefaultRotaryEmbeddingsConfig)

    # Modular pieces
    layer_cls: Type[ModularDecoderLayer] = ModularDecoderLayer
    attention_cls: Type[Attention] = Attention
    mlp_cls: Type[eqx.Module] = LlamaMlp
    norm_config: LayerNormConfigBase | None = None

    def with_(self, **updates) -> "ModularConfig":
        return dataclasses.replace(self, **updates)

    def with_attention(
        self,
        *,
        attention_cls: Type[Attention],
        attention_config_cls: Type[AttentionConfig],
        **attention_config_kwargs: Any,
    ) -> "ModularConfig":
        merged_kwargs = {**self.attention_config_kwargs, **attention_config_kwargs}
        return dataclasses.replace(
            self,
            attention_cls=attention_cls,
            attention_config_cls=attention_config_cls,
            attention_config_kwargs=merged_kwargs,
        )

    def __post_init__(self):
        if self.norm_config is None:
            object.__setattr__(
                self,
                "norm_config",
                RmsNormConfig(
                    use_weight=self.use_layer_norm_weight, use_bias=self.use_bias, eps=self.layer_norm_epsilon
                ),
            )
        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads={self.num_heads} not divisible by num_kv_heads={self.num_kv_heads}."
        )

    # Axis helpers
    Pos = property(lambda self: Axis(name="position", size=self.seq_len))
    KeyPos = property(lambda self: self.Pos.alias("key_position"))
    Embed = property(lambda self: Axis(name="embed", size=self.hidden_dim))
    Layers = property(lambda self: Axis(name="layers", size=self.num_layers))
    Mlp = property(lambda self: Axis(name="mlp", size=self.intermediate_dim))

    @property
    def model_type(self) -> Type["ModularLMHeadModel"]:
        return ModularLMHeadModel

    def mk_LayerNorm(self, axis: AxisSpec):
        assert self.norm_config is not None
        return self.norm_config.build(axis)

    def attention_config(self) -> AttentionConfig:
        """Build an :class:`AttentionConfig` for the attention module."""

        return AttentionConfig(
            Embed=self.Embed,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            use_bias=self.use_bias,
            use_output_bias=self.use_bias,
            upcast_attn=self.upcast_attn,
            attn_backend=self.attn_backend,
            flash_attention_block_size=self.flash_attention_block_size,
            rope=self.rope,
        )

    def flops_per_token(self, vocab_size: int):
        logger.warning("FLOPs per token is likely extremely approximate for custom choices in Modular Transformer.")
        return lm_flops_per_token(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            seq_len=self.seq_len,
            vocab_size=vocab_size,
            glu=True,
        )

    def total_trainable_params(self, vocab_size: int):  # type: ignore[override]
        logger.warning("Trainable params is likely extremely approximate for custom choices in Modular Transformer.")
        token_embedding = vocab_size * self.hidden_dim

        head_size = self.hidden_dim // self.num_heads
        q_proj = self.hidden_dim * head_size * self.num_heads
        kv_proj = 2 * self.hidden_dim * head_size * self.num_kv_heads
        o_proj = head_size * self.num_heads * self.hidden_dim
        attn = q_proj + kv_proj + o_proj

        mlp = 3 * self.hidden_dim * self.intermediate_dim

        transformer_layer = attn + mlp + 2 * self.hidden_dim
        transformer = self.num_layers * transformer_layer + self.hidden_dim
        if self.input_embedding_norm:
            transformer += self.hidden_dim

        return transformer + token_embedding * 2


class ModularTransformer(eqx.Module):
    """A stack of decoder layers defined by :class:`ModularConfig`."""

    config: ModularConfig = eqx.field(static=True)
    layers: BlockFoldable[ModularDecoderLayer]
    norm: hnn.RmsNorm

    @staticmethod
    def init(config: ModularConfig, *, key) -> "ModularTransformer":
        S = Stacked
        if not config.scan_layers:
            from haliax.nn.scan import BlockSeq

            S = BlockSeq

        layers = S.init(config.Layers, config.layer_cls, gradient_checkpointing=config.gradient_checkpointing)(
            config,
            key=shaped_rng_split(key, config.num_layers),
        )
        ln_f = config.mk_LayerNorm(config.Embed)
        return ModularTransformer(config, layers, ln_f)

    @named_call
    def __call__(
        self, x: NamedArray, attn_mask: Optional[NamedArray | AttentionMask], *, key, pos_ids: NamedArray | None = None
    ) -> NamedArray:
        keys = maybe_rng_split(key, self.config.num_layers) if key is not None else None
        x = self.layers.fold(x, mask=attn_mask, key=keys, pos_ids=pos_ids)
        x = self.norm(x)
        return x


class ModularEmbedding(ModuleWithStateDictSerialization, eqx.Module):
    """Token embedding layer with optional normalization."""

    token_embeddings: hnn.Embedding
    norm: Optional[hnn.RmsNorm] = None

    @staticmethod
    def init(Vocab: Axis, config: ModularConfig, *, key) -> "ModularEmbedding":
        token_embeddings = hnn.Embedding.init(Vocab, config.Embed, key=key)
        norm = config.mk_LayerNorm(config.Embed) if config.input_embedding_norm else None
        return ModularEmbedding(token_embeddings, norm)

    @property
    def Vocab(self) -> Axis:
        return self.token_embeddings.Vocab

    @property
    def Embed(self) -> Axis:
        return self.token_embeddings.Embed

    @named_call
    def embed(self, input_ids):
        input_embeds = self.token_embeddings(input_ids)
        if self.norm is not None:
            input_embeds = self.norm(input_embeds)
        return input_embeds

    def unembed(self, x: NamedArray):
        return self.token_embeddings.unembed(x)

    def resize_embeddings(self, new_size: int, key: Optional[PRNGKeyArray] = None):
        new_weights = self.token_embeddings.resize_embeddings(new_size, key=key)
        return dataclasses.replace(self, token_embeddings=new_weights)


class ModularLMHeadModel(ModuleWithStateDictSerialization, LmHeadModel[ModularConfig]):
    """A language model with a configurable Transformer backbone."""

    transformer: ModularTransformer
    embeddings: ModularEmbedding
    lm_head: Optional[hnn.Linear]

    @property
    def config(self):
        return self.transformer.config

    @property
    def vocab_size(self) -> int:
        return self.Vocab.size

    @property
    def Vocab(self) -> Axis:
        return self.embeddings.Vocab

    @classmethod
    def init(cls, Vocab: Axis, config: ModularConfig, *, key) -> "ModularLMHeadModel":
        k_e, k_t, k_lm = jrandom.split(key, 3)
        embeddings = ModularEmbedding.init(Vocab, config, key=k_e)
        transformer = ModularTransformer.init(config, key=k_t)
        lm_head = None
        if not config.tie_word_embeddings:
            lm_head = hnn.Linear.init(config.Embed, Vocab, key=k_lm, use_bias=config.use_bias)
        return ModularLMHeadModel(transformer, embeddings, lm_head)

    @named_call
    def __call__(self, input_ids, attn_mask, *, key=None, pos_ids=None):
        x = self.embeddings.embed(input_ids)
        x = self.transformer(x, attn_mask, key=key, pos_ids=pos_ids)
        if self.lm_head is None:
            logits = self.embeddings.unembed(x)
        else:
            logits = self.lm_head(x)
        return logits

    def activations(
        self,
        input_ids: NamedArray,
        attn_mask: Optional[AttentionMask | NamedArray] = None,
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        x = self.embeddings.embed(input_ids)
        x = self.transformer(x, attn_mask=attn_mask, key=key, pos_ids=pos_ids)
        return x

    def get_lm_head(self) -> hax.NamedArray:
        if self.lm_head is None:
            return self.embeddings.token_embeddings.weight
        else:
            return self.lm_head.weight

    def resize_vocab(self, new_size: int, key: Optional[PRNGKeyArray] = None) -> "LmHeadModel[ModularConfig]":
        new_Vocab = self.Vocab.resize(new_size)
        k1, k2 = maybe_rng_split(key, 2)
        new_embeddings = self.embeddings.resize_embeddings(new_size, key=k1)
        if self.lm_head is not None:
            new_lm_matrix = hax.tree_util.resize_axis(self.lm_head.weight, self.Vocab, new_size, key=k2)
            new_lm_head = dataclasses.replace(self.lm_head, Out=new_Vocab, weight=new_lm_matrix)
            return dataclasses.replace(self, embeddings=new_embeddings, lm_head=new_lm_head)
        else:
            return dataclasses.replace(self, embeddings=new_embeddings)
