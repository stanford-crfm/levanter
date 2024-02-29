import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Optional, Type

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import PRNGKeyArray

import haliax as hax
import haliax.jax_utils
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import named_call, shaped_rng_split
from haliax.nn.scan import Stacked

from levanter.compat.hf_checkpoints import HFCheckpointConverter, HFCompatConfig, LmWithHfSerializationMixin
from levanter.compat.torch_serialization import (
    StateDict,
    StateDictSerializationMixin,
    apply_prefix,
    flatten_linear_layers,
    stack_state_dict,
    unflatten_linear_layers,
    unstack_state_dict,
)
from levanter.logging import silence_transformer_nag
from levanter.models.attention import AttentionMask, dot_product_attention
from levanter.models.lm_model import LmConfig
from levanter.utils.py_utils import cached_classproperty


silence_transformer_nag()
from transformers import PretrainedConfig as HfConfig  # noqa: E402
from transformers import WhisperConfig as HfWhisperConfig  # noqa: E402
from transformers import WhisperFeatureExtractor  # noqa: E402


class WhisperMlp(eqx.Module):
    c_fc: hnn.Linear  # projection from Embed to Intermediate (typically 4x Embed)
    c_proj: hnn.Linear  # projection from Intermediate to Embed
    act: Callable = eqx.static_field()

    @staticmethod
    def init(Embed: Axis, Mlp: Axis, activation_fn, *, key, use_bias: bool = True) -> "WhisperMlp":
        k_fc, k_proj = jrandom.split(key, 2)
        c_fc = hnn.Linear.init(Out=Mlp, In=Embed, key=k_fc, use_bias=use_bias)
        c_proj = hnn.Linear.init(Out=Embed, In=Mlp, key=k_proj, use_bias=use_bias)
        if isinstance(activation_fn, str):
            activation_fn = ACT2FN[activation_fn]
        act = activation_fn  # type: ignore

        return WhisperMlp(c_fc, c_proj, act)

    @named_call
    def __call__(self, x: NamedArray, *, key=None):
        k1, k2 = haliax.jax_utils.maybe_rng_split(key, 2)
        x = self.c_fc(x, key=k1)
        x = self.act(x)
        x = self.c_proj(x, key=k2)
        return x


class WhisperAttention(StateDictSerializationMixin, eqx.Module):
    config: WhisperConfig = eqx.static_field()

    # Unlike other attention blocks, splits query to support cross-attn
    q_lin: hnn.Linear  # input projection from [embed] -> [(q, k, v), heads, head_dim]
    kv_lin: hnn.Linear  # input projection from [embed] -> [(q, k, v), heads, head_dim]

    c_proj: hnn.Linear  # output projection from [heads, head_dim] -> [embed]
    inference: bool

    @staticmethod
    def init(config: WhisperConfig, *, key) -> "WhisperAttention":
        Kv = Axis("kv", size=3)
        use_bias = config.use_bias
        Embed = config.Embed

        k_c, k_proj = jrandom.split(key, 2)
        q_lin = hnn.Linear.init(In=Embed, Out=(config.Heads, config.HeadSize), key=k_c, use_bias=use_bias)
        kv_lin = hnn.Linear.init(In=Embed, Out=(Kv, config.Heads, config.HeadSize), key=k_c, use_bias=use_bias)
        c_proj = hnn.Linear.init(In=(config.Heads, config.HeadSize), Out=Embed, key=k_proj, use_bias=use_bias)

        return WhisperAttention(config, c_attn, c_proj, inference=False)

    @named_call
    def __call__(
        self, x: NamedArray, xa: Optional[NamedArray], mask: Optional[AttentionMask | NamedArray], layer_idx, *, key
    ):
        k_kv, k_q, k_out = hax.jax_utils.maybe_rng_split(key, 3)
        q = self.q_lin(x, key=k_q).rearrange((..., "heads", "position", "head_size"))
        kv_in = x if xa is None else xa
        kv_out = self.kv_lin(kv_in, key=k_kv).rearrange((..., "kv", "heads", "position", "head_size"))
        k, v = kv_out.unbind("kv")

        # Rename k and v's Pos as haliax doesn't support unnamed axes or duplicate axes
        k = k.rename({"position": "key_position"})
        v = v.rename({"position": "key_position"})

        # attention scores can overflow FP16, or just be too imprecise, so upcast to FP32
        if self.config.scale_attn_by_inverse_layer_idx:
            q = q / (layer_idx + 1.0)

        attn_output = dot_product_attention(
            "position",
            "key_position",
            "head_size",
            q,
            k,
            v,
            mask=mask,
            inference=self.inference,
            use_flash=self.config.use_flash_attention,
            flash_block_size=self.config.flash_attention_block_size,
            prng=k_drop,
            attention_dtype=jnp.float32 if self.config.upcast_attn else None,
        )
        attn_output = self.c_proj(attn_output, key=k_out)

        if self.config.upcast_attn:
            attn_output = attn_output.astype(x.dtype)

        return attn_output


class WhisperBlock(eqx.Module):
    attn: WhisperAttention
    attn_ln: hnn.LayerNorm

    cross_attn: Optional[WhisperAttention]
    cross_attn_ln: Optional[hnn.LayerNorm]

    mlp: WhisperMlp
    mlp_ln: hnn.LayerNorm

    @staticmethod
    def init(has_cross: bool, config: WhisperConfig, *, key) -> "WhisperBlock":
        k_attn, k_cross, k_mlp = jrandom.split(key, 3)

        attn_ln = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)
        attn = WhisperAttention.init(config, key=k_attn)

        if has_cross:
            cross_attn_ln = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)
            cross_attn = WhisperAttention.init(config, key=k_attn)
        else:
            cross_attn_ln = None
            cross_attn = None

        mlp = WhisperMlp.init(
            config.Embed, config.Mlp, config.activation_function, key=k_mlp, use_bias=config.use_bias
        )
        cross_attn_ln = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)

        return WhisperBlock(attn, attn_ln, cross_attn, cross_attn_ln, mlp, mlp_ln)

    @named_call
    def __call__(self, x: NamedArray, xa: NamedArray, mask: Optional[AttentionMask | NamedArray], layer_idx, *, key):
        k1, k2, k3 = haliax.jax_utils.maybe_rng_split(key, 3)

        attn_output = self.attn(self.ln_1(x), mask=mask, layer_idx=layer_idx, key=k1)
        x = x + attn_output

        if self.cross_attn:
            cross_attn_output = self.cross_attn(self.cross_attn_ln(x), xa, layer_idx=layer_idx, key=k2)
            x = x + cross_attn_output

        ff_output = self.mlp(self.ln_2(x), key=k3)
        x = x + ff_output

        return x


class WhisperTransformer(eqx.Module):
    config: WhisperConfig = eqx.static_field()
    blocks: Stacked[WhisperBlock]
    ln_f: hnn.LayerNorm

    @staticmethod
    def init(config: WhisperConfig, *, key):
        # vectorize the blocks
        blocks = Stacked.init(config.Layers, WhisperBlock, gradient_checkpointing=config.gradient_checkpointing)(
            config,
            key=shaped_rng_split(key, config.num_layers),
        )
        ln_f = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)

        return WhisperTransformer(config, blocks, ln_f)

    @named_call
    def __call__(self, x: NamedArray, attn_mask: Optional[AttentionMask | NamedArray], *, key=None) -> NamedArray:
        keys = hax.jax_utils.maybe_rng_split(key, self.config.num_layers) if key is not None else None
        x = self.blocks.fold(x, attn_mask, hax.arange(self.config.Layers), key=keys)
        x = self.ln_f(x)

        return x


class WhisperSinusoidEmbedding(eqx.Module, StateDictSerializationMixin):
    Pos: Axis = eqx.field(static=True)
    pos_emb: NamedArray

    def __init__(self, Channels: Axis, Pos: Axis, base: int = 10000):
        self.Pos = Pos
        # this must be compile-time b/c we want to store them in a static field
        with jax.ensure_compile_time_eval():
            self.pos_emb = self._get_pos_emb(Pos=Pos, Channels=Channels, base=base)

    @staticmethod
    def _get_pos_emb(Channels: Axis, Pos: Axis, base: int) -> Tuple[NamedArray, NamedArray]:
        log_timescale_increment = hax.log(base) / (Channels // 2 - 1)
        inv_timescales = hax.exp(-log_timescale_increment * hax.arange(channels // 2))

        position_ids: NamedArray = hax.arange(Pos)

        freqs = position_ids * inv_timescales.broadcast_axis(Pos)
        emb = hax.concatenate(HeadSize, (hax.sin(freqs), hax.cos(freqs)))
        return emb

    def __call__(self, seq_len: int) -> Tuple[NamedArray, NamedArray]:
        return jax.lax.stop_gradient((self.pos_emb[self.Pos, :seq_len],))

    # TODO: maybe add a "persistent" option to eqx.field that we use for state dict serialization
    # if we do that, consider moving the key remapping stuff there too?
    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None):
        return self

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        return state_dict


class WhisperEncoder(eqx.Module):
    conv1: hnn.Conv
    conv2: hnn.Conv
    act: Callable = eqx.static_field()
    pos_emb = WhisperSinusoidEmbedding

    transformer: WhisperTransformer

    @property
    def config(self):
        return self.transformer.config

    @property
    def Vocab(self) -> Axis:
        return self.embeddings.Vocab

    @property
    def Pos(self) -> Axis:
        return self.config.Pos

    @classmethod
    def init(cls, Vocab: Axis, config: WhisperConfig, *, key) -> "WhisperDecoder":
        k_conv1, k_conv2 = jrandom.split(key, 2)
        conv1 = hnn.Conv.init(config.Mels, config.Mels, config.State, kernel_size=3, padding=1, key=k_conv1)
        conv_2 = hnn.Conv.init(config.Mels, config.Mels, config.State, kernel_size=3, stride=2, padding=1, key=k_conv2)
        if isinstance(activation_fn, str):
            activation_fn = ACT2FN[activation_fn]
        act = activation_fn  # type: ignore
        pos_emb = WhisperSinusoidEmbedding.init(config.Ctx, config.State)
        transformer = WhisperTransformer.init(config, key=k_t)

        return WhisperEncoder(conv1, conv2, act, pos_emb, transformer)

    def __call__(self, spec: NamedArray, *, key=None) -> NamedArray:
        k_conv1, k_conv2, k_transformer = jrandom.split(key, 3)
        x = self.act(self.conv1(spec, key=k_conv1))
        x = self.act(self.conv2(x, key=k_conv2))
        x = x.rearrange((..., "PLACEHOLDER", "PLACEHOLDER"))

        x = x + self.pos_emb(x.axis_size("position"))
        x = self.transformer(x, key=k_transformer)
        return x

    def resize_vocab(self, new_size: int, key: Optional[PRNGKeyArray] = None) -> "WhisperDecoder":
        new_embeddings = self.embeddings.resize_embeddings(new_size, key=key)
        return dataclasses.replace(self, embeddings=new_embeddings)


class WhisperDecoderEmbeddings(eqx.Module):
    Vocab: Axis = eqx.static_field()
    config: WhisperConfig = eqx.static_field()

    token_embeddings: hnn.Embedding
    position_embeddings: hnn.Embedding

    @staticmethod
    def init(Vocab: Axis, config: WhisperConfig, *, key) -> "WhisperEmbeddings":
        k_wte, k_wpe, k_out = jrandom.split(key, 3)

        token_embeddings = hnn.Embedding.init(
            Vocab, config.Embed, key=k_wte, initializer_range=config.initializer_range
        )

        # Whisper Initializes the Positional Embeddings as Empty
        position_embeddings = hnn.Embedding.init(config.Pos, config.Embed, key=k_wpe, initializer_range=0)

        return WhisperDecoderEmbeddings(Vocab, config, token_embeddings, position_embeddings, dropout)

    @named_call
    def embed(self, input_ids, *, key):
        input_embeds = self.token_embeddings(input_ids)
        input_Pos = input_ids.resolve_axis("position")
        position_embeds = self.position_embeddings.embed(hax.arange(input_Pos))
        x = input_embeds + position_embeds

        return x

    def unembed(self, x: NamedArray):
        return hax.dot("embed", x, self.token_embeddings.weight)

    def resize_embeddings(self, new_size: int, key: Optional[PRNGKeyArray] = None):
        new_token_embeddings = self.token_embeddings.resize_embeddings(new_size, key=key)
        return dataclasses.replace(self, Vocab=self.Vocab.resize(new_size), token_embeddings=new_token_embeddings)


class WhisperDecoder(eqx.Module):
    transformer: WhisperTransformer
    embeddings: WhisperDecoderEmbeddings

    @property
    def config(self):
        return self.transformer.config

    @property
    def Vocab(self) -> Axis:
        return self.embeddings.Vocab

    @property
    def Pos(self) -> Axis:
        return self.config.Pos

    @classmethod
    def init(cls, Vocab: Axis, config: WhisperConfig, *, key) -> "WhisperDecoder":
        k_t, k_embeddings = jrandom.split(key, 2)
        transformer = WhisperTransformer.init(config, key=k_t)
        embeddings = WhisperDecoderEmbeddings.init(Vocab, config, key=k_embeddings)

        return WhisperDecoder(transformer, embeddings)

    def __call__(
        self, input_ids: NamedArray, attn_mask: Optional[AttentionMask | NamedArray] = None, *, key=None
    ) -> NamedArray:
        k_embed, k_transformer = haliax.jax_utils.maybe_rng_split(key, 2)
        x = self.embeddings.embed(input_ids, key=k_embed)
        x = self.transformer(x, attn_mask, key=k_transformer)
        lm_logits = self.embeddings.unembed(x)

        return lm_logits

    def resize_vocab(self, new_size: int, key: Optional[PRNGKeyArray] = None) -> "WhisperDecoder":
        new_embeddings = self.embeddings.resize_embeddings(new_size, key=key)
        return dataclasses.replace(self, embeddings=new_embeddings)


class WhisperModel(eqx.Module):
    audio_encoder: WhisperEncoder
    text_decoder: WhisperDecoder

    @property
    def Vocab(self) -> Axis:
        return self.text_decoder.Vocab

    @property
    def Pos(self) -> Axis:
        return self.config.Pos

    @classmethod
    def init(cls, Vocab: Axis, config: WhisperConfig, *, key) -> "WhisperModel":
        k_t, k_embeddings = jrandom.split(key, 2)
        encoder = WhisperEncoder.init(config, key=k_embeddings)
        decpder = WhisperDecoder.init(Vocab, config, key=k_t)

        return WhisperModel(encoder, decoder)

    def __call__(
        self,
        mel: NamedArray,
        input_ids: NamedArray,
        attn_mask: Optional[AttentionMask | NamedArray] = None,
        *,
        key=None,
    ) -> NamedArray:
        k_encoder, k_decoder = haliax.jax_utils.maybe_rng_split(key, 2)
        audio_features = self.encoder(mel, key=k_encoder)
        lm_logits = self.decoder(attn_mask, audio_features, key=k_decoder)

        return lm_logits

    def resize_vocab(self, new_size: int, key: Optional[PRNGKeyArray] = None) -> "WhisperModel":
        new_embeddings = self.embeddings.resize_embeddings(new_size, key=key)
        return dataclasses.replace(self, embeddings=new_embeddings)


ACT2FN: Dict[str, Callable] = {
    "relu": hnn.relu,
    "silu": hnn.silu,
    "swish": hnn.swish,
    "gelu": partial(hnn.gelu, approximate=False),
    "gelu_new": partial(hnn.gelu, approximate=True),
    "quick_gelu": hnn.quick_gelu,
}
