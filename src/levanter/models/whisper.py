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
from transformers import WhisperFeatureExtractor  # noqa: E402
from transformers import WhisperConfig as HfWhisperConfig  # noqa: E402
from transformers import PretrainedConfig as HfConfig  # noqa: E402


class Gpt2Block(StateDictSerializationMixin, eqx.Module):
    attn: Gpt2Attention
    attn_ln: hnn.LayerNorm

    cross_attn: Gpt2Attention
    cross_attn_ln: hnn.LayerNorm

    mlp: Gpt2Mlp
    mlp_ln: hnn.LayerNorm

    @staticmethod
    def init(config: Gpt2Config, *, key) -> "Gpt2Block":
        k_attn, k_cross, k_mlp = jrandom.split(key, 3)

        ln_1 = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)
        attn = Gpt2Attention.init(config, key=k_attn)
        ln_2 = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)
        mlp = Gpt2Mlp.init(config.Embed, config.Mlp, config.activation_function, key=k_mlp, use_bias=config.use_bias)
        resid_dropout = hnn.Dropout(pdrop=config.resid_pdrop)

        return Gpt2Block(ln_1, attn, ln_2, mlp, resid_dropout)

    @named_call
    def __call__(self, x: NamedArray, mask: Optional[AttentionMask | NamedArray], layer_idx, *, key):
        k1, k2, k3, k4 = haliax.jax_utils.maybe_rng_split(key, 4)

        attn_output = self.attn(self.ln_1(x), mask=mask, layer_idx=layer_idx, key=k1)
        attn_output = self.resid_dropout(attn_output, key=k2)
        x = x + attn_output

        ff_output = self.mlp(self.ln_2(x), key=k3)
        ff_output = self.resid_dropout(ff_output, key=k4)
        x = x + ff_output

        return x


class WhisperTransformer(StateDictSerializationMixin, eqx.Module):
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

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"blocks": "h"}

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None):
        stacked = stack_state_dict(state_dict, prefix=apply_prefix(prefix, "h"))
        out = super().from_state_dict(stacked, prefix=prefix)
        return out

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        my_state_dict: StateDict = {}
        super().update_state_dict(my_state_dict, prefix)

        stacked_dict = unstack_state_dict(my_state_dict, apply_prefix(prefix, "h"))
        state_dict.update(stacked_dict)

        return state_dict


class WhisperDecoderEmbeddings(StateDictSerializationMixin, eqx.Module):
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

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"token_embeddings": "wte", "position_embeddings": "wpe"}

    def resize_embeddings(self, new_size: int, key: Optional[PRNGKeyArray] = None):
        new_token_embeddings = self.token_embeddings.resize_embeddings(new_size, key=key)
        return dataclasses.replace(self, Vocab=self.Vocab.resize(new_size), token_embeddings=new_token_embeddings)


class WhisperDecoder(eqx.Module, LmWithHfSerializationMixin[WhisperConfig]):
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

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"transformer": None, "embeddings": None}


class WhisperModel(eqx.Module, LmWithHfSerializationMixin[WhisperConfig]):
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

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"encoder": None, "decoder": None}


ACT2FN: Dict[str, Callable] = {
    "relu": hnn.relu,
    "silu": hnn.silu,
    "swish": hnn.swish,
    "gelu": partial(hnn.gelu, approximate=False),
    "gelu_new": partial(hnn.gelu, approximate=True),
    "quick_gelu": hnn.quick_gelu,
}
