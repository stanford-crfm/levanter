import re
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union, cast
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy

import haliax as hax
import haliax.jax_utils
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import named_call, shaped_rng_split
from levanter.compat.torch_serialization import (
    StateDict,
    StateDictSerializationMixin,
    apply_prefix,
    reshape_linear_layer,
    reshape_mlp_linear_layer,
)
from levanter.models.gpt2 import Gpt2Transformer, Gpt2Embeddings, Gpt2Config


sharded_normal = hax.random.generate_sharded(hax.random.normal)


@dataclass(frozen=True)
class BackpackConfig:
    seq_len: int = 512
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12

    # how much to scale the embedding dim for the mlp layer
    mlp_scale: int = 4

    initializer_range: float = 0.02
    # dropout doesn't really help so we 0 it out by default
    embed_pdrop: float = 0.0
    resid_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    layer_norm_epsilon: float = 1e-5
    activation_function: str = "gelu_new"

    # mistral tweaks:
    scale_attn_by_inverse_layer_idx: bool = True
    upcast_attn: bool = False

    gradient_checkpointing: bool = True  # better to just always use this
    gradient_checkpointing_block_size: int = 5

    use_bias: bool = True

    # Backpack-specific terms
    num_senses: int = 16
    sense_intermediate_scale: int = 4

    # Axes
    @property
    def SeqLen(self) -> Axis:
        return Axis(name="seqlen", size=self.seq_len)

    @property
    def KeySeqLen(self) -> Axis:
        return self.SeqLen.alias(f"key_{self.SeqLen.name}")

    @property
    def Embed(self) -> Axis:
        return Axis(name="embed", size=self.hidden_dim)

    @property
    def Heads(self) -> Axis:
        return Axis(name="heads", size=self.num_heads)

    @property
    def Layers(self) -> Axis:
        return Axis(name="layers", size=self.num_layers)

    @property
    def Mlp(self) -> Axis:
        return Axis(name="mlp", size=self.hidden_dim * 4)

    @property
    def HeadSize(self) -> Axis:
        return Axis(name="head", size=self.hidden_dim // self.num_heads)

    @property
    def SenseHeadSize(self) -> Axis:
        return Axis(name="head", size=self.hidden_dim // self.num_senses)

    # Backpack-specific axes
    @property
    def Senses(self) -> Axis:
      return Axis(name="senses", size=self.num_senses)

    @property
    def SenseIntermediate(self) -> Axis:
      return Axis(name="concat_senses", size=self.sense_intermediate_scale * self.hidden_dim)


class BackpackMlp(StateDictSerializationMixin, eqx.Module):
    act: Callable = eqx.static_field()
    c_fc: hnn.Linear  # projection from Embed to Intermediate (typically 4x Embed)
    c_proj: hnn.Linear  # projection from Intermediate to out
    Out: Union[Axis, List[Axis]] = eqx.static_field()

    def __init__(
        self, Embed: Axis, Intermediate: Axis, Out: Union[Axis, List[Axis]], activation_fn: Union[str, Callable], *, key, use_bias: bool = True
    ):
        k_fc, k_proj = jrandom.split(key, 2)
        self.Out = Out
        self.c_fc = hnn.Linear.init(Out=Intermediate, In=Embed, key=k_fc, use_bias=use_bias)
        self.c_proj = hnn.Linear.init(Out=Out, In=Intermediate, key=k_proj, use_bias=use_bias)
        if isinstance(activation_fn, str):
            activation_fn = ACT2FN[activation_fn]
        self.act = activation_fn  # type: ignore

    @named_call
    def __call__(self, hidden_states: NamedArray):
        hidden_states = hax.auto_sharded(hidden_states)
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> "WeightsOnlyAttention":
        # our c_attn is [embed] -> [3, heads, head_dim] and hf's is the flattened [embed] -> [3 * heads * head_dim]
        # so we need to reshape the one in the dict before forwarding to the linear
        # keep in mind that everything is vectorized in our implementation, so there's a leading num_layers dim

        es = cast(Axis, self.c_proj.In).size
        if isinstance(self.Out, Axis):
          return super().from_state_dict(state_dict, prefix)
          sizes = (es, self.Out.size)
        else:
          sizes = tuple([x.size for x in self.Out])
          #sizes = tuple([x.size for x in [es] + list(self.Out)])
        d = {}
        d.update(
            reshape_mlp_linear_layer(
                state_dict, apply_prefix(prefix, "c_proj"), (es,), sizes
            )
        )
        d.update(
            reshape_mlp_linear_layer(
                state_dict, apply_prefix(prefix, "c_fc"), (self.c_fc.In.size,), (self.c_fc.Out.size,)
            )
        )

        return super().from_state_dict(d, prefix)

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        # need to undo the reshape we did in from_state_dict
        # reminder that everything is vectorized
        my_dict: StateDict = {}
        super().update_state_dict(my_dict, prefix)

        es = cast(Axis, self.c_proj.In).size
        if isinstance(self.Out, Axis):
          out_size = self.Out.size
        else:
          out_size = math.prod([x.size for x in self.Out])
        my_dict.update(
            reshape_mlp_linear_layer(
                my_dict, apply_prefix(prefix, "c_proj"), (self.c_proj.In.size,), (out_size,)
            )
        )
        my_dict.update(
            reshape_mlp_linear_layer(
                my_dict, apply_prefix(prefix, "c_fc"), (self.c_fc.In.size,), (self.c_fc.Out.size,)
            )
        )

        state_dict.update(my_dict)
        return state_dict


class WeightsOnlyAttention(StateDictSerializationMixin, eqx.Module):
    c_attn: hnn.Linear  # input projection from [embed] -> [(q, k, v), heads, head_dim]
    dropout: hnn.Dropout

    SeqLen: Axis = eqx.static_field()
    SenseHeadSize: Axis = eqx.static_field()
    Heads: Axis = eqx.static_field()
    Qkv: Axis = eqx.static_field()
    KeySeqLen: Axis = eqx.static_field()

    # Mistral stability tweaks
    scale_by_inverse_layer_idx: bool = eqx.static_field()
    upcast: bool = eqx.static_field()

    def __init__(
        self,
        SeqLen: Axis,
        KeySeqLen: Axis,
        Embed: Axis,
        Heads: Axis,
        SenseHeadSize: Axis,
        dropout_prob: float,
        scale_by_inverse_layer_idx: bool,
        upcast: bool,
        *,
        key,
        use_bias: bool = True,
    ):
        self.Heads = Heads
        self.SenseHeadSize = SenseHeadSize
        self.SeqLen = SeqLen
        self.Qkv = Axis("qkv", 2)
        self.KeySeqLen = KeySeqLen

        k_c, k_proj = jrandom.split(key, 2)
        self.c_attn = hnn.Linear.init(In=Embed, Out=(self.Qkv, self.Heads, self.SenseHeadSize), key=k_c, use_bias=use_bias)
        self.dropout = hnn.Dropout(dropout_prob)

        self.scale_by_inverse_layer_idx = scale_by_inverse_layer_idx
        self.upcast = upcast

    @named_call
    def __call__(
        self, hidden_states: NamedArray, mask: Optional[NamedArray], layer_idx, inference: bool = True, *, key
    ):
        qkv_out = self.c_attn(hidden_states)
        q, k = qkv_out.unbind(self.Qkv)

        # Rename k and v's SeqLen as haliax doesn't support unnamed axes or duplicate axes
        k = k.rename({self.SeqLen: self.KeySeqLen})

        # mistral tweak: scale norms by 1/sqrt(layer_idx) to prevent blowup
        scale = jax.lax.rsqrt(float(self.SenseHeadSize.size))
        #if self.scale_by_inverse_layer_idx:
        #    scale /= layer_idx + 1.0

        # do this first to help keep FP values small
        q = q * scale

        # mistral tweak: attention scores can overflow FP16, or just be too imprecise, so upcast to FP32
        if self.upcast:
            q = q.astype(jnp.float32)
            k = k.astype(jnp.float32)

        attn_scores = hax.dot(self.SenseHeadSize, q, k)

        if mask is not None:
            attn_scores = attn_scores + (1.0 - mask) * -1e15

        attn_weights = hnn.softmax(attn_scores, axis=self.KeySeqLen).astype(hidden_states.dtype)
        attn_weights = self.dropout(attn_weights, key=key, inference=inference)
        return attn_weights

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> "WeightsOnlyAttention":
        # our c_attn is [embed] -> [3, heads, head_dim] and hf's is the flattened [embed] -> [3 * heads * head_dim]
        # so we need to reshape the one in the dict before forwarding to the linear
        # keep in mind that everything is vectorized in our implementation, so there's a leading num_layers dim

        es = cast(Axis, self.c_attn.In).size
        d = {}
        d.update(
            reshape_mlp_linear_layer(
                state_dict, apply_prefix(prefix, "c_attn"), (es,), (2, self.Heads.size, self.SenseHeadSize.size)
            )
        )


        return super().from_state_dict(d, prefix)

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        # need to undo the reshape we did in from_state_dict
        # reminder that everything is vectorized
        my_dict: StateDict = {}
        super().update_state_dict(my_dict, prefix)

        es = cast(Axis, self.c_attn.In).size
        my_dict.update(
            reshape_mlp_linear_layer(
                my_dict, apply_prefix(prefix, "c_attn"), (es,), (2 * self.Heads.size * self.SenseHeadSize.size,)
            )
        )

        state_dict.update(my_dict)
        return state_dict


class NoMixBlock(StateDictSerializationMixin, eqx.Module):
    ln_1: hnn.LayerNorm
    ln_2: hnn.LayerNorm
    mlp: BackpackMlp
    resid_dropout1: hnn.Dropout
    resid_dropout2: hnn.Dropout

    def __init__(self, config: BackpackConfig, *, key):
        k_attn, k_cross, k_mlp = jrandom.split(key, 3)

        self.ln_1 = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon)
        self.resid_dropout1 = hnn.Dropout(pdrop=config.resid_pdrop)
        self.resid_dropout2 = hnn.Dropout(pdrop=config.resid_pdrop)
        self.ln_2 = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon)

        self.mlp = BackpackMlp(
            Embed=config.Embed,
            Intermediate=config.Mlp,
            Out=config.Embed,
            activation_fn=config.activation_function,
            key=k_mlp,
            use_bias=config.use_bias,
        )

    @named_call
    def __call__(self, hidden_states: NamedArray, residual: NamedArray, inference, *, key):
        #
        k1, k2 = haliax.jax_utils.maybe_rng_split(key, 2)

        residual = self.resid_dropout1(hidden_states, key=k1, inference=inference) + residual
        hidden_states = self.ln_1(residual)
        mlp_out = self.mlp(hidden_states)
        residual = self.resid_dropout2(mlp_out, key=k2, inference=inference) + residual
        hidden_states = self.ln_2(residual)

        return hidden_states


class BackpackSenses(StateDictSerializationMixin, eqx.Module):
    dropout: hnn.Dropout
    block: NoMixBlock
    ln: hnn.LayerNorm
    final_mlp: BackpackMlp

    # axes
    Vocab: Axis = eqx.static_field()
    SeqLen: Axis = eqx.static_field()
    Embed: Axis = eqx.static_field()
    Senses: Axis = eqx.static_field()

    def __init__(
        self,
        Embed: Axis,
        Vocab: Axis,
        SeqLen: Axis,
        initializer_range: float,
        dropout_prob: float,
        embeddings: Gpt2Embeddings,
        config,
        *,
        key,
    ):
        super().__init__()
        k_wte, k_wpe, k_out, k_block, k_mlp = jrandom.split(key, 5)

        self.Vocab = Vocab
        self.SeqLen = SeqLen
        self.Embed = Embed
        self.Senses = config.Senses

        self.dropout = hnn.Dropout(pdrop=dropout_prob)
        self.block = NoMixBlock(config, key=k_block)
        self.ln = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon)
        self.final_mlp = BackpackMlp(
            Embed=config.Embed,
            Intermediate=config.SenseIntermediate,
            Out=(config.Senses, config.Embed),
            activation_fn=config.activation_function,
            key=k_mlp,
            use_bias=config.use_bias,
        )


    @named_call
    def sense_embed(self, input_embeds, inference, *, key):
        hidden_states = self.ln(input_embeds)
        hidden_states = self.block(hidden_states, input_embeds, inference=inference, key=key)
        senses = self.final_mlp(hidden_states)

        return senses



class BackpackLMHeadModel(StateDictSerializationMixin, eqx.Module):
    transformer: Gpt2Transformer
    embeddings: Gpt2Embeddings
    sense_net: BackpackSenses
    kq_selfattention : WeightsOnlyAttention

    @property
    def config(self):
        return self.transformer.config

    @property
    def vocab_size(self) -> int:
        return self.embeddings.Vocab.size

    @property
    def Vocab(self) -> Axis:
        return self.embeddings.Vocab

    @property
    def SeqLen(self) -> Axis:
        return self.embeddings.SeqLen

    def __init__(self, Vocab: Axis, config: BackpackConfig, *, key):
        k_t, k_embeddings, k_attn = jrandom.split(key, 3)
        self.transformer = Gpt2Transformer.init(config, key=k_t)
        gpt2_config = Gpt2Config(
            hidden_dim=config.hidden_dim,
            seq_len=config.seq_len,
            initializer_range=config.initializer_range,
            embed_pdrop=config.embed_pdrop,
        )
        self.embeddings = Gpt2Embeddings.init(
            Vocab=Vocab,
            config=gpt2_config,
            key=k_embeddings,
        )
        self.sense_net = BackpackSenses(
            Vocab=Vocab,
            Embed=config.Embed,
            SeqLen=config.SeqLen,
            initializer_range=config.initializer_range,
            dropout_prob=config.embed_pdrop,
            embeddings=self.embeddings,
            key=k_embeddings,
            config=config,
        )
        self.kq_selfattention = WeightsOnlyAttention(
            SeqLen=config.SeqLen,
            KeySeqLen=config.KeySeqLen,
            Embed=config.Embed,
            Heads=config.Senses,
            SenseHeadSize=config.SenseHeadSize,
            dropout_prob=config.attn_pdrop,
            key=k_attn,
            scale_by_inverse_layer_idx=config.scale_attn_by_inverse_layer_idx,
            upcast=config.upcast_attn,
            use_bias=config.use_bias,
        )

    def __call__(self, input_ids: NamedArray, attn_mask: Optional[NamedArray], *, inference, key):
        if not inference and key is None:
            raise ValueError("key must be provided for training")

        k_embed, k_transformer, k_senses, k_sa = haliax.jax_utils.maybe_rng_split(key, 4)

        # Compute contextualization weights
        hidden_states = self.embeddings.embed(input_ids, inference=inference, key=k_embed)
        hidden_states = self.transformer(hidden_states, attn_mask, inference=inference, key=k_transformer)
        contextualization_weights = self.kq_selfattention(hidden_states, mask=attn_mask, inference=inference, layer_idx = self.config.num_layers, key=k_sa) # (seq, seq, senses)

        ## Compute sense vectors
        sense_input_embeds  = self.embeddings.embed(input_ids, inference=None, key=None, input_embeds_only=True) # (seq, embed
        sense_vectors = self.sense_net.sense_embed(sense_input_embeds, inference=inference, key=k_senses) # (seq, senses, embed)
        sense_vectors = sense_vectors.rename({self.SeqLen: self.kq_selfattention.KeySeqLen})

        ## Weight-and-sum
        hidden_states = hax.dot(self.kq_selfattention.KeySeqLen, contextualization_weights, sense_vectors) #(seq, senses, embed)
        hidden_states = hax.sum(hidden_states, axis=self.config.Senses)
        # divide by 1/senses
        scale = self.config.Senses.size
        hidden_states = hidden_states / scale

        lm_logits = self.embeddings.unembed(hidden_states) 

        return lm_logits

    def _state_dict_key_map(self) -> Optional[Dict[str, Optional[str]]]:
        return {"transformer": None, "embeddings": None}


ACT2FN: Dict[str, Callable] = {
    "relu": hnn.relu,
    "silu": hnn.silu,
    "swish": hnn.swish,
    "gelu": partial(hnn.gelu, approximate=False),
    "gelu_new": partial(hnn.gelu, approximate=True),
    "quick_gelu": hnn.quick_gelu,
}
