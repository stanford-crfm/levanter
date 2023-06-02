from dataclasses import dataclass
from typing import Callable, Dict, Optional, Union, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

import haliax as hax
import haliax.jax_utils
import haliax.nn as hnn
from haliax import Axis, AxisSpec, NamedArray
from haliax.jax_utils import named_call
from levanter.compat.torch_serialization import (
    StateDict,
    StateDictSerializationMixin,
    apply_prefix,
    reshape_linear_layer,
    reshape_mlp_linear_layer,
)
from levanter.models.gpt2 import ACT2FN, Gpt2Config, Gpt2Embeddings, Gpt2Mlp, Gpt2Transformer


@dataclass(frozen=True)
class BackpackConfig(Gpt2Config):
    # Backpack-specific terms
    num_senses: int = 16
    sense_intermediate_scale: int = 4

    # Axes
    SenseHeadDim = property(lambda self: Axis(name="head_dim", size=self.hidden_dim // self.num_senses))
    Senses = property(lambda self: Axis(name="senses", size=self.num_senses))
    SenseIntermediate = property(
        lambda self: Axis(name="concat_senses", size=self.sense_intermediate_scale * self.hidden_dim)
    )


class BackpackMlp(StateDictSerializationMixin, Gpt2Mlp):
    @staticmethod
    def init(
        Embed: Axis,
        Mlp: Axis,
        Out: AxisSpec,
        activation_fn: Union[str, Callable],
        *,
        key,
        use_bias: bool = True,
    ) -> "BackpackMlp":
        k_fc, k_proj = jrandom.split(key, 2)
        c_fc = hnn.Linear.init(Out=Mlp, In=Embed, key=k_fc, use_bias=use_bias)
        c_proj = hnn.Linear.init(Out=Out, In=Mlp, key=k_proj, use_bias=use_bias)
        if isinstance(activation_fn, str):
            activation_fn = ACT2FN[activation_fn]
        act = activation_fn  # type: ignore

        return BackpackMlp(c_fc=c_fc, c_proj=c_proj, act=act)

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> "BackpackMlp":
        # our c_attn is [embed] -> [3, heads, head_dim] and hf's is the flattened [embed] -> [3 * heads * head_dim]
        # so we need to reshape the one in the dict before forwarding to the linear
        # keep in mind that everything is vectorized in our implementation, so there's a leading num_layers dim

        d = {}
        d.update(
            reshape_mlp_linear_layer(
                state_dict, apply_prefix(prefix, "c_proj"), (self.c_proj.In.size,), (self.c_proj.Out.size,)
            )
        )
        d.update(
            reshape_mlp_linear_layer(
                state_dict, apply_prefix(prefix, "c_fc"), (self.c_fc.In.size,), (self.c_fc.Out.size,)
            )
        )
        return super().from_state_dict(d, prefix)

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        my_dict: StateDict = {}
        super().update_state_dict(my_dict, prefix)

        my_dict.update(
            reshape_mlp_linear_layer(
                my_dict, apply_prefix(prefix, "c_proj"), (self.c_proj.In.size,), (self.c_proj.Out.size,)
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
    """
    Changes from Gpt2Attention:
    1. No projection; it returns the attention weights
    2. Use SenseHeadDim instead of HeadDim, use Senses instead of Heads
    """

    # No projection
    config: Gpt2Config = eqx.static_field()

    c_attn: hnn.Linear  # input projection from [embed] -> [(q, k, v), heads, head_dim]
    dropout: hnn.Dropout
    ln: hnn.LayerNorm

    @staticmethod
    def init(config: Gpt2Config, *, key) -> "WeightsOnlyAttention":
        Qk = Axis("qk", size=2)
        use_bias = config.use_bias
        Embed = config.Embed

        k_c, _ = jrandom.split(key, 2)
        c_attn = hnn.Linear.init(In=Embed, Out=(Qk, config.Senses, config.SenseHeadDim), key=k_c, use_bias=use_bias)
        dropout = hnn.Dropout(config.attn_pdrop)
        ln = hnn.LayerNorm.init(config.Senses, eps=config.layer_norm_epsilon)

        return WeightsOnlyAttention(config, c_attn, dropout, ln)

    @named_call
    def __call__(self, x: NamedArray, mask: Optional[NamedArray], layer_idx, inference: bool = True, *, key):
        qk_out = self.c_attn(x)
        q, k = qk_out.unbind("qk")

        # Rename k's Pos as haliax doesn't support unnamed axes or duplicate axes
        k = k.rename({"position": "key_position"})

        # mistral tweak: scale norms by 1/sqrt(layer_idx) to prevent blowup
        scale = jax.lax.rsqrt(float(self.config.SenseHeadDim.size))

        # do this first to help keep FP values small
        q = q * scale

        # mistral tweak: attention scores can overflow FP16, or just be too imprecise, so upcast to FP32
        if self.config.upcast_attn:
            q = q.astype(jnp.float32)
            k = k.astype(jnp.float32)

        attn_scores = hax.dot("head_dim", q, k)

        if mask is not None:
            attn_scores = attn_scores + (1.0 - mask) * -1e15

        attn_weights = hnn.softmax(attn_scores, axis="key_position").astype(x.dtype)
        attn_weights = self.dropout(attn_weights, key=key, inference=inference)
        attn_weights = self.ln(attn_weights)
        return attn_weights

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> "WeightsOnlyAttention":
        es = cast(Axis, self.c_attn.In).size
        d = {}
        num_heads = self.config.Senses.size
        sense_head_size = self.config.SenseHeadDim.size
        d.update(
            reshape_linear_layer(state_dict, apply_prefix(prefix, "c_attn"), (es,), (2, num_heads, sense_head_size))
        )
        return super().from_state_dict(d, prefix)

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        # need to undo the reshape we did in from_state_dict
        # reminder that everything is vectorized
        my_dict: StateDict = {}
        super().update_state_dict(my_dict, prefix)

        es = cast(Axis, self.c_attn.In).size
        num_heads = self.config.Senses.size
        sense_head_size = self.config.SenseHeadDim.size

        my_dict.update(
            reshape_linear_layer(my_dict, apply_prefix(prefix, "c_attn"), (es,), (2 * num_heads * sense_head_size,))
        )

        state_dict.update(my_dict)
        return state_dict


class NoMixBlock(StateDictSerializationMixin, eqx.Module):
    ln_1: hnn.LayerNorm
    ln_2: hnn.LayerNorm
    mlp: BackpackMlp
    resid_dropout1: hnn.Dropout
    resid_dropout2: hnn.Dropout

    @staticmethod
    def init(config: BackpackConfig, *, key) -> "NoMixBlock":
        k_mlp = jrandom.split(key, 1)[0]

        ln_1 = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon)
        resid_dropout1 = hnn.Dropout(pdrop=config.resid_pdrop)
        resid_dropout2 = hnn.Dropout(pdrop=config.resid_pdrop)
        ln_2 = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon)

        mlp = BackpackMlp.init(
            Embed=config.Embed,
            Mlp=config.Mlp,
            Out=config.Embed,
            activation_fn=config.activation_function,
            key=k_mlp,
            use_bias=config.use_bias,
        )

        return NoMixBlock(ln_1=ln_1, ln_2=ln_2, mlp=mlp, resid_dropout1=resid_dropout1, resid_dropout2=resid_dropout2)

    @named_call
    def __call__(self, hidden_states: NamedArray, residual: NamedArray, inference, *, key):
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

    Pos: Axis = eqx.static_field()
    ln_2: hnn.LayerNorm

    @staticmethod
    def init(
        config,
        dropout_prob: float,
        *,
        key,
    ):
        k_block, k_mlp = jrandom.split(key, 2)

        dropout = hnn.Dropout(pdrop=dropout_prob)
        block = NoMixBlock.init(config, key=k_block)
        ln = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon)
        final_mlp = BackpackMlp.init(
            Embed=config.Embed,
            Mlp=config.SenseIntermediate,
            Out=(config.Senses, config.Embed),
            activation_fn=config.activation_function,
            key=k_mlp,
            use_bias=config.use_bias,
        )
        ln_2 = hnn.LayerNorm.init(config.Senses, eps=config.layer_norm_epsilon)

        return BackpackSenses(
            dropout=dropout,
            block=block,
            ln=ln,
            final_mlp=final_mlp,
            Pos=config.Pos,
            ln_2=ln_2,
        )

    @named_call
    def sense_embed(self, input_embeds, inference, *, key):
        hidden_states = self.ln(input_embeds)
        hidden_states = self.block(hidden_states, input_embeds, inference=inference, key=key)
        senses = self.final_mlp(hidden_states)
        senses = self.ln_2(senses)
        return senses


class BackpackGpt2Embeddings(Gpt2Embeddings):
    """
    We want to re-use the Gpt2Embeddings class, but we need to add a new method to only embed the input_ids.
    """

    @staticmethod
    def init(Vocab: Axis, config: Gpt2Config, *, key) -> "BackpackGpt2Embeddings":
        k_wte, k_wpe, k_out = jrandom.split(key, 3)

        token_embeddings = hax.random.normal(k_wte, (Vocab, config.Embed)) * config.initializer_range
        position_embeddings = hax.random.normal(k_wpe, (config.Pos, config.Embed)) * (config.initializer_range / 2)
        dropout = hnn.Dropout(pdrop=config.embed_pdrop)

        return BackpackGpt2Embeddings(Vocab, config, token_embeddings, position_embeddings, dropout)

    @named_call
    def embed_input_ids(self, input_ids: NamedArray) -> NamedArray:
        return self.token_embeddings.take("vocab", input_ids)


class BackpackLMHeadModel(StateDictSerializationMixin, eqx.Module):
    transformer: Gpt2Transformer
    embeddings: BackpackGpt2Embeddings
    sense_net: BackpackSenses
    kq_selfattention: WeightsOnlyAttention
    rescaler

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
    def Pos(self) -> Axis:
        return self.sense_net.Pos

    @staticmethod
    def init(Vocab: Axis, config: BackpackConfig, *, key):
        k_t, k_embeddings, k_attn = jrandom.split(key, 3)
        transformer = Gpt2Transformer.init(config, key=k_t)
        gpt2_config = Gpt2Config(
            hidden_dim=config.hidden_dim,
            seq_len=config.seq_len,
            initializer_range=config.initializer_range,
            embed_pdrop=config.embed_pdrop,
        )
        embeddings = BackpackGpt2Embeddings.init(
            Vocab=Vocab,
            config=gpt2_config,
            key=k_embeddings,
        )
        sense_net = BackpackSenses.init(
            config=config,
            dropout_prob=config.embed_pdrop,
            key=k_embeddings,
        )
        kq_selfattention = WeightsOnlyAttention.init(
            config=config,
            key=k_attn,
        )
        rescaler = jnp.array( 1.0 / config.Senses.size)

        return BackpackLMHeadModel(
            transformer=transformer,
            embeddings=embeddings,
            sense_net=sense_net,
            kq_selfattention=kq_selfattention,
            rescaler=rescaler,
        )

    def __call__(self, input_ids: NamedArray, attn_mask: Optional[NamedArray], *, inference, key):
        if not inference and key is None:
            raise ValueError("key must be provided for training")

        k_embed, k_transformer, k_senses, k_sa = haliax.jax_utils.maybe_rng_split(key, 4)

        # Compute contextualization weights
        hidden_states = self.embeddings.embed(input_ids, inference=inference, key=k_embed)
        hidden_states = self.transformer(hidden_states, attn_mask, inference=inference, key=k_transformer)
        contextualization_weights = self.kq_selfattention(
            hidden_states, mask=attn_mask, inference=inference, layer_idx=self.config.num_layers, key=k_sa
        )  # (seq, seq, senses)

        ## Compute sense vectors
        sense_input_embeds = self.embeddings.embed_input_ids(input_ids)  # (seq, embed
        sense_vectors = self.sense_net.sense_embed(
            sense_input_embeds, inference=inference, key=k_senses
        )  # (seq, senses, embed)
        sense_vectors = sense_vectors.rename({self.Pos: self.config.KeyPos})

        ## Weight-and-sum
        hidden_states = hax.dot(self.config.KeyPos, contextualization_weights, sense_vectors)  # (seq, senses, embed)
        hidden_states = hax.sum(hidden_states, axis=self.config.Senses)

        # Rescale 
        hidden_states = hidden_states * self.rescaler

        lm_logits = self.embeddings.unembed(hidden_states)

        return lm_logits

    def _state_dict_key_map(self) -> Optional[Dict[str, Optional[str]]]:
        return {"transformer": None, "embeddings": None}
