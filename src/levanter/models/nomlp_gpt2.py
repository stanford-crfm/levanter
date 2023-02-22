from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

import haliax as hax
import haliax.jax_utils
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import named_call, shaped_rng_split
from levanter.models.gpt2 import Gpt2Config, Gpt2Embeddings


sharded_normal = hax.random.generate_sharded(hax.random.normal)


class NoMlpGpt2Attention(eqx.Module):
    c_k: hnn.Linear  # input projection from [embed] -> [heads, head_dim]
    c_q: hnn.Linear  # input projection from [embed] -> [heads, head_dim]
    c_v_ff: hnn.Linear  # input projection from [embed] -> [heads, v_dim]
    c_v_gate: hnn.Linear  # input projection from [embed] -> [heads, v_dim]
    c_proj: hnn.Linear  # output projection from [heads, v_dim] -> [embed]
    dropout: hnn.Dropout

    config: Gpt2Config = eqx.static_field()
    VDim: Axis = eqx.static_field()

    # Mistral stability tweaks
    scale_by_inverse_layer_idx: bool = eqx.static_field()
    upcast: bool = eqx.static_field()

    def __init__(
        self,
        config: Gpt2Config,
        dropout_prob: float,
        scale_by_inverse_layer_idx: bool,
        upcast: bool,
        *,
        key,
        use_bias: bool = True,
    ):
        self.config = config

        Embed, Heads, HeadDim = config.Embed, config.Heads, config.HeadDim
        self.VDim = Axis("v_dim", HeadDim.size * config.mlp_scale // 2)

        k_k, k_q, k_proj, k_v_ff, k_v_g = jrandom.split(key, 4)
        self.c_k = hnn.Linear(In=Embed, Out=(Heads, HeadDim), key=k_k, use_bias=use_bias)
        self.c_q = hnn.Linear(In=Embed, Out=(Heads, HeadDim), key=k_q, use_bias=use_bias)
        self.c_v_ff = hnn.Linear(In=Embed, Out=(Heads, self.VDim), key=k_v_ff, use_bias=False)
        self.c_v_gate = hnn.Linear(In=Embed, Out=(Heads, self.VDim), key=k_v_g, use_bias=False)
        self.c_proj = hnn.Linear(In=(Heads, self.VDim), Out=Embed, key=k_proj, use_bias=use_bias)

        self.dropout = hnn.Dropout(dropout_prob)

        self.scale_by_inverse_layer_idx = scale_by_inverse_layer_idx
        self.upcast = upcast

    @named_call
    def __call__(
        self, hidden_states: NamedArray, mask: Optional[NamedArray], layer_idx, inference: bool = True, *, key
    ):
        q = self.c_q(hidden_states)
        k = self.c_k(hidden_states)
        v_ff = self.c_v_ff(hidden_states)
        v_gate = self.c_v_gate(hidden_states)

        SeqLen, KeySeqLen, HeadDim = self.config.SeqLen, self.config.KeySeqLen, self.config.HeadDim

        # Rename k and v's SeqLen as haliax doesn't support unnamed axes or duplicate axes
        k = k.rename({SeqLen: KeySeqLen})
        v_ff = v_ff.rename({SeqLen: KeySeqLen})
        v_gate = v_gate.rename({SeqLen: KeySeqLen})

        # mistral tweak: scale norms by 1/sqrt(layer_idx) to prevent blowup
        scale = jax.lax.rsqrt(float(HeadDim.size))
        if self.scale_by_inverse_layer_idx:
            scale /= layer_idx + 1.0

        # do this first to help keep FP values small
        q = q * scale

        # mistral tweak: attention scores can overflow FP16, or just be too imprecise, so upcast to FP32
        if self.upcast:
            q = q.astype(jnp.float32)
            k = k.astype(jnp.float32)

        attn_scores = hax.dot(HeadDim, q, k)

        if mask is not None:
            attn_scores = attn_scores + (1.0 - mask) * -1e9

        attn_weights = hnn.softmax(attn_scores, axis=KeySeqLen).astype(hidden_states.dtype)
        attn_weights = self.dropout(attn_weights, key=key, inference=inference)

        # do quasi-mlp to v:
        v = hnn.relu(v_gate) * v_ff

        attn_output = hax.dot(KeySeqLen, attn_weights, v)  # [heads, seq_len, v_dim]

        attn_output = self.c_proj(attn_output)  # [seq_len, embed]
        return attn_output


class NoMlpGpt2Block(eqx.Module):
    ln_1: hnn.LayerNorm
    attn: NoMlpGpt2Attention
    ln_2: hnn.LayerNorm
    resid_dropout: hnn.Dropout

    def __init__(self, config: Gpt2Config, *, key):
        k_attn, k_cross, k_mlp = jrandom.split(key, 3)

        assert (
            config.Embed.size % config.num_heads == 0
        ), f"embed_dim={config.Embed} must be divisible by num_heads={config.num_heads}"

        self.ln_1 = hnn.LayerNorm(config.Embed, eps=config.layer_norm_epsilon)
        self.attn = NoMlpGpt2Attention(
            config,
            dropout_prob=config.attn_pdrop,
            key=k_attn,
            scale_by_inverse_layer_idx=config.scale_attn_by_inverse_layer_idx,
            upcast=config.upcast_attn,
            use_bias=config.use_bias,
        )
        self.resid_dropout = hnn.Dropout(pdrop=config.resid_pdrop)

    @named_call
    def __call__(self, hidden_states: NamedArray, mask: Optional[NamedArray], inference, layer_idx, *, key):
        k1, k2, k3 = haliax.jax_utils.maybe_rng_split(key, 3)

        hidden_states = hax.auto_sharded(hidden_states)
        attn_output = self.attn(self.ln_1(hidden_states), mask=mask, inference=inference, layer_idx=layer_idx, key=k1)
        attn_output = self.resid_dropout(attn_output, key=k2, inference=inference)
        hidden_states = hidden_states + attn_output

        return hidden_states


class NoMlpGpt2Transformer(eqx.Module):
    config: Gpt2Config = eqx.static_field()
    blocks: NoMlpGpt2Block
    ln_f: hnn.LayerNorm

    @property
    def Layers(self) -> Axis:
        return self.config.Layers

    def __init__(self, config: Gpt2Config, *, key):
        super().__init__()
        self.config = config

        # vectorize the blocks
        self.blocks = hax.vmap(NoMlpGpt2Block, self.Layers)(config, key=shaped_rng_split(key, config.num_layers))
        self.ln_f = hnn.LayerNorm(config.Embed, eps=config.layer_norm_epsilon)

    @named_call
    def __call__(self, hidden_states: NamedArray, attn_mask: Optional[NamedArray], *, inference, key) -> NamedArray:
        def do_block(hidden_states, block, layer_idx, key):
            return block(hidden_states, attn_mask, inference=inference, layer_idx=layer_idx, key=key)

        if self.config.gradient_checkpointing:
            do_block = jax.checkpoint(do_block, prevent_cse=False)

        keys = hax.jax_utils.maybe_rng_split(key, self.config.num_layers) if key is not None else None
        hidden_states = hax.fold(do_block, self.Layers)(  # type: ignore
            hidden_states, self.blocks, hax.arange(self.Layers), key=keys  # type: ignore
        )
        hidden_states = hax.auto_sharded(hidden_states)
        hidden_states = self.ln_f(hidden_states)

        return hidden_states


class NoMlpGpt2LMHeadModel(eqx.Module):
    transformer: NoMlpGpt2Transformer
    embeddings: Gpt2Embeddings

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

    def __init__(self, Vocab: Axis, config: Gpt2Config, *, key):
        k_t, k_embeddings = jrandom.split(key, 2)
        self.transformer = NoMlpGpt2Transformer(config, key=k_t)
        self.embeddings = Gpt2Embeddings(
            Vocab=Vocab,
            Embed=config.Embed,
            SeqLen=config.SeqLen,
            initializer_range=config.initializer_range,
            tie_word_embeddings=True,
            dropout_prob=config.embed_pdrop,
            key=k_embeddings,
        )

    def __call__(self, input_ids: NamedArray, attn_mask: Optional[NamedArray], *, inference, key):
        if not inference and key is None:
            raise ValueError("key must be provided for training")

        k_embed, k_transformer = haliax.jax_utils.maybe_rng_split(key, 2)
        hidden_states = self.embeddings.embed(input_ids, inference=inference, key=k_embed)
        hidden_states = self.transformer(hidden_states, attn_mask, inference=inference, key=k_transformer)
        lm_logits = self.embeddings.unembed(hidden_states)

        return lm_logits
