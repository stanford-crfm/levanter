from dataclasses import dataclass

import equinox as eqx
import jax
from jax.random import PRNGKey

import haliax as hax
import haliax.nn as hnn
from haliax import Axis
from haliax.jax_utils import named_call, shaped_rng_split
from haliax.nn.attention import causal_sliding_window_attention


@dataclass(frozen=True)
class YaConfig:
    seq_len: int = 512
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    window_size: int = 512

    # how much to scale the embedding dim for the mlp layer
    mlp_scale: int = 4

    initializer_range: float = 0.02

    gradient_checkpointing: bool = False

    # TODO: non_alibi_heads
    # non_alibi_heads: int = 0  # heads that don't get the alibi bias
    # NB that heads without bias will not get any positional info at all

    # Axes
    SeqLen = property(lambda self: Axis(name="seqlen", size=self.seq_len))
    Window = property(lambda self: Axis(name="window", size=self.window_size))
    Embed = property(lambda self: Axis(name="embed", size=self.hidden_dim))
    Heads = property(lambda self: Axis(name="heads", size=self.num_heads))
    Layers = property(lambda self: Axis(name="layers", size=self.num_layers))
    Mlp = property(lambda self: Axis(name="mlp", size=self.hidden_dim * self.mlp_scale))
    HeadDim = property(lambda self: Axis(name="head", size=self.hidden_dim // self.num_heads))


class SwiGluMLP(eqx.Module):
    ff: hnn.Linear
    gate: hnn.Linear
    proj: hnn.Linear

    def __init__(self, Embed: Axis, Intermediate: Axis, key: PRNGKey):
        super().__init__()
        k_ff, k_gate, k_proj = jax.random.split(key, 3)
        self.ff = hnn.Linear(Embed, Intermediate, key=k_ff, use_bias=False)
        self.gate = hnn.Linear(Embed, Intermediate, key=k_gate, use_bias=False)
        self.proj = hnn.Linear(Intermediate, Embed, key=k_proj, use_bias=False)

    @named_call
    def __call__(self, x):
        ff = self.ff(x)
        gate = self.gate(x)
        return self.proj(ff * hax.nn.swish(gate))


class YaAttention(eqx.Module):
    """Longformer-esque causal sliding window attention with alibi bias."""

    p_q: hnn.Linear
    p_k: hnn.Linear
    p_v: hnn.Linear
    p_out: hnn.Linear

    config: YaConfig = eqx.static_field()

    def __init__(self, config: YaConfig, key: PRNGKey):
        super().__init__()
        Embed, Heads, HeadDim = config.Embed, config.Heads, config.HeadDim
        k_q, k_k, k_v, k_out = jax.random.split(key, 4)
        # PaLM-style single-key attention
        # TODO: normal multiheaded attention?
        self.p_q = hnn.Linear(Embed, (Heads, HeadDim), key=k_q, use_bias=False)
        self.p_k = hnn.Linear(Embed, HeadDim, key=k_k, use_bias=False)
        self.p_v = hnn.Linear(Embed, HeadDim, key=k_v, use_bias=False)
        self.p_out = hnn.Linear((Heads, HeadDim), Embed, key=k_out, use_bias=False)

        self.config = config

    @named_call
    def __call__(self, x, bias):
        q, k, v = self.p_q(x), self.p_k(x), self.p_v(x)

        attn = causal_sliding_window_attention(
            self.config.SeqLen, self.config.Window, self.config.HeadDim, q, k, v, bias=bias
        )

        return self.p_out(attn)


class YaBlock(eqx.Module):
    attn: YaAttention
    mlp: SwiGluMLP
    ln_attn: hnn.LayerNorm
    ln_mlp: hnn.LayerNorm

    # re-zero style residual connections arxiv.org/abs/2003.04887
    a_attn: hax.NamedArray
    a_mlp: hax.NamedArray

    def __init__(self, config: YaConfig, key: PRNGKey):
        super().__init__()
        k_attn, k_mlp = jax.random.split(key, 2)
        self.attn = YaAttention(config, key=k_attn)
        self.mlp = SwiGluMLP(config.Embed, config.Mlp, key=k_mlp)
        # TODO: use elementwise_affine=True?
        self.ln_attn = hnn.LayerNorm(config.Embed, elementwise_affine=False)
        self.ln_mlp = hnn.LayerNorm(config.Embed, elementwise_affine=False)
        self.a_attn = hax.zeros(())
        self.a_mlp = hax.zeros(())

    @named_call
    def __call__(self, x, bias):
        # TODO: parallel attention/mlp?
        x = x + self.attn(self.ln_attn(x), bias) * self.a_attn
        x = x + self.mlp(self.ln_mlp(x)) * self.a_mlp
        return x


class YaTransformer(eqx.Module):
    config: YaConfig = eqx.static_field()
    blocks: YaBlock
    ln_f: hnn.LayerNorm

    def __init__(self, config: YaConfig, key: PRNGKey):
        super().__init__()
        self.config = config
        # vectorize the blocks
        self.blocks = hax.vmap(YaBlock, config.Layers)(config, key=shaped_rng_split(key, config.num_layers))
        self.ln_f = hnn.LayerNorm(config.Embed)

    @named_call
    def __call__(self, x):
        alibi_bias = hnn.attention.alibi_bias(self.config.Heads, self.config.SeqLen)
        do_block = lambda x, block: block(x, alibi_bias)  # noqa: E731

        if self.config.gradient_checkpointing:
            do_block = jax.checkpoint(do_block, prevent_cse=False)

        x = hax.fold(do_block, self.config.Layers)(x, self.blocks)
        x = self.ln_f(x)
        return self.ln_f(x)


class YaLMHeadModel(eqx.Module):
    """A YaT model with a language model head."""

    config: YaConfig = eqx.static_field()
    Vocab: Axis = eqx.static_field()
    transformer: YaTransformer
    embed: hax.NamedArray

    def __init__(self, Vocab: Axis, config: YaConfig, key: PRNGKey):
        super().__init__()
        self.config = config
        self.Vocab = Vocab
        k_t, k_h = jax.random.split(key, 2)
        self.transformer = YaTransformer(config, key=k_t)
        self.embed = hax.random.normal(k_h, (self.Vocab, config.Embed)) * 0.02

    @named_call
    def __call__(self, input_ids):
        x = self.embed.take(self.Vocab, input_ids)
        x = self.transformer(x)
        x = x.dot(self.config.Embed, self.embed)
        return x
