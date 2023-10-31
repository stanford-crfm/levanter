import dataclasses
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Type, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import PRNGKeyArray
from transformers import LlamaConfig as HfLlamaConfig
from transformers import PretrainedConfig as HfConfig

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, AxisSpec, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call, shaped_rng_split
from haliax.nn.scan import Stacked

from levanter.compat.hf_checkpoints import HFCheckpointConverter, HFCompatConfig
from levanter.compat.torch_serialization import (
    StateDict,
    StateDictSerializationMixin,
    apply_prefix,
    flatten_linear_layers,
    stack_state_dict,
    unflatten_linear_layers,
    unstack_state_dict,
)
from levanter.models.attention import AttentionMask, dot_product_attention
from levanter.models.gpt2 import ACT2FN
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.utils.py_utils import cached_classproperty


@LmConfig.register_subclass("llama")
@dataclass(frozen=True)
class LlamaConfig(HFCompatConfig):
    """Config for LlamaModel

    Args:
        seq_len (int, optional): maximum length of the input sequence. Defaults to 2048.
        hidden_dim (int, optional): dimension of the hidden state. Defaults to 4096.
        intermediate_dim (int, optional): dimension of the intermediate state. Defaults to 11008.
        num_layers (int, optional): number of hidden layers in the Transformer encoder. Defaults to 32.
        num_heads (int, optional): number of attention heads for each attention layer. Defaults to 32.
        activation_function (str, optional): activation function for the hidden layer. Defaults to "silu".
        rope_scaling (Dict, optional): dict containing the scaling configuration for the Rotary Positional Embedding.
    """

    seq_len: int = 2048
    hidden_dim: int = 4096
    intermediate_dim: int = 11008
    num_layers: int = 32
    num_heads: int = 32
    activation_function: str = "silu"
    initializer_range: float = 0.02
    layer_norm_epsilon: float = 1e-5

    # Attention-related config
    upcast_attn: bool = False
    use_flash_attention: bool = False
    flash_attention_block_size: Optional[int] = None

    gradient_checkpointing: bool = True
    gradient_checkpointing_block_size: int = 5

    use_bias: bool = False
    rope_scaling: Optional[dict] = None

    # Axis
    Pos = property(lambda self: Axis(name="position", size=self.seq_len))
    KeyPos = property(lambda self: self.Pos.alias("key_position"))
    Embed = property(lambda self: Axis(name="embed", size=self.hidden_dim))
    Heads = property(lambda self: Axis(name="heads", size=self.num_heads))
    Layers = property(lambda self: Axis(name="layers", size=self.num_layers))
    Mlp = property(lambda self: Axis(name="mlp", size=self.intermediate_dim))
    HeadSize = property(lambda self: Axis(name="head_size", size=self.hidden_dim // self.num_heads))

    @cached_classproperty
    def default_hf_checkpoint_converter(cls) -> HFCheckpointConverter["LlamaConfig"]:  # type: ignore
        return HFCheckpointConverter(
            cls,  # type: ignore
            "meta-llama/Llama-2-7b-hf",
            trust_remote_code=True,
            tokenizer="hf-internal-testing/llama-tokenizer",
            HfConfigClass=HfLlamaConfig,
        )

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig):
        return LlamaConfig(
            seq_len=hf_config.max_position_embeddings,
            hidden_dim=hf_config.hidden_size,
            intermediate_dim=hf_config.intermediate_size,
            num_layers=hf_config.num_hidden_layers,
            num_heads=hf_config.num_attention_heads,
            activation_function=hf_config.hidden_act,
            initializer_range=hf_config.initializer_range,
            layer_norm_epsilon=hf_config.rms_norm_eps,
            rope_scaling=hf_config.rope_scaling,
        )

    def to_hf_config(self, vocab_size: int, config_overrides: Optional[Dict] = None) -> HfLlamaConfig:
        """Convert to HuggingFace's LlamaConfig

        Args:
            vocab_size (int, optional): Vocabulary size of the tokenizer. Defaults to 32000.
            config_overrides (dict, optional): Overrides for the config. Defaults to None.

        Returns:
            HfLlamaConfig: HuggingFace's LlamaConfig
        """
        if config_overrides is None:
            config_overrides = {}

        return HfLlamaConfig(
            max_position_embeddings=self.seq_len,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_dim,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            hidden_act=self.activation_function,
            initializer_range=self.initializer_range,
            rms_norm_eps=self.layer_norm_epsilon,
            rope_scaling=self.rope_scaling,
            vocab_size=vocab_size,
            **config_overrides,
        )

    @property
    def model_type(cls) -> Type["LlamaLMHeadModel"]:
        return LlamaLMHeadModel


class LlamaMlp(eqx.Module, StateDictSerializationMixin):
    """Multi-layer Perceptron
    In comparison with GPT2, LlamaMlp adds an up-proj that multiplies with activated gate_proj,
    before down-proj.
    """

    gate_proj: hnn.Linear  # projection from Embed to Mlp
    up_proj: hnn.Linear  # projection from Embed to Mlp
    down_proj: hnn.Linear  # projection from Mlp to Embed
    act: Callable = eqx.static_field()

    @staticmethod
    def init(
        Embed: Axis, Mlp: Axis, activation_fn: Union[str, Callable], *, key, use_bias: bool = False
    ) -> "LlamaMlp":
        k_fc, k_up_proj, k_down_proj = jrandom.split(key, 3)
        gate_proj = hnn.Linear.init(Out=Mlp, In=Embed, key=k_fc, use_bias=use_bias)
        up_proj = hnn.Linear.init(Out=Mlp, In=Embed, key=k_up_proj, use_bias=use_bias)
        down_proj = hnn.Linear.init(Out=Embed, In=Mlp, key=k_down_proj, use_bias=use_bias)
        if isinstance(activation_fn, str):
            activation_fn = ACT2FN[activation_fn]
        act = activation_fn  # type: ignore
        return LlamaMlp(gate_proj, up_proj, down_proj, act)

    @named_call
    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        k_gate, k_up, k_down = maybe_rng_split(key, 3)
        hidden_states = self.gate_proj(x, key=k_gate)
        hidden_states = self.act(hidden_states)
        hidden_states = hidden_states * self.up_proj(x, key=k_up)
        outputs = self.down_proj(hidden_states, key=k_down)
        return outputs

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None):
        # unflatten the linear layers of HF state_dict to match the shape of LlamaMlp
        d = {}
        d.update(
            unflatten_linear_layers(
                apply_prefix(prefix, "gate_proj"), state_dict, self.gate_proj, out_dims_first_in_dict=True
            )
        )
        d.update(
            unflatten_linear_layers(
                apply_prefix(prefix, "up_proj"), state_dict, self.up_proj, out_dims_first_in_dict=True
            )
        )
        d.update(
            unflatten_linear_layers(
                apply_prefix(prefix, "down_proj"), state_dict, self.down_proj, out_dims_first_in_dict=True
            )
        )

        return super().from_state_dict(d, prefix)

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        my_dict: StateDict = {}
        super().update_state_dict(my_dict, prefix=prefix)

        my_dict.update(
            flatten_linear_layers(apply_prefix(prefix, "gate_proj"), self.gate_proj, out_dims_first_in_dict=True)
        )
        my_dict.update(
            flatten_linear_layers(apply_prefix(prefix, "up_proj"), self.up_proj, out_dims_first_in_dict=True)
        )
        my_dict.update(
            flatten_linear_layers(apply_prefix(prefix, "down_proj"), self.down_proj, out_dims_first_in_dict=True)
        )

        state_dict.update(my_dict)
        return state_dict


class LlamaRotaryEmbedding(eqx.Module, StateDictSerializationMixin):
    Pos: Axis = eqx.field(static=True)
    cos_cached: NamedArray
    sin_cached: NamedArray

    def __init__(self, HeadSize: Axis, Pos: Axis, base: int = 10000):
        self.Pos = Pos
        # this must be compile-time b/c we want to store them in a static field
        with jax.ensure_compile_time_eval():
            self.cos_cached, self.sin_cached = self._get_cos_sin_cache(Pos=Pos, HeadSize=HeadSize, base=base)

    @staticmethod
    def _get_cos_sin_cache(HeadSize: hax.Axis, Pos: hax.Axis, base: float) -> Tuple[NamedArray, NamedArray]:
        HeadHalfSize = HeadSize.resize(HeadSize.size // 2)
        inv_freq: NamedArray = 1.0 / (base ** (hax.arange(HeadHalfSize, step=2) / HeadSize.size))

        position_ids: NamedArray = hax.arange(Pos)

        freqs = position_ids * inv_freq.broadcast_axis(Pos)
        # This is different from the paper but aligns with HF implementation:
        # It uses a different permutation in order to obtain the same calculation
        emb = hax.concatenate(HeadSize, (freqs, freqs))
        cos_cached = hax.cos(emb)
        sin_cached = hax.sin(emb)
        # This is different from the paper but aligns with HF implementation:
        return cos_cached, sin_cached

    def __call__(self, seq_len: int) -> Tuple[NamedArray, NamedArray]:
        return jax.lax.stop_gradient(
            (
                self.cos_cached[self.Pos, :seq_len],
                self.sin_cached[self.Pos, :seq_len],
            )
        )

    # TODO: maybe add a "persistent" option to eqx.field that we use for state dict serialization
    # if we do that, consider moving the key remapping stuff there too?
    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None):
        return self

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        return state_dict


class LlamaAttention(StateDictSerializationMixin, eqx.Module):
    config: LlamaConfig = eqx.static_field()
    q_proj: hnn.Linear  # projection from Embed to query
    k_proj: hnn.Linear  # projection from Embed to key
    v_proj: hnn.Linear  # projection from Embed to value
    o_proj: hnn.Linear  # projection from Heads to output
    rotary_emb: LlamaRotaryEmbedding  # rotary embedding

    @staticmethod
    def init(config: LlamaConfig, *, key) -> "LlamaAttention":
        use_bias = config.use_bias
        Embed = config.Embed
        k_q, k_k, k_v, k_o = jrandom.split(key, 4)
        q_proj = hnn.Linear.init(In=Embed, Out=(config.Heads, config.HeadSize), key=k_q, use_bias=use_bias)
        k_proj = hnn.Linear.init(In=Embed, Out=(config.Heads, config.HeadSize), key=k_k, use_bias=use_bias)
        v_proj = hnn.Linear.init(In=Embed, Out=(config.Heads, config.HeadSize), key=k_v, use_bias=use_bias)
        o_proj = hnn.Linear.init(In=(config.Heads, config.HeadSize), Out=Embed, key=k_o, use_bias=use_bias)
        rotary_emb = LlamaRotaryEmbedding(config.HeadSize, config.Pos)
        return LlamaAttention(config, q_proj, k_proj, v_proj, o_proj, rotary_emb)

    @named_call
    def __call__(self, x: NamedArray, mask: Optional[NamedArray], *, key=None) -> NamedArray:
        key_q, key_k, key_v, key_o = maybe_rng_split(key, 4)

        # reorder heads and position for better training throughput
        q = self.q_proj(x, key=key_q).rearrange((..., "heads", "position", "head_size"))
        k = self.k_proj(x, key=key_k).rearrange((..., "heads", "position", "head_size"))
        v = self.v_proj(x, key=key_v).rearrange((..., "heads", "position", "head_size"))

        cos, sin = self.rotary_emb(seq_len=x.axis_size("position"))

        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        if self.config.upcast_attn:
            q = q.astype(jnp.float32)
            k = k.astype(jnp.float32)
        k = k.rename({"position": "key_position"})
        v = v.rename({"position": "key_position"})

        c = self.config
        attn_output = dot_product_attention(
            "position",
            "key_position",
            "head_size",
            q,
            k,
            v,
            mask,
            attention_dtype=jnp.float32 if self.config.upcast_attn else x.dtype,
            use_flash=c.use_flash_attention,
            flash_block_size=c.flash_attention_block_size,
        )

        if self.config.upcast_attn:
            attn_output = attn_output.astype(x.dtype)

        attn_output = self.o_proj(attn_output, key=key_o)
        return attn_output

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None):
        # unflatten the linear layers of HF state_dict to match the shape of LlamaAttention
        d = {}
        d.update(unflatten_linear_layers(apply_prefix(prefix, "q_proj"), state_dict, self.q_proj, True))
        d.update(unflatten_linear_layers(apply_prefix(prefix, "k_proj"), state_dict, self.k_proj, True))
        d.update(unflatten_linear_layers(apply_prefix(prefix, "v_proj"), state_dict, self.v_proj, True))
        d.update(unflatten_linear_layers(apply_prefix(prefix, "o_proj"), state_dict, self.o_proj, True))

        return super().from_state_dict(d, prefix)

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        # flatten the linear layers of LlamaAttention to match the shape of HF state_dict
        my_dict: StateDict = {}
        super().update_state_dict(my_dict, prefix)

        my_dict.update(flatten_linear_layers(apply_prefix(prefix, "q_proj"), self.q_proj, True))
        my_dict.update(flatten_linear_layers(apply_prefix(prefix, "k_proj"), self.k_proj, True))
        my_dict.update(flatten_linear_layers(apply_prefix(prefix, "v_proj"), self.v_proj, True))
        my_dict.update(flatten_linear_layers(apply_prefix(prefix, "o_proj"), self.o_proj, True))

        state_dict.update(my_dict)
        return state_dict


class LlamaRMSNorm(hnn.LayerNorm):
    """It is a modified version of LayerNorm.
    The main changes are:
    1. The variance is defined as the average of square, versus the original
    definition as the average of the squared deviations from the mean.
    2. The output is defined as x * inv, without minusing the mean.
    3. The default value of eps is set to 1e-6 and use_bias to False.
    """

    @staticmethod
    def init(axis: AxisSpec, eps: float = 1e-6, use_weight: bool = True, use_bias: bool = False):
        if use_weight:
            weight = hax.ones(axis)
        else:
            weight = None
        if use_bias:
            bias = hax.zeros(axis)
        else:
            bias = None

        return LlamaRMSNorm(axis, weight, bias, eps)

    def __call__(self, x: NamedArray) -> NamedArray:
        # This gives a different result than jnp.var(), which is
        # defined as the average of the squared deviations from the mean
        var = hax.mean(hax.square(x), axis=self.axis)
        inv = hax.rsqrt(var + self.eps)
        out = x * inv

        if self.weight is not None:
            out = self.weight * out
        if self.bias is not None:
            out = out + self.bias
        return out


class LlamaDecoderLayer(StateDictSerializationMixin, eqx.Module):
    config: LlamaConfig = eqx.static_field()
    self_attn: LlamaAttention
    mlp: LlamaMlp
    input_layernorm: LlamaRMSNorm
    post_attention_layernorm: LlamaRMSNorm

    @staticmethod
    def init(config: LlamaConfig, *, key) -> "LlamaDecoderLayer":
        k_attn, k_mlp = jrandom.split(key, 2)

        attn = LlamaAttention.init(config, key=k_attn)
        mlp = LlamaMlp.init(
            config.Embed,
            config.Mlp,
            config.activation_function,
            key=k_mlp,
            use_bias=config.use_bias,
        )
        ln_1 = LlamaRMSNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)
        ln_2 = LlamaRMSNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)

        return LlamaDecoderLayer(config, attn, mlp, ln_1, ln_2)

    @named_call
    def __call__(self, x: NamedArray, mask: Optional[NamedArray], *, key=None) -> NamedArray:
        k_attn, k_mlp = maybe_rng_split(key, 2)
        # self attention and skip connection
        residual = x
        x = self.input_layernorm(x)
        attn_output = self.self_attn(x=x, mask=mask, key=k_attn)
        x = residual + attn_output

        # MLP and skip connection
        residual = x
        x = self.post_attention_layernorm(x)
        mlp_output = self.mlp(x, key=k_mlp)
        output = residual + mlp_output
        return output


class LlamaTransformer(StateDictSerializationMixin, eqx.Module):
    config: LlamaConfig = eqx.static_field()
    layers: Stacked[LlamaDecoderLayer]
    norm: LlamaRMSNorm

    @staticmethod
    def init(config: LlamaConfig, *, key) -> "LlamaTransformer":
        layers = Stacked.init(config.Layers, LlamaDecoderLayer, gradient_checkpointing=config.gradient_checkpointing)(
            config,
            key=shaped_rng_split(key, config.num_layers),
        )
        ln_f = LlamaRMSNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)

        return LlamaTransformer(config, layers, ln_f)

    @named_call
    def __call__(self, x: NamedArray, attn_mask: Optional[NamedArray], *, key) -> NamedArray:
        keys = maybe_rng_split(key, self.config.num_layers) if key is not None else None
        x = self.layers.fold(x, mask=attn_mask, key=keys)
        x = self.norm(x)

        return x

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None):
        stacked = stack_state_dict(state_dict, prefix=apply_prefix(prefix, "layers"))
        out = super().from_state_dict(stacked, prefix=prefix)
        return out

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        my_state_dict: StateDict = {}
        super().update_state_dict(my_state_dict, prefix=prefix)

        stacked_dict = unstack_state_dict(my_state_dict, prefix=apply_prefix(prefix, "layers"))
        state_dict.update(stacked_dict)

        return state_dict


class LlamaEmbedding(StateDictSerializationMixin, eqx.Module):
    """Similar to GPT2 Embedding, except that:
    - Llama doesn't have position embedding in the Embedding layer.
    - Llama doesn't use dropout.
    """

    Vocab: Axis = eqx.static_field()
    config: LlamaConfig = eqx.static_field()
    token_embeddings: NamedArray

    @staticmethod
    def init(Vocab: Axis, config: LlamaConfig, *, key) -> "LlamaEmbedding":
        k_wte = jrandom.split(key, 1)

        token_embeddings = hax.random.normal(k_wte, (Vocab, config.Embed))
        return LlamaEmbedding(Vocab, config, token_embeddings)

    @named_call
    def embed(self, input_ids, *args):
        input_embeds = self.token_embeddings.take("vocab", input_ids)
        x = input_embeds
        return x

    def unembed(self, x: NamedArray):
        return hax.dot("embed", x, self.token_embeddings)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"token_embeddings": "model.embed_tokens.weight"}

    def resize_embeddings(self, new_size: int, key: Optional[PRNGKeyArray] = None):
        new_weights = hax.tree_util.resize_axis(self.token_embeddings, self.Vocab, new_size, key=key)
        return dataclasses.replace(self, Vocab=self.Vocab.resize(new_size), token_embeddings=new_weights)


class LlamaLMHeadModel(eqx.Module, LmHeadModel[LlamaConfig], StateDictSerializationMixin):
    transformer: LlamaTransformer
    embeddings: LlamaEmbedding
    lm_head: hnn.Linear

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
    def init(cls, Vocab: Axis, config: LlamaConfig, *, key) -> "LlamaLMHeadModel":
        k_t, k_emb = jrandom.split(key, 2)
        transformer = LlamaTransformer.init(config, key=k_t)
        embeddings = LlamaEmbedding.init(Vocab, config, key=k_emb)
        lm_head = hnn.Linear.init(In=config.Embed, Out=Vocab, key=k_emb, use_bias=False, out_first=True)
        return LlamaLMHeadModel(transformer, embeddings, lm_head)

    def __call__(
        self,
        input_ids: NamedArray,
        attn_mask: Optional[Union[NamedArray, AttentionMask]] = None,
        *,
        key=None,
    ) -> NamedArray:
        """
        Args:
            input_ids (NamedArray): [batch, position]
                Indices of input sequence tokens in the vocabulary.
            attn_mask (Union[NamedArray, AttentionMask], optional): [batch, position]
                Mask to avoid performing attention on the padding token indices of the encoder input.
                The attn_mask from training pipeline may be an AttentionMask object instead of NamedArray
        """
        k_t, k_head = maybe_rng_split(key, 2)
        x = self.embeddings.embed(input_ids)
        x = self.transformer(x, attn_mask=attn_mask, key=k_t)
        lm_logits = self.lm_head(x, key=k_head)
        return lm_logits

    def resize_vocab(self, new_size: int, key=None) -> "LmHeadModel[LlamaConfig]":
        new_Vocab = self.Vocab.resize(new_size)
        k1, k2 = maybe_rng_split(key, 2)
        new_embeddings = self.embeddings.resize_embeddings(new_size, key=k1)
        new_lm_matrix = hax.tree_util.resize_axis(self.lm_head.weight, self.Vocab, new_size, key=k2)
        new_lm_head = dataclasses.replace(self.lm_head, Out=new_Vocab, weight=new_lm_matrix)

        return dataclasses.replace(self, embeddings=new_embeddings, lm_head=new_lm_head)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"transformer": "model", "embeddings": None}

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None):
        # unflatten the linear layers of HF state_dict to match the shape of LlamaMlp
        d = state_dict.copy()
        d.update(
            unflatten_linear_layers(
                apply_prefix(prefix, "lm_head"), state_dict, self.lm_head, out_dims_first_in_dict=True
            )
        )
        return super().from_state_dict(d, prefix)

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        my_dict: StateDict = {}
        super().update_state_dict(my_dict, prefix=prefix)

        my_dict.update(
            flatten_linear_layers(apply_prefix(prefix, "lm_head"), self.lm_head, out_dims_first_in_dict=True)
        )

        state_dict.update(my_dict)
        return state_dict


def _rotate_half(x: NamedArray) -> NamedArray:
    """Rotates half of the hidden dims of the input and concatenates them."""
    HeadSize = x.axes[-1]
    x1 = x[HeadSize, : HeadSize.size // 2]
    x2 = x[HeadSize, HeadSize.size // 2 :]
    out = hax.concatenate(HeadSize, (-x2, x1))
    return out


def _apply_rotary_pos_emb(
    q: NamedArray,  # [batch, position, heads, head_size]
    k: NamedArray,  # [batch, position, kv_heads, head_size]
    cos: NamedArray,  # [position, head_size]
    sin: NamedArray,  # [position, head_size]
) -> Tuple[NamedArray, NamedArray]:
    """Applies rotary position embedding to q and k."""
    q_embed = q * cos + _rotate_half(q) * sin
    k_embed = k * cos + _rotate_half(k) * sin
    return q_embed, k_embed
