import dataclasses
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Type, Union

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from haliax import Axis, AxisSpec, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call, shaped_rng_split
from haliax.nn.scan import Stacked
from haliax.util import ensure_tuple
from jaxtyping import PRNGKeyArray, PyTree

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
from levanter.logging import silence_transformer_nag
from levanter.models.attention import AttentionMask, dot_product_attention
from levanter.models.gpt2 import ACT2FN
from levanter.models.llama import LlamaRMSNorm
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.types import BlockFoldable
from levanter.utils.jax_utils import leaf_key_paths

silence_transformer_nag()
from transformers import LlamaConfig as HfLlamaConfig  # noqa: E402
from transformers import PretrainedConfig as HfConfig  # noqa: E402


@LmConfig.register_subclass("factorized_llama")
@dataclass(frozen=True)
class FactorizedLlamaConfig(HFCompatConfig):
    """Config for FactorizedLlamaModel

    Args:
        seq_len (int, optional): maximum length of the input sequence. Defaults to 2048.
        hidden_dim (int, optional): dimension of the hidden state. Defaults to 4096.
        intermediate_dim (int, optional): dimension of the intermediate state. Defaults to 11008.
        num_layers (int, optional): number of hidden layers in the Transformer encoder. Defaults to 32.
        num_heads (int, optional): number of attention heads for each attention layer. Defaults to 32.
        num_kv_heads (int, optional): number of attention heads for keys and values in each attention layer.
            Setting to 1 means MQA. Setting to num_heads means MHA. Otherwise GQA.
            Note that num_heads must be divisible by this number. Defaults to 32.
        activation_function (str, optional): activation function for the hidden layer. Defaults to "silu".
        rope_scaling (Dict, optional): dict containing the scaling configuration for the Rotary Positional Embedding.
    """

    reference_checkpoint: Optional[str] = None

    seq_len: int = 2048
    hidden_dim: int = 4096
    factor_dim: int = 512
    intermediate_dim: int = 11008
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 32
    activation_function: str = "silu"
    initializer_range: float = 0.02
    layer_norm_epsilon: float = 1e-5

    # Attention-related config
    upcast_attn: bool = False
    use_flash_attention: bool = True
    flash_attention_block_size: Optional[int] = None

    gradient_checkpointing: bool = True
    gradient_checkpointing_block_size: int = 5
    scan_layers: bool = True

    use_bias: bool = False
    rope_scaling: Optional[dict] = None

    # Axis
    Pos = property(lambda self: Axis(name="position", size=self.seq_len))
    KeyPos = property(lambda self: self.Pos.alias("key_position"))
    Factor = property(lambda self: Axis(name="factor", size=self.factor_dim))
    Embed = property(lambda self: Axis(name="embed", size=self.hidden_dim))
    Heads = property(lambda self: Axis(name="heads", size=self.num_heads))
    KVHeads = property(lambda self: Axis(name="kv_heads", size=self.num_kv_heads))
    Layers = property(lambda self: Axis(name="layers", size=self.num_layers))
    Mlp = property(lambda self: Axis(name="mlp", size=self.intermediate_dim))
    HeadSize = property(lambda self: Axis(name="head_size", size=self.hidden_dim // self.num_heads))

    def __post_init__(self):
        assert (
            self.num_heads % self.num_kv_heads == 0
        ), f"num_heads={self.num_heads} not divisible by num_kv_heads={self.num_kv_heads}."

    @property
    def default_hf_checkpoint_converter(self) -> HFCheckpointConverter["FactorizedLlamaConfig"]:  # type: ignore
        assert self.reference_checkpoint, "Must specify HF model id to convert from."
        return HFCheckpointConverter(
            self.__class__,  # type: ignore
            self.reference_checkpoint,
            trust_remote_code=True,
            HfConfigClass=HfLlamaConfig,
        )

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig):
        return FactorizedLlamaConfig(
            seq_len=hf_config.max_position_embeddings,
            hidden_dim=hf_config.hidden_size,
            intermediate_dim=hf_config.intermediate_size,
            num_layers=hf_config.num_hidden_layers,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            activation_function=hf_config.hidden_act,
            initializer_range=hf_config.initializer_range,
            layer_norm_epsilon=hf_config.rms_norm_eps,
            rope_scaling=hf_config.rope_scaling,
            factor_dim=getattr(hf_config, "factor_dim", 512),
        )

    def to_hf_config(self, vocab_size: int, config_overrides: Optional[Dict] = None) -> HfLlamaConfig:
        """Convert to HuggingFace's FactorizedLlamaConfig

        Args:
            vocab_size (int, optional): Vocabulary size of the tokenizer. Defaults to 32000.
            config_overrides (dict, optional): Overrides for the config. Defaults to None.

        Returns:
            HfLlamaConfig: HuggingFace's FactorizedLlamaConfig
        """
        if config_overrides is None:
            config_overrides = {}

        return HfLlamaConfig(
            max_position_embeddings=self.seq_len,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_dim,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            hidden_act=self.activation_function,
            initializer_range=self.initializer_range,
            rms_norm_eps=self.layer_norm_epsilon,
            rope_scaling=self.rope_scaling,
            factor_dim=self.factor_dim,
            vocab_size=vocab_size,
            **config_overrides,
        )

    @property
    def model_type(cls) -> Type["FactorizedLlamaLMHeadModel"]:
        return FactorizedLlamaLMHeadModel


def low_rank_approximation(matrix, rank):
    """
    Approximates the input matrix using Singular Value Decomposition (SVD) to a lower rank.

    Args:
        matrix (jax.numpy.ndarray): Input matrix of shape (N, M)
        rank (int): Desired rank of the approximation (H)

    Returns:
        jax.numpy.ndarray, jax.numpy.ndarray: Two matrices of shape (N, H) and (H, M)
    """
    from jax.numpy.linalg import svd

    # Perform SVD
    U, S, Vh = svd(matrix, full_matrices=False)

    S = S[..., :rank]
    # Truncate U, S, and Vh to the desired rank
    U_truncated = U[..., :rank]
    if len(S.shape) == 1:
        S_truncated = jnp.diag(S)
    else:
        S_truncated = jax.vmap(jnp.diag, 0)(S[..., :rank])

    Vh_truncated = Vh[..., :rank, :]  # Note: Vh is already the conjugate transpose of V

    # Reconstruct the low-rank approximation
    down_proj = U_truncated @ S_truncated  # Shape (N, H)
    up_proj = Vh_truncated  # Shape (H, M)

    return down_proj, up_proj


class FactorizedLinear(StateDictSerializationMixin, eqx.Module):
    """Factorized Linear Layer"""

    down_proj: hnn.Linear
    up_proj: hnn.Linear
    out_first: bool = eqx.static_field()

    @staticmethod
    def init(
        Out: Axis, In: Axis, Hidden: Axis, *, key, use_bias: bool = False, out_first: bool = False
    ) -> "FactorizedLinear":
        assert Hidden.size <= np.prod([a.size for a in ensure_tuple(Out)]), (
            "Hidden size must be less than or equal to output size.",
            Hidden,
            Out,
        )

        k_down, k_up = jrandom.split(key, 2)
        down_proj = hnn.Linear.init(Out=Hidden, In=In, key=k_up, use_bias=use_bias)
        up_proj = hnn.Linear.init(Out=Out, In=Hidden, key=k_down, use_bias=use_bias)
        return FactorizedLinear(down_proj, up_proj, out_first=out_first)

    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        k_down, k_up = maybe_rng_split(key, 2)
        return self.up_proj(self.down_proj(x, key=k_up), key=k_down)

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None):
        d = state_dict.copy()

        # Initial factorized linear with the SVD approximation of the original input matrix.
        weights = state_dict[prefix + ".weight"]
        down_proj, up_proj = low_rank_approximation(weights, self.down_proj.Out.size)
        if self.out_first:
            d[prefix + ".down_proj.weight"] = up_proj
            d[prefix + ".up_proj.weight"] = down_proj
        else:
            d[prefix + ".down_proj.weight"] = down_proj
            d[prefix + ".up_proj.weight"] = up_proj

        d.update(
            unflatten_linear_layers(
                apply_prefix(prefix, "down_proj"),
                d,
                self.down_proj,
                out_dims_first_in_dict=self.out_first,
            )
        )
        d.update(
            unflatten_linear_layers(
                apply_prefix(prefix, "up_proj"),
                d,
                self.up_proj,
                out_dims_first_in_dict=self.out_first,
            )
        )

        return super().from_state_dict(d, prefix)

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        # We override the default state dict generation, as we don't want to output
        # weights for the factorized linear layers.
        # super().update_state_dict(my_dict, prefix=prefix)
        my_dict: StateDict = {}

        my_dict.update(
            flatten_linear_layers(prefix + ".down_proj", self.down_proj, out_dims_first_in_dict=self.out_first)
        )
        my_dict.update(flatten_linear_layers(prefix + ".up_proj", self.up_proj, out_dims_first_in_dict=self.out_first))

        if self.out_first:
            my_dict[prefix + ".weight"] = jnp.transpose(
                jnp.dot(
                    jnp.transpose(my_dict[prefix + ".down_proj.weight"]),
                    jnp.transpose(my_dict[prefix + ".up_proj.weight"]),
                )
            )
        else:
            my_dict[prefix + ".weight"] = jnp.dot(
                my_dict[prefix + ".down_proj.weight"],
                my_dict[prefix + ".up_proj.weight"],
            )

        del my_dict[prefix + ".down_proj.weight"]
        del my_dict[prefix + ".up_proj.weight"]

        state_dict.update(my_dict)
        return state_dict


class FactorizedLlamaMlp(eqx.Module, StateDictSerializationMixin):
    """Multi-layer Perceptron
    In comparison with GPT2, FactorizedLlamaMlp adds an up-proj that multiplies with activated gate_proj,
    before down-proj.
    """

    gate_proj: FactorizedLinear  # projection from Embed to Mlp
    up_proj: FactorizedLinear  # projection from Embed to Mlp
    down_proj: FactorizedLinear  # projection from Mlp to Embed
    act: Callable = eqx.static_field()

    @staticmethod
    def init(
        Embed: Axis, Mlp: Axis, Factor: Axis, activation_fn: Union[str, Callable], *, key, use_bias: bool = False
    ) -> "FactorizedLlamaMlp":
        k_fc, k_up_proj, k_down_proj = jrandom.split(key, 3)
        gate_proj = FactorizedLinear.init(
            Out=Mlp, In=Embed, Hidden=Factor, key=k_fc, use_bias=use_bias, out_first=True
        )
        up_proj = FactorizedLinear.init(
            Out=Mlp, In=Embed, Hidden=Factor, key=k_up_proj, use_bias=use_bias, out_first=True
        )
        down_proj = FactorizedLinear.init(
            Out=Embed, In=Mlp, Hidden=Factor, key=k_down_proj, use_bias=use_bias, out_first=True
        )
        if isinstance(activation_fn, str):
            activation_fn = ACT2FN[activation_fn]
        act = activation_fn  # type: ignore
        return FactorizedLlamaMlp(gate_proj, up_proj, down_proj, act)

    @named_call
    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        k_gate, k_up, k_down = maybe_rng_split(key, 3)
        hidden_states = self.gate_proj(x, key=k_gate)
        hidden_states = self.act(hidden_states)
        hidden_states = hidden_states * self.up_proj(x, key=k_up)
        outputs = self.down_proj(hidden_states, key=k_down)
        return outputs


class FactorizedLlamaAttention(StateDictSerializationMixin, eqx.Module):
    config: FactorizedLlamaConfig = eqx.static_field()
    q_proj: FactorizedLinear  # projection from Embed to query
    k_proj: FactorizedLinear  # projection from Embed to key
    v_proj: FactorizedLinear  # projection from Embed to value
    o_proj: FactorizedLinear  # projection from Heads to output

    @staticmethod
    def init(config: FactorizedLlamaConfig, *, key) -> "FactorizedLlamaAttention":
        use_bias = config.use_bias
        Embed = config.Embed
        QHeadsPerGroup = hax.Axis("q_heads_per_group", config.num_heads // config.num_kv_heads)

        k_q, k_k, k_v, k_o = jrandom.split(key, 4)
        q_proj = FactorizedLinear.init(
            In=Embed,
            Hidden=config.Factor,
            Out=(config.KVHeads, QHeadsPerGroup, config.HeadSize),
            key=k_q,
            use_bias=use_bias,
        )
        k_proj = FactorizedLinear.init(
            In=Embed,
            Hidden=config.Factor,
            Out=(config.KVHeads, config.HeadSize),
            key=k_k,
            use_bias=use_bias,
            out_first=True,
        )
        v_proj = FactorizedLinear.init(
            In=Embed,
            Hidden=config.Factor,
            Out=(config.KVHeads, config.HeadSize),
            key=k_v,
            use_bias=use_bias,
            out_first=True,
        )
        o_proj = FactorizedLinear.init(
            In=(config.Heads, config.HeadSize), Hidden=config.Factor, Out=Embed, key=k_o, use_bias=use_bias
        )
        return FactorizedLlamaAttention(config, q_proj, k_proj, v_proj, o_proj)

    @named_call
    def __call__(self, x: NamedArray, mask: Optional[NamedArray | AttentionMask], *, key=None) -> NamedArray:
        key_q, key_k, key_v, key_o = maybe_rng_split(key, 4)

        # reorder heads and position for better training throughput
        q = self.q_proj(x, key=key_q).rearrange((..., "kv_heads", "q_heads_per_group", "position", "head_size"))
        k = self.k_proj(x, key=key_k).rearrange((..., "kv_heads", "position", "head_size"))
        v = self.v_proj(x, key=key_v).rearrange((..., "kv_heads", "position", "head_size"))

        cos, sin = llama_rotary_pos_emb(self.config.HeadSize, x.resolve_axis("position"))
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

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

        attn_output = attn_output.flatten_axes(("kv_heads", "q_heads_per_group"), "heads")
        attn_output = attn_output.astype(x.dtype)

        attn_output = self.o_proj(attn_output, key=key_o)
        return attn_output


class FactorizedLlamaDecoderLayer(StateDictSerializationMixin, eqx.Module):
    config: FactorizedLlamaConfig = eqx.static_field()
    self_attn: FactorizedLlamaAttention
    mlp: FactorizedLlamaMlp
    input_layernorm: LlamaRMSNorm
    post_attention_layernorm: LlamaRMSNorm

    @staticmethod
    def init(config: FactorizedLlamaConfig, *, key) -> "FactorizedLlamaDecoderLayer":
        k_attn, k_mlp = jrandom.split(key, 2)

        attn = FactorizedLlamaAttention.init(config=config, key=k_attn)
        mlp = FactorizedLlamaMlp.init(
            Embed=config.Embed,
            Mlp=config.Mlp,
            Factor=config.Factor,
            activation_fn=config.activation_function,
            key=k_mlp,
            use_bias=config.use_bias,
        )
        ln_1 = LlamaRMSNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)
        ln_2 = LlamaRMSNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)

        return FactorizedLlamaDecoderLayer(config, attn, mlp, ln_1, ln_2)

    @named_call
    def __call__(self, x: NamedArray, mask: Optional[NamedArray | AttentionMask], *, key=None) -> NamedArray:
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


class FactorizedLlamaTransformer(StateDictSerializationMixin, eqx.Module):
    config: FactorizedLlamaConfig = eqx.static_field()
    layers: BlockFoldable[FactorizedLlamaDecoderLayer]
    norm: LlamaRMSNorm

    @staticmethod
    def init(config: FactorizedLlamaConfig, *, key) -> "FactorizedLlamaTransformer":
        S = Stacked
        if not config.scan_layers:
            from haliax.nn.scan import BlockSeq

            S = BlockSeq

        layers = S.init(
            config.Layers, FactorizedLlamaDecoderLayer, gradient_checkpointing=config.gradient_checkpointing
        )(
            config,
            key=shaped_rng_split(key, config.num_layers),
        )
        ln_f = LlamaRMSNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)

        return FactorizedLlamaTransformer(config, layers, ln_f)

    @named_call
    def __call__(self, x: NamedArray, attn_mask: Optional[NamedArray | AttentionMask], *, key) -> NamedArray:
        keys = maybe_rng_split(key, self.config.num_layers) if key is not None else None
        x = self.layers.fold(x, mask=attn_mask, key=keys)
        x = self.norm(x)

        return x

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None):
        if isinstance(self.layers, Stacked):
            state_dict = stack_state_dict(state_dict, prefix=apply_prefix(prefix, "layers"))

        out = super().from_state_dict(state_dict, prefix=prefix)
        return out

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        my_state_dict: StateDict = {}
        super().update_state_dict(my_state_dict, prefix=prefix)

        if isinstance(self.layers, Stacked):
            stacked_dict = unstack_state_dict(my_state_dict, prefix=apply_prefix(prefix, "layers"))
            state_dict.update(stacked_dict)
        else:
            state_dict.update(my_state_dict)

        return state_dict


class FactorizedLlamaEmbedding(StateDictSerializationMixin, eqx.Module):
    """Similar to GPT2 Embedding, except that:
    - FactorizedLlama doesn't have position embedding in the Embedding layer.
    - FactorizedLlama doesn't use dropout.
    """

    Vocab: Axis = eqx.static_field()
    config: FactorizedLlamaConfig = eqx.static_field()
    token_embeddings: NamedArray

    @staticmethod
    def init(Vocab: Axis, config: FactorizedLlamaConfig, *, key) -> "FactorizedLlamaEmbedding":
        token_embeddings = hax.random.normal(key, (Vocab, config.Embed))
        return FactorizedLlamaEmbedding(Vocab, config, token_embeddings)

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


class FactorizedLlamaLMHeadModel(eqx.Module, LmHeadModel[FactorizedLlamaConfig], StateDictSerializationMixin):
    transformer: FactorizedLlamaTransformer
    embeddings: FactorizedLlamaEmbedding
    lm_head: FactorizedLinear

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
    def init(cls, Vocab: Axis, config: FactorizedLlamaConfig, *, key) -> "FactorizedLlamaLMHeadModel":
        k_t, k_emb = jrandom.split(key, 2)
        transformer = FactorizedLlamaTransformer.init(config, key=k_t)
        embeddings = FactorizedLlamaEmbedding.init(Vocab, config, key=k_emb)
        lm_head = FactorizedLinear.init(
            In=config.Embed, Hidden=config.Factor, Out=Vocab, key=k_emb, use_bias=False, out_first=True
        )
        return FactorizedLlamaLMHeadModel(transformer, embeddings, lm_head)

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

    def resize_vocab(self, new_size: int, key=None) -> "LmHeadModel[FactorizedLlamaConfig]":
        new_Vocab = self.Vocab.resize(new_size)
        k1, k2 = maybe_rng_split(key, 2)
        new_embeddings = self.embeddings.resize_embeddings(new_size, key=k1)
        new_lm_matrix = hax.tree_util.resize_axis(self.lm_head.weight, self.Vocab, new_size, key=k2)
        new_lm_head = dataclasses.replace(self.lm_head, Out=new_Vocab, weight=new_lm_matrix)

        return dataclasses.replace(self, embeddings=new_embeddings, lm_head=new_lm_head)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"transformer": "model", "embeddings": None}

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None):
        return super().from_state_dict(state_dict, prefix)

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        super().update_state_dict(state_dict, prefix=prefix)


def _rotate_half(x: NamedArray) -> NamedArray:
    """Rotates half of the hidden dims of the input and concatenates them."""
    HeadSize = x.axes[-1]
    x1 = x[HeadSize, : HeadSize.size // 2]
    x2 = x[HeadSize, HeadSize.size // 2 :]
    out = hax.concatenate(HeadSize, (-x2, x1))
    return out


def _apply_rotary_pos_emb(
    q: NamedArray,  # [batch, position, kv_heads, q_heads_per_group, head_size]
    k: NamedArray,  # [batch, position, kv_heads, head_size]
    cos: NamedArray,  # [position, head_size]
    sin: NamedArray,  # [position, head_size]
) -> Tuple[NamedArray, NamedArray]:
    """Applies rotary position embedding to q and k."""
    q_embed = q * cos + _rotate_half(q) * sin
    k_embed = k * cos + _rotate_half(k) * sin
    return q_embed, k_embed


def llama_rotary_pos_emb(HeadSize: Axis, Pos: Axis, base: int = 10000) -> Tuple[NamedArray, NamedArray]:
    with jax.ensure_compile_time_eval():
        HeadHalfSize = HeadSize.resize(HeadSize.size // 2)
        inv_freq: NamedArray = 1.0 / (base ** (hax.arange(HeadHalfSize, step=2) / HeadSize.size))

        position_ids: NamedArray = hax.arange(Pos)

        freqs = position_ids * inv_freq.broadcast_axis(Pos)
        # This is different from the paper but aligns with HF implementation:
        # It uses a different permutation in order to obtain the same calculation
        emb = hax.concatenate(HeadSize, (freqs, freqs))
        cos = hax.cos(emb)
        sin = hax.sin(emb)
        # This is different from the paper but aligns with HF implementation:
        return cos, sin
