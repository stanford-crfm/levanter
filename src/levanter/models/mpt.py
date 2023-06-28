# implements the necessary variations of GPT-2 to work with https://huggingface.co/mosaicml/mpt-7b in haliax/levanter
import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.random import PRNGKey
from transformers import AutoModelForCausalLM

import haliax
import haliax as hax
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import filter_eval_shape, named_call, shaped_rng_split
from haliax.nn.scan import Stacked
from levanter.compat.hf_checkpoints import HFCheckpointConverter, HFCompatConfig, LmWithHfSerializationMixin
from levanter.compat.torch_serialization import (
    StateDict,
    StateDictSerializationMixin,
    apply_prefix,
    flatten_linear_layer,
    stack_state_dict,
    unflatten_linear_layer,
    unstack_state_dict,
)
from levanter.models.lm_model import LmConfig
from levanter.utils.py_utils import cached_classproperty


init_config_defaults: Dict = {
    "emb_init_std": None,
    "emb_init_uniform_lim": None,
    "fan_mode": "fan_in",
    "init_div_is_residual": True,
    "init_gain": 0.0,
    "init_nonlinearity": "relu",
    "init_std": None,
    "name": "kaiming_normal_",
    "verbose": 0,
}

LazyHfMPTConfig: Optional[Type] = None


def _load_hf_mpt_config():
    global LazyHfMPTConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    LazyHfMPTConfig = get_class_from_dynamic_module("modeling_mpt.MPTConfig", "mosaicml/mpt-7b", "modeling_mpt.py")


@dataclass
class MptAttentionConfig:
    attn_type: str = "multihead_attention"
    attn_impl: str = "torch"
    attn_pdrop: float = 0.0
    attn_uses_sequence_id: bool = False
    prefix_lm: bool = False
    clip_qkv: Optional[float] = None
    softmax_scale: Optional[float] = None
    qk_ln: bool = False
    alibi: bool = False
    alibi_bias_max: Optional[int] = 8

    def __post_init__(self):
        assert self.attn_impl in ["torch"], f"attn_impl={self.attn_impl} not implemented yet."
        assert self.attn_pdrop == 0.0, f"attn_pdrop={self.attn_pdrop} not implemented yet."
        assert (
            not self.attn_uses_sequence_id
        ), f"attn_uses_sequence_id={self.attn_uses_sequence_id} not implemented yet."
        # assert not self.alibi, f'alibi={self.alibi} not implemented yet.'
        assert not self.prefix_lm, f"prefix_lm={self.prefix_lm} not implemented yet."
        assert self.clip_qkv is None, f"clip_qkv={self.clip_qkv} not implemented yet."
        assert not self.softmax_scale, f"softmax_scale={self.softmax_scale} not implemented yet."
        assert not self.qk_ln, f"qk_ln={self.qk_ln} not implemented yet."

    @staticmethod
    def from_dict(d):
        return MptAttentionConfig(**d)


# Haliax-style data class version


@LmConfig.register_subclass("mpt")
@dataclass
class MptConfig(HFCompatConfig):
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    expansion_ratio: int = 4
    max_seq_len: int = 2048
    learned_pos_emb: bool = True
    attn_config: MptAttentionConfig = MptAttentionConfig()
    logit_scale: Optional[Union[float, str]] = None
    use_bias: bool = True

    # these aren't supported but are here to detect incompatible configs
    embedding_fraction: float = 1.0
    resid_pdrop: float = 0.0
    emb_pdrop: float = 0.0
    init_config: Dict[str, Any] = field(default_factory=lambda: {})

    Embed = property(lambda self: Axis("embed", self.d_model))
    Head = property(lambda self: Axis("head", self.n_heads))
    Layers = property(lambda self: Axis("layer", self.n_layers))
    Pos = property(lambda self: Axis("position", self.max_seq_len))
    KeyPos = property(lambda self: Axis("key_position", self.max_seq_len))
    Mlp = property(lambda self: Axis("mlp", self.expansion_ratio * self.d_model))
    HeadDim = property(lambda self: Axis("head_dim", self.d_model // self.n_heads))

    _logit_scale = property(
        lambda self: self.logit_scale if isinstance(self.logit_scale, float) else 1 / jnp.sqrt(self.d_model)
    )

    def __post_init__(self):
        if self.embedding_fraction != 1.0:
            raise ValueError("embedding_fraction not supported yet.")

        if self.resid_pdrop != 0.0:
            raise ValueError("resid_pdrop not supported yet.")

        if self.emb_pdrop != 0.0:
            raise ValueError("emb_pdrop not supported yet.")

        if isinstance(self.logit_scale, str) and self.logit_scale != "inv_sqrt_d_model":
            raise ValueError(
                f"self.logit_scale={self.logit_scale!r} is not recognized as an option; use numeric value or"
                " 'inv_sqrt_d_model'."
            )

        # if self.init_config and self.init_config != init_config_defaults:
        #     raise ValueError("init_config_defaults not supported yet.")

    @property
    def model_type(self) -> Type["MptLmHeadModel"]:
        return MptLmHeadModel

    @cached_classproperty
    def default_hf_checkpoint_converter(cls) -> HFCheckpointConverter["MptConfig"]:  # type: ignore
        return HFCheckpointConverter(
            cls, "mosaicml/mpt-7b@68e1a8e0ebb9b30f3c45c1ef6195980f29063ae2", trust_remote_code=True
        )

    @classmethod
    def from_hf_config(cls, config):
        return MptConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            expansion_ratio=config.expansion_ratio,
            max_seq_len=config.max_seq_len,
            resid_pdrop=config.resid_pdrop,
            emb_pdrop=config.emb_pdrop,
            learned_pos_emb=config.learned_pos_emb,
            attn_config=MptAttentionConfig.from_dict(config.attn_config),
            use_bias=not config.no_bias,
            embedding_fraction=config.embedding_fraction,
            logit_scale=config.logit_scale,
            init_config=config.init_config,
        )

    def to_hf_config(self, vocab_size, config_overrides=None):
        if LazyHfMPTConfig is None:
            _load_hf_mpt_config()

        if config_overrides is None:
            config_overrides = {}

        return LazyHfMPTConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            expansion_ratio=self.expansion_ratio,
            max_seq_len=self.max_seq_len,
            resid_pdrop=self.resid_pdrop,
            emb_pdrop=self.emb_pdrop,
            learned_pos_emb=self.learned_pos_emb,
            attn_config=dataclasses.asdict(self.attn_config),
            no_bias=not self.use_bias,
            embedding_fraction=self.embedding_fraction,
            logit_scale=self.logit_scale,
            init_config=self.init_config,
            vocab_size=vocab_size,
            **config_overrides,
        )


class MptMlp(eqx.Module, StateDictSerializationMixin):
    up_proj: hnn.Linear  # projection from Embed to Intermediate (typically 4x Embed)
    down_proj: hnn.Linear  # projection from Intermediate to Embed

    @staticmethod
    def init(Embed: Axis, Intermediate: Axis, *, key, use_bias: bool = False):
        k_fc, k_proj = jrandom.split(key, 2)
        up_proj = hnn.Linear.init(Out=Intermediate, In=Embed, key=k_fc, use_bias=use_bias)
        down_proj = hnn.Linear.init(Out=Embed, In=Intermediate, key=k_proj, use_bias=use_bias)
        return MptMlp(up_proj=up_proj, down_proj=down_proj)

    @named_call
    def __call__(self, hidden_states: NamedArray):
        hidden_states = self.up_proj(hidden_states)
        hidden_states = hnn.gelu(hidden_states, approximate=False)
        hidden_states = self.down_proj(hidden_states)
        return hidden_states

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None):
        # this is a bit annoying, Torch's Linear is the opposite of ours, so we need to transpose
        ret_dict = {}

        for k, v in state_dict.items():
            if prefix is None or k.startswith(prefix):
                if k.endswith("weight"):
                    ret_dict[k] = v.swapaxes(-1, -2)
                elif k.endswith("bias"):
                    ret_dict[k] = v

        return super().from_state_dict(ret_dict, prefix=prefix)

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None):
        # this is a bit annoying, Torch's Linear is the opposite of ours, so we need to transpose
        ret_dict: StateDict = {}

        super().update_state_dict(ret_dict, prefix=None)

        for k, v in ret_dict.items():
            if k.endswith("weight"):
                state_dict[apply_prefix(prefix, k)] = v.swapaxes(-1, -2)
            else:
                assert k.endswith("bias")
                state_dict[apply_prefix(prefix, k)] = v

        return state_dict


# Attention is the same as GPT-2 Attention, modulo alibi
class MptAttention(StateDictSerializationMixin, eqx.Module):
    Wqkv: hnn.Linear  # input projection from [embed] -> [(q, k, v), heads, head_dim]
    out_proj: hnn.Linear  # output projection from [heads, head_dim] -> [embed]

    config: MptConfig = eqx.static_field()

    @staticmethod
    def init(
        config: MptConfig,
        *,
        key,
        use_bias: bool = True,
    ):
        k_c, k_proj = jrandom.split(key, 2)
        qkv = Axis("qkv", 3)
        Wqkv = hnn.Linear.init(In=config.Embed, Out=(qkv, config.Head, config.HeadDim), key=k_c, use_bias=use_bias)
        out_proj = hnn.Linear.init(In=(config.Head, config.HeadDim), Out=config.Embed, key=k_proj, use_bias=use_bias)
        return MptAttention(config=config, Wqkv=Wqkv, out_proj=out_proj)

    def __call__(
        self, hidden_states: NamedArray, mask: Optional[NamedArray] = None, bias: Optional[NamedArray] = None
    ):
        qkv_out = self.Wqkv(hidden_states)
        q, k, v = qkv_out.unbind("qkv")

        # Rename k and v's SeqLen as haliax doesn't support unnamed axes or duplicate axes
        k = k.rename({self.config.Pos: self.config.KeyPos})
        v = v.rename({self.config.Pos: self.config.KeyPos})

        # mistral tweak: scale norms by 1/sqrt(layer_idx) to prevent blowup
        scale = jax.lax.rsqrt(float(self.config.HeadDim.size))
        # if self.scale_by_inverse_layer_idx:
        #     scale /= layer_idx + 1.0

        # do this first to help keep FP values small
        q = q * scale

        attn_scores = hax.dot(self.config.HeadDim, q, k)

        if bias is not None:
            attn_scores = attn_scores + bias

        if mask is not None:
            attn_scores = attn_scores + (1.0 - mask) * -1e9

        attn_weights = hnn.softmax(attn_scores, axis="key_position").astype(hidden_states.dtype)
        # attn_weights = self.dropout(attn_weights, key=key, inference=inference)

        attn_output = hax.dot("key_position", attn_weights, v)  # [heads, seq_len, head_dim]

        attn_output = self.out_proj(attn_output)
        return attn_output

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None):
        # our c_attn is [embed] -> [3, heads, head_dim] and hf's is the flattened [embed] -> [3 * heads * head_dim]
        # and our c_proj is [heads, head_dim] -> [embed] and hf's is the flattened [heads * head_dim] -> [embed]
        # so we need to reshape the one in the dict before forwarding to the linear
        # keep in mind that everything is vectorized in our implementation, so there's a leading num_layers dim

        d = {}
        d.update(
            unflatten_linear_layer(apply_prefix(prefix, "Wqkv"), state_dict, self.Wqkv, out_dims_first_in_dict=True)
        )
        d.update(
            unflatten_linear_layer(
                apply_prefix(prefix, "out_proj"), state_dict, self.out_proj, out_dims_first_in_dict=True
            )
        )

        return super().from_state_dict(d, prefix)

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        # need to undo the reshape we did in from_state_dict
        # reminder that everything is vectorized
        state_dict.update(flatten_linear_layer(apply_prefix(prefix, "Wqkv"), self.Wqkv, out_dims_first_in_dict=True))
        state_dict.update(
            flatten_linear_layer(apply_prefix(prefix, "out_proj"), self.out_proj, out_dims_first_in_dict=True)
        )
        return state_dict


# Block is broadly similar to GPT-2 Block, with the following changes:
# * fancy layer norm type (we ignore this)
# pdrop seems to be off so we won't use it


class MptBlock(eqx.Module):
    norm_1: eqx.Module
    norm_2: eqx.Module
    attn: MptAttention
    ffn: MptMlp

    @staticmethod
    def init(config: MptConfig, *, key):
        kattn, kmlp = jrandom.split(key, 2)
        norm_1 = hnn.LayerNorm.init(config.Embed, use_bias=config.use_bias)
        attn = MptAttention.init(config, key=kattn, use_bias=config.use_bias)
        norm_2 = hnn.LayerNorm.init(config.Embed, use_bias=config.use_bias)
        ffn = MptMlp.init(config.Embed, config.Mlp, key=kmlp, use_bias=config.use_bias)
        return MptBlock(norm_1, norm_2, attn, ffn)

    @named_call
    def __call__(
        self, hidden_states: NamedArray, attn_bias: Optional[NamedArray], attention_mask: Optional[NamedArray]
    ):
        a = self.norm_1(hidden_states)
        b = self.attn(a, bias=attn_bias, mask=attention_mask)
        hidden_states = hidden_states + b
        m = self.norm_2(hidden_states)
        n = self.ffn(m)
        hidden_states = hidden_states + n
        return hidden_states


class MptTransformer(StateDictSerializationMixin, eqx.Module):
    config: MptConfig = eqx.static_field()
    blocks: Stacked[MptBlock]
    norm_f: hnn.LayerNorm

    @property
    def Layers(self) -> Axis:
        return self.config.Layers

    @staticmethod
    def init(config: MptConfig, *, key):
        blocks = Stacked.init(config.Layers, MptBlock, gradient_checkpointing=True)(
            config, key=shaped_rng_split(key, config.n_layers)
        )
        norm_f = hnn.LayerNorm.init(config.Embed, use_bias=config.use_bias)

        return MptTransformer(config, blocks, norm_f)

    @named_call
    def __call__(self, hidden_states: NamedArray, attention_mask: Optional[NamedArray]) -> NamedArray:
        if self.config.attn_config.alibi:
            bias = _mpt_build_alibi_bias(self.config.Head, self.config.KeyPos, self.config.attn_config.alibi_bias_max)
        else:
            bias = None

        hidden_states = self.blocks.fold(hidden_states, attn_bias=bias, attention_mask=attention_mask)
        hidden_states = self.norm_f(hidden_states)

        return hidden_states

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None):
        # We use a vectorized set of blocks, meaning that we have 1 GptBlock,
        # whereas in hf we have numlayers GptBlocks. So we need to build one GptBlock from numlayers GptBlocks.
        # the individual blocks are named h.0.FOO, h.1.FOO, etc.
        # we want to vectorize them to h.FOO, h.FOO, etc.
        stacked = stack_state_dict(state_dict, prefix=apply_prefix(prefix, "blocks"))
        out = super().from_state_dict(stacked, prefix=prefix)
        return out

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        # this method needs to "devectorize" the blocks, so that we have a list of blocks h.0.FOO, h.1.FOO, etc.
        # first just do the normal thing with our own dict, which we'll post-process
        my_state_dict: StateDict = {}
        super().update_state_dict(my_state_dict, prefix)

        stacked_dict = unstack_state_dict(my_state_dict, apply_prefix(prefix, "blocks"))
        state_dict.update(stacked_dict)

        return state_dict


class MptLmHeadModel(eqx.Module, LmWithHfSerializationMixin):
    wte: hnn.Embedding
    transformer: MptTransformer
    _config: MptConfig = eqx.static_field()

    @property
    def Vocab(self) -> Axis:
        return self.wte.Vocab

    @property
    def config(self) -> MptConfig:
        return self._config

    @classmethod
    def init(cls, Vocab: Axis, config: MptConfig, *, key):
        k_transformer, k_wte = jrandom.split(key, 2)
        wte = hnn.Embedding.init(Vocab, config.Embed, key=k_wte)
        transformer = MptTransformer.init(config, key=k_transformer)

        assert config.emb_pdrop == 0.0, "embedding dropout not supported"
        assert config.resid_pdrop == 0.0, "residual dropout not supported"
        assert config.attn_config.alibi, "alibi attention is required for now"

        return MptLmHeadModel(wte, transformer, config)

    @named_call
    def __call__(self, input_ids: NamedArray, attn_mask: Optional[NamedArray], *, inference, key=None) -> NamedArray:
        # TODO: add back in dropout
        del key
        del inference
        hidden_states = self.wte.embed(input_ids)
        hidden_states = self.transformer(hidden_states, attention_mask=attn_mask)
        output_logits = self.wte.unembed(hidden_states)

        return output_logits

    def _state_dict_key_map(self) -> Optional[Dict[str, Optional[str]]]:
        return {
            "wte": "transformer.wte",
        }

    @staticmethod
    def from_hf_pretrained(
        model_name_or_path="mosaicml/mpt-7b",
        axis_mapping: Optional[Dict[str, str]] = None,
        config=None,
    ) -> "MptLmHeadModel":
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, config=config)
        state_dict = model.state_dict()
        # move to cpu
        state_dict = {k: v.cpu().numpy() for k, v in state_dict.items()}
        config = model.config

        del model

        lev_config = MptConfig.from_hf_config(config)  # type: ignore
        Vocab = haliax.Axis("vocab", config.vocab_size)  # type: ignore

        with jax.default_device(jax.devices("cpu")[0]):
            lev_model = filter_eval_shape(MptLmHeadModel.init, Vocab, lev_config, key=PRNGKey(0))
            lev_model = lev_model.from_state_dict(state_dict)

        if axis_mapping is not None:
            lev_model = haliax.shard_with_axis_mapping(lev_model, axis_mapping)

        return lev_model


def _mpt_alibi_gen_slopes(n_heads, alibi_bias_max=8):
    _n_heads = 2 ** math.ceil(math.log2(n_heads))
    m = jnp.arange(1, _n_heads + 1)
    m = m * (alibi_bias_max / _n_heads)
    slopes = 1.0 / jnp.power(2, m)
    if _n_heads != n_heads:
        slopes = jnp.concat([slopes[1::2], slopes[::2]])[:n_heads]
    return slopes


def _mpt_build_alibi_bias(Heads, KSeqLen, alibi_bias_max=8):

    alibi_bias = jnp.arange(1 - KSeqLen.size, 1, dtype=jnp.int32)
    slopes = _mpt_alibi_gen_slopes(Heads.size, alibi_bias_max)

    slopes = hax.named(slopes, Heads)
    positions = hax.named(alibi_bias, KSeqLen).broadcast_axis(Heads)

    return slopes * positions
