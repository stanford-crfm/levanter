# implements the necessary variations of GPT-2 to work with https://huggingface.co/mosaicml/mpt-7b in haliax/levanter
import dataclasses
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Type, Union

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
from jax.random import PRNGKey

import haliax
import haliax as hax
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call, shaped_rng_split
from haliax.nn.scan import Stacked

import levanter.models.attention
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
from levanter.models.attention import AttentionMask
from levanter.models.lm_model import LmConfig
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.jax_utils import use_cpu_device


silence_transformer_nag()
from transformers.models.mpt.configuration_mpt import MptAttentionConfig as HfMptAttentionConfig  # noqa: E402
from transformers.models.mpt.configuration_mpt import MptConfig as HfMptConfig  # noqa: E402


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
    alibi: bool = True
    alibi_bias_max: Optional[int] = 8
    flash_attention_block_size: Optional[int] = None

    def __post_init__(self):
        assert self.attn_type in ["multihead_attention"], f"attention_type={self.attn_type} not implemented yet."
        assert self.attn_impl in ["torch", "flash"], f"attn_impl={self.attn_impl} not implemented yet."
        assert self.attn_pdrop == 0.0, f"attn_pdrop={self.attn_pdrop} not implemented yet."
        assert (
            not self.attn_uses_sequence_id
        ), f"attn_uses_sequence_id={self.attn_uses_sequence_id} not implemented yet."
        # assert not self.alibi, f'alibi={self.alibi} not implemented yet.'
        assert not self.prefix_lm, f"prefix_lm={self.prefix_lm} not implemented yet."
        assert self.clip_qkv is None, f"clip_qkv={self.clip_qkv} not implemented yet."
        assert not self.softmax_scale, f"softmax_scale={self.softmax_scale} not implemented yet."
        assert not self.qk_ln, f"qk_ln={self.qk_ln} not implemented yet."

    def to_hf(self):
        return HfMptAttentionConfig(
            attn_type=self.attn_type,
            attn_impl=self.attn_impl,
            attn_pdrop=self.attn_pdrop,
            attn_uses_sequence_id=self.attn_uses_sequence_id,
            prefix_lm=self.prefix_lm,
            clip_qkv=self.clip_qkv,
            softmax_scale=self.softmax_scale,
            qk_ln=self.qk_ln,
            alibi=self.alibi,
            alibi_bias_max=self.alibi_bias_max,
        )

    @staticmethod
    def from_hf(config: HfMptAttentionConfig):
        if isinstance(config, dict):
            config = HfMptAttentionConfig(**config)
        try:
            flash_attention_block_size = config.flash_attention_block_size
        except AttributeError:
            flash_attention_block_size = None
        return MptAttentionConfig(
            attn_type=config.attn_type,
            attn_impl=config.attn_impl,
            attn_pdrop=config.attn_pdrop,
            attn_uses_sequence_id=config.attn_uses_sequence_id,
            prefix_lm=config.prefix_lm,
            clip_qkv=config.clip_qkv,
            softmax_scale=config.softmax_scale,
            qk_ln=config.qk_ln,
            alibi=config.alibi,
            alibi_bias_max=config.alibi_bias_max,
            flash_attention_block_size=flash_attention_block_size,
        )


# Haliax-style data class version


@LmConfig.register_subclass("mpt")
@dataclass
class MptConfig(HFCompatConfig):
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    expansion_ratio: int = 4
    max_seq_len: int = 2048
    learned_pos_emb: bool = False
    attn_config: MptAttentionConfig = field(default_factory=MptAttentionConfig)
    logit_scale: Optional[Union[float, str]] = None
    use_bias: bool = True

    # these aren't supported but are here to detect incompatible configs
    embedding_fraction: float = 1.0
    resid_pdrop: float = 0.0
    emb_pdrop: float = 0.0

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

    @property
    def model_type(self) -> Type["MptLmHeadModel"]:
        return MptLmHeadModel

    def hf_checkpoint_converter(self) -> HFCheckpointConverter["MptConfig"]:  # type: ignore
        return HFCheckpointConverter(self, "mosaicml/mpt-7b", trust_remote_code=False)

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
            attn_config=MptAttentionConfig.from_hf(config.attn_config),
            use_bias=not config.no_bias,
            embedding_fraction=config.embedding_fraction,
            logit_scale=config.logit_scale,
        )

    def to_hf_config(self, vocab_size, config_overrides=None):
        if config_overrides is None:
            config_overrides = {}

        attn_config = self.attn_config.to_hf()

        return HfMptConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            expansion_ratio=self.expansion_ratio,
            max_seq_len=self.max_seq_len,
            resid_pdrop=self.resid_pdrop,
            emb_pdrop=self.emb_pdrop,
            learned_pos_emb=self.learned_pos_emb,
            attn_config=attn_config,
            no_bias=not self.use_bias,
            embedding_fraction=self.embedding_fraction,
            logit_scale=self.logit_scale,
            vocab_size=vocab_size,
            **config_overrides,
        )

    def flops_per_token(self, vocab_size: int) -> Optional[float]:
        return lm_flops_per_token(
            hidden_dim=self.d_model,
            intermediate_dim=self.d_model * self.expansion_ratio,
            num_layers=self.n_layers,
            num_kv_heads=self.n_heads,
            num_heads=self.n_heads,
            seq_len=self.max_seq_len,
            vocab_size=vocab_size,
            glu=False,
        )


class MptMlp(eqx.Module, StateDictSerializationMixin):
    up_proj: hnn.Linear  # projection from Embed to Intermediate (typically 4x Embed)
    down_proj: hnn.Linear  # projection from Intermediate to Embed

    @staticmethod
    def init(Embed: Axis, Intermediate: Axis, *, key, use_bias: bool = False):
        k_fc, k_proj = jrandom.split(key, 2)
        up_proj = hnn.Linear.init(Out=Intermediate, In=Embed, key=k_fc, use_bias=use_bias, out_first=True)
        down_proj = hnn.Linear.init(Out=Embed, In=Intermediate, key=k_proj, use_bias=use_bias, out_first=True)
        return MptMlp(up_proj=up_proj, down_proj=down_proj)

    @named_call
    def __call__(self, hidden_states: NamedArray, *, key):
        k_up, k_down = maybe_rng_split(key, 2)
        hidden_states = self.up_proj(hidden_states, key=k_up)
        hidden_states = hnn.gelu(hidden_states, approximate=False)
        hidden_states = self.down_proj(hidden_states, key=k_down)
        return hidden_states


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
        Wqkv = hnn.Linear.init(
            In=config.Embed, Out=(qkv, config.Head, config.HeadDim), key=k_c, use_bias=use_bias, out_first=True
        )
        out_proj = hnn.Linear.init(
            In=(config.Head, config.HeadDim), Out=config.Embed, key=k_proj, use_bias=use_bias, out_first=True
        )
        return MptAttention(config=config, Wqkv=Wqkv, out_proj=out_proj)

    def __call__(
        self,
        hidden_states: NamedArray,
        mask: Optional[AttentionMask | NamedArray],
        bias: Optional[NamedArray],
        key: Optional[PRNGKey],
    ) -> NamedArray:
        k_qkv, k_out = maybe_rng_split(key, 2)
        qkv_out = self.Wqkv(hidden_states, key=k_qkv)
        q, k, v = qkv_out.unbind("qkv")

        # Rename k and v's SeqLen as haliax doesn't support unnamed axes or duplicate axes
        k = k.rename({"position": "key_position"})
        v = v.rename({"position": "key_position"})

        if bias is not None:
            # slice out the bias for this input
            if bias.has_axis("key_position"):
                bias = bias["key_position", hax.dslice(0, k.axis_size("key_position"))]

            if bias.has_axis("position"):
                bias = bias["position", hax.dslice(0, q.axis_size("position"))]

        attn_output = levanter.models.attention.dot_product_attention(
            "position",
            "key_position",
            "head_dim",
            q,
            k,
            v,
            mask=mask,
            bias=bias,
            inference=True,
            use_flash=self.config.attn_config.attn_impl == "flash",
            flash_block_size=self.config.attn_config.flash_attention_block_size,
        )

        attn_output = self.out_proj(attn_output, key=k_out)
        attn_output = attn_output.astype(hidden_states.dtype)

        return attn_output

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None):
        # our c_attn is [embed] -> [3, heads, head_dim] and hf's is the flattened [embed] -> [3 * heads * head_dim]
        # and our c_proj is [heads, head_dim] -> [embed] and hf's is the flattened [heads * head_dim] -> [embed]
        # so we need to reshape the one in the dict before forwarding to the linear
        # keep in mind that everything is vectorized in our implementation, so there's a leading num_layers dim

        d = unflatten_linear_layers(apply_prefix(prefix, "Wqkv"), state_dict, self.Wqkv, out_dims_first_in_dict=True)
        d.update(
            unflatten_linear_layers(
                apply_prefix(prefix, "out_proj"), state_dict, self.out_proj, out_dims_first_in_dict=True
            )
        )

        return super().from_state_dict(d, prefix)

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        # need to undo the reshape we did in from_state_dict
        # reminder that everything is vectorized
        state_dict.update(flatten_linear_layers(apply_prefix(prefix, "Wqkv"), self.Wqkv, out_dims_first_in_dict=True))
        state_dict.update(
            flatten_linear_layers(apply_prefix(prefix, "out_proj"), self.out_proj, out_dims_first_in_dict=True)
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
        self,
        hidden_states: NamedArray,
        attn_bias: Optional[NamedArray],
        attention_mask: Optional[AttentionMask | NamedArray],
        *,
        key,
    ):
        k_attn, k_ffn = maybe_rng_split(key, 2)
        a = self.norm_1(hidden_states)
        b = self.attn(a, bias=attn_bias, mask=attention_mask, key=k_attn)
        hidden_states = hidden_states + b
        m = self.norm_2(hidden_states)
        n = self.ffn(m, key=k_ffn)
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
    def __call__(
        self, hidden_states: NamedArray, attention_mask: Optional[AttentionMask | NamedArray], *, key
    ) -> NamedArray:
        if self.config.attn_config.alibi:
            bias = _mpt_build_alibi_bias(self.config.Head, self.config.KeyPos, self.config.attn_config.alibi_bias_max)
        else:
            bias = None

        key = maybe_rng_split(key, self.Layers.size) if key is not None else None

        hidden_states = self.blocks.fold(hidden_states, attn_bias=bias, attention_mask=attention_mask, key=key)
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
    def Pos(self) -> Axis:
        return self.config.Pos

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
    def __call__(
        self, input_ids: NamedArray, attn_mask: Optional[AttentionMask | NamedArray], *, key=None
    ) -> NamedArray:
        hidden_states = self.wte.embed(input_ids)
        hidden_states = self.transformer(hidden_states, attention_mask=attn_mask, key=key)
        output_logits = self.wte.unembed(hidden_states)

        return output_logits

    def resize_vocab(self, new_size: int, key: Optional[PRNGKey] = None) -> "MptLmHeadModel":
        if new_size == self.vocab_size:
            return self

        return dataclasses.replace(self, wte=self.wte.resize_embeddings(new_size, key=key))

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"wte": "transformer.wte"}

    @staticmethod
    def from_hf_pretrained(
        model_name_or_path="mosaicml/mpt-7b",
        axis_mapping: Optional[Dict[str, str]] = None,
        config=None,
    ) -> "MptLmHeadModel":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, config=config)
        state_dict = model.state_dict()
        # move to cpu
        state_dict = {k: v.cpu().numpy() for k, v in state_dict.items()}
        config = model.config

        del model

        lev_config = MptConfig.from_hf_config(config)  # type: ignore
        Vocab = haliax.Axis("vocab", config.vocab_size)  # type: ignore

        with use_cpu_device():
            lev_model = eqx.filter_eval_shape(MptLmHeadModel.init, Vocab, lev_config, key=PRNGKey(0))
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
        slopes = jnp.concatenate([slopes[1::2], slopes[::2]])[:n_heads]
    return slopes


def _mpt_build_alibi_bias(Heads, KSeqLen, alibi_bias_max=8):

    alibi_bias = jnp.arange(1 - KSeqLen.size, 1, dtype=jnp.int32)
    slopes = _mpt_alibi_gen_slopes(Heads.size, alibi_bias_max)

    slopes = hax.named(slopes, Heads)
    positions = hax.named(alibi_bias, KSeqLen).broadcast_axis(Heads)

    return slopes * positions
