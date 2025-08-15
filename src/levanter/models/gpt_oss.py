from dataclasses import dataclass
from functools import partial
from typing import Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import Array

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call, shaped_rng_split
from haliax.nn.scan import Stacked
from haliax.state_dict import ModuleWithStateDictSerialization

import levanter.tracker
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.layers.attention import AttentionMask, AttentionWithSink
from levanter.layers.rotary import RotaryEmbeddingsConfig
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.models.mistral import MistralConfig
from levanter.models.llama import LlamaEmbedding
from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.types import BlockFoldable
from transformers import (
    AutoConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    PretrainedConfig as HfConfig,
)


class HfGptOssConfig(HfConfig):
    """Minimal HuggingFace-style config for GPT-OSS used in tests."""

    model_type = "gpt_oss"

    def __init__(
        self,
        num_hidden_layers: int = 2,
        num_local_experts: int = 8,
        vocab_size: int = 32000,
        hidden_size: int = 512,
        intermediate_size: int = 512,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 8,
        sliding_window: int | None = None,
        num_experts_per_tok: int = 2,
        router_aux_loss_coef: float = 0.0,
        output_router_logits: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        rms_norm_eps: float = 1e-6,
        max_position_embeddings: int = 2048,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.num_hidden_layers = num_hidden_layers
        self.num_local_experts = num_local_experts
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.sliding_window = sliding_window
        self.num_experts_per_tok = num_experts_per_tok
        self.router_aux_loss_coef = router_aux_loss_coef
        self.output_router_logits = output_router_logits
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings


AutoConfig.register("gpt_oss", HfGptOssConfig)


@LmConfig.register_subclass("gpt_oss")
@dataclass(frozen=True)
class GptOssConfig(MistralConfig):
    """Configuration for the GPT-OSS model."""

    num_experts_per_tok: int = 2
    num_local_experts: int = 8
    router_aux_loss_coef: Optional[float] = 0.01
    output_router_logits: bool = False
    sliding_window: Optional[int] = None
    reference_checkpoint: str = "openai/gpt-oss"
    tokenizer: Optional[str] = None

    # Axis helpers
    Experts = property(lambda self: Axis(name="experts", size=self.num_local_experts))
    TopExperts = property(lambda self: Axis(name="top_experts", size=self.num_experts_per_tok))
    Mlp = property(lambda self: Axis(name="mlp", size=self.intermediate_dim))

    def __post_init__(self):
        super().__post_init__()
        assert (
            self.num_experts_per_tok <= self.num_local_experts
        ), "num_experts_per_tok must be <= num_local_experts"

    def flops_per_token(self, vocab_size: int):
        return lm_flops_per_token(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            seq_len=self.seq_len,
            vocab_size=vocab_size,
            glu=True,
        )

    def hf_checkpoint_converter(
        self,
        ref_checkpoint: Optional[str] = None,
        tokenizer: Optional[Union[str, "PreTrainedTokenizer", "PreTrainedTokenizerFast"]] = None,
    ) -> HFCheckpointConverter["GptOssConfig"]:  # type: ignore[name-defined]
        return HFCheckpointConverter(
            self.__class__,
            reference_checkpoint=self.reference_checkpoint if ref_checkpoint is None else ref_checkpoint,
            trust_remote_code=True,
            tokenizer=self.tokenizer if tokenizer is None else tokenizer,
            HfConfigClass=HfGptOssConfig,
        )

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig):
        rope_theta = getattr(hf_config, "rope_theta", 10000.0)
        rope_scaling = getattr(hf_config, "rope_scaling", None)
        rope_config = RotaryEmbeddingsConfig.from_hf_config(rope_theta, rope_scaling)
        return GptOssConfig(
            seq_len=hf_config.max_position_embeddings,
            hidden_dim=hf_config.hidden_size,
            intermediate_dim=hf_config.intermediate_size,
            num_layers=hf_config.num_hidden_layers,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            activation_function=ActivationFunctionEnum(hf_config.hidden_act),
            initializer_range=hf_config.initializer_range,
            layer_norm_epsilon=hf_config.rms_norm_eps,
            sliding_window=getattr(hf_config, "sliding_window", None),
            num_experts_per_tok=hf_config.num_experts_per_tok,
            num_local_experts=hf_config.num_local_experts,
            router_aux_loss_coef=getattr(hf_config, "router_aux_loss_coef", None),
            output_router_logits=getattr(hf_config, "output_router_logits", False),
            rope=rope_config,
            tie_word_embeddings=getattr(hf_config, "tie_word_embeddings", False),
        )

    def to_hf_config(self, vocab_size: int, config_overrides: Optional[dict] = None) -> "HfGptOssConfig":
        if config_overrides is None:
            config_overrides = {}
        rope_theta, rope_scaling = self.rope.to_hf_config()
        return HfGptOssConfig(
            max_position_embeddings=self.seq_len,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_dim,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            sliding_window=self.sliding_window if self.sliding_window is not None else 0,
            num_local_experts=self.num_local_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            router_aux_loss_coef=self.router_aux_loss_coef or 0.0,
            output_router_logits=self.output_router_logits,
            vocab_size=vocab_size,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            rms_norm_eps=self.layer_norm_epsilon,
            tie_word_embeddings=self.tie_word_embeddings,
            **config_overrides,
        )

    @property
    def model_type(cls) -> type["GptOssLMHeadModel"]:
        return GptOssLMHeadModel

    def mk_LayerNorm(self, axis: Axis) -> hnn.RmsNorm:
        return hnn.RmsNorm.init(axis, eps=self.layer_norm_epsilon, use_bias=self.use_bias)


class GptOssExperts(eqx.Module):
    """MoE expert block used in GPT-OSS."""

    gate_up_proj: hnn.MoELinear
    down_proj: hnn.MoELinear
    Embed: Axis = eqx.field(static=True)
    Mlp: Axis = eqx.field(static=True)
    alpha: float = eqx.field(static=True, default=1.702)
    limit: float = eqx.field(static=True, default=7.0)

    @staticmethod
    def init(Experts: Axis, Embed: Axis, Mlp: Axis, *, key, use_bias: bool = True) -> "GptOssExperts":
        k1, k2 = jrandom.split(key, 2)
        Mlp2 = Mlp.resize(Mlp.size * 2)
        gate_up_proj = hnn.MoELinear.init(Experts=Experts, Out=Mlp2, In=Embed, key=k1, use_bias=use_bias)
        down_proj = hnn.MoELinear.init(Experts=Experts, Out=Embed, In=Mlp, key=k2, use_bias=use_bias)
        return GptOssExperts(gate_up_proj, down_proj, Embed, Mlp)

    @named_call
    def __call__(self, x: NamedArray, group_sizes: NamedArray, *, key=None) -> NamedArray:
        k1, k2 = maybe_rng_split(key, 2)
        Mlp2 = self.Mlp.resize(self.Mlp.size * 2)
        gate_up = self.gate_up_proj(x, group_sizes, key=k1).rename({self.gate_up_proj.Out: Mlp2})
        gate, up = hax.split(gate_up, Mlp2, [self.Mlp.size, self.Mlp.size])
        gate = hax.clip(gate, -self.limit, self.limit)
        up = hax.clip(up, -self.limit, self.limit)
        glu = gate * hnn.sigmoid(self.alpha * gate)
        hidden = (up + 1.0) * glu
        out = self.down_proj(hidden, group_sizes, key=k2)
        return out


class GptOssSparseMoeBlock(eqx.Module):
    config: GptOssConfig = eqx.field(static=True)
    gate: hnn.Linear
    experts: GptOssExperts

    @staticmethod
    def init(config: GptOssConfig, *, key) -> "GptOssSparseMoeBlock":
        k_gate, k_exp = jrandom.split(key, 2)
        gate = hnn.Linear.init(config.Embed, config.Experts, key=k_gate, use_bias=config.use_bias)
        experts = GptOssExperts.init(
            Experts=config.Experts,
            Embed=config.Embed,
            Mlp=config.Mlp,
            key=k_exp,
            use_bias=config.use_bias,
        )
        return GptOssSparseMoeBlock(config, gate, experts)

    def _route(self, router_probs: NamedArray, Token: Axis, TopExperts: Axis):
        topk = jax.lax.top_k(router_probs.array, self.config.num_experts_per_tok)
        selected_weights_ = topk[0]
        selected_experts_ = topk[1]
        selected_weights = NamedArray(selected_weights_, axes=(Token, TopExperts))
        selected_experts = NamedArray(selected_experts_, axes=(Token, TopExperts))
        selected_weights = selected_weights / hax.sum(selected_weights, axis=TopExperts, keepdims=True)
        return selected_weights, selected_experts

    def _permute(self, x_flat: NamedArray, topk_idx_flat: NamedArray, TokenRepeat: Axis):
        Experts = self.config.Experts

        @partial(
            hax.shard_map,
            mesh=hax.partitioning._get_mesh(),
            in_specs=(
                hax.partitioning.pspec_for_axis(x_flat.axes),
                hax.partitioning.pspec_for_axis(topk_idx_flat.axes),
            ),
            out_specs=(
                hax.partitioning.pspec_for_axis((TokenRepeat, self.config.Embed)),
                hax.partitioning.pspec_for_axis((Experts,)),
                hax.partitioning.pspec_for_axis((TokenRepeat,)),
            ),
            check_rep=False,
        )
        def permute_sharded(x_flat_: Array, topk_idx_flat_: Array):
            sort_idx_ = jnp.argsort(topk_idx_flat_, axis=-1)
            x_repeat_sort_ = jnp.take(x_flat_, sort_idx_ // self.config.num_experts_per_tok, axis=0)
            group_sizes_ = jnp.bincount(topk_idx_flat_, length=self.config.num_local_experts)
            return x_repeat_sort_, group_sizes_, sort_idx_

        with jax.named_scope("permute"):
            x_repeat_sort_, group_sizes_, sort_idx_ = permute_sharded(x_flat.array, topk_idx_flat.array)
            x_repeat_sort = NamedArray(x_repeat_sort_, axes=(TokenRepeat, self.config.Embed))
            group_sizes = NamedArray(group_sizes_, axes=(Experts,))
            sort_idx = NamedArray(sort_idx_, axes=(TokenRepeat,))
        return x_repeat_sort, group_sizes, sort_idx

    def _unpermute(
        self,
        out_repeat_sort: NamedArray,
        sort_idx: NamedArray,
        topk_weights: NamedArray,
        Token: Axis,
        TokenRepeat: Axis,
        TopExperts: Axis,
    ):
        @partial(
            hax.shard_map,
            mesh=hax.partitioning._get_mesh(),
            in_specs=(
                hax.partitioning.pspec_for_axis(out_repeat_sort.axes),
                hax.partitioning.pspec_for_axis(sort_idx.axes),
            ),
            out_specs=hax.partitioning.pspec_for_axis((Token, TopExperts, self.config.Embed)),
            check_rep=False,
        )
        def unpermute_sharded(out_repeat_sort_: Array, sort_idx_: Array):
            inv_sort_idx_ = jnp.argsort(sort_idx_)
            out_repeat_ = jnp.take(out_repeat_sort_, inv_sort_idx_, axis=0)
            out_repeat_unflat_ = jnp.reshape(
                out_repeat_, (-1, self.config.num_experts_per_tok, self.config.hidden_dim)
            )
            return out_repeat_unflat_

        with jax.named_scope("unpermute"):
            out_repeat_unflat_ = unpermute_sharded(out_repeat_sort.array, sort_idx.array)
            out_repeat_unflat = NamedArray(out_repeat_unflat_, axes=(Token, TopExperts, self.config.Embed))
        return out_repeat_unflat

    @named_call
    def __call__(self, x: NamedArray, *, key=None):
        if x.has_axis("batch"):
            squash_axes = [x.resolve_axis("batch"), x.resolve_axis(self.config.Pos.name)]
        else:
            squash_axes = [x.resolve_axis(self.config.Pos.name)]
        Experts = self.config.Experts
        TopExperts = self.config.TopExperts

        k_gate, k_experts, key = maybe_rng_split(key, 3)

        x_flat = hax.flatten_axes(x, old_axes=squash_axes, new_axis="token")
        Token = x_flat.resolve_axis("token")

        router_logits = self.gate(x_flat, key=k_gate)
        router_probs = hnn.softmax(router_logits, axis=Experts)
        topk_weights, topk_idx = self._route(router_probs, Token, TopExperts)

        topk_idx_flat = hax.flatten_axes(topk_idx, old_axes=[Token, TopExperts], new_axis="token_repeat")
        TokenRepeat = topk_idx_flat.resolve_axis("token_repeat")
        x_repeat_sort, group_sizes, sort_idx = self._permute(x_flat, topk_idx_flat, TokenRepeat)

        out_repeat_sort = self.experts(x_repeat_sort, group_sizes, key=k_experts)

        out_repeat_unflat = self._unpermute(
            out_repeat_sort, sort_idx, topk_weights, Token, TokenRepeat, TopExperts
        )

        out = out_repeat_unflat.dot(topk_weights, axis=TopExperts)

        extras = {}
        expert_loads = group_sizes / hax.sum(group_sizes, axis=Experts)
        extras = {"expert_loads": expert_loads}
        if self.config.router_aux_loss_coef is not None:
            f = expert_loads * self.config.num_local_experts / self.config.num_experts_per_tok
            p = hax.mean(router_probs, axis=Token)
            extras["load_balancing_loss"] = self.config.router_aux_loss_coef * hax.sum(f * p, axis=Experts)
        return hax.unflatten_axis(out, axis=Token, new_axes=squash_axes), extras


class GptOssDecoderLayer(eqx.Module):
    config: GptOssConfig = eqx.field(static=True)
    self_attn: AttentionWithSink
    block_sparse_moe: GptOssSparseMoeBlock
    input_layernorm: hnn.RmsNorm
    post_attention_layernorm: hnn.RmsNorm

    @staticmethod
    def init(config: GptOssConfig, *, key) -> "GptOssDecoderLayer":
        k_attn, k_moe = jrandom.split(key, 2)
        attn_config = config.attention_config()
        attn = AttentionWithSink.init(attn_config, key=k_attn)
        block_sparse_moe = GptOssSparseMoeBlock.init(config, key=k_moe)
        ln_1 = hnn.RmsNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)
        ln_2 = hnn.RmsNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)
        return GptOssDecoderLayer(config, attn, block_sparse_moe, ln_1, ln_2)

    @named_call
    def __call__(self, x: NamedArray, mask: Optional[NamedArray | AttentionMask], *, key=None):
        k_attn, k_mlp = maybe_rng_split(key, 2)
        residual = x
        x = self.input_layernorm(x)
        attn_output = self.self_attn(x=x, mask=mask, key=k_attn)
        x = residual + attn_output

        residual = x
        x = self.post_attention_layernorm(x)
        moe_output, extras = self.block_sparse_moe(x, key=k_mlp)
        output = residual + moe_output
        return output, extras


class GptOssTransformer(eqx.Module):
    config: GptOssConfig = eqx.field(static=True)
    layers: BlockFoldable[GptOssDecoderLayer]
    norm: hnn.RmsNorm

    @staticmethod
    def init(config: GptOssConfig, *, key) -> "GptOssTransformer":
        S = Stacked
        if not config.scan_layers:
            from haliax.nn.scan import BlockSeq

            S = BlockSeq
        layers = S.init(config.Layers, GptOssDecoderLayer, gradient_checkpointing=config.gradient_checkpointing)(
            config, key=shaped_rng_split(key, config.num_layers)
        )
        ln_f = config.mk_LayerNorm(config.Embed)
        return GptOssTransformer(config, layers, ln_f)

    @named_call
    def __call__(self, x: NamedArray, attn_mask: Optional[NamedArray], *, key, pos_ids: NamedArray | None = None):
        keys = maybe_rng_split(key, self.config.num_layers) if key is not None else None
        x, extras = self.layers.scan(x, mask=attn_mask, key=keys)
        x = self.norm(x)

        expert_loads = extras["expert_loads"]
        entropy = -hax.sum(expert_loads * hax.log(expert_loads + 1e-6), axis=self.config.Experts)
        stats = {}
        for i in range(self.config.num_layers):
            stats[f"moe/layer{i}/routing_entropy"] = entropy.array[i]
            for j in range(self.config.num_local_experts):
                stats[f"moe/layer{i}/expert{j}_load"] = expert_loads.array[i, j]
        if self.config.router_aux_loss_coef is not None:
            extras["load_balancing_loss"] = hax.sum(extras["load_balancing_loss"], axis=self.config.Layers)
            stats["train/load_balancing_loss"] = extras["load_balancing_loss"].array
        levanter.tracker.jit_log(stats)
        return x, extras


class GptOssLMHeadModel(ModuleWithStateDictSerialization, LmHeadModel[GptOssConfig]):
    transformer: GptOssTransformer
    embeddings: LlamaEmbedding
    lm_head: Optional[hnn.Linear]

    @property
    def config(self) -> GptOssConfig:  # type: ignore[override]
        return self.transformer.config

    @property
    def vocab_size(self) -> int:  # type: ignore[override]
        return self.Vocab.size

    @property
    def Vocab(self) -> Axis:  # type: ignore[override]
        return self.embeddings.Vocab

    @classmethod
    def init(cls, Vocab: Axis, config: GptOssConfig, *, key) -> "GptOssLMHeadModel":
        k_t, k_emb = jrandom.split(key, 2)
        transformer = GptOssTransformer.init(config, key=k_t)
        embeddings = LlamaEmbedding.init(Vocab, config, key=k_emb)
        if config.tie_word_embeddings:
            lm_head = None
        else:
            lm_head = hnn.Linear.init(In=config.Embed, Out=Vocab, key=k_emb, use_bias=False, out_first=True)
        return GptOssLMHeadModel(transformer, embeddings, lm_head)

    def __call__(
        self,
        input_ids: NamedArray,
        attn_mask: Optional[Union[NamedArray, AttentionMask]] = None,
        pos_ids: NamedArray | None = None,
        *,
        key=None,
    ) -> NamedArray:
        k_t, k_head = maybe_rng_split(key, 2)
        x = self.embeddings.embed(input_ids)
        x, _ = self.transformer(x, attn_mask=attn_mask, key=k_t, pos_ids=pos_ids)
        if self.lm_head is None:
            lm_logits = self.embeddings.project(x)
        else:
            lm_logits = self.lm_head(x, key=k_head)
        return lm_logits

    def activations(
        self,
        input_ids: NamedArray,
        attn_mask: Optional[AttentionMask | NamedArray] = None,
        pos_ids: NamedArray | None = None,
        *,
        key=None,
    ) -> NamedArray:
        x = self.embeddings.embed(input_ids)
        x, extras = self.transformer(x, attn_mask=attn_mask, key=key, pos_ids=pos_ids)
        aux_loss = 0
        if self.config.router_aux_loss_coef is not None:
            aux_loss += extras.get("load_balancing_loss", 0)
        return x, aux_loss

    def _state_dict_key_map(self) -> dict[str, Optional[str]]:
        return {"transformer": "model", "embeddings": None}
