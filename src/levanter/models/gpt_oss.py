from dataclasses import dataclass
import dataclasses
from typing import Dict, Optional, Type, Union

import equinox as eqx
import jax
import jax.lax
import jax.random as jrandom

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call, shaped_rng_split
from haliax.nn.scan import Stacked
from haliax.state_dict import ModuleWithStateDictSerialization
from haliax.specialized_fns import top_k
from haliax.ops import bincount

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
from transformers import GptOssConfig as HfGptOssConfig


@LmConfig.register_subclass("gpt_oss")
@dataclass(frozen=True)
class GptOssConfig(MistralConfig):
    """Configuration for the GPT-OSS model."""

    num_experts_per_tok: int = 2
    num_local_experts: int = 8
    router_aux_loss_coef: Optional[float] = 0.01
    output_router_logits: bool = False
    sliding_window: Optional[int] = None
    layer_types: Optional[tuple[str, ...]] = None
    reference_checkpoint: str = "unsloth/gpt-oss-20b-BF16"
    tokenizer: Optional[str] = None
    
    # GPT-OSS specific: Explicit head dimension (not hidden_dim // num_heads)
    head_dim: int = 64

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

    @property
    def model_type(cls) -> type["GptOssLMHeadModel"]:
        return GptOssLMHeadModel

    def mk_LayerNorm(self, axis: Axis) -> hnn.RmsNorm:
        return hnn.RmsNorm.init(axis, eps=self.layer_norm_epsilon, use_bias=False)

    def attention_config(self) -> "AttentionConfig":
        """Override attention_config to use explicit head_dim instead of calculated value.
        
        GPT-OSS uses explicit head_dim from HF config, not hidden_dim // num_heads.
        This ensures correct projection shapes for grouped query attention.
        """
        from levanter.layers.attention import AttentionConfig
        
        return AttentionConfig(
            Embed=self.Embed,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            use_bias=self.use_bias,
            upcast_attn=self.upcast_attn,
            attn_backend=self.attn_backend,
            flash_attention_block_size=self.flash_attention_block_size,
            rope=self.rope,
        )

    # HF compatibility -----------------------------------------------------------------

    def hf_checkpoint_converter(
        self, ref_checkpoint: Optional[str] = None, tokenizer: Optional[str] = None
    ) -> HFCheckpointConverter["GptOssConfig"]:  # type: ignore[misc]

        return HFCheckpointConverter(
            self.__class__,
            reference_checkpoint=self.reference_checkpoint if ref_checkpoint is None else ref_checkpoint,
            trust_remote_code=True,
            tokenizer=self.tokenizer if tokenizer is None else tokenizer,
            HfConfigClass=HfGptOssConfig,
        )

    @classmethod
    def from_hf_config(cls, hf_config):
        rope_config = RotaryEmbeddingsConfig.from_hf_config(
            hf_config.rope_theta, getattr(hf_config, "rope_scaling", None)
        )
        layer_types = tuple(hf_config.layer_types) if getattr(hf_config, "layer_types", None) is not None else None
        # GPT-OSS models use bias parameters for attention and MoE components
        use_bias = getattr(hf_config, "attention_bias", True)
        # GPT-OSS models have explicit head_dim (not hidden_size // num_heads)
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        return cls(
            seq_len=hf_config.max_position_embeddings,
            hidden_dim=hf_config.hidden_size,
            intermediate_dim=hf_config.intermediate_size,
            num_layers=hf_config.num_hidden_layers,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            activation_function=ActivationFunctionEnum(hf_config.hidden_act),
            initializer_range=hf_config.initializer_range,
            layer_norm_epsilon=hf_config.rms_norm_eps,
            sliding_window=hf_config.sliding_window,
            rope=rope_config,
            tie_word_embeddings=hf_config.tie_word_embeddings,
            num_experts_per_tok=hf_config.num_experts_per_tok,
            num_local_experts=hf_config.num_local_experts,
            router_aux_loss_coef=getattr(hf_config, "router_aux_loss_coef", None),
            output_router_logits=getattr(hf_config, "output_router_logits", False),
            layer_types=layer_types,
            use_bias=use_bias,
            head_dim=head_dim,
        )

    def to_hf_config(self, vocab_size: int, config_overrides: Optional[Dict] = None):
        if config_overrides is None:
            config_overrides = {}

        rope_theta, rope_scaling = self.rope.to_hf_config()

        return HfGptOssConfig(
            num_hidden_layers=self.num_layers,
            num_local_experts=self.num_local_experts,
            vocab_size=vocab_size,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_dim,
            head_dim=self.head_dim,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            sliding_window=self.sliding_window if self.sliding_window is not None else 0,
            rope_theta=rope_theta,
            tie_word_embeddings=self.tie_word_embeddings,
            hidden_act=self.activation_function.name,
            initializer_range=self.initializer_range,
            max_position_embeddings=self.seq_len,
            rms_norm_eps=self.layer_norm_epsilon,
            rope_scaling=rope_scaling,
            attention_dropout=0.0,
            num_experts_per_tok=self.num_experts_per_tok,
            router_aux_loss_coef=self.router_aux_loss_coef if self.router_aux_loss_coef is not None else 0.0,
            output_router_logits=self.output_router_logits,
            layer_types=list(self.layer_types) if self.layer_types is not None else None,
            **config_overrides,
        )


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
        gate, up = hax.split(gate_up, Mlp2, [self.Mlp, self.Mlp])
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
        selected_weights, selected_experts = top_k(
            router_probs, 
            axis=self.config.Experts, 
            k=self.config.num_experts_per_tok,
            new_axis=TopExperts
        )
        normalizer = hax.sum(selected_weights, axis=TopExperts).broadcast_axis(TopExperts)
        selected_weights = selected_weights / normalizer
        return selected_weights, selected_experts

    def _permute(self, x_flat: NamedArray, topk_idx_flat: NamedArray, TokenRepeat: Axis):
        Experts = self.config.Experts
        Token = x_flat.axes[0]  # Get the token axis statically
        
        with jax.named_scope("permute"):
            # Sort indices by expert assignment
            sort_idx = hax.argsort(topk_idx_flat, axis=TokenRepeat)
            
            # Create expert assignments for tokens by dividing by num_experts_per_tok
            # Use raw array for integer division to avoid tracer leaks
            token_assignments_raw = sort_idx.array // self.config.num_experts_per_tok
            token_assignments = hax.named(token_assignments_raw, sort_idx.axes)
            x_repeat_sort = hax.take(x_flat, axis=Token, index=token_assignments)
            
            # Count how many tokens are assigned to each expert
            group_sizes = bincount(
                topk_idx_flat, 
                Counts=Experts, 
                minlength=self.config.num_local_experts
            )
            
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
        with jax.named_scope("unpermute"):
            # Get the inverse sort indices to restore original order
            inv_sort_idx = hax.argsort(sort_idx, axis=TokenRepeat)
            
            # Restore the original token order
            out_repeat = hax.take(out_repeat_sort, axis=TokenRepeat, index=inv_sort_idx)
            
            # Reshape to (Token, TopExperts, Embed) dimensions
            out_repeat_unflat = hax.unflatten_axis(
                out_repeat, 
                axis=TokenRepeat, 
                new_axes=(Token, TopExperts)
            )
            
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

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        """Map from Levanter MoE block names to HF checkpoint names."""
        return {
            "gate": "router",
        }


class GptOssDecoderLayer(ModuleWithStateDictSerialization, eqx.Module):
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
        ln_1 = hnn.RmsNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=False)
        ln_2 = hnn.RmsNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=False)
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

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        """Map from Levanter decoder layer names to HF checkpoint names."""
        return {
            "block_sparse_moe": "mlp",
        }


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
        # GPT-OSS specific: Create per-layer attention masks based on layer_types
        if self.config.layer_types is not None and attn_mask is not None:
            # Create per-layer masks based on layer_types configuration
            layer_masks = []
            for i in range(self.config.num_layers):
                layer_type = self.config.layer_types[i % len(self.config.layer_types)]
                if layer_type == "sliding_attention" and self.config.sliding_window is not None:
                    # Apply sliding window to this layer
                    layer_mask = attn_mask.with_sliding_window(self.config.sliding_window)
                else:
                    # Use base attention mask (full attention)
                    layer_mask = attn_mask
                layer_masks.append(layer_mask)
            
            # Since we now default to BlockSeq, we can use its natural per-layer capabilities
            # BlockSeq iterates through layers explicitly, so we can apply different masks
            carry = x
            all_extras = {"expert_loads": [], "load_balancing_loss": []}
            
            # Manually iterate through BlockSeq layers with per-layer masks
            for i in range(self.config.num_layers):
                layer = self.layers.blocks[i]  # BlockSeq provides direct access
                layer_key = keys[i] if keys is not None else None
                carry, layer_extras = layer(carry, mask=layer_masks[i], key=layer_key)
                
                # Accumulate extras
                if "expert_loads" in layer_extras:
                    all_extras["expert_loads"].append(layer_extras["expert_loads"])
                if "load_balancing_loss" in layer_extras:
                    all_extras["load_balancing_loss"].append(layer_extras["load_balancing_loss"])
            
            # Stack the extras along the layer axis
            if all_extras["expert_loads"]:
                all_extras["expert_loads"] = hax.stack(self.config.Layers, all_extras["expert_loads"])
            if all_extras["load_balancing_loss"]:
                all_extras["load_balancing_loss"] = hax.stack(self.config.Layers, all_extras["load_balancing_loss"])
            
            x, extras = carry, all_extras
        else:
            # Standard scan for uniform masks
            x, extras = self.layers.scan(x, mask=attn_mask, key=keys)
        
        x = self.norm(x)

        expert_loads = extras["expert_loads"]
        entropy = -hax.sum(expert_loads * hax.log(expert_loads + 1e-6), axis=self.config.Experts)
        stats = {}
        # Use stop_gradient to prevent tracer leaks when logging
        entropy_stopped = jax.lax.stop_gradient(entropy.array)
        expert_loads_stopped = jax.lax.stop_gradient(expert_loads.array)
        for i in range(self.config.num_layers):
            stats[f"moe/layer{i}/routing_entropy"] = entropy_stopped[i]
            for j in range(self.config.num_local_experts):
                stats[f"moe/layer{i}/expert{j}_load"] = expert_loads_stopped[i, j]
        if self.config.router_aux_loss_coef is not None:
            extras["load_balancing_loss"] = hax.sum(extras["load_balancing_loss"], axis=self.config.Layers)
            stats["train/load_balancing_loss"] = jax.lax.stop_gradient(extras["load_balancing_loss"].array)
        levanter.tracker.jit_log(stats)
        return x, extras

    def from_state_dict(self, state_dict, prefix: str | None = None):
        """Custom state dict loading to handle GPT-OSS specific transformations.
        
        1. Handles sinks tensor conversion from (layers, num_heads) to (layers, kv_heads, q_heads_per_group)
        2. Adds .weight/.bias suffixes to MoE expert parameters to match Haliax expectations
        3. Transforms router bias key: mlp.router.bias -> block_sparse_moe.gate.bias
        """
        from haliax._src.state_dict import default_eqx_module_from_state_dict
        import jax.numpy as jnp
        
        # DEBUG: Log entry to the method
        print(f"🔍 GPT-OSS Transformer from_state_dict called with prefix: {prefix}", flush=True)
        print(f"📊 State dict has {len(state_dict)} keys", flush=True)
        
        # DEBUG: Show sample of keys to understand structure
        all_keys = list(state_dict.keys())
        expert_keys = [k for k in all_keys if 'experts.' in k]
        router_keys = [k for k in all_keys if 'router' in k or 'gate' in k]
        
        print(f"🔧 Expert keys found: {len(expert_keys)}", flush=True)
        if expert_keys:
            print(f"   First expert key: {expert_keys[0]}", flush=True)
            print(f"   Sample expert key shapes:", flush=True)
            for key in expert_keys[:3]:  # Show first 3
                if key in state_dict:
                    tensor = state_dict[key]
                    print(f"     {key} -> {tensor.shape}", flush=True)
        
        print(f"🎯 Router/gate keys found: {len(router_keys)}", flush=True)
        if router_keys:
            for key in router_keys[:3]:  # Show first 3
                if key in state_dict:
                    tensor = state_dict[key]
                    print(f"   {key} -> {tensor.shape}", flush=True)
        
        # Make a copy to avoid mutating the original
        state_dict = dict(state_dict)
        
        # STEP 1: Add .weight/.bias suffixes to MoE expert parameters
        # GPT-OSS checkpoint has: experts.gate_up_proj, experts.gate_up_proj_bias  
        # Haliax expects: experts.gate_up_proj.weight, experts.gate_up_proj.bias
        print("🔄 Starting MoE parameter transformations...", flush=True)
        keys_to_transform = list(state_dict.keys())
        transformation_count = 0
        
        for key in keys_to_transform:
            if 'experts.' in key:
                print(f"🔧 Processing expert key: {key}", flush=True)
                if key in state_dict:
                    tensor = state_dict[key]
                    print(f"   Original shape: {tensor.shape}", flush=True)
                
                transformation_count += 1
                
                # Handle expert bias parameters: experts.gate_up_proj_bias -> experts.gate_up_proj.bias
                if key.endswith('_bias'):
                    new_key = key[:-5] + '.bias'  # Replace '_bias' with '.bias'
                    print(f"   Bias transformation: {key} -> {new_key}", flush=True)
                    bias_tensor = state_dict.pop(key)
                    print(f"   Bias tensor shape: {bias_tensor.shape}", flush=True)
                    
                    # HF format: (num_experts, feature_dim) -> Levanter format: shared bias (feature_dim,)
                    # In Levanter MoELinear, all experts share the same bias parameter
                    if len(bias_tensor.shape) == 2:  # (num_experts, feature_dim)
                        # Use first expert's bias as the shared bias (MoELinear design)
                        shared_bias = bias_tensor[0]  # Shape: (feature_dim,)
                        state_dict[new_key] = shared_bias
                        print(f"   Converted to shared bias shape: {shared_bias.shape}", flush=True)
                    else:
                        state_dict[new_key] = bias_tensor
                        print(f"   Direct bias shape: {bias_tensor.shape}", flush=True)
                    
                # Handle expert weight parameters: experts.gate_up_proj -> experts.gate_up_proj.weight
                elif not key.endswith('.weight') and not key.endswith('.bias'):
                    # Only add .weight if it doesn't already have a suffix
                    new_key = key + '.weight'
                    print(f"   Weight transformation: {key} -> {new_key}", flush=True)
                    weight_tensor = state_dict.pop(key)
                    print(f"   Weight tensor shape: {weight_tensor.shape}", flush=True)
                    state_dict[new_key] = weight_tensor
        
        print(f"✅ Completed {transformation_count} MoE parameter transformations", flush=True)
        
        # STEP 2: Convert sinks tensors that need reshaping  
        print("🔄 Starting sinks tensor reshaping...", flush=True)
        sinks_count = 0
        # Look for pattern: model.layers.{N}.self_attn.sinks
        for key in list(state_dict.keys()):
            if 'sinks' in key and key.endswith('.self_attn.sinks'):
                print(f"🔧 Processing sinks tensor: {key}", flush=True)
                sinks_count += 1
                sinks_tensor = state_dict[key]
                
                if hasattr(sinks_tensor, 'shape') and len(sinks_tensor.shape) == 1:
                    # This is a 1D tensor (num_heads,) that needs to become (kv_heads, q_heads_per_group)
                    heads_dim = sinks_tensor.shape[0]
                    expected_heads = self.config.num_heads
                    print(f"   Sinks shape: {sinks_tensor.shape}, expected heads: {expected_heads}", flush=True)
                    
                    if heads_dim == expected_heads:
                        # Reshape from (num_heads,) to (kv_heads, q_heads_per_group)
                        kv_heads = self.config.num_kv_heads
                        q_heads_per_group = self.config.num_heads // self.config.num_kv_heads
                        
                        reshaped = jnp.reshape(sinks_tensor, (kv_heads, q_heads_per_group))
                        state_dict[key] = reshaped
                        print(f"   Reshaped to: {reshaped.shape}", flush=True)
                    else:
                        print(f"   ⚠️ Heads dimension mismatch: {heads_dim} != {expected_heads}", flush=True)
        
        print(f"✅ Processed {sinks_count} sinks tensors", flush=True)
                    
        # STEP 3: Load with the normalized state dict
        print(f"🔄 Calling default_eqx_module_from_state_dict with {len(state_dict)} keys...", flush=True)
        
        # DEBUG: Show final key sample before calling default loader
        final_expert_keys = [k for k in state_dict.keys() if 'experts.' in k]
        print(f"📊 Final expert keys count: {len(final_expert_keys)}", flush=True)
        if final_expert_keys:
            print(f"   Example final expert key: {final_expert_keys[0]}", flush=True)
            if final_expert_keys[0] in state_dict:
                tensor = state_dict[final_expert_keys[0]]
                print(f"   Example final expert tensor shape: {tensor.shape}", flush=True)
        
        try:
            result = default_eqx_module_from_state_dict(self, state_dict, prefix)
            print("✅ default_eqx_module_from_state_dict completed successfully", flush=True)
            return result
        except Exception as e:
            print(f"❌ ERROR in default_eqx_module_from_state_dict: {e}", flush=True)
            print(f"❌ Error type: {type(e)}", flush=True)
            if "Shape mismatch" in str(e):
                print(f"❌ Shape mismatch details: {str(e)}", flush=True)
                # Try to find which parameter is causing the issue
                for key, tensor in state_dict.items():
                    if hasattr(tensor, 'shape') and 'experts.' in key:
                        print(f"   Expert tensor: {key} -> {tensor.shape}", flush=True)
            raise


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

    def get_lm_head(self) -> hax.NamedArray:
        if self.lm_head is None:
            return self.embeddings.token_embeddings.weight
        else:
            return self.lm_head.weight

    def resize_vocab(self, new_size: int, key=None) -> "LmHeadModel[GptOssConfig]":
        new_Vocab = self.Vocab.resize(new_size)
        k1, k2 = maybe_rng_split(key, 2)
        new_embeddings = self.embeddings.resize_embeddings(new_size, key=k1)
        if self.lm_head is not None:
            new_lm_matrix = hax.tree_util.resize_axis(self.lm_head.weight, self.Vocab, new_size, key=k2)
            new_lm_head = dataclasses.replace(self.lm_head, Out=new_Vocab, weight=new_lm_matrix)
            return dataclasses.replace(self, embeddings=new_embeddings, lm_head=new_lm_head)
        else:
            return dataclasses.replace(self, embeddings=new_embeddings)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        """Map from Levanter model names to HF checkpoint names.
        
        Key insight from checkpoint inspector:
        - HF checkpoint uses: layers.N.mlp.experts.*, layers.N.mlp.router.*
        - Levanter model uses: layers.N.block_sparse_moe.experts.*, layers.N.block_sparse_moe.gate.*
        
        We need to map:
        - Levanter's block_sparse_moe → HF's mlp  
        - Levanter's gate → HF's router
        """
        # Map transformer to HF's 'model' and drop the 'embeddings' prefix so LlamaEmbedding can map to 'model.embed_tokens'.
        return {
            "transformer": "model",
            "embeddings": None,
        }
