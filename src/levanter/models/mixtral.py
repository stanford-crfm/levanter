import dataclasses
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Type, Union

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call, shaped_rng_split
from haliax.nn.scan import Stacked

from levanter.compat.hf_checkpoints import HFCheckpointConverter
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
from levanter.models.gpt2 import ACT2FN
from levanter.models.llama import LlamaAttention, LlamaEmbedding, LlamaRMSNorm
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.models.mistral import MistralConfig
from levanter.utils.py_utils import cached_classproperty


silence_transformer_nag()
from transformers import MixtralConfig as HfMixtralConfig  # noqa: E402
from transformers import PretrainedConfig as HfConfig  # noqa: E402


@LmConfig.register_subclass("mixtral")
@dataclass(frozen=True)
class MixtralConfig(MistralConfig):
    """Config for MistralModel

    Args:
        seq_len (int, optional): maximum length of the input sequence. Defaults to 8192.
        hidden_dim (int, optional): dimension of the hidden state. Defaults to 4096.
        intermediate_dim (int, optional): dimension of the intermediate state. Defaults to 14336.
        num_layers (int, optional): number of hidden layers in the Transformer encoder. Defaults to 32.
        num_heads (int, optional): number of attention heads for each attention layer. Defaults to 32.
        num_kv_heads (int, optional): number of attention heads for keys and values in each attention layer.
            Setting to 1 means MQA. Setting to num_heads means MHA. Otherwise GQA.
            Note that num_heads must be divisible by this number. Defaults to 8.
        activation_function (str, optional): activation function for the hidden layer. Defaults to "silu".
        sliding_window (int, optional): window size of sliding window attention. Defaults to 4096.
        num_experts_per_tok (int, optional: number of experts to root per-token, can be also interpreted as the `top-p` routing parameter.
        num_local_experts (int, optional): number of experts per Sparse MLP layer.
        router_aux_loss_coef (`float`, optional): aux loss factor for the total loss. Defaults to 0.001
    """

    seq_len: int = 8192
    hidden_dim: int = 4096
    intermediate_dim: int = 14336
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 8
    activation_function: str = "silu"
    initializer_range: float = 0.02
    layer_norm_epsilon: float = 1e-6
    sliding_window: int = 4096
    num_experts_per_tok: int = 2
    num_local_experts: int = 8
    router_aux_loss_coef: float = 0.02
    chunk_size_coef: float = num_experts_per_tok / num_local_experts  # perf

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
    KVHeads = property(lambda self: Axis(name="kv_heads", size=self.num_kv_heads))
    Layers = property(lambda self: Axis(name="layers", size=self.num_layers))
    Experts = property(lambda self: Axis(name="experts", size=self.num_local_experts))
    TopExperts = property(lambda self: Axis(name="top_experts", size=self.num_experts_per_tok))
    Mlp = property(lambda self: Axis(name="mlp", size=self.intermediate_dim))
    HeadSize = property(lambda self: Axis(name="head_size", size=self.hidden_dim // self.num_heads))

    def __post_init__(self):
        super().__post_init__()
        assert self.num_experts_per_tok <= self.num_local_experts, (
            f"num_experts_per_tok={self.num_experts_per_tok} greater than by"
            f" num_local_experts={self.num_local_experts}."
        )

    @cached_classproperty
    def default_hf_checkpoint_converter(cls) -> HFCheckpointConverter["MixtralConfig"]:  # type: ignore
        return HFCheckpointConverter(
            cls,  # type: ignore
            "mistralai/Mixtral-8x7B-v0.1",
            trust_remote_code=True,
            tokenizer="mistralai/Mixtral-8x7B-v0.1",
            HfConfigClass=HfMixtralConfig,
        )

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig):
        return MixtralConfig(
            seq_len=hf_config.max_position_embeddings,
            hidden_dim=hf_config.hidden_size,
            intermediate_dim=hf_config.intermediate_size,
            num_layers=hf_config.num_hidden_layers,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            activation_function=hf_config.hidden_act,
            initializer_range=hf_config.initializer_range,
            layer_norm_epsilon=hf_config.rms_norm_eps,
            sliding_window=hf_config.sliding_window,
            num_experts_per_tok=hf_config.num_experts_per_tok,
            num_local_experts=hf_config.num_local_experts,
            router_aux_loss_coef=hf_config.router_aux_loss_coef,
        )

    def to_hf_config(self, vocab_size: int, config_overrides: Optional[Dict] = None) -> HfMixtralConfig:
        """Convert to HuggingFace's MistralConfig

        Args:
            vocab_size (int, optional): Vocabulary size of the tokenizer. Defaults to 32000.
            config_overrides (dict, optional): Overrides for the config. Defaults to None.

        Returns:
            HfMistralConfig: HuggingFace's MistralConfig
        """
        if config_overrides is None:
            config_overrides = {}

        return HfMixtralConfig(
            max_position_embeddings=self.seq_len,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_dim,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            hidden_act=self.activation_function,
            initializer_range=self.initializer_range,
            rms_norm_eps=self.layer_norm_epsilon,
            sliding_window=self.sliding_window,
            num_experts_per_tok=self.num_experts_per_tok,
            num_local_experts=self.num_local_experts,
            router_aux_loss_coef=self.router_aux_loss_coef,
            vocab_size=vocab_size,
            **config_overrides,
        )

    @property
    def model_type(cls) -> Type["MixtralLMHeadModel"]:
        return MixtralLMHeadModel


class MixtralMlp(eqx.Module, StateDictSerializationMixin):
    """Multi-layer Perceptron"""

    w1: hnn.Linear  # (gate_proj) projection from Embed to Mlp
    w2: hnn.Linear  # (down_proj) projection from Mlp to Embed
    w3: hnn.Linear  # (up_proj) projection from Embed to Mlp
    act: Callable = eqx.static_field()

    @staticmethod
    def init(
        Embed: Axis, Mlp: Axis, activation_fn: Union[str, Callable], *, key, use_bias: bool = False
    ) -> "MixtralMlp":
        k1, k2, k3 = jrandom.split(key, 3)
        w1 = hnn.Linear.init(Out=Mlp, In=Embed, key=k1, use_bias=use_bias)
        w2 = hnn.Linear.init(Out=Embed, In=Mlp, key=k2, use_bias=use_bias)
        w3 = hnn.Linear.init(Out=Mlp, In=Embed, key=k3, use_bias=use_bias)
        if isinstance(activation_fn, str):
            activation_fn = ACT2FN[activation_fn]
        act = activation_fn  # type: ignore
        return MixtralMlp(w1, w2, w3, act)

    @named_call
    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        k1, k2, k3 = maybe_rng_split(key, 3)
        hidden_states = self.w1(x, key=k1)
        hidden_states = self.act(hidden_states)
        hidden_states = hidden_states * self.w3(x, key=k3)
        outputs = self.w2(hidden_states, key=k2)
        return outputs

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None):
        # unflatten the linear layers of HF state_dict to match the shape of MistralMlp
        d = {}
        d.update(unflatten_linear_layers(apply_prefix(prefix, "w1"), state_dict, self.w1, out_dims_first_in_dict=True))
        d.update(unflatten_linear_layers(apply_prefix(prefix, "w2"), state_dict, self.w2, out_dims_first_in_dict=True))
        d.update(unflatten_linear_layers(apply_prefix(prefix, "w3"), state_dict, self.w3, out_dims_first_in_dict=True))

        return super().from_state_dict(d, prefix)

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        my_dict: StateDict = {}
        super().update_state_dict(my_dict, prefix=prefix)

        my_dict.update(flatten_linear_layers(apply_prefix(prefix, "w1"), self.w1, out_dims_first_in_dict=True))
        my_dict.update(flatten_linear_layers(apply_prefix(prefix, "w2"), self.w2, out_dims_first_in_dict=True))
        my_dict.update(flatten_linear_layers(apply_prefix(prefix, "w3"), self.w3, out_dims_first_in_dict=True))

        state_dict.update(my_dict)
        return state_dict


class MixtralSparseMoeBlock(eqx.Module, StateDictSerializationMixin):
    """Mixture-of-Experts"""

    config: MistralConfig = eqx.static_field()
    gate: hnn.Linear  # projection from Embed to Experts
    experts: List[MixtralMlp]

    @staticmethod
    def init(config: MistralConfig, *, key) -> "MixtralSparseMoeBlock":
        k_gate, key = jrandom.split(key, 2)
        k_experts = jrandom.split(key, config.num_local_experts)

        gate = hnn.Linear.init(config.Embed, config.Experts, key=k_gate, use_bias=config.use_bias)
        experts = [
            MixtralMlp.init(
                Embed=config.Embed,
                Mlp=config.Mlp,
                activation_fn=config.activation_function,
                key=k_experts[i],
                use_bias=config.use_bias,
            )
            for i in range(config.num_local_experts)
        ]

        return MixtralSparseMoeBlock(config, gate, experts)

    @named_call
    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        if x.has_axis("batch"):
            squash_axes = [x.resolve_axis("batch"), x.resolve_axis(self.config.Pos)]
        else:
            squash_axes = [
                x.resolve_axis(self.config.Pos),
            ]
        Experts = self.config.Experts
        TopExperts = self.config.TopExperts

        k_gate, key = maybe_rng_split(key, 2)
        k_experts = maybe_rng_split(key, self.config.num_local_experts)

        x_flat = hax.flatten_axes(x, old_axes=squash_axes, new_axis="token")  # [Batch, Pos, Embed] -> [Token, Embed]
        router_logits = self.gate(x_flat, key=k_gate)  # [Token, Embed] -> [Token, Experts]
        routing_weights = hnn.softmax(router_logits, axis=Experts)  # [Token, Experts] distribution
        selected_weights, selected_experts = hax.top_k(
            routing_weights,
            axis=Experts,
            k=self.config.num_experts_per_tok,
            new_axis=TopExperts,
        )  # [Token, Experts] -> [Token, TopExperts]
        selected_weights /= selected_weights.sum(axis=TopExperts)  # [Token, TopExperts] distribution
        # HF cast dtype to fp32 during softmax and cast it back in this line. not sure if we should do it...
        expert_mask = hax.nn.one_hot(selected_experts, Experts)  # [Token, TopExperts, Experts] one hot

        Token = x_flat.resolve_axis("token")
        result = hax.zeros((Token, self.config.Embed))
        chunk_size = int(Token.size * self.config.chunk_size_multiplier)

        def cond(carry):
            step = carry[0]
            token_idx = carry[4]
            return jnp.logical_or(step * chunk_size < Token.size, token_idx["token", step * chunk_size] == -1)

        def body(carry):
            step, result, x, selected_weights, token_idx, top_x, expert_i, k_expert = carry

            chunk_token_idx = token_idx["token", hax.ds.block(step, chunk_size)]
            chunk_top_x = top_x["token", hax.ds.block(step, chunk_size)]

            chunk_x = x["token", chunk_token_idx]
            chunk_selected_weights = selected_weights["token", chunk_token_idx, "top_experts", chunk_top_x]
            chunk_z = chunk_selected_weights * expert_i(chunk_x, key=k_expert)

            # set -1 elements to an out of bound value so that .at[...].add() will drop those indices
            oob_chunk_token_idx = chunk_token_idx + (Token.size + 1) * (chunk_token_idx == -1)
            # TODO: support index_add() in haliax. until then we would have to directly modify its .array
            result = hax.NamedArray(
                result.array.at[oob_chunk_token_idx.array].add(
                    chunk_z.array, mode="drop", indices_are_sorted=True, unique_indices=True
                ),
                (Token, self.config.Embed),
            )

            return step + 1, result, x, selected_weights, token_idx, top_x, expert_i, k_expert

        for i in range(self.config.num_local_experts):
            expert_i = self.experts[i]
            expert_mask_i = expert_mask[{"experts": i}]  # [Token, TopExperts] boolean mask
            token_idx, top_x = hax.where(expert_mask_i, fill_value=-1, new_axis=Token)  # [Token,] indices

            with hax.enable_shape_checks(False):
                result = eqx.internal.while_loop(
                    cond_fun=cond,
                    body_fun=body,
                    init_val=(0, result, x_flat, selected_weights, token_idx, top_x, expert_i, k_experts[i]),
                    max_steps=Token.size // chunk_size,
                    kind="checkpointed",
                )[1]

        return hax.unflatten_axis(result, "token", squash_axes)

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None):
        d = state_dict.copy()
        d.update(
            unflatten_linear_layers(apply_prefix(prefix, "gate"), state_dict, self.gate, out_dims_first_in_dict=True)
        )

        return super().from_state_dict(d, prefix)

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        my_dict: StateDict = {}
        super().update_state_dict(my_dict, prefix=prefix)

        my_dict.update(flatten_linear_layers(apply_prefix(prefix, "gate"), self.gate, out_dims_first_in_dict=True))

        state_dict.update(my_dict)
        return state_dict


class MixtralDecoderLayer(StateDictSerializationMixin, eqx.Module):
    config: MixtralConfig = eqx.static_field()
    self_attn: LlamaAttention
    block_sparse_moe: MixtralMlp
    input_layernorm: LlamaRMSNorm
    post_attention_layernorm: LlamaRMSNorm

    @staticmethod
    def init(config: MistralConfig, *, key) -> "MixtralDecoderLayer":
        k_attn, k_mlp = jrandom.split(key, 2)

        attn = LlamaAttention.init(config, key=k_attn)
        mlp = MixtralSparseMoeBlock.init(config, key=k_mlp)
        ln_1 = LlamaRMSNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)
        ln_2 = LlamaRMSNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)

        return MixtralDecoderLayer(config, attn, mlp, ln_1, ln_2)

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
        mlp_output = self.block_sparse_moe(x, key=k_mlp)
        output = residual + mlp_output
        return output


class MixtralTransformer(StateDictSerializationMixin, eqx.Module):
    config: MistralConfig = eqx.static_field()
    layers: Stacked[MixtralDecoderLayer]
    norm: LlamaRMSNorm

    @staticmethod
    def init(config: MistralConfig, *, key) -> "MixtralTransformer":
        layers = Stacked.init(
            config.Layers, MixtralDecoderLayer, gradient_checkpointing=config.gradient_checkpointing
        )(
            config,
            key=shaped_rng_split(key, config.num_layers),
        )
        ln_f = LlamaRMSNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)

        return MixtralTransformer(config, layers, ln_f)

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


class MixtralLMHeadModel(eqx.Module, LmHeadModel[MixtralConfig], StateDictSerializationMixin):
    transformer: MixtralTransformer
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
    def init(cls, Vocab: Axis, config: MistralConfig, *, key) -> "MixtralLMHeadModel":
        k_t, k_emb = jrandom.split(key, 2)
        transformer = MixtralTransformer.init(config, key=k_t)
        embeddings = LlamaEmbedding.init(Vocab, config, key=k_emb)
        lm_head = hnn.Linear.init(In=config.Embed, Out=Vocab, key=k_emb, use_bias=False, out_first=True)
        return MixtralLMHeadModel(transformer, embeddings, lm_head)

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

    def resize_vocab(self, new_size: int, key=None) -> "LmHeadModel[MixtralConfig]":
        new_Vocab = self.Vocab.resize(new_size)
        k1, k2 = maybe_rng_split(key, 2)
        new_embeddings = self.embeddings.resize_embeddings(new_size, key=k1)
        new_lm_matrix = hax.tree_util.resize_axis(self.lm_head.weight, self.Vocab, new_size, key=k2)
        new_lm_head = dataclasses.replace(self.lm_head, Out=new_Vocab, weight=new_lm_matrix)

        return dataclasses.replace(self, embeddings=new_embeddings, lm_head=new_lm_head)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"transformer": "model", "embeddings": None}

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None):
        # unflatten the linear layers of HF state_dict to match the shape of MistralMlp
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
