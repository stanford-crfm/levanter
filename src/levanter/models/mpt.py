# implements the necessary variations of GPT-2 to work with https://huggingface.co/mosaicml/mpt-7b in haliax/levanter
import math
from dataclasses import dataclass
from typing import Dict, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from transformers import PretrainedConfig

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import named_call, shaped_rng_split
from haliax.nn.scan import Stacked
from levanter.compat.torch_serialization import (
    StateDict,
    StateDictSerializationMixin,
    apply_prefix,
    flatten_linear_layer,
    stack_state_dict,
    unflatten_linear_layer,
    unstack_state_dict,
)


attn_config_defaults: Dict = {
    "attn_type": "multihead_attention",
    "attn_pdrop": 0.0,
    "attn_impl": "triton",
    "qk_ln": False,
    "clip_qkv": None,
    "softmax_scale": None,
    "prefix_lm": False,
    "attn_uses_sequence_id": False,
    "alibi": False,
    "alibi_bias_max": 8,
}
init_config_defaults: Dict = {"name": "kaiming_normal_", "fan_mode": "fan_in", "init_nonlinearity": "relu"}


class MPTConfig(PretrainedConfig):
    # copied from https://huggingface.co/mosaicml/mpt-7b/blob/main/configuration_mpt.py
    model_type = "mpt"

    def __init__(
        self,
        d_model: int = 2048,
        n_heads: int = 16,
        n_layers: int = 24,
        expansion_ratio: int = 4,
        max_seq_len: int = 2048,
        vocab_size: int = 50368,
        resid_pdrop: float = 0.0,
        emb_pdrop: float = 0.0,
        learned_pos_emb: bool = True,
        attn_config: Dict = attn_config_defaults,
        init_device: str = "cpu",
        logit_scale: Optional[Union[float, str]] = None,
        no_bias: bool = False,
        verbose: int = 0,
        embedding_fraction: float = 1.0,
        norm_type: str = "low_precision_layernorm",
        use_cache: bool = False,
        init_config: Dict = init_config_defaults,
        **kwargs,
    ):
        """The MPT configuration class.
        Args:
            d_model (int): The size of the embedding dimension of the model.
            n_heads (int): The number of attention heads.
            n_layers (int): The number of layers in the model.
            expansion_ratio (int): The ratio of the up/down scale in the MLP.
            max_seq_len (int): The maximum sequence length of the model.
            vocab_size (int): The size of the vocabulary.
            resid_pdrop (float): The dropout probability applied to the attention output before combining with residual.
            emb_pdrop (float): The dropout probability for the embedding layer.
            learned_pos_emb (bool): Whether to use learned positional embeddings
            attn_config (Dict):  A dictionary used to configure the model's attention module:
                attn_type (str): type of attention to use. Options: multihead_attention, multiquery_attention
                attn_pdrop (float): The dropout probability for the attention layers.
                attn_impl (str): The attention implementation to use. One of 'torch', 'flash', or 'triton'.
                qk_ln (bool): Whether to apply layer normalization to the queries and keys in the attention layer.
                clip_qkv (Optional[float]): If not None, clip the queries, keys, and values in the attention layer to
                    this value.
                softmax_scale (Optional[float]): If not None, scale the softmax in the attention layer by this value. If None,
                    use the default scale of ``1/sqrt(d_keys)``.
                prefix_lm (Optional[bool]): Whether the model should operate as a Prefix LM. This requires passing an
                    extra `prefix_mask` argument which indicates which tokens belong to the prefix. Tokens in the prefix
                    can attend to one another bi-directionally. Tokens outside the prefix use causal attention.
                attn_uses_sequence_id (Optional[bool]): Whether to restrict attention to tokens that have the same sequence_id.
                    When the model is in `train` mode, this requires passing an extra `sequence_id` argument which indicates
                    which sub-sequence each token belongs to.
                    Defaults to ``False`` meaning any provided `sequence_id` will be ignored.
                alibi (bool): Whether to use the alibi bias instead of position embeddings.
                alibi_bias_max (int): The maximum value of the alibi bias.
            init_device (str): The device to use for parameter initialization.
            logit_scale (Optional[Union[float, str]]): If not None, scale the logits by this value.
            no_bias (bool): Whether to use bias in all layers.
            verbose (int): The verbosity level. 0 is silent.
            embedding_fraction (float): The fraction to scale the gradients of the embedding layer by.
            norm_type (str): choose type of norm to use
            multiquery_attention (bool): Whether to use multiquery attention implementation.
            use_cache (bool): Whether or not the model should return the last key/values attentions
            init_config (Dict): A dictionary used to configure the model initialization:
                init_config.name: The parameter initialization scheme to use. Options: 'default_', 'baseline_',
                    'kaiming_uniform_', 'kaiming_normal_', 'neox_init_', 'small_init_', 'xavier_uniform_', or
                    'xavier_normal_'. These mimic the parameter initialization methods in PyTorch.
                init_div_is_residual (Union[int, float, str, bool]): Value to divide initial weights by if ``module._is_residual`` is True.
                emb_init_std (Optional[float]): The standard deviation of the normal distribution used to initialize the embedding layer.
                emb_init_uniform_lim (Optional[Union[Tuple[float, float], float]]): The lower and upper limits of the uniform distribution
                    used to initialize the embedding layer. Mutually exclusive with ``emb_init_std``.
                init_std (float): The standard deviation of the normal distribution used to initialize the model,
                    if using the baseline_ parameter initialization scheme.
                init_gain (float): The gain to use for parameter initialization with kaiming or xavier initialization schemes.
                fan_mode (str): The fan mode to use for parameter initialization with kaiming initialization schemes.
                init_nonlinearity (str): The nonlinearity to use for parameter initialization with kaiming initialization schemes.
                ---
                See llmfoundry.models.utils.param_init_fns.py for info on other param init config options
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.expansion_ratio = expansion_ratio
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.resid_pdrop = resid_pdrop
        self.emb_pdrop = emb_pdrop
        self.learned_pos_emb = learned_pos_emb
        self.attn_config = attn_config
        self.init_device = init_device
        self.logit_scale = logit_scale
        self.no_bias = no_bias
        self.verbose = verbose
        self.embedding_fraction = embedding_fraction
        self.norm_type = norm_type
        self.use_cache = use_cache
        self.init_config = init_config
        if "name" in kwargs:
            del kwargs["name"]
        if "loss_fn" in kwargs:
            del kwargs["loss_fn"]
        super().__init__(**kwargs)
        self._validate_config()

    def _set_config_defaults(self, config, config_defaults):
        for (k, v) in config_defaults.items():
            if k not in config:
                config[k] = v
        return config

    def _validate_config(self):
        self.attn_config = self._set_config_defaults(self.attn_config, attn_config_defaults)
        self.init_config = self._set_config_defaults(self.init_config, init_config_defaults)
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if any((prob < 0 or prob > 1 for prob in [self.attn_config["attn_pdrop"], self.resid_pdrop, self.emb_pdrop])):
            raise ValueError(
                "self.attn_config['attn_pdrop'], resid_pdrop, emb_pdrop are probabilities and must be between 0 and 1"
            )
        if self.attn_config["attn_impl"] not in ["torch", "flash", "triton"]:
            raise ValueError(f"Unknown attn_impl={self.attn_config['attn_impl']}")
        # if self.attn_config['prefix_lm'] and self.attn_config['attn_impl'] not in ['torch', 'triton']:
        #     raise NotImplementedError('prefix_lm only implemented with torch and triton attention.')
        # no prefix_lm for now
        if self.attn_config["prefix_lm"]:
            raise NotImplementedError("prefix_lm not implemented yet.")
        if self.attn_config["alibi"] and self.attn_config["attn_impl"] not in ["torch", "triton"]:
            raise NotImplementedError("alibi only implemented with torch and triton attention.")
        # if self.attn_config['attn_uses_sequence_id'] and self.attn_config['attn_impl'] not in ['torch', 'triton']:
        #     raise NotImplementedError('attn_uses_sequence_id only implemented with torch and triton attention.')
        # no attn_uses_sequence_id for now
        if self.attn_config["attn_uses_sequence_id"]:
            raise NotImplementedError("attn_uses_sequence_id not implemented yet.")
        # no clip_qkv for now
        if self.attn_config["clip_qkv"]:
            raise NotImplementedError("clip_qkv not implemented yet.")
        # no softmax_scale
        if self.attn_config["softmax_scale"]:
            raise NotImplementedError("softmax_scale not implemented yet.")
        # no qk_ln
        if self.attn_config["qk_ln"]:
            raise NotImplementedError("qk_ln not implemented yet.")
        if self.embedding_fraction > 1 or self.embedding_fraction <= 0:
            raise ValueError("model.embedding_fraction must be between 0 (exclusive) and 1 (inclusive)!")
        if isinstance(self.logit_scale, str) and self.logit_scale != "inv_sqrt_d_model":
            raise ValueError(
                f"self.logit_scale={self.logit_scale!r} is not recognized as an option; use numeric value or"
                " 'inv_sqrt_d_model'."
            )
        if self.init_config.get("name", None) is None:
            raise ValueError(f"self.init_config={self.init_config!r} 'name' needs to be set.")
        if not self.learned_pos_emb and (not self.attn_config["alibi"]):
            raise ValueError(
                "Positional information must be provided to the model using either learned_pos_emb or alibi."
            )


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
@dataclass
class MptConfig:
    d_model: int
    n_heads: int
    n_layers: int
    expansion_ratio: int
    max_seq_len: int
    resid_pdrop: float = 0.0
    emb_pdrop: float = 0.0
    learned_pos_emb: bool = True
    attn_config: MptAttentionConfig = MptAttentionConfig()

    use_bias: bool = True

    Embed = property(lambda self: Axis("embed", self.d_model))
    Head = property(lambda self: Axis("head", self.n_heads))
    Layers = property(lambda self: Axis("layer", self.n_layers))
    SeqLen = property(lambda self: Axis("seqlen", self.max_seq_len))
    KeySeqLen = property(lambda self: Axis("key_seqlen", self.max_seq_len))
    Mlp = property(lambda self: Axis("mlp", self.expansion_ratio * self.d_model))
    HeadDim = property(lambda self: Axis("head_dim", self.d_model // self.n_heads))

    @staticmethod
    def from_torch_config(config):
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
        )


# Torch code:
# class MPTMLP(nn.Module):
#
#     def __init__(self, d_model: int, expansion_ratio: int, device: Optional[str]=None):
#         super().__init__()
#         self.up_proj = nn.Linear(d_model, expansion_ratio * d_model, device=device)
#         self.act = nn.GELU(approximate='none')
#         self.down_proj = nn.Linear(expansion_ratio * d_model, d_model, device=device)
#         self.down_proj._is_residual = True
#
#     def forward(self, x):
#         return self.down_proj(self.act(self.up_proj(x)))
#
# class MPTBlock(nn.Module):
#
#     def __init__(self, d_model: int, n_heads: int, expansion_ratio: int, attn_config: Dict={'attn_type': 'multihead_attention', 'attn_pdrop': 0.0, 'attn_impl': 'triton', 'qk_ln': False, 'clip_qkv': None, 'softmax_scale': None, 'prefix_lm': False, 'attn_uses_sequence_id': False, 'alibi': False, 'alibi_bias_max': 8}, resid_pdrop: float=0.0, norm_type: str='low_precision_layernorm', device: Optional[str]=None, **kwargs):
#         del kwargs
#         super().__init__()
#         norm_class = NORM_CLASS_REGISTRY[norm_type.lower()]
#         attn_class = ATTN_CLASS_REGISTRY[attn_config['attn_type']]
#         self.norm_1 = norm_class(d_model, device=device)
#         self.attn = attn_class(attn_impl=attn_config['attn_impl'], clip_qkv=attn_config['clip_qkv'], qk_ln=attn_config['qk_ln'], softmax_scale=attn_config['softmax_scale'], attn_pdrop=attn_config['attn_pdrop'], d_model=d_model, n_heads=n_heads, device=device)
#         self.norm_2 = norm_class(d_model, device=device)
#         self.ffn = MPTMLP(d_model=d_model, expansion_ratio=expansion_ratio, device=device)
#         self.resid_attn_dropout = nn.Dropout(resid_pdrop)
#         self.resid_ffn_dropout = nn.Dropout(resid_pdrop)
#
#     def forward(self, x: torch.Tensor, past_key_value: Optional[Tuple[torch.Tensor]]=None, attn_bias: Optional[torch.Tensor]=None, attention_mask: Optional[torch.ByteTensor]=None, is_causal: bool=True) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
#         a = self.norm_1(x)
#         (b, _, past_key_value) = self.attn(a, past_key_value=past_key_value, attn_bias=attn_bias, attention_mask=attention_mask, is_causal=is_causal)
#         x = x + self.resid_attn_dropout(b)
#         m = self.norm_2(x)
#         n = self.ffn(m)
#         x = x + self.resid_ffn_dropout(n)
#         return (x, past_key_value)

# Haliax code:
# the MLP is the same as GPT-2 MLP with the following settings:
# * non-approximate GELU
# * no bias terms by default


class MptMlp(eqx.Module, StateDictSerializationMixin):
    up_proj: hnn.Linear  # projection from Embed to Intermediate (typically 4x Embed)
    down_proj: hnn.Linear  # projection from Intermediate to Embed

    def __init__(self, Embed: Axis, Intermediate: Axis, *, key, use_bias: bool = False):
        k_fc, k_proj = jrandom.split(key, 2)
        self.up_proj = hnn.Linear(Out=Intermediate, In=Embed, key=k_fc, use_bias=use_bias)
        self.down_proj = hnn.Linear(Out=Embed, In=Intermediate, key=k_proj, use_bias=use_bias)

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

    def __init__(
        self,
        config: MptConfig,
        *,
        key,
        use_bias: bool = True,
    ):
        self.config = config

        k_c, k_proj = jrandom.split(key, 2)
        qkv = Axis("qkv", 3)
        self.Wqkv = hnn.Linear(In=config.Embed, Out=(qkv, config.Head, config.HeadDim), key=k_c, use_bias=use_bias)
        self.out_proj = hnn.Linear(In=(config.Head, config.HeadDim), Out=config.Embed, key=k_proj, use_bias=use_bias)

    def __call__(
        self, hidden_states: NamedArray, mask: Optional[NamedArray] = None, bias: Optional[NamedArray] = None
    ):
        qkv_out = self.Wqkv(hidden_states)
        q, k, v = qkv_out.unbind("qkv")

        # Rename k and v's SeqLen as haliax doesn't support unnamed axes or duplicate axes
        k = k.rename({self.config.SeqLen: self.config.KeySeqLen})
        v = v.rename({self.config.SeqLen: self.config.KeySeqLen})

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

        attn_weights = hnn.softmax(attn_scores, axis="key_seqlen").astype(hidden_states.dtype)
        # attn_weights = self.dropout(attn_weights, key=key, inference=inference)

        attn_output = hax.dot("key_seqlen", attn_weights, v)  # [heads, seq_len, head_dim]

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

    def __init__(self, config: MptConfig, *, key):
        super().__init__()
        kattn, kmlp = jrandom.split(key, 2)
        self.norm_1 = hnn.LayerNorm(config.Embed, use_bias=config.use_bias)
        self.attn = MptAttention(config, key=kattn)
        self.norm_2 = hnn.LayerNorm(config.Embed, use_bias=config.use_bias)
        self.ffn = MptMlp(config.Embed, config.Mlp, key=kmlp, use_bias=config.use_bias)

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

    def __init__(self, Vocab, config: MptConfig, *, key):
        super().__init__()
        self.config = config

        # vectorize the blocks
        self.blocks = Stacked(
            self.Layers, MptBlock, config, key=shaped_rng_split(key, config.n_layers), gradient_checkpointing=False
        )
        self.norm_f = hnn.LayerNorm(config.Embed, use_bias=config.use_bias)

    @named_call
    def __call__(self, hidden_states: NamedArray, attention_mask: Optional[NamedArray]) -> NamedArray:
        if self.config.attn_config.alibi:
            bias = mpt_build_alibi_bias(
                self.config.Head, self.config.KeySeqLen, self.config.attn_config.alibi_bias_max
            )
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


class MptLmHeadModel(StateDictSerializationMixin, eqx.Module):
    wte: hnn.Embedding
    transformer: MptTransformer

    @property
    def config(self) -> MptConfig:
        return self.transformer.config

    def __init__(self, Vocab: Axis, config: MptConfig, *, key):
        super().__init__()
        k_transformer, k_wte = jrandom.split(key, 2)
        self.wte = hnn.Embedding(Vocab, config.Embed, key=k_wte)
        self.transformer = MptTransformer(Vocab, config, key=k_transformer)

        assert config.emb_pdrop == 0.0, "embedding dropout not supported"
        assert config.resid_pdrop == 0.0, "residual dropout not supported"
        assert config.attn_config.alibi, "alibi attention is required for now"

    @named_call
    def __call__(self, input_ids: NamedArray, attention_mask: Optional[NamedArray] = None) -> NamedArray:
        hidden_states = self.wte.embed(input_ids)
        causal = hnn.attention.causal_mask(self.config.SeqLen, self.config.KeySeqLen)
        attention_mask = hnn.attention.combine_masks_and(causal, attention_mask)
        hidden_states = self.transformer(hidden_states, attention_mask=attention_mask)
        output_logits = self.wte.unembed(hidden_states)

        return output_logits

    def _state_dict_key_map(self) -> Optional[Dict[str, Optional[str]]]:
        return {
            "wte": "transformer.wte",
        }


def mpt_alibi_gen_slopes(n_heads, alibi_bias_max=8):
    _n_heads = 2 ** math.ceil(math.log2(n_heads))
    m = jnp.arange(1, _n_heads + 1)
    m = m * (alibi_bias_max / _n_heads)
    slopes = 1.0 / jnp.power(2, m)
    if _n_heads != n_heads:
        slopes = jnp.concat([slopes[1::2], slopes[::2]])[:n_heads]
    return slopes


def mpt_build_alibi_bias(Heads, KSeqLen, alibi_bias_max=8):
    alibi_bias = jnp.arange(1 - KSeqLen.size, 1, dtype=jnp.int32)
    slopes = mpt_alibi_gen_slopes(Heads.size, alibi_bias_max)

    slopes = hax.named(slopes, Heads)
    positions = hax.named(alibi_bias, KSeqLen).broadcast_axis(Heads)

    return slopes * positions
