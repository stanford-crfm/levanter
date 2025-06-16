import dataclasses
from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
from litellm.litellm_core_utils.prompt_templates.common_utils import convert_openai_message_to_only_content_messages

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, AxisSpec, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call, shaped_rng_split
from haliax.nn.normalization import LayerNormBase
from haliax.nn.scan import Stacked
from haliax.state_dict import ModuleWithStateDictSerialization

from levanter.compat.hf_checkpoints import HFCheckpointConverter, HFCompatConfig
from levanter.layers.attention import Attention, AttentionBackend, AttentionConfig, AttentionMask
from levanter.layers.normalization import LayerNormConfigBase
from levanter.models.llama import LlamaEmbedding, LlamaMlp  # Gemma attention and MLP is identical to LLama
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig, RotaryEmbeddingsConfig
from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.logging import silence_transformer_nag
from levanter.utils.types import BlockFoldable


silence_transformer_nag()
from transformers import GemmaConfig as HfGemmaConfig  # noqa: E402
from transformers import Gemma2Config as HfGemma2Config  # noqa: E402
from transformers import Gemma3Config as HfGemma3Config  # noqa: E402
from transformers import PretrainedConfig as HfConfig  # noqa: E402


# Gemma is... very similar to Llama, so we use much of the same modeling code.
#
# The key differences are:
# * Activation is changed to approximate gelu
# * Embedding weights are tied to the LM head
# * Gemma allows specifying a head dimension independently of the hidden and intermediate dims.

# Gemma 2 adds logit capping and alternating sliding window attention
# Gemma 3 removes logit capping and replaces with qk-norm, uses 5-1 global/local attention,


@LayerNormConfigBase.register_subclass("gemma")
@dataclass
class GemmaNormConfig(LayerNormConfigBase):
    """Configuration for Gemma's custom RMS normalization."""
    def build(self, axis: AxisSpec) -> "GemmaRMSNorm":
        return GemmaRMSNorm.init(axis, eps=self.eps, use_weight=self.use_weight, use_bias=self.use_bias)

@LmConfig.register_subclass("gemma")
@dataclass(frozen=True)
class GemmaConfig(HFCompatConfig):
    """Config for GemmaModel.

    Defaults are set for Gemma-2B.

    Args:
        seq_len (int, optional): maximum length of the input sequence. Defaults to 8192.
        hidden_dim (int, optional): dimension of the hidden state. Defaults to 2048.
        intermediate_dim (int, optional): dimension of the intermediate state. Defaults to 16384.
        num_layers (int, optional): number of hidden layers in the Transformer encoder. Defaults to 18.
        num_heads (int, optional): number of attention heads for each attention layer. Defaults to 8.
        num_kv_heads (int, optional): number of attention heads for keys and values in each attention layer.
            Setting to 1 means MQA. Setting to num_heads means MHA. Otherwise GQA.
            Note that num_heads must be divisible by this number. Defaults to 1.
        activation_function (str, optional): activation function for the hidden layer. Defaults to "gelu".
        rope_scaling (Dict, ignored): dict containing the scaling configuration for the Rotary Positional Embedding.
    """

    activation_function: ActivationFunctionEnum = ActivationFunctionEnum.gelu_new
    initializer_range: float = 0.02
    layer_norm_epsilon: float = 1e-5
    tie_word_embeddings: bool = False

    seq_len: int = 8192
    hidden_dim: int = 2048
    intermediate_dim: int = 16384
    vocab_size: int = 256_000
    num_layers: int = 18
    num_heads: int = 8
    head_dim: int = 256
    num_kv_heads: int = 1
    attn_dropout = 0.0

    use_bias: bool = False
    input_embedding_norm: bool = False

    # Attention-related config
    upcast_attn: bool = False
    use_flash_attention: bool | None = None
    attn_backend: AttentionBackend | None = None
    flash_attention_block_size: int | None = None
    use_qk_norm: bool = False

    rope: RotaryEmbeddingsConfig = dataclasses.field(default_factory=DefaultRotaryEmbeddingsConfig)
    rope_local_base_freq: float = 10_000.0
    query_pre_attn_scalar: float | None = None

    gradient_checkpointing: bool = True
    scan_layers: bool = True


    @property
    def local_rope(self) -> RotaryEmbeddingsConfig:
        """Local rope config for Gemma."""
        # HF doesn't document this particularly well, but there are two different Rope configs for
        # Gemma. The `rope` is used for global attention, and the `local_rope` is used for local attention.
        return dataclasses.replace(self.rope, theta=self.rope_local_base_freq)

    final_logit_softcapping: float | None = None
    attn_logit_softcapping: float | None = None

    # Axis
    Pos = property(lambda self: Axis(name="position", size=self.seq_len))
    KeyPos = property(lambda self: self.Pos.alias("key_position"))
    Embed = property(lambda self: Axis(name="embed", size=self.hidden_dim))
    Heads = property(lambda self: Axis(name="heads", size=self.num_heads))
    KVHeads = property(lambda self: Axis(name="kv_heads", size=self.num_kv_heads))
    Layers = property(lambda self: Axis(name="layers", size=self.num_layers))
    Mlp = property(lambda self: Axis(name="mlp", size=self.intermediate_dim))
    HeadSize = property(lambda self: Axis(name="head_size", size=self.head_dim))

    def __post_init__(self):
        assert (
            self.num_heads % self.num_kv_heads == 0
        ), f"num_heads={self.num_heads} not divisible by num_kv_heads={self.num_kv_heads}."

    def hf_checkpoint_converter(
        self, ref_checkpoint: str = "google/gemma-2b"
    ) -> HFCheckpointConverter["GemmaConfig"]:  # type: ignore
        return HFCheckpointConverter(
            self,
            reference_checkpoint=ref_checkpoint,
            trust_remote_code=True,
            HfConfigClass=HfGemmaConfig,
        )

    # The activation function specified in the Gemma HF repo is "gelu", but this is incorrect, it should
    # be "gelu_pytorch_tanh". For backwards compatibility, HF did not change the value in the repo, but
    # instead patches around it. We mimic this behavior and use the approximate gelu internally, and
    # specify the approximate gelu for HF when appropriate.
    # See https://github.com/huggingface/transformers/pull/29402 for more detail.
    @classmethod
    def from_hf_config(cls, hf_config: HfConfig):
        if hf_config.hidden_activation is None:
            activation_function = "gelu_pytorch_tanh"
        else:
            activation_function = "gelu_pytorch_tanh"

        if activation_function == "gelu_pytorch_tanh":
            activation_function = "gelu_new"

        assert activation_function is not None, "No activation function found in HF configuration."
        activation_function_enum = getattr(ActivationFunctionEnum, activation_function)

        rope_theta = hf_config.rope_theta
        rope_config = RotaryEmbeddingsConfig.from_hf_config(rope_theta, hf_config.rope_scaling)

        return GemmaConfig(
            seq_len=hf_config.max_position_embeddings,
            activation_function=activation_function_enum,
            hidden_dim=hf_config.hidden_size,
            intermediate_dim=hf_config.intermediate_size,
            num_layers=hf_config.num_hidden_layers,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            initializer_range=hf_config.initializer_range,
            layer_norm_epsilon=hf_config.rms_norm_eps,
            tie_word_embeddings=hf_config.tie_word_embeddings,
            rope=rope_config,
            head_dim=hf_config.head_dim,
            query_pre_attn_scalar=getattr(hf_config, "query_pre_attn_scalar", hf_config.head_dim),
            final_logit_softcapping=getattr(hf_config, "final_logit_softcapping", None),
            attn_logit_softcapping=getattr(hf_config, "attn_logit_softcapping", None),
        )

    def to_hf_config(self, vocab_size: int, config_overrides: dict | None = None) -> HfGemmaConfig | HfGemma2Config | HfGemma3Config:
        """Convert to HuggingFace's GemmaConfig

        Args:
            vocab_size (int, optional): Vocabulary size of the tokenizer. Defaults to 32000.
            config_overrides (dict, optional): Overrides for the config. Defaults to None.

        Returns:
            HfGemmaConfig: HuggingFace's GemmaConfig
        """
        if config_overrides is None:
            config_overrides = {}

        rope = self.rope
        assert isinstance(rope, DefaultRotaryEmbeddingsConfig), \
            "Only DefaultRotaryEmbeddingsConfig is supported for Gemma."
        rope_theta, rope_scaling = rope.to_hf_config()

        config_class = HfGemmaConfig

        common_args = dict(
            max_position_embeddings=self.seq_len,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_dim,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            hidden_activation=(
                "gelu_pytorch_tanh" if self.activation_function == "gelu_new" else self.activation_function
            ),
            initializer_range=self.initializer_range,
            rms_norm_eps=self.layer_norm_epsilon,
            tie_word_embeddings=self.tie_word_embeddings,
            vocab_size=vocab_size,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
        )

        if self.use_qk_norm:
            config_class = HfGemma3Config
            common_args["use_qk_norm"] = self.use_qk_norm
            common_args["rope_local_base_freq"] = self.rope_local_base_freq
            common_args["query_pre_attn_scalar"] = self.query_pre_attn_scalar or self.head_dim
            common_args["final_logit_softcapping"] = self.final_logit_softcapping
            common_args["attn_logit_softcapping"] = self.attn_logit_softcapping
        elif self.final_logit_softcapping or self.query_pre_attn_scalar:
            config_class = HfGemma2Config
            common_args["final_logit_softcapping"] = self.final_logit_softcapping
            common_args["attn_logit_softcapping"] = self.attn_logit_softcapping
            common_args["query_pre_attn_scalar"] = self.query_pre_attn_scalar or self.head_dim

        config = config_class(
            **common_args,
            **config_overrides,
        )
        return config

    @property
    def model_type(cls) -> type["GemmaLMHeadModel"]:
        return GemmaLMHeadModel

    def flops_per_token(self, vocab_size: int) -> float | None:
        return lm_flops_per_token(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            seq_len=self.seq_len,
            vocab_size=vocab_size,
            glu=False,
        )

    def to_attention_config(self) -> AttentionConfig:
        """Convert this GemmaConfig to an AttentionConfig for use with Attention."""
        return AttentionConfig(
            Embed=self.Embed,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            use_bias=self.use_bias,
            upcast_attn=self.upcast_attn,
            attn_backend=self.attn_backend,
            flash_attention_block_size=self.flash_attention_block_size,
            rope=self.rope,
            qk_norm=self.norm_config if self.use_qk_norm else None,
        )

    @property
    def norm_config(self) -> LayerNormConfigBase:
        """Get the normalization configuration for Gemma."""
        return GemmaNormConfig(
            eps=self.layer_norm_epsilon,
            use_weight=True,  # GemmaRMSNorm requires use_weight=True
            use_bias=False,   # GemmaRMSNorm doesn't support bias
        )

    def mk_LayerNorm(self, axis: AxisSpec):
        """Create a layer normalization module using the config."""
        return self.norm_config.build(axis)


class GemmaRMSNorm(LayerNormBase):
    """
    Like Llama, Gemma uses an RMSNorm instead of a layer norm.

    The canonical Gemma model computes the variances calculation in fp32 explicitly, so
    we do the same for compatibility.

    Also note that Gemma's RMSNorm adds 1 to the output, which is different from most RMSNorm implementations.
    """

    @classmethod
    def init(cls, axis: AxisSpec, eps: float = 1e-6, use_weight: bool = True, use_bias: bool = False, dtype=None):
        assert use_weight, "GemmaRMSNorm does not support use_weight=False"
        assert not use_bias, "GemmaRMSNorm does not support use_bias=True"

        weight = hax.zeros(axis)
        bias = None

        return GemmaRMSNorm(axis, weight, bias, eps, dtype=jnp.float32)

    def __call__(self, x: NamedArray) -> NamedArray:
        # Gemma's norm is calculated in fp32 explicitly
        # See https://github.com/google/gemma_pytorch/blob/main/gemma/model.py#L173
        dtype = x.dtype
        x = x.astype(jnp.float32)

        var = hax.mean(hax.square(x), axis=self.axis)
        inv = hax.rsqrt(var + self.eps)
        out = x * inv
        # NB: this differs from most RMS Norms by adding 1.
        # This is probably so we can weight decay to 1?
        out = out * (1.0 + self.weight)
        return out.astype(dtype)


class GemmaDecoderLayer(ModuleWithStateDictSerialization):
    config: GemmaConfig = eqx.field(static=True)
    self_attn: Attention
    mlp: LlamaMlp
    input_layernorm: LayerNormBase
    post_attention_layernorm: LayerNormBase

    @staticmethod
    def init(config: GemmaConfig, *, key) -> "GemmaDecoderLayer":
        k_attn, k_mlp = jrandom.split(key, 2)

        attn_config = config.to_attention_config()
        attn = Attention.init(attn_config, key=k_attn)
        mlp = LlamaMlp.init(
            config.Embed,
            config.Mlp,
            config.activation_function,
            key=k_mlp,
            use_bias=config.use_bias,
        )
        ln_1 = config.mk_LayerNorm(config.Embed)
        ln_2 = config.mk_LayerNorm(config.Embed)

        return GemmaDecoderLayer(config, attn, mlp, ln_1, ln_2)

    @named_call
    def __call__(self, x: NamedArray, mask: NamedArray | AttentionMask | None, *, key=None) -> NamedArray:
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


class GemmaTransformer(ModuleWithStateDictSerialization):
    config: GemmaConfig = eqx.field(static=True)
    layers: BlockFoldable[GemmaDecoderLayer]
    norm: LayerNormBase

    @staticmethod
    def init(config: GemmaConfig, *, key) -> "GemmaTransformer":
        S = Stacked
        if not config.scan_layers:
            from haliax.nn.scan import BlockSeq
            S = BlockSeq

        layers = S.init(config.Layers, GemmaDecoderLayer, gradient_checkpointing=config.gradient_checkpointing)(
            config,
            key=shaped_rng_split(key, config.num_layers),
        )
        ln_f = config.mk_LayerNorm(config.Embed)

        return GemmaTransformer(config, layers, ln_f)

    @named_call
    def __call__(self, x: NamedArray, attn_mask: NamedArray | AttentionMask | None, *, key) -> NamedArray:
        keys = maybe_rng_split(key, self.config.num_layers) if key is not None else None
        x = self.layers.fold(x, mask=attn_mask, key=keys)
        x = self.norm(x)

        return x


class GemmaLMHeadModel(LmHeadModel[GemmaConfig], ModuleWithStateDictSerialization):
    transformer: GemmaTransformer

    # Gemma ties the weights of the embedding matrix and LM head.  Rather than
    # use eqx.Shared which is a bit cumbersome, we juse re-use the embedding matrix
    # as we do in GPT-2.
    embeddings: LlamaEmbedding
    lm_head: hnn.Linear | None

    @property
    def config(self):
        return self.transformer.config

    @property
    def vocab_size(self) -> int:
        return self.Vocab.size

    @property
    def Vocab(self) -> Axis:
        return self.embeddings.Vocab

    def get_lm_head(self) -> hax.NamedArray:
        if self.lm_head is None:
            return self.embeddings.token_embeddings.weight
        else:
            return self.lm_head.weight

    @classmethod
    def init(cls, Vocab: Axis, config: GemmaConfig, *, key) -> "GemmaLMHeadModel":
        k_t, k_emb = jrandom.split(key, 2)
        transformer = GemmaTransformer.init(config, key=k_t)
        embeddings = LlamaEmbedding.init(Vocab, config, key=k_emb)
        if config.tie_word_embeddings:
            lm_head = None
        else:
            lm_head = hnn.Linear.init(In=config.Embed, Out=Vocab, key=k_emb, use_bias=False, out_first=True)
        return GemmaLMHeadModel(transformer, embeddings, lm_head)

    def activations(
        self,
        input_ids: NamedArray,
        attn_mask: NamedArray | AttentionMask | None = None,
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
        x = self.embeddings.embed(input_ids)
        normalizer = jnp.sqrt(self.config.hidden_dim).astype(x.dtype)
        x = x * normalizer
        x = self.transformer(x, attn_mask=attn_mask, key=key)
        return x

    def resize_vocab(self, new_size: int, key=None) -> "LmHeadModel[GemmaConfig]":
        new_Vocab = self.Vocab.resize(new_size)
        k1, k2 = maybe_rng_split(key, 2)
        new_embeddings = self.embeddings.resize_embeddings(new_size, key=k1)
        if self.lm_head is not None:
            new_lm_matrix = hax.tree_util.resize_axis(self.lm_head.weight, self.Vocab, new_size, key=k2)
            new_lm_head = dataclasses.replace(self.lm_head, Out=new_Vocab, weight=new_lm_matrix)
            return dataclasses.replace(self, embeddings=new_embeddings, lm_head=new_lm_head)
        else:
            return dataclasses.replace(self, embeddings=new_embeddings)

    def _state_dict_key_map(self) -> dict[str, str | None]:
        """Map from Levanter model names to HF."""
        return {"transformer": "model", "embeddings": None}
