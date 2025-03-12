"""The GPT2 architecture, but with Hyena instead of Attention / Transformer."""

import dataclasses
from dataclasses import dataclass
from typing import Dict, Optional, Type

import equinox as eqx
import jax.random as jrandom
from jaxtyping import PRNGKeyArray

import haliax as hax
import haliax.jax_utils
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import named_call, shaped_rng_split
from haliax.nn.scan import Stacked
from haliax.state_dict import ModuleWithStateDictSerialization

from levanter.models.attention import AttentionMask
from levanter.models.gpt2 import Gpt2Embeddings, Gpt2Mlp
from levanter.models.hyena import HyenaConfig, HyenaOperator
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.utils.activation import ActivationFunctionName


@LmConfig.register_subclass("gpt2_hyena")
@dataclass(frozen=True)
class Gpt2HyenaConfig(LmConfig):
    seq_len: int = 1024
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12

    # how much to scale the embedding dim for the mlp layer
    mlp_scale: int = 4

    initializer_range: float = 0.02
    # dropout doesn't really help so we 0 it out by default
    embed_pdrop: float = 0.0
    resid_pdrop: float = 0.0
    hyena_pdrop: float = 0.0
    layer_norm_epsilon: float = 1e-5
    activation_function: ActivationFunctionName = ActivationFunctionName.GELU_NEW

    gradient_checkpointing: bool = True  # better to just always use this
    gradient_checkpointing_block_size: int = 5

    use_bias: bool = True

    # Hyena-specific parameters
    hyena_order: int = 2  # depth of the Hyena recurrence
    hyena_filter_order: int = 64  # width of the FFN parametrizing the implicit filter
    hyena_inner_factor: int = 1  # inner dimension multiplier
    hyena_short_filter_order: int = 3  # length of the explicit input convolutional filter
    hyena_outer_mixing: bool = False  # whether to use outer mixing

    # Hyena filter parameters
    hyena_emb_dim: int = 3  # dim of input to MLP, augments with positional encoding

    # Hyena modulation parameters
    hyena_fast_decay_pct: float = 0.3
    hyena_slow_decay_pct: float = 1.5
    hyena_target: float = 1e-2
    hyena_modulate: bool = True
    hyena_shift: float = 0.0

    # Additional Hyena options
    hyena_post_order_ffn: bool = False  # Apply a dense layer between steps of the recurrence
    hyena_return_state: bool = False  # Whether to return state information

    # Axes
    Pos = property(lambda self: Axis(name="position", size=self.seq_len))
    KeyPos = property(lambda self: self.Pos.alias("key_position"))
    Embed = property(lambda self: Axis(name="embed", size=self.hidden_dim))
    Heads = property(lambda self: Axis(name="heads", size=self.num_heads))
    Layers = property(lambda self: Axis(name="layers", size=self.num_layers))
    Mlp = property(lambda self: Axis(name="mlp", size=self.hidden_dim * self.mlp_scale))
    HeadSize = property(lambda self: Axis(name="head_size", size=self.hidden_dim // self.num_heads))

    @property
    def model_type(cls) -> Type["Gpt2HyenaModel"]:
        return Gpt2HyenaModel

    def flops_per_token(self, vocab_size: int) -> Optional[float]:
        # TODO: implement
        return None


class Gpt2HyenaBlock(eqx.Module):
    config: Gpt2HyenaConfig = eqx.field(static=True)
    ln_1: hnn.LayerNorm
    hyena_operator: HyenaOperator
    ln_2: hnn.LayerNorm
    mlp: Gpt2Mlp
    resid_dropout: hnn.Dropout

    @staticmethod
    def init(config: Gpt2HyenaConfig, *, key) -> "Gpt2HyenaBlock":
        k_hyena, k_mlp = jrandom.split(key, 2)

        ln_1 = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)

        # Create HyenaConfig from Gpt2HyenaConfig
        hyena_config = HyenaConfig(
            seq_len=config.seq_len,
            hidden_dim=config.hidden_dim,
            order=config.hyena_order,
            filter_order=config.hyena_filter_order,
            short_filter_order=config.hyena_short_filter_order,
            outer_mixing=config.hyena_outer_mixing,
            activation=config.activation_function,
            filter_emb_dim=config.hyena_emb_dim,
            filter_dropout=config.hyena_pdrop,
            fast_decay_pct=config.hyena_fast_decay_pct,
            slow_decay_pct=config.hyena_slow_decay_pct,
            target=config.hyena_target,
            modulate=config.hyena_modulate,
            shift=config.hyena_shift,
            resid_pdrop=config.resid_pdrop,
            use_bias=config.use_bias,
        )

        # Initialize the HyenaOperator directly
        hyena_operator = HyenaOperator.init(hyena_config, key=k_hyena)

        ln_2 = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)
        mlp = Gpt2Mlp.init(config.Embed, config.Mlp, config.activation_function, key=k_mlp, use_bias=config.use_bias)
        resid_dropout = hnn.Dropout(pdrop=config.resid_pdrop)

        return Gpt2HyenaBlock(config, ln_1, hyena_operator, ln_2, mlp, resid_dropout)

    @named_call
    def __call__(self, x: NamedArray, *, key):
        k1, k2, k3, k4 = haliax.jax_utils.maybe_rng_split(key, 4)

        # Scale by inverse layer idx if configured (similar to Mistral tweak from the original code)
        # We'll apply this to the input of the Hyena operator
        x_for_hyena = self.ln_1(x)

        hyena_output = self.hyena_operator(x_for_hyena, key=k1)
        hyena_output = self.resid_dropout(hyena_output, key=k2)
        x = x + hyena_output

        ff_output = self.mlp(self.ln_2(x), key=k3)
        ff_output = self.resid_dropout(ff_output, key=k4)
        x = x + ff_output

        return x


class Gpt2HyenaBackbone(ModuleWithStateDictSerialization):
    config: Gpt2HyenaConfig = eqx.field(static=True)
    blocks: Stacked[Gpt2HyenaBlock]
    ln_f: hnn.LayerNorm

    @staticmethod
    def init(config: Gpt2HyenaConfig, *, key):
        # vectorize the blocks
        blocks = Stacked.init(config.Layers, Gpt2HyenaBlock, gradient_checkpointing=config.gradient_checkpointing)(
            config,
            key=shaped_rng_split(key, config.num_layers),
        )
        ln_f = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)

        return Gpt2HyenaBackbone(config, blocks, ln_f)

    @named_call
    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        keys = hax.jax_utils.maybe_rng_split(key, self.config.num_layers) if key is not None else None
        x = self.blocks.fold(x, key=keys)
        x = self.ln_f(x)

        return x

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"blocks": "h"}


class Gpt2HyenaModel(LmHeadModel[Gpt2HyenaConfig]):
    backbone: Gpt2HyenaBackbone
    embeddings: Gpt2Embeddings

    @property
    def config(self):
        return self.backbone.config

    @property
    def Vocab(self) -> Axis:
        return self.embeddings.Vocab

    @property
    def Pos(self) -> Axis:
        return self.config.Pos

    @classmethod
    def init(cls, Vocab: Axis, config: Gpt2HyenaConfig, *, key) -> "Gpt2HyenaModel":
        k_t, k_embeddings = jrandom.split(key, 2)
        backbone = Gpt2HyenaBackbone.init(config, key=k_t)
        embeddings = Gpt2Embeddings.init(
            Vocab,
            # Our config type has everything it needs, but is not a subclass of Gpt2Config
            config,  # type: ignore
            key=k_embeddings,
        )

        return Gpt2HyenaModel(backbone, embeddings)

    def activations(
        self, input_ids: NamedArray, attn_mask: Optional[AttentionMask | NamedArray] = None, *, key=None
    ) -> NamedArray:
        # NOTE: attn_mask not used since we use the Hyena operator instead of attention.
        # TODO: double check HyenaOperator is causal by default.
        # See test in standalone_hyena.py.
        k_embed, k_backbone = haliax.jax_utils.maybe_rng_split(key, 2)
        x = self.embeddings.embed(input_ids, key=k_embed)

        x = self.backbone(x, key=k_backbone)

        return x

    def get_lm_head(self) -> hax.NamedArray:
        return self.embeddings.token_embeddings.weight

    def resize_vocab(self, new_size: int, key: Optional[PRNGKeyArray] = None) -> "Gpt2HyenaModel":
        new_embeddings = self.embeddings.resize_embeddings(new_size, key=key)
        return dataclasses.replace(self, embeddings=new_embeddings)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"backbone": None, "embeddings": None}
