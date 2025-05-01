import dataclasses
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Type, Union

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import PRNGKeyArray

import haliax as hax
import haliax.jax_utils
import haliax.nn as hnn
from haliax import Axis, AxisSpec, NamedArray
from haliax.jax_utils import named_call
from haliax.state_dict import ModuleWithStateDictSerialization, StateDict, with_prefix

from levanter.compat.hf_checkpoints import HFCheckpointConverter, LmWithHfSerializationMixin
from levanter.models.attention import AttentionMask, materialize_mask
from levanter.models.gpt2 import Gpt2Config, Gpt2Transformer
from levanter.models.lm_model import LmConfig
from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.logging import silence_transformer_nag


silence_transformer_nag()
from transformers import PretrainedConfig  # noqa: E402


@LmConfig.register_subclass("backpack")
@dataclass(frozen=True)
class BackpackConfig(Gpt2Config):
    # Backpack-specific terms
    num_senses: int = 16
    sense_intermediate_scale: int = 4

    @property
    def model_type(self) -> Type["BackpackLMHeadModel"]:
        return BackpackLMHeadModel

    def hf_checkpoint_converter(self) -> HFCheckpointConverter["BackpackConfig"]:  # type: ignore
        # We trust this code because it's in our hub repo
        return HFCheckpointConverter(
            self,
            reference_checkpoint="stanford-crfm/levanter-backpack-1b@9face7bd6182155fe3f1a6a5a14ca1c4810bb079",
            trust_remote_code=True,
        )

    # Axes
    SenseHeadDim = property(lambda self: Axis(name="head_dim", size=self.hidden_dim // self.num_senses))
    Senses = property(lambda self: Axis(name="senses", size=self.num_senses))
    SenseIntermediate = property(
        lambda self: Axis(name="concat_senses", size=self.sense_intermediate_scale * self.hidden_dim)
    )

    def to_hf_config(self, vocab_size, config_overrides=None):
        if config_overrides is None:
            config_overrides = {}

        return PretrainedConfig(
            vocab_size=vocab_size,
            n_positions=self.seq_len,
            n_layer=self.num_layers,
            n_head=self.num_heads,
            n_embd=self.hidden_dim,
            initializer_range=self.initializer_range,
            attn_pdrop=self.attn_pdrop,
            embd_pdrop=self.embed_pdrop,
            resid_pdrop=self.resid_pdrop,
            layer_norm_epsilon=self.layer_norm_epsilon,
            activation_function=self.activation_function,
            scale_attn_by_inverse_layer_idx=self.scale_attn_by_inverse_layer_idx,
            reorder_and_upcast_attn=self.upcast_attn,
            num_senses=self.num_senses,
            sense_intermediate_scale=self.sense_intermediate_scale,
            **config_overrides,
        )

    @classmethod
    def from_hf_config(cls, hf_config: PretrainedConfig):
        return cls(
            seq_len=hf_config.n_positions,
            num_layers=hf_config.n_layer,
            num_heads=hf_config.n_head,
            hidden_dim=hf_config.n_embd,
            initializer_range=hf_config.initializer_range,
            attn_pdrop=hf_config.attn_pdrop,
            embed_pdrop=hf_config.embd_pdrop,
            resid_pdrop=hf_config.resid_pdrop,
            layer_norm_epsilon=hf_config.layer_norm_epsilon,
            activation_function=ActivationFunctionEnum(hf_config.activation_function),
            scale_attn_by_inverse_layer_idx=hf_config.scale_attn_by_inverse_layer_idx,
            upcast_attn=hf_config.reorder_and_upcast_attn,
            num_senses=hf_config.num_senses,
            sense_intermediate_scale=hf_config.sense_intermediate_scale,
        )


class BackpackMlp(eqx.Module):
    c_fc: hnn.Linear  # projection from Embed to Intermediate (typically 4x Embed)
    c_proj: hnn.Linear  # projection from Intermediate to Embed
    act: Callable = eqx.field(static=True)

    @staticmethod
    def init(
        Embed: Axis,
        Mlp: Axis,
        Out: AxisSpec,
        activation_fn: Union[ActivationFunctionEnum, Callable],
        *,
        key,
        use_bias: bool = True,
    ) -> "BackpackMlp":
        k_fc, k_proj = jrandom.split(key, 2)
        c_fc = hnn.Linear.init(Out=Mlp, In=Embed, key=k_fc, use_bias=use_bias, out_first=False)
        c_proj = hnn.Linear.init(Out=Out, In=Mlp, key=k_proj, use_bias=use_bias, out_first=False)
        if isinstance(activation_fn, ActivationFunctionEnum):
            activation_fn = activation_fn.to_fn()

        return BackpackMlp(c_fc=c_fc, c_proj=c_proj, act=activation_fn)

    @named_call
    def __call__(self, x: NamedArray):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x


class WeightsOnlyAttention(ModuleWithStateDictSerialization):
    """
    Changes from Gpt2Attention:
    1. No projection; it returns the attention weights
    2. Use SenseHeadDim instead of HeadDim, use Senses instead of Heads
    """

    # No projection
    config: Gpt2Config = eqx.field(static=True)

    c_attn: hnn.Linear  # input projection from [embed] -> [(q, k, v), heads, head_dim]
    dropout: hnn.Dropout

    @staticmethod
    def init(config: Gpt2Config, *, key) -> "WeightsOnlyAttention":
        Qk = Axis("qk", size=2)
        use_bias = config.use_bias
        Embed = config.Embed

        k_c, _ = jrandom.split(key, 2)
        # NB: out_first=True b/c the torch implementation uses Linear
        c_attn = hnn.Linear.init(
            In=Embed, Out=(Qk, config.Senses, config.SenseHeadDim), key=k_c, use_bias=use_bias, out_first=True
        )
        dropout = hnn.Dropout(config.attn_pdrop)

        return WeightsOnlyAttention(config, c_attn, dropout)

    @named_call
    def __call__(self, x: NamedArray, mask: Optional[AttentionMask | NamedArray], layer_idx, *, key):
        qk_out = self.c_attn(x)
        q, k = qk_out.unbind("qk")

        # Rename k's Pos as haliax doesn't support unnamed axes or duplicate axes
        k = k.rename({"position": "key_position"})
        QPos = q.resolve_axis("position")
        KPos = k.resolve_axis("key_position")

        mask = materialize_mask(mask, q.resolve_axis("position"), k.resolve_axis("key_position"))

        attn_weights = hnn.attention.dot_product_attention_weights(
            "head_dim",
            "key_position",
            q,
            k,
            mask=materialize_mask(mask, QPos, KPos),
            attention_dtype=jnp.float32 if self.config.upcast_attn else None,
        )

        attn_weights = self.dropout(attn_weights, key=key)
        return attn_weights


class NoMixBlock(eqx.Module):
    ln_1: hnn.LayerNorm
    ln_2: hnn.LayerNorm
    mlp: BackpackMlp
    resid_dropout1: hnn.Dropout
    resid_dropout2: hnn.Dropout

    @staticmethod
    def init(config: BackpackConfig, *, key) -> "NoMixBlock":
        k_mlp = jrandom.split(key, 1)[0]

        ln_1 = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=True)
        resid_dropout1 = hnn.Dropout(pdrop=config.resid_pdrop)
        resid_dropout2 = hnn.Dropout(pdrop=config.resid_pdrop)
        ln_2 = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=True)

        mlp = BackpackMlp.init(
            Embed=config.Embed,
            Mlp=config.Mlp,
            Out=config.Embed,
            activation_fn=config.activation_function,
            key=k_mlp,
            use_bias=config.use_bias,
        )

        return NoMixBlock(ln_1=ln_1, ln_2=ln_2, mlp=mlp, resid_dropout1=resid_dropout1, resid_dropout2=resid_dropout2)

    @named_call
    def __call__(self, hidden_states: NamedArray, residual: NamedArray, *, key):
        k1, k2 = haliax.jax_utils.maybe_rng_split(key, 2)

        residual = self.resid_dropout1(hidden_states, key=k1) + residual
        hidden_states = self.ln_1(residual)
        mlp_out = self.mlp(hidden_states)
        residual = self.resid_dropout2(mlp_out, key=k2) + residual
        hidden_states = self.ln_2(residual)

        return hidden_states


class BackpackSenses(eqx.Module):
    dropout: hnn.Dropout
    block: NoMixBlock
    ln: hnn.LayerNorm
    final_mlp: BackpackMlp

    Pos: Axis = eqx.field(static=True)

    @staticmethod
    def init(
        config,
        dropout_prob: float,
        *,
        key,
    ):
        k_block, k_mlp = jrandom.split(key, 2)

        dropout = hnn.Dropout(pdrop=dropout_prob)
        block = NoMixBlock.init(config, key=k_block)
        ln = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=True)
        final_mlp = BackpackMlp.init(
            Embed=config.Embed,
            Mlp=config.SenseIntermediate,
            Out=(config.Senses, config.Embed),
            activation_fn=config.activation_function,
            key=k_mlp,
            use_bias=config.use_bias,
        )

        return BackpackSenses(
            dropout=dropout,
            block=block,
            ln=ln,
            final_mlp=final_mlp,
            Pos=config.Pos,
        )

    @named_call
    def sense_embed(self, input_embeds, *, key):
        hidden_states = self.ln(input_embeds)
        hidden_states = self.block(hidden_states, input_embeds, key=key)
        senses = self.final_mlp(hidden_states)

        return senses


class BackpackGpt2Embeddings(eqx.Module):
    Vocab: Axis = eqx.field(static=True)
    config: Gpt2Config = eqx.field(static=True)

    token_embeddings: NamedArray
    position_embeddings: NamedArray
    dropout: hnn.Dropout

    @staticmethod
    def init(Vocab: Axis, config: Gpt2Config, *, key) -> "BackpackGpt2Embeddings":
        k_wte, k_wpe, k_out = jrandom.split(key, 3)

        token_embeddings = hax.random.normal(k_wte, (Vocab, config.Embed)) * config.initializer_range
        position_embeddings = hax.random.normal(k_wpe, (config.Pos, config.Embed)) * (config.initializer_range / 2)
        dropout = hnn.Dropout(pdrop=config.embed_pdrop)

        return BackpackGpt2Embeddings(Vocab, config, token_embeddings, position_embeddings, dropout)

    @named_call
    def embed_input_ids(self, input_ids: NamedArray) -> NamedArray:
        return self.token_embeddings.take("vocab", input_ids)

    @named_call
    def embed(self, input_ids, *, key):
        input_embeds = self.token_embeddings.take("vocab", input_ids)
        position_embeds = self.position_embeddings
        input_len = input_ids.resolve_axis("position").size
        x = input_embeds + position_embeds["position", hax.dslice(0, input_len)]
        x = self.dropout(x, key=key)

        return x

    def unembed(self, x: NamedArray):
        return hax.dot(x, self.token_embeddings, axis="embed")

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"token_embeddings": "wte.weight", "position_embeddings": "wpe.weight"}

    def resize_embeddings(self, new_size: int, key: Optional[PRNGKeyArray] = None):
        new_weights = hax.tree_util.resize_axis(self.token_embeddings, self.Vocab, new_size, key=key)
        return dataclasses.replace(self, Vocab=self.Vocab.resize(new_size), token_embeddings=new_weights)


class BackpackLMHeadModel(LmWithHfSerializationMixin, ModuleWithStateDictSerialization):
    transformer: Gpt2Transformer
    embeddings: BackpackGpt2Embeddings
    sense_net: BackpackSenses
    kq_selfattention: WeightsOnlyAttention

    @property
    def config(self):
        return self.transformer.config

    @property
    def Vocab(self) -> Axis:
        return self.embeddings.Vocab

    @property
    def Pos(self) -> Axis:
        return self.sense_net.Pos

    @staticmethod
    def init(Vocab: Axis, config: BackpackConfig, *, key):
        k_t, k_embeddings, k_attn = jrandom.split(key, 3)
        transformer = Gpt2Transformer.init(config, key=k_t)

        embeddings = BackpackGpt2Embeddings.init(
            Vocab=Vocab,
            config=config,
            key=k_embeddings,
        )
        sense_net = BackpackSenses.init(
            config=config,
            dropout_prob=config.embed_pdrop,
            key=k_embeddings,
        )
        kq_selfattention = WeightsOnlyAttention.init(
            config=config,
            key=k_attn,
        )

        return BackpackLMHeadModel(
            transformer=transformer,
            embeddings=embeddings,
            sense_net=sense_net,
            kq_selfattention=kq_selfattention,
        )

    @named_call
    def activations(
        self, input_ids: NamedArray, attn_mask: Optional[AttentionMask | NamedArray] = None, *, key=None
    ) -> NamedArray:
        k_embed, k_transformer, k_senses, k_sa = haliax.jax_utils.maybe_rng_split(key, 4)

        # Compute contextualization weights
        hidden_states = self.embeddings.embed(input_ids, key=k_embed)
        hidden_states = self.transformer(hidden_states, attn_mask, key=k_transformer)
        contextualization_weights = self.kq_selfattention(
            hidden_states, mask=attn_mask, layer_idx=self.config.num_layers, key=k_sa
        )  # (seq, seq, senses)

        ## Compute sense vectors
        sense_input_embeds = self.embeddings.embed_input_ids(input_ids)  # (seq, embed
        sense_vectors = self.sense_net.sense_embed(sense_input_embeds, key=k_senses)  # (seq, senses, embed)
        sense_vectors = sense_vectors.rename({self.Pos: self.config.KeyPos})

        ## Weight-and-sum
        hidden_states = hax.dot(
            contextualization_weights, sense_vectors, axis=self.config.KeyPos
        )  # (seq, senses, embed)
        hidden_states = hax.sum(hidden_states, axis=self.config.Senses)

        # Rescale - this is important for large num_senses
        scale = self.config.Senses.size
        hidden_states = hidden_states / scale

        return hidden_states

    def get_lm_head(self) -> hax.NamedArray:
        return self.embeddings.token_embeddings

    def resize_vocab(self, new_size: int, key: Optional[PRNGKeyArray] = None):
        new_embeddings = self.embeddings.resize_embeddings(new_size, key=key)
        return dataclasses.replace(self, embeddings=new_embeddings)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {
            "transformer": "backpack.gpt2_model",
            "embeddings": "backpack.gpt2_model",
            "sense_net": "backpack.sense_network",
            "kq_selfattention": "backpack.sense_weight_net",
        }

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        state_dict = super().update_state_dict(state_dict, prefix=prefix)
        # In levanter's implementation, we have a shared embedding matrix for both the word
        # embeddings and the sense embeddings
        state_dict[with_prefix(prefix, "backpack.word_embeddings.weight")] = state_dict[
            with_prefix(prefix, "backpack.gpt2_model.wte.weight")
        ]
        state_dict[with_prefix(prefix, "backpack.position_embeddings.weight")] = state_dict[
            with_prefix(prefix, "backpack.gpt2_model.wpe.weight")
        ]
        return state_dict
