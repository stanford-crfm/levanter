import dataclasses
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import PRNGKeyArray

import haliax as hax
import haliax.nn as hnn
from haliax.nn import cross_entropy_loss
from haliax import Axis, AxisSpec, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call, shaped_rng_split
from haliax.nn.scan import BlockSeq

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
from levanter.models.attention import AttentionBackend, AttentionMask, simple_attention_with_dropout
from levanter.models.gpt2 import ACT2FN
from levanter.models.lm_model import LmConfig, LmHeadModel, MaskedLmExample
from levanter.types import BlockFoldable
from levanter.utils.flop_utils import lm_flops_per_token

silence_transformer_nag()
from transformers import PretrainedConfig as HfConfig
from transformers import RobertaConfig as HfRobertaConfig



@LmConfig.register_subclass("roberta")
@dataclass(frozen=True)
class RobertaConfig(HFCompatConfig):
    r"""

    Adapted from HuggingFace RobertaConfig, description below


    This is the configuration class to store the configuration of a [`RobertaModel`] or a [`TFRobertaModel`]. It is
    used to instantiate a RoBERTa model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the RoBERTa
    [FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50265):
            Vocabulary size of the RoBERTa model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`RobertaModel`] or [`TFRobertaModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`RobertaModel`] or [`TFRobertaModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.

    Examples:

    ```python
    >>> from transformers import RobertaConfig, RobertaModel

    >>> # Initializing a RoBERTa configuration
    >>> configuration = RobertaConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = RobertaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    vocab_size: int = 50265
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2
    position_embedding_type: Optional[str] = "absolute"
    use_cache: bool = False
    classifier_dropout: Optional[float] = None

    scan_layers: bool = True
    gradient_checkpointing: bool = True

    reference_checkpoint: str = "FacebookAI/roberta-base"
    tokenizer: Optional[str] = None

    # Axes
    Pos = property(lambda self: Axis(name="position", size=self.max_position_embeddings))
    KeyPos = property(lambda self: self.Pos.alias("key_position"))
    Embed = property(lambda self: Axis(name="embed", size=self.hidden_size))
    EmbedAtt = property(lambda self: self.Embed.alias("embed_att"))
    FinalEmbed = property(lambda self: self.Embed.alias("final_embed"))
    Heads = property(lambda self: Axis(name="heads", size=self.num_attention_heads))
    Layers = property(lambda self: Axis(name="layers", size=self.num_hidden_layers))
    Mlp = property(lambda self: Axis(name="mlp", size=self.intermediate_size))
    HeadSize = property(lambda self: Axis(name="head_size", size=self.hidden_size // self.num_attention_heads))


    @classmethod
    def from_hf_config(cls, hf_config: HfConfig) -> "RobertaConfig":
        return RobertaConfig(
            vocab_size = hf_config.vocab_size,
            hidden_size = hf_config.hidden_size,
            num_hidden_layers = hf_config.num_hidden_layers,
            num_attention_heads = hf_config.num_attention_heads,
            intermediate_size = hf_config.intermediate_size,
            hidden_act = hf_config.hidden_act,
            hidden_dropout_prob= hf_config.hidden_dropout_prob,
            attention_probs_dropout_prob = hf_config.attention_probs_dropout_prob,
            max_position_embeddings = hf_config.max_position_embeddings,
            type_vocab_size = hf_config.type_vocab_size,
            initializer_range = hf_config.initializer_range,
            layer_norm_eps = hf_config.layer_norm_eps,
            pad_token_id = hf_config.pad_token_id,
            bos_token_id = hf_config.bos_token_id,
            eos_token_id = hf_config.eos_token_id,
            position_embedding_type = hf_config.position_embedding_type,
            use_cache = hf_config.use_cache,
            classifier_dropout = hf_config.classifier_dropout,
        )
    
    def hf_checkpoint_converter(self) -> HFCheckpointConverter["RobertaConfig"]:  # type: ignore
        return HFCheckpointConverter(
            self.__class__,
            reference_checkpoint=self.reference_checkpoint,
            trust_remote_code=True,
            tokenizer=self.tokenizer if self.tokenizer else self.reference_checkpoint,
            HfConfigClass=HfRobertaConfig,
        )

    def to_hf_config(self, vocab_size: int, config_overrides: Optional[Dict] = None) -> HfRobertaConfig:
        """Convert to HuggingFace's LlamaConfig

        Args:
            vocab_size (int, optional): Vocabulary size of the tokenizer. Defaults to 32000.
            config_overrides (dict, optional): Overrides for the config. Defaults to None.

        Returns:
            HfRobertaConfig: HuggingFace's RobertaConfig
        """

        if config_overrides is None:
            config_overrides = {}

        return HfRobertaConfig(
            vocab_size = vocab_size,
            hidden_size = self.hidden_size,
            num_hidden_layers = self.num_hidden_layers,
            num_attention_heads = self.num_attention_heads,
            intermediate_size = self.intermediate_size,
            hidden_act = self.hidden_act,
            hidden_dropout_prob = self.hidden_dropout_prob,
            attention_probs_dropout_prob = self.attention_probs_dropout_prob,
            max_position_embeddings = self.max_position_embeddings,
            type_vocab_size = self.type_vocab_size,
            initializer_range = self.initializer_range,
            layer_norm_eps = self.layer_norm_eps,
            pad_token_id = self.pad_token_id,
            bos_token_id = self.bos_token_id,
            eos_token_id = self.eos_token_id,
            position_embedding_type = self.position_embedding_type,
            use_cache = self.use_cache,
            classifier_dropout = self.classifier_dropout,
        )

    @property
    def model_type(self) -> Type["RobertaModel"]:
        return RobertaModel
    
    def flops_per_token(self, vocab_size: int):
        return lm_flops_per_token(
            hidden_dim=self.hidden_size,
            intermediate_dim=self.intermediate_size,
            num_layers=self.num_hidden_layers,
            num_kv_heads=self.num_attention_heads,
            num_heads=self.num_attention_heads,
            seq_len=self.max_position_embeddings,
            vocab_size=vocab_size,
            glu=True,
        )

class RobertaSelfAttention(eqx.Module, StateDictSerializationMixin):

    config: RobertaConfig
    Heads: Axis
    HeadSize: Axis
    EmbedAtt: Axis

    q_proj: hnn.Linear
    k_proj: hnn.Linear
    v_proj: hnn.Linear
    
    dropout: hnn.Dropout
    position_embedding_type: Optional[str]

    Pos: Axis
    KeyPos: Axis
    distance_embedding: Optional[hnn.Embedding]

    @staticmethod
    def init(config: RobertaConfig, *, key) -> "RobertaSelfAttention":        
        Embed = config.Embed
        EmbedAtt = config.EmbedAtt

        k_q, k_k, k_v, k_e = jrandom.split(key, 4)
        q_proj = hnn.Linear.init(In=Embed, Out=EmbedAtt, key=k_q, out_first=True)
        k_proj = hnn.Linear.init(In=Embed, Out=EmbedAtt, key=k_k, out_first=True)
        v_proj = hnn.Linear.init(In=Embed, Out=EmbedAtt, key=k_v, out_first=True)

        dropout = hnn.Dropout(config.attention_probs_dropout_prob)

        distance_embedding = None
        position_embedding_type = config.position_embedding_type

        if position_embedding_type == "relative_key" or position_embedding_type == "relative_key_query":
            RelPos = Axis("rel_pos", 2 * config.max_position_embeddings - 1)
            distance_embedding = hnn.Embedding.init(RelPos, config.HeadSize, k_e)

        return RobertaSelfAttention(config, config.Heads, config.HeadSize, EmbedAtt,
                                    q_proj, k_proj, v_proj,
                                    dropout, position_embedding_type,
                                    config.Pos, config.KeyPos, distance_embedding,
                                    )

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"q_proj": "query", "k_proj": "key", "v_proj": "value"}

    def _rope_scale_factor(self) -> float:
        # hasattr for gemma and I'm feeling lazy
        if hasattr(self.config, "rope_scaling") and self.config.rope_scaling is not None:
            assert self.config.rope_scaling["type"] == "linear"
            return self.config.rope_scaling["factor"]
        return 1.0

    def transpose_for_scores(self, x: NamedArray) -> NamedArray:
        # Makes sure to have the correct output order as well
        y = hax.rearrange(x, "... position (embed_att: heads head_size) -> ... heads position head_size", heads=self.Heads, head_size=self.HeadSize)
        return y

    @named_call
    def __call__(
        self,
        hidden_states: NamedArray,
        attention_mask: Optional[NamedArray] = None,
        *,
        key = None
    ) -> Tuple[NamedArray]:

        query_layer = self.transpose_for_scores(self.q_proj(hidden_states))
        key_layer = self.transpose_for_scores(self.k_proj(hidden_states))
        value_layer = self.transpose_for_scores(self.v_proj(hidden_states))

        if self.position_embedding_type == "rope":
            cos, sin = llama_rotary_pos_emb(
                self.config.HeadSize, hidden_states.resolve_axis("position"), scale=self._rope_scale_factor()
            )
            query_layer, key_layer = _apply_rotary_pos_emb(query_layer, key_layer, cos, sin)

        key_layer = key_layer.rename({"position": "key_position"})
        value_layer = value_layer.rename({"position": "key_position"})

        attention_scores = hax.dot(query_layer, key_layer, axis=self.HeadSize) # aka hax.einsum("bhld, bhrd -> bhlr")

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            Left = self.Pos # Queries
            Right = self.KeyPos # Keys
            
            position_ids_l = hax.arange(Left).broadcast_to((Left,Right))
            position_ids_r = hax.arange(Right).broadcast_to((Left,Right))

            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.Pos.size)

            if self.position_embedding_type == "relative_key":
                relative_position_scores = hax.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = hax.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = hax.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key
        
        attention_scores /= jnp.sqrt(self.HeadSize.size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            # Attention_mask should have shape Batch Pos, so it should broadcast to shape Batch Heads Pos KeyPos for summation
            attention_scores = attention_scores + attention_mask 
        
        attention_probs = hnn.softmax(attention_scores, axis=self.KeyPos)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs, key=key)

        hax.dot(query_layer, key_layer, axis=self.HeadSize)

        context_layer = hax.dot(attention_probs, value_layer, axis=self.KeyPos)
        
        outputs = hax.rearrange(context_layer, ("... heads position head_size -> ... position (embed_att: heads head_size)"), heads=self.Heads, head_size=self.HeadSize)

        return outputs

class RobertaSelfOutput(eqx.Module, StateDictSerializationMixin):
    dense: hnn.Linear
    LayerNorm: hnn.LayerNorm
    dropout: hnn.Dropout

    @staticmethod
    def init(config: RobertaConfig, *, key) -> "RobertaSelfOutput":
        Embed = config.Embed
        EmbedAtt = config.EmbedAtt
        dense = hnn.Linear.init(In=EmbedAtt, Out=Embed, key=key, out_first=True)
        LayerNorm = hnn.LayerNorm.init(axis=Embed, eps=config.layer_norm_eps)
        dropout = hnn.Dropout(config.hidden_dropout_prob)
        return RobertaSelfOutput(dense, LayerNorm, dropout)
    
    @named_call
    def __call__(self, hidden_states: NamedArray, input: NamedArray,*, key) -> NamedArray:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, key=key)
        hidden_states = self.LayerNorm(hidden_states + input)
        return hidden_states

class RobertaAttention(eqx.Module, StateDictSerializationMixin):
    self_attn: RobertaSelfAttention
    output: RobertaSelfOutput

    @staticmethod
    def init(config: RobertaConfig, *, key) -> "RobertaAttention":
        k_a, k_o = jrandom.split(key, 2)

        self_attn = RobertaSelfAttention.init(config, key=k_a)
        output = RobertaSelfOutput.init(config, key=k_o)

        return RobertaAttention(self_attn, output)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"self_attn": "self"}

    @named_call
    def __call__(
        self,
        hidden_states: NamedArray,
        attention_mask: Optional[NamedArray] = None,
        *,
        key
    ) -> NamedArray:
        k_a, k_o = maybe_rng_split(key, 2)
        
        self_outputs = self.self_attn(
            hidden_states,
            attention_mask,
            key=k_a
        )
        attention_output = self.output(self_outputs, hidden_states, key=k_o)
        return attention_output
    
class RobertaIntermediate(eqx.Module, StateDictSerializationMixin):
    dense: hnn.Linear
    intermediate_act_fn: Callable = eqx.static_field()

    @staticmethod
    def init(config, *, key) -> "RobertaIntermediate":
        dense = hnn.Linear.init(config.Embed, config.Mlp, key=key, out_first=True)
        if isinstance(config.hidden_act, str):
            intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            intermediate_act_fn = config.hidden_act

        return RobertaIntermediate(dense, intermediate_act_fn)

    @named_call
    def __call__(self, hidden_states: NamedArray, *, key = None) -> NamedArray:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class RobertaOutput(eqx.Module, StateDictSerializationMixin):
    dense: hnn.Linear
    LayerNorm: hnn.LayerNorm
    dropout: hnn.Dropout

    @staticmethod
    def init(config: RobertaConfig, *, key) -> "RobertaSelfOutput":
        Embed = config.Embed
        dense = hnn.Linear.init(In=config.Mlp, Out=Embed, key=key, out_first=True)
        LayerNorm = hnn.LayerNorm.init(axis=Embed, eps=config.layer_norm_eps)
        dropout = hnn.Dropout(config.hidden_dropout_prob)
        return RobertaSelfOutput(dense, LayerNorm, dropout)

    @named_call
    def __call__(self, hidden_states: NamedArray, input: NamedArray, *, key) -> NamedArray:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, key=key)
        hidden_states = self.LayerNorm(hidden_states + input)
        return hidden_states
    
class RobertaLayer(eqx.Module, StateDictSerializationMixin):
    attention: RobertaAttention
    intermediate: RobertaIntermediate
    output: RobertaOutput
    
    @staticmethod
    def init(config: RobertaConfig, *, key) -> "RobertaLayer":
        k_a, k_i, k_o = jrandom.split(key, 3)

        attention = RobertaAttention.init(config, key=k_a)
        intermediate = RobertaIntermediate.init(config, key=k_i)
        output = RobertaOutput.init(config, key=k_o)

        return RobertaLayer(attention, intermediate, output)
    
    @named_call
    def __call__(
        self,
        hidden_states: NamedArray,
        attention_mask: Optional[NamedArray] = None,
        *,
        key
    ) -> Tuple[NamedArray]:
        k_a, k_o = maybe_rng_split(key, 2)

        attention_output = self.attention(
            hidden_states,
            attention_mask,
            key=k_a, 
        )

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output, key=k_o)

        # jax.debug.print("{layer_output}", layer_output=layer_output)

        return (layer_output, layer_output)


class RobertaEncoder(eqx.Module, StateDictSerializationMixin):
    config: RobertaConfig
    layer: BlockFoldable[RobertaLayer]
    output_hidden_states: bool

    @staticmethod
    def init(config: RobertaConfig, output_hidden_states: bool = False, *, key) -> "RobertaEncoder":
        S = BlockSeq

        layer = S.init(config.Layers, RobertaLayer, gradient_checkpointing=config.gradient_checkpointing)(
            config,
            key=shaped_rng_split(key, config.num_hidden_layers), #TODO: config.gradient_checkpointing
        )

        return RobertaEncoder(config, layer, output_hidden_states)

    @named_call
    def __call__(
        self,
        hidden_states: NamedArray,
        attention_mask: Optional[NamedArray] = None,
        *,
        key
    ) -> Tuple[NamedArray]:
        
        keys = maybe_rng_split(key, self.config.num_hidden_layers) if key is not None else None

        x, intermediates = self.layer.scan(hidden_states, attention_mask, key=keys)

        if not self.output_hidden_states:
            return x, None
        else:
             return x, intermediates
    
    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None):
        out = super().from_state_dict(state_dict, prefix=prefix)
        return out

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        my_state_dict: StateDict = {}
        super().update_state_dict(my_state_dict, prefix=prefix)

        state_dict.update(my_state_dict)

        return state_dict

class RobertaEmbedding(eqx.Module, StateDictSerializationMixin):
    Vocab: Axis = eqx.static_field()
    Pos: Axis = eqx.static_field()

    word_embeddings: hnn.Embedding
    position_embeddings: hnn.Embedding
    token_type_embeddings: Optional[hnn.Embedding]
    padding_idx: NamedArray

    LayerNorm: hnn.LayerNorm
    dropout: hnn.Dropout
    position_embedding_type: Optional[str]

    @staticmethod
    def init(Vocab: Axis, config: RobertaConfig, *, key) -> "RobertaEmbedding":
        key_w, key_p, key_t = jrandom.split(key, 3)

        padding_idx = config.pad_token_id

        word_embeddings = hnn.Embedding.init(Vocab, config.Embed, key=key_w) # padding_idx not specified
        position_embeddings = hnn.Embedding.init(config.Pos, config.Embed, key=key_p)

        Token = hax.Axis("token", config.type_vocab_size)

        token_type_embeddings = hnn.Embedding.init(Token, config.Embed, key=key_t)
        
        LayerNorm = hnn.LayerNorm.init(config.Embed, config.layer_norm_eps)
        dropout = hnn.Dropout(config.hidden_dropout_prob)

        return RobertaEmbedding(Vocab, config.Pos, word_embeddings, position_embeddings, token_type_embeddings, padding_idx, LayerNorm, dropout, config.position_embedding_type)

    def create_position_ids_from_input_ids(self, input_ids, past_key_values_length=0):
        mask = hax.not_equal(input_ids, self.padding_idx) * 1
        incremental_indices = (hax.cumsum(mask, axis=self.Pos).astype(mask) + past_key_values_length) * mask
        return incremental_indices + self.padding_idx

    def create_position_ids_from_inputs_embeds(self, input_axes, PosInput):
        # position_ids = hax.arange(axis = PosInput, start = 0, dtype=jnp.int32)
        position_ids = hax.arange(axis = PosInput, start = self.padding_idx + 1, dtype=jnp.int32)

        return hax.broadcast_to(position_ids, input_axes)

    @named_call
    def embed(self, input_ids=None, token_type_ids=None, position_ids=None, input_embeds=None, past_key_values_length=0, *, key = None):
        print(input_ids.dtype)
        print(token_type_ids.dtype)
        print(position_ids.dtype)
        print(input_embeds.dtype)
        
        """
        Note: When inputting your own embeds into input_embeds, make sure that the embeds axis has the name "embed"
        for compatibility with the position_id creation function. Make sures its length is not equal to 
        """
        
        # Get Axes
        if input_ids is not None:
            input_axes = input_ids.axes
        else:
            input_axes = hax.eliminate_axes(input_embeds.axes, "embed")

        # Get position_ids
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = self.create_position_ids_from_input_ids(input_ids, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(input_axes, input_embeds.resolve_axis("position"))
        
        # Get token_type_ids
        if token_type_ids is None:
            token_type_ids = hax.zeros(input_axes, dtype=jnp.int32)

        if input_embeds is None:
            input_embeds = self.word_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = input_embeds + token_type_embeddings

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings, key=key)
        return embeddings

class RobertaPooler(eqx.Module, StateDictSerializationMixin):
    dense: hnn.Linear
    config: RobertaConfig

    @staticmethod
    def init(config: RobertaConfig, *, key):
        dense = hnn.Linear.init(In=config.Embed, Out=config.FinalEmbed, key=key, out_first=True)

        return RobertaPooler(dense, config)

    @named_call
    def __call__(self, hidden_states: NamedArray, *, key=None) -> NamedArray:
        first_token = hidden_states[{"position" : 0}]
        x = self.dense(first_token, key=key).rename({self.config.FinalEmbed: self.config.Embed})
        x = hax.tanh(x)
        return x


class RobertaModel(eqx.Module, StateDictSerializationMixin):
    encoder: RobertaEncoder
    embeddings: RobertaEmbedding
    pooler : Optional[RobertaPooler]
    output_hidden_states: bool

    @staticmethod
    def init(Vocab: Axis, config: RobertaConfig, add_pooling_layer: bool = True, output_hidden_states: bool = False, *, key) -> "RobertaModel":
        k_t, k_emb, k_p = jrandom.split(key, 3)
        encoder = RobertaEncoder.init(config=config, output_hidden_states=output_hidden_states, key=k_t)
        embeddings = RobertaEmbedding.init(Vocab, config, key=k_emb)

        pooler = RobertaPooler.init(config, key=k_p) if add_pooling_layer else None
        return RobertaModel(encoder, embeddings, pooler, output_hidden_states)

    @property
    def config(self):
        return self.encoder.config

    @property
    def vocab_size(self) -> int:
        return self.Vocab.size

    @property
    def Vocab(self) -> Axis:
        return self.embeddings.Vocab

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    @named_call
    def __call__(
        self,
        input_ids: Optional[NamedArray] = None,
        token_type_ids: Optional[NamedArray] = None, 
        position_ids: Optional[NamedArray] = None, 
        input_embeds: Optional[NamedArray] = None,
        attention_mask: Optional[NamedArray] = None,
        *,
        key,
    ) -> Tuple[NamedArray]:
        """
        Not Used: meant to be used to improve performance in decoder implementations

        head_mask: Optional[NamedArray] = None,
        encoder_hidden_states: Optional[NamedArray] = None,
        encoder_attention_mask: Optional[NamedArray] = None,
        past_key_values_length = 0,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        """
        k_emb, k_e, k_p = maybe_rng_split(key, 3)

        if input_ids is not None and input_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_axes = input_ids.axes
        elif input_embeds is not None:
            input_axes = hax.eliminate_axes(input_embeds.axes, "embed")
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = hax.ones(input_axes)
        
        # Attention mask from mask to actual numbers 0 -> -inf
        attention_mask = (attention_mask == 0) * jnp.finfo(jnp.bfloat16).min
        
        embedding_output = self.embeddings.embed(input_ids, token_type_ids, position_ids, input_embeds, key=k_emb)

        encoder_outputs = self.encoder(embedding_output, attention_mask=attention_mask, key=k_e)
        
        sequence_output = encoder_outputs[0]

        pooled_output = self.pooler(sequence_output, key=k_p) if self.pooler is not None else None

        return (sequence_output, pooled_output) + encoder_outputs[1:] if self.output_hidden_states else (sequence_output, pooled_output)

class RobertaLMHead(eqx.Module, StateDictSerializationMixin):
    """Roberta Head for masked language modeling."""

    dense: hnn.Linear
    layer_norm: hnn.LayerNorm
    decoder: hnn.Linear
    config: RobertaConfig

    @staticmethod
    def init(Vocab: Axis, config: RobertaConfig, *, key):
        k_dense, k_decoder = jrandom.split(key, 2)
        Embed = config.Embed

        dense = hnn.Linear.init(In=Embed, Out=config.FinalEmbed, key=k_dense, out_first=True)
        layer_norm = hnn.LayerNorm.init(axis=Embed, eps=config.layer_norm_eps)

        decoder = hnn.Linear.init(Embed, Vocab, key=k_decoder, out_first=True)

        # idk what this is: TODO
        # self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        # self.decoder.bias = self.bias

        return RobertaLMHead(dense, layer_norm, decoder, config)

    @named_call
    def __call__(self, features: NamedArray, *, key=None) -> NamedArray:
        x = self.dense(features).rename({self.config.FinalEmbed: self.config.Embed})
        x = hnn.gelu(x, approximate=False)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

class RobertaForMaskedLM(eqx.Module, StateDictSerializationMixin):
    roberta: RobertaModel
    lm_head: RobertaLMHead
    Vocab: Axis
    output_hidden_states: bool

    @classmethod
    def init(self, Vocab: Axis, config: RobertaConfig, output_hidden_states: bool = False, *, key):

        # if config.is_decoder:
        #     raise AttributeError("Model is being run as a MaskedLM aka an encoder model, but is_decoder is true")

        key_rob, key_head = jrandom.split(key, 2)
        roberta = RobertaModel.init(Vocab, config, add_pooling_layer=False, output_hidden_states=output_hidden_states, key=key_rob)
        lm_head = RobertaLMHead.init(Vocab, config, key=key_head)

        return RobertaForMaskedLM(roberta, lm_head, Vocab, output_hidden_states)

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    @named_call
    def __call__(
        self,
        input_ids: Optional[NamedArray] = None,
        attention_mask: Optional[NamedArray] = None,
        token_type_ids: Optional[NamedArray] = None,
        position_ids: Optional[NamedArray] = None,
        input_embeds: Optional[NamedArray] = None,
        *,
        key=None
    ) -> Tuple[NamedArray]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """

        k_rob, k_lm = maybe_rng_split(key, 2)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            input_embeds=input_embeds,
            key=k_rob
        )

        prediction_scores = self.lm_head(outputs[0], key=k_lm)

        return (prediction_scores,) + outputs[2:]
    
    def compute_loss(
            self,
            example: MaskedLmExample,
            *,
            key=None,
            reduction: Optional[hax.ReductionFunction] = hax.mean,
            reduction_axis: Optional[hax.AxisSelection] = None,
    ) -> jnp.ndarray | NamedArray:
        logits = self(example.tokens, example.attn_mask, key=key)
        logits = logits.astype(jnp.float32)
        targets = example.targets

        target_y = hax.nn.one_hot(targets, self.Vocab, dtype=logits.dtype)
        target_y = jax.debug.breakpoint(token=target_y)
        loss = cross_entropy_loss(
            logits, self.Vocab, target_y, reduction, reduction_axis=reduction_axis, where=example.loss_mask
        )

        return loss
    
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


def llama_rotary_pos_emb(
    HeadSize: Axis, Pos: Axis, base: float = 10000, scale: float = 1.0
) -> Tuple[NamedArray, NamedArray]:
    with jax.ensure_compile_time_eval():
        HeadHalfSize = HeadSize.resize(HeadSize.size // 2)
        inv_freq: NamedArray = 1.0 / (base ** (hax.arange(HeadHalfSize, step=2) / HeadSize.size))

        position_ids: NamedArray = hax.arange(Pos) / scale

        freqs = position_ids * inv_freq.broadcast_axis(Pos)
        # This is different from the paper but aligns with HF implementation:
        # It uses a different permutation in order to obtain the same calculation
        emb = hax.concatenate(HeadSize, (freqs, freqs))
        cos = hax.cos(emb)
        sin = hax.sin(emb)
        # This is different from the paper but aligns with HF implementation:
        return cos, sin
