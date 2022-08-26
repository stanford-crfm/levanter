from dataclasses import dataclass
from typing import Callable, Dict, Optional

import equinox as eqx
import equinox.nn as nn
import jax
import jax.lax as lax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import jmp
from equinox.custom_types import Array

import haliax as hax
import levanter.nn as pnn
from haliax import Axis, NamedArray
from levanter import jax_utils
from levanter.compat.torch_serialization import (
    StateDict,
    TorchSerializationMixin,
    apply_prefix,
    eqx_module_from_torch_dict,
    jax_to_torch,
    torch_to_jax,
    update_torch_dict_with_jax_tree,
)
from levanter.jax_utils import named_call
from levanter.modeling_utils import ACT2FN
from levanter.nn.linear import NamedLinear


@dataclass(frozen=True)
class Gpt2Config:
    seq_len: int = 512
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12

    initializer_range: float = 0.02
    embed_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    activation_function: str = "gelu_new"

    # mistral tweak:
    scale_attn_by_inverse_layer_idx: bool = False

    gradient_checkpointing: bool = False
    gradient_checkpointing_block_size: int = 5

    # Axes
    @property
    def seqlen(self) -> Axis:
        return Axis(name="seqlen", size=self.seq_len)

    @property
    def embed(self) -> Axis:
        return Axis(name="embed", size=self.hidden_dim)


class Gpt2Mlp(eqx.Module):
    act: Callable = eqx.static_field()
    c_fc: NamedLinear
    c_proj: NamedLinear

    def __init__(self, hidden: Axis, intermediate: Axis, activation_fn, *, key, mp: jmp.Policy):
        k_fc, k_proj = jrandom.split(key, 2)
        self.c_fc = NamedLinear(out_axis=intermediate, in_axis=hidden, key=k_fc, mp=mp)
        self.c_proj = NamedLinear(out_axis=hidden, in_axis=intermediate, key=k_proj, mp=mp)
        self.act = ACT2FN[activation_fn]  # type: ignore

    @named_call
    def __call__(self, hidden_states: NamedArray):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = jax.tree_util.tree_map(self.act, hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states


class Gpt2Attention(TorchSerializationMixin, eqx.Module):
    c_attn: NamedLinear
    c_proj: NamedLinear
    dropout: pnn.Dropout

    causal: bool = eqx.static_field()
    seqlen: Axis = eqx.static_field()
    head_dim: Axis = eqx.static_field()
    heads: Axis = eqx.static_field()
    qkv: Axis = eqx.static_field()
    total_head_dim: Axis = eqx.static_field()

    scale_by_inverse_layer_idx: bool = eqx.static_field()
    mp: jmp.Policy = eqx.static_field()

    def __init__(
        self,
        seqlen: Axis,
        in_dim: Axis,
        heads: Axis,
        head_dim: Axis,
        dropout_prob: float,
        scale_by_inverse_layer_idx: bool,
        *,
        key,
        mp: jmp.Policy,
        causal: bool = True,
    ):
        self.causal = causal
        self.heads = heads
        self.head_dim = head_dim
        # TODO: we only need this for hf checkpoint compat
        self.seqlen = seqlen

        self.total_head_dim = Axis("total_head_dim", self.head_dim.size * self.heads.size)
        self.qkv = Axis("qkv", 3 * self.total_head_dim.size)
        self.scale_by_inverse_layer_idx = scale_by_inverse_layer_idx
        self.mp = mp

        k_c, k_proj = jrandom.split(key, 2)

        # we could have this if we didn't need hf checkpoint compat
        # self.c_attn = NamedLinear(out_axis=(self.qkv, self.heads, self.head_dim), in_axis=in_dim, key=k_c, mp=mp)
        # self.c_proj = NamedLinear(out_axis=in_dim, in_axis=(self.heads, self.head_dim), key=k_proj, mp=mp)
        self.c_attn = NamedLinear(out_axis=self.qkv, in_axis=in_dim, key=k_c, mp=mp)
        self.c_proj = NamedLinear(out_axis=in_dim, in_axis=self.total_head_dim, key=k_proj, mp=mp)
        self.dropout = pnn.Dropout(dropout_prob)

    # TODO: cross-attention
    # TODO: reorder_and_upcast_attn
    # @eqx.filter_jit
    @named_call
    def __call__(self, hidden_states: NamedArray, layer_idx, inference: bool = True, *, key):
        # hidden_states has shape [seq_len, embed_dim]
        rng_key = key

        qkv_out = self.c_attn(hidden_states)  # [seq_len, 3 * embed_dim]
        three = Axis("3", 3)
        qkv_out = qkv_out.unflatten_axis(self.qkv, (three, self.heads, self.head_dim))
        query, key, value = qkv_out.unbind(three)

        key_seqlen = self.seqlen.alias("key_seqlen")
        key = key.rename({self.seqlen: key_seqlen})
        value = value.rename({self.seqlen: key_seqlen})

        # mistral tweak
        scale = lax.rsqrt(float(self.head_dim.size))
        if self.scale_by_inverse_layer_idx:
            scale /= layer_idx + 1.0

        #  I strongly suspect jax can fuse the next two ops so we don't need to do that mistral tweak
        # TODO: verify
        attn_weights = hax.dot(self.head_dim, query, key)
        # TODO(haliax): add elemwise ops to hax
        attn_weights = hax.rearrange(attn_weights, (..., self.heads, self.seqlen, key_seqlen))
        attn_axes = attn_weights.axes
        attn_weights = attn_weights.array
        attn_weights = attn_weights * scale

        if self.causal:
            causal_mask = jnp.tril(jnp.ones((self.seqlen.size, key_seqlen.size), dtype=jnp.bool_))

            # TODO(haliax): add where ops to hax
            attn_weights = jnp.where(causal_mask, attn_weights, -1e9)

        attn_weights = jnn.softmax(attn_weights)  # heads, seqlen, seqlen
        attn_weights = self.mp.cast_to_compute(attn_weights)
        attn_weights = self.dropout(attn_weights, key=rng_key, inference=inference)

        attn_weights = NamedArray(attn_weights, attn_axes)

        attn_output = hax.dot(key_seqlen, attn_weights, value)  # [heads, seq_len, head_dim]

        attn_output = hax.flatten_axes(attn_output, (self.heads, self.head_dim), self.total_head_dim)
        attn_output = self.c_proj(attn_output)

        assert attn_output.dtype == self.mp.compute_dtype

        return attn_output


class Gpt2Block(TorchSerializationMixin, eqx.Module):
    mp: jmp.Policy = eqx.static_field()
    ln_1: nn.LayerNorm
    attn: Gpt2Attention
    ln_2: nn.LayerNorm
    mlp: Gpt2Mlp
    resid_dropout: pnn.Dropout

    seqlen: Axis = eqx.static_field()
    embed: Axis = eqx.static_field()

    def __init__(self, config: Gpt2Config, *, key, mp: jmp.Policy):
        k_attn, k_cross, k_mlp = jrandom.split(key, 3)

        embed = config.embed
        self.embed = embed
        self.seqlen = config.seqlen
        inner_dim = Axis("mlp", 4 * embed.size)
        heads = Axis("heads", config.num_heads)
        head_dim = Axis("head", embed.size // config.num_heads)

        assert (
            embed.size % config.num_heads == 0
        ), f"embed_dim={embed} must be divisible by num_heads={config.num_heads}"

        self.mp = mp

        self.ln_1 = nn.LayerNorm(embed.size, eps=config.layer_norm_epsilon)
        self.attn = Gpt2Attention(
            seqlen=config.seqlen,
            in_dim=embed,
            heads=heads,
            head_dim=head_dim,
            dropout_prob=config.attn_pdrop,
            key=k_attn,
            causal=True,
            scale_by_inverse_layer_idx=config.scale_attn_by_inverse_layer_idx,
            mp=mp,
        )
        self.resid_dropout = pnn.Dropout(p=config.resid_pdrop)
        self.ln_2 = nn.LayerNorm(embed.size, eps=config.layer_norm_epsilon)

        self.mlp = Gpt2Mlp(
            hidden=embed,
            intermediate=inner_dim,
            activation_fn=config.activation_function,
            key=k_mlp,
            mp=mp,
        )

    # @eqx.filter_jit
    @named_call
    def __call__(self, hidden_states: Array, inference, layer_idx, *, key):
        k1, k2, k3 = jax_utils.maybe_rng_split(key, 3)

        residual = hidden_states
        hidden_states = jax.vmap(self.ln_1)(hidden_states)
        hidden_states = self.mp.cast_to_compute(hidden_states)
        h = NamedArray(hidden_states, (self.seqlen, self.embed))
        attn_output = self.attn(h, inference=inference, layer_idx=layer_idx, key=k1)
        dout = self.resid_dropout(attn_output.array, key=k2, inference=inference)
        hidden_states = residual + dout

        residual = hidden_states
        hidden_states = jax.vmap(self.ln_2)(hidden_states)
        hidden_states = self.mp.cast_to_compute(hidden_states)
        h = NamedArray(hidden_states, (self.seqlen, self.embed))
        ff_output = self.mlp(h)
        dout = self.resid_dropout(ff_output.array, key=k3, inference=inference)
        hidden_states = residual + dout

        assert attn_output.dtype == self.mp.compute_dtype

        return hidden_states


class Gpt2Transformer(TorchSerializationMixin, eqx.Module):
    config: Gpt2Config = eqx.static_field()
    mp: jmp.Policy = eqx.static_field()
    blocks: Gpt2Block
    ln_f: nn.LayerNorm

    layers: Axis = eqx.static_field()

    def __init__(self, config: Gpt2Config, *, key, mp: jmp.Policy):
        super().__init__()
        self.config = config
        self.mp = mp

        self.layers = Axis("block", config.num_layers)

        self.blocks = hax.vmap(lambda key: Gpt2Block(config, key=key, mp=mp), self.layers)(
            jax_utils.shaped_rng_split(key, config.num_layers),
        )
        self.ln_f = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_epsilon)

    # @eqx.filter_jit
    @named_call
    def __call__(self, hidden_states: Array, inference=True, *, key) -> Array:
        keys = jax_utils.maybe_rng_split(key, self.layers.size)

        if inference:

            def do_block(states, block_and_layer):
                block, layer_idx = block_and_layer
                return block(states, inference=inference, layer_idx=layer_idx, key=None)

            # unlikely we'll call grad with inference=True, but just in case
            if self.config.gradient_checkpointing:
                do_block = jax.checkpoint(do_block)

            hidden_states = hax.fold_left(
                do_block, self.layers, hidden_states, (self.blocks, jnp.arange(self.layers.size))
            )
        else:

            def do_block_train(states, block_layer_idx_key):
                block: Gpt2Block
                block, layer_idx, key = block_layer_idx_key
                return block(states, inference=inference, layer_idx=layer_idx, key=key)

            if self.config.gradient_checkpointing:
                do_block_train = jax.checkpoint(do_block_train)

            hidden_states = hax.fold_left(
                do_block_train, self.layers, hidden_states, (self.blocks, jnp.arange(self.layers.size), keys)
            )

        hidden_states = jax.vmap(self.ln_f)(hidden_states)
        hidden_states = self.mp.cast_to_compute(hidden_states)

        return hidden_states

    def _torch_key_map(self) -> Optional[Dict[str, Optional[str]]]:
        return {"blocks": "h"}

    def from_torch_dict(self, torch_dict: StateDict, prefix: Optional[str] = None):
        # this method is a bit of a pain because we use a vectorized set of blocks, meaning that we have 1 GptBlock,
        # whereas in torch we have numlayers GptBlocks. So we need to build one GptBlock from numlayers GptBlocks.
        def vectorized_load(tree, torch_dict, pattern: Optional[str]):
            if isinstance(tree, eqx.Module):
                # TODO: should vectorized_load know how to deal with custom from_torch_dict?
                # could solve by doing gross surgery on the state dict
                if hasattr(tree, "key_map"):
                    key_map = tree.key_map
                else:
                    key_map = {}

                old_values, mod_state = tree.tree_flatten()
                dyn_keys = mod_state[0]

                new_values = []
                for k, old in zip(dyn_keys, old_values):
                    k = key_map.get(k, k)
                    new_values.append(vectorized_load(old, torch_dict, apply_prefix(pattern, k)))

                return tree.tree_unflatten(mod_state, new_values)
            elif isinstance(tree, NamedArray):
                inner_array = vectorized_load(tree.array, torch_dict, pattern)
                return NamedArray(inner_array, tree.axes)
            else:
                assert pattern is not None
                my_tensors = [torch_to_jax(torch_dict[pattern.format(i)]) for i in range(self.config.num_layers)]
                return jnp.stack(my_tensors, axis=0)

        blocks = vectorized_load(self.blocks, torch_dict, apply_prefix(prefix, "h.{}"))
        ln_f = eqx_module_from_torch_dict(self.ln_f, torch_dict, apply_prefix(prefix, "ln_f"))
        mod = eqx.tree_at(lambda x: (x.blocks, x.ln_f), self, replace=(blocks, ln_f))

        return mod

    def update_torch_dict(self, torch_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        import torch

        # this method is also a bit of a pain for the same reasons
        def devectorized_save(tree, torch_dict, pattern: Optional[str]):
            if isinstance(tree, eqx.Module):
                # TODO: should devectorized_save know how to deal with custom to_torch_dict?
                # could solve by doing gross surgery on the returned dicts?
                if hasattr(tree, "key_map"):
                    key_map = tree.key_map
                else:
                    key_map = {}

                values, mod_state = tree.tree_flatten()
                dyn_keys = mod_state[0]

                for k, value in zip(dyn_keys, values):
                    k = key_map.get(k, k)
                    devectorized_save(value, torch_dict, apply_prefix(pattern, k))
            elif isinstance(tree, NamedArray):
                devectorized_save(tree.array, torch_dict, pattern)
            else:
                assert pattern is not None
                torch_tensors = torch.unbind(jax_to_torch(tree))
                assert len(torch_tensors) == self.config.num_layers
                for i in range(self.config.num_layers):
                    torch_dict[pattern.format(i)] = torch_tensors[i]

        devectorized_save(self.blocks, torch_dict, apply_prefix(prefix, "h.{}"))
        update_torch_dict_with_jax_tree(self.ln_f, torch_dict, apply_prefix(prefix, "ln_f"))

        return torch_dict


class Gpt2Embeddings(TorchSerializationMixin, eqx.Module):
    token_embeddings: NamedArray
    position_embeddings: NamedArray
    token_out_embeddings: Optional[NamedArray]
    dropout: pnn.Dropout

    # axes
    vocab: Axis = eqx.static_field()
    seqlen: Axis = eqx.static_field()
    hidden: Axis = eqx.static_field()

    mp: jmp.Policy = eqx.static_field()

    def __init__(
        self,
        embed: Axis,
        vocab: Axis,
        seqlen: Axis,
        initializer_range: float,
        tie_word_embeddings: bool,
        dropout_prob: float,
        *,
        key,
        mp: jmp.Policy,
    ):
        super().__init__()
        k_wte, k_wpe, k_out = jrandom.split(key, 3)

        self.vocab = vocab
        self.seqlen = seqlen
        self.hidden = embed

        self.mp = mp

        self.token_embeddings = hax.random.normal(key=k_wte, shape=(vocab, embed)) * initializer_range
        self.position_embeddings = hax.random.normal(key=k_wpe, shape=(seqlen, embed)) * (initializer_range / 2)
        self.dropout = pnn.Dropout(p=dropout_prob)

        if tie_word_embeddings:
            self.token_out_embeddings = None
        else:
            self.token_out_embeddings = hax.random.normal(key=k_out, shape=(vocab, embed)) * initializer_range

    @named_call
    def embed(self, input_ids, inference, *, key):
        # TODO: select
        # input_embeds = self.token_embeddings.select(self.vocab, input_ids)
        # position_embeds = self.position_embeddings.select(self.seqlen, jnp.arange(input_ids.shape[-1], dtype="i4"))
        input_embeds = self.token_embeddings.array[input_ids]
        input_embeds = self.mp.cast_to_compute(input_embeds)

        position_embeds = self.position_embeddings.array[jnp.arange(input_ids.shape[-1], dtype="i4")]
        position_embeds = self.mp.cast_to_compute(position_embeds)

        hidden_states = input_embeds + position_embeds
        hidden_states = self.dropout(hidden_states, inference=inference, key=key)

        return hidden_states

    def unembed(self, hidden_states: Array):
        embeddings = self.token_out_embeddings or self.token_embeddings
        # return hax.dot(self.hidden, hidden_states, embeddings)
        return jnp.einsum("... l h, ... v h -> ... l v", hidden_states, embeddings.array)

    def _torch_key_map(self) -> Optional[Dict[str, Optional[str]]]:
        assert self.token_out_embeddings is None
        return {"token_embeddings": "wte.weight", "position_embeddings": "wpe.weight"}


class Gpt2LMHeadModel(TorchSerializationMixin, eqx.Module):
    transformer: Gpt2Transformer
    embeddings: Gpt2Embeddings

    @property
    def config(self):
        return self.transformer.config

    @property
    def vocab_size(self) -> int:
        return self.embeddings.vocab.size

    def __init__(self, vocab: Axis, config: Gpt2Config, *, key, mp: jmp.Policy):
        k_t, k_embeddings = jrandom.split(key, 2)
        self.transformer = Gpt2Transformer(config, key=k_t, mp=mp)
        self.embeddings = Gpt2Embeddings(
            vocab=vocab,
            embed=config.embed,
            seqlen=config.seqlen,
            initializer_range=config.initializer_range,
            tie_word_embeddings=True,
            dropout_prob=config.embed_pdrop,
            key=k_embeddings,
            mp=mp,
        )

    def __call__(self, input_ids, *, inference, key):
        if not inference and key is None:
            raise ValueError("key must be provided for training")

        k_embed, k_transformer = jax_utils.maybe_rng_split(key, 2)
        hidden_states = self.embeddings.embed(input_ids, inference=inference, key=k_embed)
        hidden_states = self.transformer(hidden_states, inference=inference, key=k_transformer)
        lm_logits = self.embeddings.unembed(hidden_states)

        return lm_logits

    def _torch_key_map(self) -> Optional[Dict[str, Optional[str]]]:
        return {"transformer": None, "embeddings": None}
