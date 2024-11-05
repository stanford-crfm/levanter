import abc
from dataclasses import dataclass
from typing import Tuple

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp

import haliax as hax
from haliax import Axis, NamedArray


def _rotate_half(x: NamedArray, HeadSize: Axis) -> NamedArray:
    """Rotates half of the hidden dims of the input and concatenates them."""
    x1 = x[HeadSize, : HeadSize.size // 2]
    x2 = x[HeadSize, HeadSize.size // 2 :]
    out = hax.concatenate(HeadSize, (-x2, x1))
    return out


class RotaryEmbeddings(eqx.Module):
    cos: NamedArray
    sin: NamedArray

    @property
    def nograd_cos(self):
        return jax.lax.stop_gradient(self.cos)

    @property
    def nograd_sin(self):
        return jax.lax.stop_gradient(self.sin)

    def __call__(self, HeadDim: Axis, q: NamedArray, k: NamedArray) -> tuple[NamedArray, NamedArray]:
        q_embed = q * self.nograd_cos + _rotate_half(q, HeadDim) * self.nograd_sin
        k_embed = k * self.nograd_cos + _rotate_half(k, HeadDim) * self.nograd_sin
        return q_embed, k_embed


@dataclass
class RotaryEmbeddingsConfig(abc.ABC, draccus.ChoiceRegistry):
    @abc.abstractmethod
    def build(self, HeadSize: Axis, Pos: Axis) -> RotaryEmbeddings:
        pass

    @staticmethod
    def from_hf_config(rope_theta, config: dict | None) -> "RotaryEmbeddingsConfig":
        if config is None:
            return DefaultRotaryEmbeddingsConfig(theta=rope_theta)
        tpe = config.get("rope_type") or config.get("type") or "default"
        return RotaryEmbeddingsConfig.get_choice_class(tpe).make_from_hf_config(rope_theta, config)

    @classmethod
    @abc.abstractmethod
    def make_from_hf_config(cls, rope_theta: float, config: dict) -> "RotaryEmbeddingsConfig":
        pass

    @abc.abstractmethod
    def to_hf_config(self) -> tuple[float, dict | None]:
        """Returns the rope_theta and config dict for the HF config."""
        pass


@dataclass
class DefaultRotaryEmbeddingsConfig(RotaryEmbeddingsConfig):
    theta: float = 10000
    factor: float = 1.0  # this should have been called scale_factor, but for hf compat

    def build(self, HeadSize: Axis, Pos: Axis) -> RotaryEmbeddings:
        with jax.ensure_compile_time_eval():
            HeadHalfSize = HeadSize.resize(HeadSize.size // 2)
            inv_freq: NamedArray = 1.0 / (self.theta ** (hax.arange(HeadHalfSize, step=2) / HeadSize.size))
            inv_freq = inv_freq / self.factor

            position_ids: NamedArray = hax.arange(Pos)

            freqs = position_ids * inv_freq.broadcast_axis(Pos)
            emb = hax.concatenate(HeadSize, (freqs, freqs))
            cos = hax.cos(emb)
            sin = hax.sin(emb)
            return RotaryEmbeddings(cos=cos, sin=sin)

    @classmethod
    def make_from_hf_config(cls, rope_theta: float, config: dict) -> "RotaryEmbeddingsConfig":
        return DefaultRotaryEmbeddingsConfig(theta=rope_theta, factor=config.get("factor", 1.0))

    def to_hf_config(self) -> tuple[float, dict | None]:
        if self.factor == 1.0:
            return self.theta, None
        return self.theta, {"factor": self.factor}


RotaryEmbeddingsConfig.register_subclass("default", DefaultRotaryEmbeddingsConfig)
RotaryEmbeddingsConfig.register_subclass("linear", DefaultRotaryEmbeddingsConfig)


@dataclass
class Llama3RotaryEmbeddingsConfig(RotaryEmbeddingsConfig):
    """
    To match this from HF:
                "rope_scaling": {
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3"
              },
    """

    theta: float = 500000
    factor: float = 8.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_position_embeddings: int = 8192

    def build(self, HeadSize: Axis, Pos: Axis) -> RotaryEmbeddings:
        # https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py#L307
        # Porting that to JAX/Haliax:
        with jax.ensure_compile_time_eval():
            HeadHalfSize = HeadSize.resize(HeadSize.size // 2)
            inv_freq: NamedArray = 1.0 / (self.theta ** (hax.arange(HeadHalfSize, step=2) / HeadSize.size))

            old_context_len = self.original_max_position_embeddings
            low_freq_wavelen = old_context_len / self.low_freq_factor
            high_freq_wavelen = old_context_len / self.high_freq_factor

            wavelen = 2 * jnp.pi / inv_freq
            inv_freq_llama = hax.where(wavelen > low_freq_wavelen, inv_freq / self.factor, inv_freq)
            smooth_factor = (old_context_len / wavelen - self.low_freq_factor) / (
                self.high_freq_factor - self.low_freq_factor
            )
            smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / self.factor + smooth_factor * inv_freq_llama
            is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
            inv_freq_llama = hax.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

            position_ids: NamedArray = hax.arange(Pos)

            freqs = position_ids * inv_freq_llama.broadcast_axis(Pos)
            emb = hax.concatenate(HeadSize, (freqs, freqs))
            cos = hax.cos(emb)
            sin = hax.sin(emb)
            return RotaryEmbeddings(cos=cos, sin=sin)

    @classmethod
    def make_from_hf_config(cls, rope_theta: float, config: dict) -> "RotaryEmbeddingsConfig":
        return Llama3RotaryEmbeddingsConfig(
            theta=rope_theta,
            factor=config.get("factor", 8.0),
            low_freq_factor=config.get("low_freq_factor", 1.0),
            high_freq_factor=config.get("high_freq_factor", 4.0),
            original_max_position_embeddings=config.get("original_max_position_embeddings", 8192),
        )

    def to_hf_config(self) -> tuple[float, dict]:
        return self.theta, {
            "factor": self.factor,
            "low_freq_factor": self.low_freq_factor,
            "high_freq_factor": self.high_freq_factor,
            "original_max_position_embeddings": self.original_max_position_embeddings,
            "rope_type": "llama3",
        }


RotaryEmbeddingsConfig.register_subclass("llama3", Llama3RotaryEmbeddingsConfig)


def rotary_pos_emb(
    HeadSize: Axis, Pos: Axis, theta: float = 10000, scale: float = 1.0
) -> Tuple[NamedArray, NamedArray]:
    with jax.ensure_compile_time_eval():
        HeadHalfSize = HeadSize.resize(HeadSize.size // 2)
        inv_freq: NamedArray = 1.0 / (theta ** (hax.arange(HeadHalfSize, step=2) / HeadSize.size)) / scale

        position_ids: NamedArray = hax.arange(Pos)

        freqs = position_ids * inv_freq.broadcast_axis(Pos)
        # This is different from the paper but aligns with HF implementation:
        # It uses a different permutation in order to obtain the same calculation
        emb = hax.concatenate(HeadSize, (freqs, freqs))
        cos = hax.cos(emb)
        sin = hax.sin(emb)
        # This is different from the paper but aligns with HF implementation:
        return cos, sin
