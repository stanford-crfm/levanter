import typing

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np


class Permutation:
    """A pseudo-random permutation for an arbitrary domain using a Feistel network and cycle-walking.

    The domain [0, length) is embedded into a power-of-two domain [0, m) where m = next_power_of_two(length).
    A Feistel network is applied on the bit-level representation, and if the result falls outside [0, length),
    the network is reapplied (cycle-walking) until the result is in-range.
    """

    def __init__(self, length: int, prng_key, rounds: int = 5):
        self.length = length
        self.m = next_power_of_two(length)  # m is a power-of-two >= length
        self.rounds = rounds
        self.bits = self.m.bit_length() - 1  # because m == 1 << bits
        # Split the bits into two halves.
        self.left_bits = self.bits // 2
        self.right_bits = self.bits - self.left_bits
        self.R_mask = (1 << self.right_bits) - 1

        # Convert JAX PRNG to numpy RNG and generate round keys in range [0, 2^(right_bits))
        self.rng = np.random.Generator(np.random.PCG64(jrandom.randint(prng_key, (), 0, 2**30).item()))
        self.keys = self.rng.integers(0, 1 << self.right_bits, size=rounds, dtype=np.uint64)

    def _F(self, right: np.ndarray, key: np.uint64) -> np.ndarray:
        """A simple round function that mixes the right half.

        Operates modulo 2^(right_bits).
        """
        return (right * np.uint64(2654435761) + key) & self.R_mask

    def _feistel(self, x: np.ndarray) -> np.ndarray:
        """Apply the Feistel network to x, where x is assumed to be in [0, m)."""
        # Split x into left and right parts.
        right = x & self.R_mask
        left = x >> self.right_bits
        for key in self.keys:
            new_left = right
            new_right = left ^ self._F(right, key)
            left, right = new_left, new_right
        return (left << self.right_bits) | right

    @typing.overload
    def __call__(self, indices: int) -> int:
        ...

    @typing.overload
    def __call__(self, indices: np.ndarray | jnp.ndarray) -> np.ndarray:
        ...

    def __call__(self, indices):
        was_int = False
        if isinstance(indices, jnp.ndarray):
            indices = np.array(indices)
        if not isinstance(indices, np.ndarray):
            if indices < 0 or indices >= self.length:
                raise IndexError(f"Index {indices} is out of bounds for length {self.length}")
            indices = np.atleast_1d(np.array(indices, dtype=np.uint64))
            was_int = True
        else:
            indices = indices.astype(np.uint64)

        if np.any(indices < 0) or np.any(indices >= self.length):
            raise IndexError(f"Index {indices} is out of bounds for length {self.length}")

        x = indices
        out = self._feistel(x)
        mask = out >= self.length
        while np.any(mask):
            out[mask] = self._feistel(out[mask])
            mask = out >= self.length

        if was_int:
            return int(out[0])
        return out


def next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()
