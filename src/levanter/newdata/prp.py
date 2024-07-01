# Pseudo-Random Permutation Code
# This is not intended to be cryptographically secure. Its job is to generate random numbers in a stateless way
from functools import partial

import jax.lax
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom

# TODO: do we make this a pytree
class PseudoRandomPermutation:
    def __init__(self, length, prng_key):
        self.length = length
        self.prng_key = prng_key
        a_key, b_key = jrandom.split(prng_key)
        self._a = jrandom.randint(a_key, (1,), 1, length)
        self._b = jrandom.randint(b_key, (1,), 0, length)

        @partial(jax.lax.while_loop, lambda a, key: jnp.gcd(a, length) != 1)
        def loop(a, key):
            this_key, key = jrandom.split(key)
            a = jrandom.randint(this_key, (1,), 1, length)
            return a, key

        self._a, _ = loop(self._a, a_key)

    def __call__(self, indices):
        return (self._a * indices + self._b) % self.length
