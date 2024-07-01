from typing import Optional

import jax
from jax import random as jrandom

from levanter.newdata.core import Sampler
from levanter.newdata.prp import PseudoRandomPermutation


class IdentitySampler(Sampler):
    def __call__(self, item_index: int, *, key) -> int:
        return item_index

    def has_len(self) -> bool:
        return False



class PermutationSampler(Sampler):
    def __init__(self, length: int, *, allow_epochs: bool = True, key: jrandom.PRNKey):
        self.length = length
        self.allow_epochs = allow_epochs
        self.permutations = [PseudoRandomPermutation(length, key)]
        self.key = key

    def __call__(self, item_index: int, *, key) -> int:
        del key

        if not self.allow_epochs and item_index >= self.length:
            raise ValueError(f"item_index {item_index} is out of bounds for length {self.length}")

        if item_index < 0:
            raise ValueError(f"item_index {item_index} is out of bounds for length {self.length}")

        epoch = item_index // self.length
        while epoch >= len(self.permutations):
            perm_key, self.key = jrandom.split(self.key)
            self.permutations.append(PseudoRandomPermutation(self.length, perm_key))

        permutation = self.permutations[epoch]

        return permutation(item_index)

    def __len__(self) -> int:
        return self.length

    def has_known_len(self) -> bool:
        return True


#
# class EraLengthPolicy:
#     untils: list[int]
#     lengths: list[int]
#

# class EraPermutationSampler(Sampler):
#     """
#     A sampler that divides the stream into "eras" of (currently) fixed length. Within each era, the indices are
#     permuted. After the era is exhausted, a new permutation is generated.
#     """
#     # TODO: add an era policy
#
#
#     def __init__(self, era_length: int, *, key: jrandom.PRNKey):
#         self.era_length = era_length
#         self.permutations = [PseudoRandomPermutation(era_length, key)]
#         self.key = key
#
#
#     def __call__(self, item_index: int, *, key) -> int:
