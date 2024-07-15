import functools

from jax import random as jrandom

from levanter.newdata.core import Sampler
from levanter.newdata.prp import Permutation


# TODO: figure out "stateless shuffle buffer"
# TODO: figure out a simpler prefix sampler


class IdentitySampler(Sampler):
    def __call__(self, idx: int) -> int:

        return idx

    def has_known_len(self) -> bool:
        return False

    def __len__(self) -> int:
        raise NotImplementedError("IdentitySampler does not support __len__")


class PermutationSampler(Sampler):
    def __init__(self, length: int, *, allow_epochs: bool = True, key: jrandom.PRNGKey):
        self.length = length
        self.allow_epochs = allow_epochs

        @functools.lru_cache(maxsize=4)  # we're mostly going to be going sequentially, so even 4 is overkill
        def gen_epoch_permutation(epoch: int) -> Permutation:
            mix_key = jrandom.fold_in(key, epoch)
            return Permutation(length, mix_key)

        self.gen_epoch_permutation = gen_epoch_permutation

    def __call__(self, idx: int) -> int:
        if not self.allow_epochs and idx >= self.length:
            raise ValueError(f"item_index {idx} is out of bounds for length {self.length}")

        if idx < 0:
            raise ValueError(f"item_index {idx} is out of bounds for length {self.length}")

        epoch = idx // self.length

        permutation = self.gen_epoch_permutation(epoch)

        return permutation(idx)

    def __len__(self) -> int:
        return self.length

    def has_known_len(self) -> bool:
        return True


# era sampling is a decent solution to the "stateless shuffle buffer" problem
# TODO: but i think we might like to let this grow the era length over time?
class EraPermutationSampler(Sampler):
    """
    A sampler that divides the stream into "eras" of (currently) fixed length. Within each era, the indices are
    permuted. After the era is exhausted, a new permutation is generated.

    Note that this sampler has no notion of "epoch" so you have to munge the returned index
    """

    def __init__(self, era_length: int, *, key: jrandom.PRNGKey):
        self.era_length = era_length
        self.key = key

        @functools.lru_cache(maxsize=4)  # we're mostly going to be going sequentially
        def gen_era_permutation(era: int) -> Permutation:
            mix_key = jrandom.fold_in(key, era)
            return Permutation(era_length, mix_key)

        self.gen_era_permutation = gen_era_permutation

    def __call__(self, idx: int) -> int:
        if idx < 0:
            raise ValueError(f"item_index {idx} is out of bounds for era length {self.era_length}")

        era = idx // self.era_length
        permutation = self.gen_era_permutation(era)

        return permutation(idx) + era * self.era_length

    def __len__(self) -> int:
        raise NotImplementedError("EraPermutationSampler does not support __len__")

    def has_known_len(self) -> bool:
        return False
