from typing import Optional, Protocol, TypeVar

T = TypeVar("T")

# We take a lot of inspiration from PyGrain
# https://github.com/google/grain/blob/main/docs/samplers.md
#
# Our approach is similar but with a key difference: We require that samplers be random access.

# TODO: figure out "stateless shuffle buffer"
# TODO: figure out a simpler prefix sampler


class Sampler(Protocol):
    """
    A source of indices for indexing into a dataset. In general, we strive for these to be stateless.

    Conceptually, a Sampler is something like a permutation, but it also allows for randomness to be injected
    if preferred.

    """

    def __call__(self, item_index: int, *, key) -> int:
        pass

    def __len__(self) -> int:
        pass

    def has_len(self) -> bool:
        pass


class Dataset(Protocol[T]):
    def __len__(self) -> int:
        pass

    def has_len(self) -> bool:
        pass

    def __getitem__(self, index: int) -> T:
        pass


