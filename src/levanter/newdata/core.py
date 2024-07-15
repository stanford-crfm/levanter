from typing import Protocol, TypeVar


T = TypeVar("T", covariant=True)

# We take a lot of inspiration from PyGrain
# https://github.com/google/grain/blob/main/docs/samplers.md
#
# Those docs seem to be quite stale: samplers are actually random access, not just sequential.
# We also require random access samplers. PyGrain seems to allow for stateful samplers, but we don't.


class Sampler(Protocol):
    """
    A source of indices for indexing into a dataset. For now we require these to be stateless.
    We may add stateful support in the future.

    Conceptually, a Sampler is something like a permutation.

    Some samplers might be infinite (e.g. the identity sampler or the era sampler)
    """

    # TODO: maybe support slices/sequences here?
    def __call__(self, idx: int) -> int:
        pass

    def __len__(self) -> int:
        """Returns the length of the sampler. May raise if the length is not known."""
        pass

    def has_known_len(self) -> bool:
        pass
