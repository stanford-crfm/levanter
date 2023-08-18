from typing import Protocol, Tuple, TypeVar


M = TypeVar("M")  # Model
X = TypeVar("X", contravariant=True)  # Input


class ValAndGradFn(Protocol[M, X]):
    def __call__(self, model: M, *inputs: X, **input_kwargs) -> Tuple[float, M]:
        ...


class ValFn(Protocol[M, X]):
    def __call__(self, model: M, *inputs: X, **input_kwargs) -> Tuple[float, M]:
        ...
