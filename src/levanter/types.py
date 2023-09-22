from typing import Any, Callable, Protocol, Tuple, TypeVar, Union


M = TypeVar("M")  # Model
X = TypeVar("X", contravariant=True)  # Input


class ValAndGradFn(Protocol[M, X]):
    def __call__(self, model: M, *inputs: X, **input_kwargs) -> Tuple[float, M]:
        ...


class ValFn(Protocol[M, X]):
    def __call__(self, model: M, *inputs: X, **input_kwargs) -> Tuple[float, M]:
        ...


FilterSpec = Union[bool, Callable[[Any], bool]]
"""
A filter specification. Typically used on a pytree to filter out certain subtrees. Boolean values are
treated as-is, while callables are called on each element of the pytree. If the callable returns True, the element
is kept, otherwise it is filtered out.
"""
