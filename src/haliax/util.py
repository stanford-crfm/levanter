from typing import Sequence, Tuple, TypeVar, Union


T = TypeVar("T")


def is_named_array(leaf):
    from .core import NamedArray

    "Typically used as the is_leaf predicate in tree_map"
    return isinstance(leaf, NamedArray)


def ensure_tuple(x: Union[Sequence[T], T]) -> Tuple[T, ...]:
    if isinstance(x, Sequence):
        return tuple(x)
    return (x,)
