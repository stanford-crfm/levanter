from typing import Callable, Sequence, Tuple, Type, TypeVar, Union

from haliax.jax_utils import is_jax_array_like


T = TypeVar("T")


py_slice = slice

slice_t = Type[slice]


def is_named_array(leaf):
    from .core import NamedArray

    "Typically used as the is_leaf predicate in tree_map"
    return isinstance(leaf, NamedArray)


def ensure_tuple(x: Union[Sequence[T], T]) -> Tuple[T, ...]:
    if isinstance(x, str):
        return (x,)  # type: ignore
    elif isinstance(x, Sequence):
        return tuple(x)
    return (x,)


class StringHolderEnum(type):
    """Like a python enum but just holds string constants, as opposed to wrapped string constants"""

    # https://stackoverflow.com/questions/62881486/a-group-of-constants-in-python

    def __new__(cls, name, bases, members):
        # this just iterates through the class dict and removes
        # all the dunder methods
        cls.members = [v for k, v in members.items() if not k.startswith("__") and not callable(v)]
        return super().__new__(cls, name, bases, members)

    # giving your class an __iter__ method gives you membership checking
    # and the ability to easily convert to another iterable
    def __iter__(cls):
        yield from cls.members


def is_jax_or_hax_array_like(x):
    return is_jax_array_like(x) or is_named_array(x)


def index_where(pred: Callable[[T], bool], xs: Sequence[T]) -> int:
    for i, x in enumerate(xs):
        if pred(x):
            return i
    raise ValueError("No element satisfies predicate")


__all__ = [
    "is_named_array",
    "ensure_tuple",
    "StringHolderEnum",
    "is_jax_or_hax_array_like",
    "index_where",
    "slice_t",
    "py_slice",
]
