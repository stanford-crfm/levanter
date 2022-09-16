from typing import Optional, Sequence, Tuple, TypeVar, Union

import jax


T = TypeVar("T")


def is_named_array(leaf):
    from .core import NamedArray

    "Typically used as the is_leaf predicate in tree_map"
    return isinstance(leaf, NamedArray)


def ensure_tuple(x: Union[Sequence[T], T]) -> Tuple[T, ...]:
    if isinstance(x, Sequence):
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


def _UNSPECIFIED():
    raise ValueError("unspecified")


def named_call(f=_UNSPECIFIED, name: Optional[str] = None):
    if f is _UNSPECIFIED:
        return lambda f: named_call(f, name)  # type: ignore
    else:
        if name is None:
            name = f.__name__
            if name == "__call__":
                if hasattr(f, "__self__"):
                    name = f.__self__.__class__.__name__  # type: ignore
                else:
                    name = f.__qualname__.rsplit(".", maxsplit=1)[0]  # type: ignore
            else:
                name = f.__qualname__

        return jax.named_scope(name)(f)
