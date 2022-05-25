import dataclasses
import inspect
from typing import List, Tuple

from jax.experimental.maps import xmap

class Array:
    def __class_getitem__(cls, item):
        if isinstance(item, str):
            return NamedArray((item, ))
        return NamedArray(item)

@dataclasses.dataclass
class NamedArray:
    names: Tuple[str, ...]


def infer_named_axes(fun):
    """Automatically makes a function a named-function using xmap. Arguments may be given type annotations
    like this: Array['batch', 'inputs', ...] and the function will be xmapped with the named axes.
    You can then re-xmap the function with resource names like this:
    xmap(fun, in_axes=[...], out_axes=[...], axis_resources={'batch': 'x'})



    """
    sig = inspect.signature(fun)

    arg_types = [p.annotation for p in sig.parameters.values()]
    ret_type = sig.return_annotation

    # arguments are named like Array["name1", "name2", ...]
    def extract_axis_names(arg_type):
        if isinstance(arg_type, NamedArray):
            return arg_type.names
        return None

    arg_types = [extract_axis_names(t) for t in arg_types]
    ret_type = extract_axis_names(ret_type)
    if ret_type is None:
        ret_type = [...]

    return xmap(fun, in_axes=arg_types, out_axes=ret_type)


def with_axis_resources(fun, **axis_resources):
    """Assigns resources to named axes.
    Syntactic sugar for xmap(fun, in_axes=[...], out_axes=[...], axis_resources=axis_resources)
    """
    return xmap(fun, in_axes=[...], out_axes=[...], axis_resources=axis_resources)
