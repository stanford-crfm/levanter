import dataclasses
import functools
import inspect
import typing
from inspect import BoundArguments
from typing import List, Tuple, Dict, Sequence, Optional, Callable, get_origin, get_args, Annotated, Union

import equinox as eqx
import jax
from equinox.custom_types import sentinel, PyTree
from jax._src.tree_util import all_leaves, _registry
from jax.experimental.maps import xmap, AxisName, ResourceSet

from psithuros.python_utils import StringHolderEnum


# Predefined axis names
# in compliance with https://github.com/google-research/t5x/blob/main/t5x/partitioning.py
class ResourceAxis(StringHolderEnum):
    MODEL = "model"
    DATA = "data"


class LogicalAxis(StringHolderEnum):
    BATCH = "batch"
    PARAMS = "params"


class Array(jax.numpy.ndarray):
    """Type annotation for arrays with axis names. Just used as a type hint. The axis names are used with
    infer_named_axes to infer the axes of the array to pass to xmap.

    Example:
        >>> class MyModule(eqx.Module):
        >>>     params: Array["x", "y"]
    """
    def __class_getitem__(cls, item):
        if not isinstance(item, tuple):
            item = (item,)
        return AxisNames(item)


@dataclasses.dataclass
class AxisNames:
    names: Tuple[AxisName, ...]

    def __call__(self, *args, **kwds):
        raise TypeError(f"Shouldn't call this. Just necessary to trick the type checker")

    def concat(self, other: Optional['AxisNames']) -> 'AxisNames':
        if other is None:
            return self
        return AxisNames(tuple(self.names) + tuple(other.names))

    def __hash__(self):
        return hash(self.names)

    def __eq__(self, other):
        if isinstance(other, AxisNames):
            return self.names == other.names
        return False


UnnamedAxes = AxisNames(names=(...,))
"""Used for types with no axes specified"""

T = typing.TypeVar('T')


class Shaped(typing.Generic[T]):
    """Supports things like Shaped["shard", Linear] or Shaped[ ("shard", "x", ...), Linear]. The returned type is
    an Annotated type, with the type of the wrapped type as the first element and the AxisNames as the second."""
    # TODO: currently PyCharm gets confused by these types and thnks it's not the underlying type T. Fix this.

    def __class_getitem__(cls, item: Tuple[typing.Union[AxisName, Tuple[AxisName, ...]], typing.Type[T]])-> typing.Type[T]:
        if len(item) != 2:
            raise ValueError("Shaped[...] only supports two-tuples. If you want to use a tuple of axes, "
                             "use Shaped[(...), ...]")
        shapes, tpe = item
        if not isinstance(shapes, tuple):
            shapes = (shapes,)

        return typing.Annotated[tpe, AxisNames(shapes)]





def infer_leaf_axes(tpe: type)-> List[AxisNames]:
    """Infers the leaves from just a type. This doesn't work well yet and you should use infer_named_axes where possible"""
    origin = get_origin(tpe)
    if origin is Annotated:
        args = get_args(tpe)
        shapeses = [s for s in args[1:] if isinstance(s, AxisNames)]
        if len(shapeses) != 1:
            raise ValueError(f"We only support one Shaped[...] in a leaf type, but got {shapeses}")
        prefix_names = shapeses[0].names

        recursive_leaf_names = infer_leaf_axes(args[0])
        return [prefix_names + n for n in recursive_leaf_names]
    elif type(tpe) == eqx.module._ModuleMeta and issubclass(tpe, eqx.Module):
        # unfortunately need to replicate the logic in Module#tree_flatten
        shapes = []
        for field_ in dataclasses.fields(tpe):  # type: ignore
            if not field_.metadata.get("static", False):
                shapes += infer_leaf_axes(field_.type)
        return shapes
    elif isinstance(tpe, AxisNames):
        return [tpe]
    elif tpe is Array:
        return [UnnamedAxes]
    else:
        return [UnnamedAxes]


def _is_named_tuple(x):
    # https://stackoverflow.com/questions/2166818/how-to-check-if-an-object-is-an-instance-of-a-namedtuple
    # Python is so dumb
    return isinstance(x, tuple) and hasattr(x, '_fields')


def infer_named_axes(value: PyTree, tpe: Optional[type])->Optional[Union[AxisNames, PyTree]]:
    """Automatically get a "pytree" of named axes for a pytree
       The leaves of this PyTree are AxisNames, which is just a wrapper around a list of names.
       To pass this to xmap, you need to unwrap the names using tree_map:
       >>> axis_names = unwrap_axis_names(axis_names)
   """
    # TODO: decide if we want an internal method that returns the AxisNames type and an external method that returns the
    # unwrapped names, which can be directly passed to xmap.
    origin = get_origin(tpe)
    if origin is Annotated:
        args = get_args(tpe)
        shapeses = [s for s in args[1:] if isinstance(s, AxisNames)]
        if len(shapeses) != 1:
            raise ValueError(f"We only support one Shaped[...] in a leaf type, but got {shapeses}")
        prefix_names = shapeses[0]

        recursive_leaf_names = infer_named_axes(value, args[0])
        if recursive_leaf_names is None:
            return prefix_names
        else:
            return jax.tree_map(lambda x: prefix_names.concat(x), recursive_leaf_names)
    elif isinstance(value, eqx.Module):
        return infer_named_axes_from_module(value)
    elif isinstance(tpe, AxisNames):
        return tpe
    elif isinstance(value, jax.numpy.ndarray):
        return UnnamedAxes
    elif tpe is Array:
        return UnnamedAxes
    elif all_leaves([value]):
        return UnnamedAxes
    elif _is_named_tuple(value):
        child_axes = [infer_named_axes(child, tpe) for child, tpe in zip(value, value.__annotations__.values())]

        return value.__class__(*child_axes)
    else:
        handler = _registry.get(type(value))
        if not handler:
            raise NotImplementedError("Don't know how to infer axes for type %s" % type(value))
        children, meta = handler.to_iter(value)
        child_axes = [infer_named_axes(child, None) for child in children]
        return handler.from_iter(meta, child_axes)


def unwrap_axis_names(tree: Union[AxisNames, PyTree])->Union[AxisNames, PyTree]:
    """Unwraps the AxisNames from a tree. This is useful for passing to xmap"""
    return jax.tree_map(lambda x: x.names if isinstance(x, AxisNames) else x, tree)


def infer_named_axes_from_module(mod: eqx.Module):
    """Automatically get a "pytree" of named axes for an equinox Module.
    The leaves of this PyTree are AxisNames, which is just a wrapper around a list of names.
    To pass this to xmap, you need to unwrap the names using tree_map:
    >>> axis_names = jax.tree_map(lambda x: x.axes, infer_named_axes(mod))
    """
    # first split into the pytree
    dynamic_values, aux = mod.tree_flatten()
    dynamic_field_names = aux[0]
    fields: Sequence[dataclasses.Field] = dataclasses.fields(mod)  # type:ignore
    fields = {f.name: f for f in fields}

    named_shapes: List[Tuple[AxisName, ...]] = []

    for name, value in zip(dynamic_field_names, dynamic_values):
        if name not in fields:
            raise ValueError(f"Could not find field {name} in {mod.__class__}")

        field = fields[name]
        shape = infer_named_axes(value=value, tpe=field.type)
        named_shapes.append(shape)

    return mod.__class__.tree_unflatten(aux, named_shapes)



def _ensure_tuple(x):
    if x is None:
        return ()
    elif isinstance(x, typing.Iterable):
        return tuple(x)
    else:
        return (x,)



def xmapped_init(cls: typing.Type[eqx.Module],
                 *,
                 static_argnums: Optional[Sequence[int]]=None,
                 static_kwarg_names: Optional[Sequence[str]] = None,
                 axis_sizes=None, axis_resources=None, backend=None
                 ):
    """
    Wraps a class's constructor to automatically infer axes from field declarations to automatically xmap the function.
    This method uses infer_named_axes to infer the axes of all arguments, as well as that of the returned module.

    :return: a function that can be called with the same arguments as the original constructor, and will return an
        xmapped module.
    """

    axis_sizes = axis_sizes or {}
    axis_resources = axis_resources or {}

    # this is pretty tricky to get right.
    # It shares a lot in common with equinox's filter_jit etc, though it's a bit less fancy (for now), using
    # static argnums etc for now. We also don't bother with making sure caching works, since we're typically
    # only doing this once per run

    static_argnums = _ensure_tuple(static_argnums)
    static_kwarg_names = _ensure_tuple(static_kwarg_names)

    sig = inspect.signature(cls.__init__)

    @functools.wraps(cls.__new__)
    def wrapper_function(*args, **kwargs):

        bound_args: BoundArguments = sig.bind_partial(*((None, ) + args), **kwargs)

        dynamic_args = []
        dynamic_arg_shapes = []
        dynamic_arg_names = []

        for i, (name, param) in enumerate(sig.parameters.items()):
            if i == 0:
                assert name == "self"
                continue

            i = i - 1 # drop one for "self"
            arg = bound_args.arguments[name]
            if name not in static_kwarg_names and i not in static_argnums:
                dynamic_args.append(arg)
                dynamic_arg_shapes.append(infer_named_axes(arg, param.annotation))
                dynamic_arg_names.append(name)


        # these have to be tuples for xmap, but they break tree_map
        dynamic_arg_shapes_as_lists = jax.tree_map(lambda x: x.names if isinstance(x, AxisNames) else x, dynamic_arg_shapes)

        # we have to call the ctor twice under xmap.
        # The first time we get the axis names for the whole thing,
        # the second time we actually xmap

        # helper function we'll use to get the instance
        def construct_object(*dynamic_args):
            # update the signature
            bound_args.arguments.update(dict(zip(dynamic_arg_names, dynamic_args)))
            bound_args.apply_defaults()
            # call the original function, again dropping the first argument which is the dummy self
            inst = cls(*bound_args.args[1:], **bound_args.kwargs)
            return inst

        # first pass: get the axis names
        out_axes = None
        @functools.wraps(cls.__new__)
        def initial_xmap(*dynamic_args):
            # in the first pass we have to remove the names axes from the dynamic args
            def remove_named_axes(value, axis_spec: AxisNames):
                axis_spec = [axis for axis in axis_spec.names if axis is not Ellipsis]
                for _ in axis_spec:
                    value = value[0]
                return value
            # dynamic_args = jax.tree_map(remove_named_axes, dynamic_args, dynamic_arg_shapes)
            inst = construct_object(*dynamic_args)
            nonlocal out_axes
            out_axes = infer_named_axes(inst, cls)
            return None

        xmap(initial_xmap, in_axes=dynamic_arg_shapes_as_lists, out_axes=[...], axis_sizes=axis_sizes, axis_resources=axis_resources)(*dynamic_args)

        out_axes = jax.tree_map(lambda x: x.names if isinstance(x, AxisNames) else x, out_axes)

        # second pass: we actually have our axes.

        # now we make the function that we will xmap
        def function_to_xmap(*dynamic_args):
            inst = construct_object(*dynamic_args)
            return inst

        # now we can call xmap
        f = xmap(function_to_xmap, in_axes=dynamic_arg_shapes_as_lists, out_axes=out_axes,
                 axis_resources=axis_resources, axis_sizes=axis_sizes, backend=backend)
        inst = f(*dynamic_args)
        return inst

    return wrapper_function


__all__ = ["xmapped_init", "infer_leaf_axes", "infer_named_axes", "Array", "AxisNames",
           "infer_named_axes_from_module", "Shaped", "LogicalAxis", "ResourceAxis", "unwrap_axis_names"]