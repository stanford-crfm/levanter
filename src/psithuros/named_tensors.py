import dataclasses
import functools
import inspect
import typing
from typing import List, Tuple, Dict, Sequence, Union, Optional, Callable, Iterable, _GenericAlias, Any

import equinox as eqx
import jax
from equinox.custom_types import sentinel
from equinox.module import _ModuleMeta
from jax import tree_structure
from jax.experimental.maps import xmap, AxisName, ResourceSet


class Array(jax.numpy.ndarray):
    def __class_getitem__(cls, item):
        if not isinstance(item, tuple):
            item = (item,)
        return NamedArray(item)


@dataclasses.dataclass
class NamedArray:
    names: Tuple[AxisName, ...]


def infer_named_axes_from_module(mod: eqx.Module):
    """Automatically get a "pytree" of named axes for an equinox Module. This can be passed to xmap."""
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
        shape = infer_named_axes(value=value, annotation=field.type)
        named_shapes.append(shape)

    return mod.__class__.tree_unflatten(aux, named_shapes)

def infer_leaf_axes(annotation: type)-> List[Tuple[AxisName, ...]]:
    if type(annotation) == eqx.module._ModuleMeta and issubclass(annotation, eqx.Module):
        # unfortunately need to replicate the logic in Module#tree_flatten
        shapes = []
        for field_ in dataclasses.fields(annotation):  # type: ignore
            if not field_.metadata.get("static", False):
                shapes += infer_leaf_axes(field_.type)
        return shapes
    elif isinstance(annotation, NamedArray):
        return [annotation.names]
    elif annotation is Array:
        return [(...,)]
    else:
        return [(...,)]

def infer_named_axes(value: Any, annotation: Optional[type]):
    if isinstance(value, eqx.Module):
        return infer_named_axes_from_module(value)
    elif isinstance(annotation, NamedArray):
        return annotation.names
    elif isinstance(value, jax.numpy.ndarray):
        return (...,)
    elif annotation is Array:
        return (...,)
    else:
        # TODO exploit tuple types: tuples, lists, dicts, etc.
        leaves, structure = jax.tree_flatten(value)
        leaf_axes = [infer_named_axes(leaf, None) for leaf in leaves]
        return jax.tree_unflatten(structure, leaf_axes)


def auto_xmap(fun: Callable = sentinel,
              *,
              axis_sizes: Dict[AxisName, int] = None,
              axis_resources: Dict[AxisName, ResourceSet] = None,
              backend: Optional[str] = None):
    if fun is sentinel:
        return functools.partial(auto_xmap, axis_sizes=axis_sizes, axis_resources=axis_resources, backend=backend)
    """Wraps xmap to automatically infer tensor names from function signature and dataclass field declarations. This
    method knows about types annotated with NamedArray as well as equinox Module dataclasses."""

    # we want to make a function that, when it is called with a Module, will:
    # 1. infer the names of the axes from the Module's dataclass
    # 2. flatten the module into leaves and treedefs
    # 3. create a new function that will take the leaves as input and unflatten it into a Module
    # 4. call xmap with the new function
    # 5. apply the xmapped function to the flattened module (which will unflatten it)

    sig = inspect.signature(fun)

    @functools.wraps(fun)
    def axis_extracting_function(*args, **kwargs):
        # inspect the function signature for all args and such.
        # We need the signature for figuring out the names of axes for passed-in arrays as well as what type
        # we're expected to return
        # infer the axes of all arguments:
        arg_shapes = [infer_named_axes(arg, param.annotation) for (arg, param) in zip(args, sig.parameters.values())]
        if len(kwargs) > 0:
            raise NotImplementedError("kwargs not yet supported")
        kwarg_shapes = {k: infer_named_axes(v, None) for k, v in kwargs.items()}
        # flatten the arguments into pytrees
        args_leaves, args_treedefs = jax.tree_flatten(args)
        kwargs_leaves_defs = {k: jax.tree_flatten(v) for k,v in kwargs.items()}
        kwargs_leaves = {k: v[0] for k,v in kwargs_leaves_defs.items()}
        kwargs_treedefs = {k: v[1] for k,v in kwargs_leaves_defs.items()}

        # attempt to figure out the return type
        # TODO: want to handle type vars...
        return_axes = infer_leaf_axes(sig.return_annotation)

        results_treedefs = None

        @functools.wraps(fun)
        def function_to_xmap(*args, **kwargs):
            # unflatten the arguments into pytrees
            # args_unflattened = [jax.tree_unflatten(treedef, leaf) for treedef, leaf in zip(args_treedefs, args_leaves)]
            # kwargs_unflattened = {k: jax.tree_unflatten(kwargs_treedefs[k], kwargs_leaves[k]) for k in kwargs_leaves}
            # call the original function
            results = fun(*args, **kwargs)
            # flatten the results into pytrees
            nonlocal results_treedefs
            results_leaves, results_treedefs = jax.tree_flatten(results)
            return results_leaves

        # now we can call xmap
        # TODO: need to do a compile cache thing!
        # TODO: make this work with the signature for plain arrays
        # TODO: need to handle return type
        # TODO: figure out how to use kwargs shapes
        f = xmap(function_to_xmap, in_axes=arg_shapes, out_axes=return_axes)
        result_leaves = f(*args, **kwargs)
        result_unflattened = jax.tree_unflatten(results_treedefs, result_leaves)
        return result_unflattened

    return axis_extracting_function




def xmapped_class(cls: typing.Type[eqx.Module]):

    # this is pretty tricky to get right.
    # we want to be able to make it appear that the class __init__ has
    # an additional named argument "axis_sizes" that is a Dict[AxisName, int],
    # analogous to the argument to xmap.
    # To the user, when we call the class, we want them to pass in an axis_sizes
    # when creating the class.
    # Underneath, we first call the original __init__ with no arguments so that we can get a PyTree
    # with the right shape to which we can apply axis_sizes. This gives us "out_axes".
    # in_axes can be derived from looking at the arguments to the original __init__ and any accompanying types.
    #
    # Then we call xmap with the original __init__, the axis_sizes dict, the in_axes, and the out_axes.

    # TODO: cache this?
    @functools.wraps(cls.__new__)
    def xmapped_new(wrapped_cls, *args, axis_sizes=None, axis_resources=None, backend=None, **kwargs):

        tree_shape = None

        def mk_instance(*args, **kwargs):
            inst = cls(*args, **kwargs)
            nonlocal tree_shape
            leaves, tree_shape = jax.tree_flatten(inst)
            return leaves
        # # first, call the original __init__ with the original arguments, which will give us the original
        # # module, from which we can derive the out_axes
        # # this gives us the right shape to which we can apply axis_sizes
        # print(kwargs)
        # first, infer the axes of all arguments:
        sig = inspect.signature(cls.__init__)
        # bind the arguments to the signature, using None as "self" because we don't have or want it
        bound_sig = sig.bind_partial(*((None,) + args), **kwargs)
        bound_sig.apply_defaults()
        # infer the axes of all arguments:
        arg_shapes = [infer_named_axes(value, sig.parameters[arg_name].annotation) for arg_name, value in bound_sig.arguments.items() if arg_name != "self"]
        # now infer the out_axes
        out_axes = infer_leaf_axes(cls)

        leaves = xmap(mk_instance, in_axes=arg_shapes, out_axes=out_axes, axis_sizes=axis_sizes or {}, axis_resources=axis_resources or {})(*args)
        return jax.tree_unflatten(tree_shape, leaves)

    # now return the class with the xmapped __init__
    return type(cls.__name__ + "Wrapped", (), {"__new__": xmapped_new, "__init__": lambda self: ()})





