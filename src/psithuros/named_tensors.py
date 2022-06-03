import dataclasses
import functools
import inspect
from typing import List, Tuple, Dict, Sequence, Union, Optional, Callable, Iterable

import equinox as eqx
import jax
from equinox.custom_types import sentinel
from equinox.module import _ModuleMeta
from jax import tree_structure
from jax.experimental.maps import xmap, AxisName, ResourceSet


class Array:
    def __class_getitem__(cls, item):
        if not isinstance(item, tuple):
            item = (item,)
        return NamedArray(item)


@dataclasses.dataclass
class NamedArray:
    names: Tuple[AxisName, ...]


def infer_named_axes_from_value(value):
    if isinstance(value, eqx.Module):
        return infer_named_axes_from_module(value)
    elif isinstance(value, jax.numpy.ndarray):
        return [...]
    elif isinstance(value, Iterable):
        return [infer_named_axes_from_value(v) for v in value]




def infer_named_axes_from_module(mod: eqx.Module):
    """Automatically get a "pytree" of named axes for an equinox Module. This can be passed to xmap."""
    # first split into the pytree
    _, (dynamic_field_names, _, _) = mod.tree_flatten()
    fields: Sequence[dataclasses.Field] = dataclasses.fields(mod)  # type:ignore
    fields = {f.name: f for f in fields}

    named_shapes: List[Tuple[AxisName, ...]] = []

    for name in dynamic_field_names:
        if name not in fields:
            raise ValueError(f"Could not find field {name} in {mod.__class__}")

        field = fields[name]
        shape = infer_named_axes_from_annotation(field.type)
        named_shapes.append(shape)

    return named_shapes


def infer_named_axes_from_annotation(annotation):
    if isinstance(annotation, NamedArray):
        return annotation.names
    elif issubclass(annotation, eqx.Module):
        # unfortunately need to replicate the logic in Module#tree_flatten
        shapes = []
        for field_ in dataclasses.fields(annotation):  # type: ignore
            if not field_.metadata.get("static", False):
                shapes.append(infer_named_axes_from_annotation(field_.type))

        return shapes
    return [...]


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
        arg_shapes = [infer_named_axes_from_value(arg) for arg in args]
        if len(kwargs) > 0:
            raise NotImplementedError("kwargs not yet supported")
        kwarg_shapes = {k: infer_named_axes_from_value(v) for k, v in kwargs.items()}
        # flatten the arguments into pytrees
        args_leaves, args_treedefs = zip(*[jax.tree_flatten(arg) for arg in args])
        kwargs_leaves_defs = {k: jax.tree_flatten(v) for k,v in kwargs.items()}
        kwargs_leaves = {k: v[0] for k,v in kwargs_leaves_defs.items()}
        kwargs_treedefs = {k: v[1] for k,v in kwargs_leaves_defs.items()}

        # attempt to figure out the return type
        # TODO: want to handle type vars...
        return_axes = infer_named_axes_from_annotation(sig.return_annotation)

        results_treedefs = None

        @functools.wraps(fun)
        def function_to_xmap(*args_leaves, **kwargs_leaves):
            # unflatten the arguments into pytrees
            args_unflattened = [jax.tree_unflatten(treedef, leaf) for treedef, leaf in zip(args_treedefs, args_leaves)]
            kwargs_unflattened = {k: jax.tree_unflatten(kwargs_treedefs[k], kwargs_leaves[k]) for k in kwargs_leaves}
            # call the original function
            results = fun(*args_unflattened, **kwargs_unflattened)
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
        result_leaves = f(*args_leaves, **kwargs_leaves)
        result_unflattened = jax.tree_unflatten(results_treedefs, result_leaves)
        return result_unflattened

    return axis_extracting_function



