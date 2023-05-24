import functools
from typing import Dict, Generic, Optional, Protocol, Type, TypeVar

import equinox as eqx

import haliax
from haliax.core import Axis
from haliax.jax_utils import filter_checkpoint


M = TypeVar("M", bound=eqx.Module, covariant=True)


class ModuleInit(Protocol[M]):
    def __call__(self, *args, **kwargs) -> M:
        ...


class Stacked(eqx.Module, Generic[M]):
    """
    A "Stacked" wraps another module and produces a "stacked" version of it, where an input is applied
    to each instance of the stacked module in sequence. This is useful for e.g. transformers
    where you have multiple instances of the same transformer block and the input is applied in a fold/for loop
    in sequence.

    It's similar in spirit to an equinox.nn.Sequential, but it must be homogeneous. In Jax, this is much cheaper to
    compile than a sequential (or moral equivalent), because Jax compiles the module's method once, instead of unrolling
    the sequential and compiling everything as a giant graph. In Jax, this pattern is often called "scan layers" or
    "scan over layers".

    A further constraint is that the elements of the stack must have the same Python control flow. This is because
    Jax's scan primitive requires that the function you pass to it is pure, and the only way to do that is to ensure
    that the function has the same control flow for every element of the stack.

    Stacked supports both "fold" and "scan" semantics. "fold" is the same as a for loop that accumulates a single
    output, while "scan" is the same as a for loop that accumulates a list of intermediates as well as the final output.

    Stacked also supports gradient checkpointing, which is useful for very large models that don't fit in memory.

    Example:
        >>> import equinox as eqx
        >>> import haliax as hax
        >>> import haliax.nn as hnn
        >>> class MyModule(eqx.Module):
        ...     def __init__(self, num_layers: int, hidden: hax.Axis, *, key):
        ...         self.axis = Axis("layer", num_layers)
        ...         self.layers = Stacked.init(self.axis, hnn.Linear)(In=hidden, Out=hidden, key=key)
        ...
        ...     def __call__(self, x):
        ...         return self.layers.fold(x)  # applies each layer in sequence
        ...
        >>> Hidden = hax.Axis("hidden", 10)
        >>> mod = MyModule(5, Hidden)
        >>> mod(hax.ones(Hidden))
    """

    # TODO: we can probably make this module support pipeline parallelism, but that's a whole project in itself

    stacked: M
    Block: Axis = eqx.static_field()
    # TODO: support fancier gradient checkpointing
    gradient_checkpointing: bool = eqx.static_field()

    @staticmethod
    def init(Block: Axis, module: Type[M], *, gradient_checkpointing: bool = False) -> ModuleInit["Stacked[M]"]:
        @functools.wraps(module)
        def fn(*args, **kwargs):
            stacked = haliax.vmap(module.init, Block)(*args, **kwargs)
            return Stacked(stacked, Block, gradient_checkpointing)

        return fn

    def scan(self, init, *extra_args, **extra_kwargs):
        if self.gradient_checkpointing:
            do_block = filter_checkpoint(self._do_block)
        else:
            do_block = self._do_block
        return haliax.scan(do_block, self.Block)(init, self.stacked, *extra_args, **extra_kwargs)

    def fold(self, init, *args, **kwargs):
        if self.gradient_checkpointing:
            do_block = filter_checkpoint(self._do_block)
        else:
            do_block = self._do_block

        return haliax.fold(do_block, self.Block)(init, self.stacked, *args, **kwargs)

    @staticmethod
    def _do_block(carry, block, *extra_args, **extra_kwargs):
        return block(carry, *extra_args, **extra_kwargs)

    # TODO: this is for logic that's in levanter. We should move that logic to haliax I guess?
    def _state_dict_key_map(self) -> Optional[Dict[str, Optional[str]]]:
        return {"stacked": None}
