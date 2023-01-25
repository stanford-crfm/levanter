# Fancier gradient checkpointing
from functools import partial
from typing import Callable, TypeVar

import jax
import jax.numpy as jnp
from equinox.custom_types import BoolAxisSpec

from haliax import Axis
from haliax.util import is_jax_or_hax_array_like, is_named_array


Carry = TypeVar("Carry")
X = TypeVar("X")
Y = TypeVar("Y")


def checkpointed_fold(
    fn: Callable[[Carry, X], Carry],
    axis: Axis,
    *,
    checkpoint_block_size: int,
    is_scanned: BoolAxisSpec = is_jax_or_hax_array_like,
):
    axis_size = axis.size
    if axis.size % checkpoint_block_size != 0:
        raise ValueError(f"axis size ({axis}) must be divisible by checkpoint_block_size ({checkpoint_block_size})")

    checkpoint_blocks = axis_size // checkpoint_block_size

    def is_scanned_with_axis(leaf):
        if is_named_array(leaf):
            return axis in leaf.axes and is_scanned(leaf)
        else:
            return is_scanned(leaf)

    def wrapped_fn(init, *args, **kwargs):
        def select_index_for_axis(leaf, index):
            if is_scanned_with_axis(leaf):
                if is_named_array(leaf):
                    return leaf.take(axis, index)
                else:
                    return leaf[index]
            else:
                return leaf

        def do_one_index(carry_index, i):
            carry, start = carry_index
            (my_args, my_kwargs) = jax.tree_util.tree_map(
                partial(select_index_for_axis, index=start+i), (args, kwargs), is_leaf=is_scanned_with_axis
            )
            return (fn(carry, *my_args, **my_kwargs), start), None

        def do_one_block(carry, i):
            carry, _ = carry
            start = i * checkpoint_block_size
            # return jax.lax.fori_loop(start, end, do_one_index, carry)
            return jax.lax.scan(do_one_index, (carry, start), jnp.arange(0, checkpoint_block_size, dtype=int))

        # return jax.lax.fori_loop(0, checkpoint_blocks, jax.checkpoint(do_one_block, prevent_cse=False), init)
        return jax.lax.scan(do_one_block, (init, 0), jnp.arange(checkpoint_blocks, dtype=int))[0][0]

    return wrapped_fn
