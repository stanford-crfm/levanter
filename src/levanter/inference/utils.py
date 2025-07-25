import haliax as hax
from haliax import NamedArray


def is_invalid(x):
    return (x < 0) | (x == INVALID)


def is_valid(x):
    """
    Returns a boolean array indicating whether each token in the input is valid.
    A token is considered valid if it is not negative and not equal to INVALID.
    """
    return (x >= 0) & (x != INVALID)


INVALID = 2_000_000


def masked_set(dest: NamedArray, selector, axis, start, src, num_to_copy) -> NamedArray:
    """
    jit-safe masked memcpy-like operation.
    Copy into dest[selector, axis, start:start+num_to_copy] the values from src[axis, :num_to_copy].

    Probably faster to not use an arange (which lowers to a scatter) and use blocks? Probably not a bottleneck

    num_to_copy may be dynamic
    """

    src_arange = hax.arange(src.resolve_axis(axis))
    dest_axis_size = dest.axis_size(axis)
    # mask out the tail
    dest_arange = hax.where(src_arange >= num_to_copy, dest_axis_size, src_arange + start)
    src_arange = hax.where(src_arange >= num_to_copy, src_arange.size, src_arange)

    return dest.at[{**selector, axis: dest_arange}].set(src[axis, src_arange], mode="drop")
