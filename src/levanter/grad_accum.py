import enum
import functools
from typing import Callable, Optional, ParamSpec, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jtu
from jax.lax import with_sharding_constraint
from jax.sharding import PartitionSpec

import haliax as hax
import haliax.quantization as hq
from haliax import Axis
from haliax.partitioning import ResourceAxis
from haliax.util import is_named_array

from levanter.utils.jax_utils import zeros_like_tree
from levanter.utils.types import ComputeLossFunction


Args = ParamSpec("Args")
R = TypeVar("R")
M_con = TypeVar("M_con", contravariant=True)  # Model
X = TypeVar("X", contravariant=True)  # Input


class ReductionType(enum.Enum):
    SUM = enum.auto()
    MEAN = enum.auto()
    # TODO: add MAX?


def apply_updates_running(acc, r, updates, overwrites):
    def _running_sum_updates(u, p):
        if u is None:
            return p
        else:
            return p * (1 - r) + u * r

    def _is_none(x):
        return x is None

    def _apply_update(tree, update, overwrite):
        if overwrite is not None:
            return overwrite

        return jtu.map(_running_sum_updates, update, tree, is_leaf=_is_none)

    def is_leaf(x):
        return x is None or isinstance(x, hq.OverwriteWithGradient)

    return jtu.map(_apply_update, acc, updates, overwrites, is_leaf=is_leaf)


# TODO: should we use a custom_jvp on microbatched?

# cf https://github.com/google-research/t5x/blob/main/t5x/trainer.py#L617
def microbatched(
    loss_fn: ComputeLossFunction[M_con, X],
    Batch: Axis,
    microbatch_size: int,
    accum_axis_mapping,
    compute_axis_mapping,
    patch_in_rng_key: Optional[str] = "key",
    accum_dtype: Optional[jnp.dtype] = None,
) -> Callable[Args, R]:
    """
    Wraps a function that takes a batch and changes it to instead take microbatches and accumulate the results
    This function has to reduce the batch axis, so it can't be used for functions that need to keep the batch axis.

    Can be used as a decorator with functools.partial, e.g.:

    >>> @functools.partial(microbatched, Batch=Batch, per_device_parallelism=4)
    >>> def my_fn(x):
    >>>     return hax.mean(x + 1)


    Args:
        fn: a function to wrap
        Batch: the batch axis
        per_device_parallelism: how many examples to process at once on each device
        accum_axis_mapping:  the axis mapping for the accumulator (typically this is the same as the params)
        compute_axis_mapping:  the axis mapping for the computation (typically this is the same as the inputs)
        patch_in_rng_key: if provided, this kwarg will be split, 1 for each accum step. It won't work if the
            PRNGKey is passed in as a positional argument.
        reduce: whether to sum or average the results
        accum_dtype: the dtype of floating point values in the accumulator. If None, this will be inferred from the return type of `fn`.

    Returns:
        a function that splits the batch into microbatches, calls the function on each microbatch, and
        accumulates the results.
    """
    batch_size = Batch.size
    data_axis_size = hax.partitioning.physical_axis_size(Batch, compute_axis_mapping)
    if data_axis_size is None:
        raise ValueError(f"{Batch} axis must be sharded")
    physical_axis_name = hax.partitioning.physical_axis_name(Batch, compute_axis_mapping)
    assert physical_axis_name is not None

    if microbatch_size <= 0:
        raise ValueError(f"Bad value for {microbatch_size=}")

    num_micro_steps = batch_size // microbatch_size

    if num_micro_steps == 1:

        @functools.wraps(loss_fn)
        def no_accum_loss_fn(*args, **kwargs):
            losses, where, extras = loss_fn(*args, **kwargs)
            return hax.mean(losses, where=where).scalar(), extras

        return eqx.filter_value_and_grad(no_accum_loss_fn, has_aux=True)

    Microbatch = Batch.resize(microbatch_size)
    AccumStep = Axis("accum_step", num_micro_steps)
    assert num_micro_steps * microbatch_size == batch_size

    @functools.wraps(loss_fn)
    def accum_loss_fn(*args, **kwargs):
        losses, where, extras = loss_fn(*args, **kwargs)
        return hax.mean(losses, where=where).scalar(), (where.sum(), extras)

    grad_fn = eqx.filter_value_and_grad(accum_loss_fn, has_aux=True)

    @functools.wraps(grad_fn)
    def wrapped_fn(*args, **kwargs):

        # first, determine the shape and make accumulator arrays
        r_shape = eqx.filter_eval_shape(grad_fn, *args, **kwargs)
        acc = zeros_like_tree(r_shape, accum_axis_mapping, accum_dtype)

        # then, reshape the inputs from (Batch, ...) to (AccumStep, Microbatch, ...)

        # Special handling for PRNGKey: it comes in as a single key, but we need to split it for each microbatch
        key = kwargs.get(patch_in_rng_key, None)
        if key is not None:
            key = jax.random.split(key, num_micro_steps)
            kwargs = kwargs.copy()
            kwargs.pop(patch_in_rng_key)

        args = _reshape_for_microbatch(Batch, Microbatch, AccumStep, args, compute_axis_mapping)

        def loop(acc, microbatch_and_key):
            (loss, (total, extras)), grads = acc
            microbatch, microbatch_kwargs, key = microbatch_and_key
            with jax.named_scope("compute"):
                microbatch_kwargs = microbatch_kwargs.copy()
                if key is not None:
                    microbatch_kwargs[patch_in_rng_key] = key
                (loss_mb, (n_mb, extras_mb)), grads_mb = grad_fn(*microbatch, **microbatch_kwargs)

            with jax.named_scope("accum"):

                # TODO: this uses the latest value for the scale for fp8, which seems not ideal but probably ok?
                overwrites, updates = hq.partition_for_grad_overwrite(grads_mb)
                r = n_mb / (total + n_mb)
                loss = loss + (loss_mb - loss) * r
                grads = apply_updates_running(grads, r, updates, overwrites)
                grads = hax.shard_with_axis_mapping(grads, accum_axis_mapping)
            print(loss, loss_mb, r)
            return (loss, (total, {k: v + extras_mb[k] for k, v in extras.items()})), grads

        with jax.named_scope("microbatched"):
            (loss, (_, extras)), grads, = hax.fold(
                loop, AccumStep
            )(acc, (args, kwargs, key))

        return (loss, extras), grads

    return wrapped_fn


def _reshape_for_microbatch(Batch: Axis, Microbatch: Axis, AccumStep: Axis, inputs, axis_mapping):
    def _reshape(x):
        if isinstance(x, hax.NamedArray):
            if not x.has_axis(Batch.name):
                return x
            x = x.unflatten_axis(Batch, (AccumStep, Microbatch))
            return hax.shard(x, axis_mapping)
        elif isinstance(x, jnp.ndarray):
            x = x.reshape((AccumStep.size, Microbatch.size) + x.shape[1:])
            return with_sharding_constraint(x, PartitionSpec(None, ResourceAxis.DATA, *(None,) * (len(x.shape) - 2)))
        else:
            # assert jnp.isscalar(x)
            return x

    return jax.tree_util.tree_map(_reshape, inputs, is_leaf=is_named_array)
