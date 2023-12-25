import functools
from typing import Callable, Optional, ParamSpec, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.lax import with_sharding_constraint
from jax.sharding import PartitionSpec

import haliax as hax
from haliax import Axis
from haliax.jax_utils import named_call
from haliax.partitioning import ResourceAxis
from haliax.util import is_jax_array_like, is_named_array

from levanter.types import M, ValAndGradFn, ValFn, X


Args = ParamSpec("Args")
R = TypeVar("R")


def microbatched_mean(
    fn: Callable[Args, R],
    Batch: Axis,
    per_device_parallelism: int,
    accum_axis_mapping,
    compute_axis_mapping,
    patch_in_rng_key: Optional[str] = "key",
) -> Callable[Args, R]:
    """
    Wraps a function that takes a batch and changes it to instead take microbatches and accumulate the results
    This function takes the *mean* of the microbatched results, so it only does what you want if the function
    is taking the mean of the batch axis.

    Args:
        fn: a function to wrap
        Batch: the batch axis
        per_device_parallelism: how many examples to process at once on each device
        accum_axis_mapping:  the axis mapping for the accumulator (typically this is the same as the params)
        compute_axis_mapping:  the axis mapping for the computation (typically this is the same as the inputs)
        patch_in_rng_key: if provided, this kwarg will be split, 1 for each accum step. It won't work if the
            PRNGKey is passed in as a positional argument.

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

    microbatch_size = data_axis_size * per_device_parallelism
    num_micro_steps = batch_size // microbatch_size
    Microbatch = Batch.resize(microbatch_size)
    AccumStep = Axis("accum_step", num_micro_steps)
    assert num_micro_steps * microbatch_size == batch_size

    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        key = kwargs.get(patch_in_rng_key, None)
        if key is not None:
            key = jax.random.split(key, num_micro_steps)
        # first, determine the shape and make accumulator arrays
        r_shape = eqx.filter_eval_shape(fn, *args, **kwargs)
        acc = jax.tree_util.tree_map(
            functools.partial(_zeros_like, accum_axis_mapping), r_shape, is_leaf=is_named_array
        )

        args = _reshape_for_microbatch(Batch, Microbatch, AccumStep, args, compute_axis_mapping)

        def loop(acc, microbatch_and_key):
            microbatch, microbatch_kwargs, key = microbatch_and_key
            with jax.named_scope("compute_microbatch"):
                microbatch_kwargs = microbatch_kwargs.copy()
                if key is not None:
                    microbatch_kwargs[patch_in_rng_key] = key
                this_r = fn(*microbatch, **microbatch_kwargs)

            with jax.named_scope("accum"):
                acc = eqx.apply_updates(acc, this_r)
                acc = hax.shard_with_axis_mapping(acc, accum_axis_mapping)

            return acc

        acc = hax.fold(loop, AccumStep)(acc, (args, kwargs, key))
        acc = jax.tree_util.tree_map(lambda x: x / num_micro_steps, acc)

        return acc

    return wrapped_fn


# cf https://github.com/google-research/t5x/blob/main/t5x/trainer.py#L617
@named_call
def accumulate_gradients_sharded(
    f: ValFn[M, X],
    Batch: Axis,
    per_device_parallelism: int,
    parameter_axis_mapping,
) -> ValAndGradFn[M, X]:
    """
    Accumulate gradients across a sharded batch, keeping a local copy of the gradient on each row of the data
     parallel axis. (If the model is not sharded, then a copy of the gradient is on each individual device.)

     Parameters:
        f: a function whose gradients are to be accumulated
        per_device_parallelism: how many examples to process at once on each device
        inputs: inputs with the batch axis. non-named arrays assume that the 0th axis is the batch axis.
        parameter_axis_mapping: the axis mapping for the model parameters
        key: an optional PRNG key for the random number generator.
        If provided, this key will be split, 1 for each accum step
        kwargs: passed to f

    """
    grad_fn = eqx.filter_value_and_grad(f, has_aux=False)
    grad_fn = microbatched_mean(grad_fn, Batch, per_device_parallelism, parameter_axis_mapping, parameter_axis_mapping)

    return grad_fn


def _reshape_for_microbatch(Batch: Axis, Microbatch: Axis, AccumStep: Axis, inputs, axis_mapping):
    def _reshape(x):
        if isinstance(x, hax.NamedArray):
            if not x.has_axis(Batch.name):
                return x
            x = x.unflatten_axis(Batch, (AccumStep, Microbatch))
            return hax.shard_with_axis_mapping(x, axis_mapping)
        elif isinstance(x, jnp.ndarray):
            x = x.reshape((AccumStep.size, Microbatch.size) + x.shape[1:])
            return with_sharding_constraint(x, PartitionSpec(None, ResourceAxis.DATA, *(None,) * (len(x.shape) - 2)))
        else:
            assert jnp.isscalar(x)
            return x

    return jax.tree_util.tree_map(_reshape, inputs, is_leaf=is_named_array)


def _zeros_like(mapping, n):
    if isinstance(n, hax.NamedArray):
        return hax.auto_sharded(hax.zeros_like(n), mapping)
    elif is_jax_array_like(n):
        return jnp.zeros_like(n)
    else:
        assert jnp.isscalar(n)
        return 0.0
