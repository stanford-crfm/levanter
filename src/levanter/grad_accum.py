import functools
from typing import Dict, Optional, Protocol, Tuple, TypeVar, Union

import jax
from jax import numpy as jnp
from jax.experimental.pjit import with_sharding_constraint
from jax.interpreters.pxla import PartitionSpec
from jaxtyping import PyTree

import haliax as hax
from haliax import Axis
from haliax.jax_utils import named_call
from haliax.partitioning import ResourceAxis, shard_with_axis_mapping
from haliax.util import is_named_array
from levanter.utils.jax_utils import reduce


M = TypeVar("M")  # Model
X = TypeVar("X", contravariant=True)  # Input


class GradAndValFn(Protocol[M, X]):
    def __call__(self, model: M, *inputs: X, **kwargs) -> Tuple[float, M]:
        ...


@named_call
def accumulate_gradients(f: GradAndValFn, model: M, *inputs: X) -> Tuple[float, M]:
    """Simple gradient accumulation that just loops over the inputs."""
    zero = (jnp.zeros(()), jax.tree_util.tree_map(lambda m: jnp.zeros_like(m), model), 0)

    def compute_and_accumulate(acc, *input):
        loss, grad = f(model, *input)
        acc_loss, acc_grad, n = acc
        return loss + acc_loss, jax.tree_map(jnp.add, acc_grad, grad), n + 1

    total_loss, total_grad, total_n = reduce(compute_and_accumulate, zero, *inputs)

    return total_loss / total_n, jax.tree_map(lambda x: x / total_n, total_grad)


# cf https://github.com/google-research/t5x/blob/main/t5x/trainer.py#L617
@named_call
def accumulate_gradients_sharded(
    f: GradAndValFn,
    Batch: Axis,
    per_device_parallelism: int,
    compute_axis_mapping,
    parameter_axis_mapping,
    batched_args: Optional[Union[bool, Tuple[bool, ...]]] = None,
    batched_kwargs: Optional[Union[bool, Dict[str, bool]]] = None,
) -> GradAndValFn[M, X]:

    """
    Accumulate gradients across a sharded batch, keeping a local copy of the gradient on each row of the data
     parallel axis. (If the model is not sharded, then a copy of the gradient is on each individual device.)

     Parameters:
        f: a function that takes a model and a batch of inputs and returns a tuple of (loss, gradient)
        per_device_parallelism: how many examples to process at once on each device
        parameter_axis_mapping: the axis mapping for the model parameters
        batched_args: a tuple of booleans indicating whether each unnamed array argument is batched. Defaults to True for each argument.
        batched_kwargs: a dictionary of booleans indicating whether each unnamed kwarg is batched. Defaults to True for each kwarg.
    """
    batch_size = Batch.size
    data_axis_size = hax.partitioning.physical_axis_size(Batch, parameter_axis_mapping)
    if data_axis_size is None:
        raise ValueError(f"Batch {Batch} axis must be sharded for sharded grad accumulation")
    physical_axis_name = hax.partitioning.physical_axis_name(Batch, parameter_axis_mapping)
    assert physical_axis_name is not None

    microbatch_size = data_axis_size * per_device_parallelism
    num_micro_steps = batch_size // microbatch_size

    assert batch_size % data_axis_size == 0, f"batch_size % data_axis_size != 0: {batch_size} % {data_axis_size} != 0"
    assert (
        batch_size % microbatch_size == 0
    ), f"batch_size % microbatch_size != 0: {batch_size} % {microbatch_size} != 0"

    Microbatch = Axis(Batch.name, microbatch_size)
    AccumStep = Axis("accum_step", num_micro_steps)

    assert num_micro_steps * microbatch_size == batch_size

    # TODO: update docstring?
    @functools.wraps(f)
    def accum_f(model: M, *args: X, **kwargs) -> Tuple[float, M]:
        # first things first, we want a copy of our gradient sharded like our model, along with a loss value
        loss = jnp.zeros(())
        with jax.named_scope("zeros"):
            grad = jax.tree_map(jnp.zeros_like, model)
            grad = shard_with_axis_mapping(grad, parameter_axis_mapping)

        if batched_args is None:
            batched_args_spec = (True,) * len(args)
        elif isinstance(batched_args, bool):
            batched_args_spec = (batched_args,) * len(args)
        elif not isinstance(batched_args, tuple):
            batched_args_spec = tuple(batched_args)
        else:
            batched_args_spec = batched_args

        if len(batched_args_spec) < len(args):
            batched_args_spec = batched_args_spec + (True,) * (len(args) - len(batched_args_spec))
        elif len(batched_args_spec) > len(args):
            raise ValueError(f"batched_args_spec must be a tuple of booleans of length {len(args)} (or less)")

        if isinstance(batched_kwargs, bool):
            batched_kwarg_spec = {k: True for k in kwargs}
        elif batched_kwargs is None:
            batched_kwarg_spec = batched_kwargs or {k: True for k in kwargs}
        else:
            batched_kwarg_spec = {k: batched_kwargs.get(k, True) for k in kwargs}

        # second, we want to reshape our data to (num_micro_steps, micro_batch_size, ...), sharded along the data axis
        (args, kwargs) = _reshape_for_microbatch(
            Batch,
            Microbatch,
            AccumStep,
            (args, kwargs),
            (batched_args_spec, batched_kwarg_spec),
            compute_axis_mapping,
        )

        # third, we want to do compute.
        def loop(acc, microbatch):
            loss, grad = acc
            with jax.named_scope("grad"):
                this_loss, this_grad = f(model, *microbatch)
                this_grad = shard_with_axis_mapping(this_grad, parameter_axis_mapping)

            with jax.named_scope("accum"):
                loss += this_loss
                grad = jax.tree_map(jnp.add, grad, this_grad)
                grad = shard_with_axis_mapping(grad, parameter_axis_mapping)

            return loss, grad

        loss, grad = hax.fold(loop, AccumStep)((loss, grad), args)

        return loss / num_micro_steps, jax.tree_map(lambda x: x / num_micro_steps, grad)

    return accum_f


def _reshape_for_microbatch(
    Batch: Axis, Microbatch: Axis, AccumStep: Axis, inputs: PyTree[X], should_batch_spec: PyTree[bool], axis_mapping
) -> PyTree[X]:
    """Reshapes the Batch axis (or 0'th axis for unnamed arrays) to (AccumStep, Microbatch, ...), with Microbatch
    sharded along the Batch axis. should_batch_spec should be a pytree prefix"""

    def _reshape(should_batch, x):
        if not should_batch:
            return x
        elif isinstance(x, hax.NamedArray):
            x = x.unflatten_axis(Batch, (AccumStep, Microbatch))
            return hax.shard_with_axis_mapping(x, axis_mapping)
        elif isinstance(x, jnp.ndarray):
            x = x.reshape((AccumStep.size, Microbatch.size) + x.shape[1:])
            return with_sharding_constraint(x, PartitionSpec(None, ResourceAxis.DATA, *(None,) * (len(x.shape) - 2)))
        elif jnp.isscalar(x):
            return x
        else:  # x is a pytree but should_batch is a bool (we hope)
            assert should_batch is True
            return jax.tree_map(functools.partial(_reshape, True), x, is_leaf=is_named_array)

    return jax.tree_map(_reshape, should_batch_spec, inputs, is_leaf=is_named_array)
