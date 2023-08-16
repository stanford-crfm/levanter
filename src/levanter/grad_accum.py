from typing import Optional, Protocol, Tuple, TypeVar

import jax
import jax.numpy as jnp
from jax.lax import with_sharding_constraint
from jax.sharding import PartitionSpec
from jaxtyping import PRNGKeyArray

import haliax as hax
from haliax import Axis
from haliax.jax_utils import named_call
from haliax.partitioning import ResourceAxis
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
    model: M,
    *inputs: X,
    key: Optional[PRNGKeyArray] = None,
    per_device_parallelism: int,
    parameter_axis_mapping,
    **kwargs,
) -> Tuple[float, M]:
    """
    Accumulate gradients across a sharded batch, keeping a local copy of the gradient on each row of the data
     parallel axis. (If the model is not sharded, then a copy of the gradient is on each individual device.)

     Parameters:
        f: a function that takes a model and a batch of inputs and returns a tuple of (loss, gradient)
        per_device_parallelism: how many examples to process at once on each device
        inputs: inputs with the batch axis. non-named arrays assume that the 0th axis is the batch axis.
        parameter_axis_mapping: the axis mapping for the model parameters
        key: an optional PRNG key for the random number generator.
        If provided, this key will be split, 1 for each accum step
        kwargs: passed to f

    """
    batch_size = Batch.size
    data_axis_size = hax.partitioning.physical_axis_size(Batch, parameter_axis_mapping)
    if data_axis_size is None:
        raise ValueError(f"{Batch} axis must be sharded")
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

    if key is not None:
        key = jax.random.split(key, num_micro_steps)

    assert num_micro_steps * microbatch_size == batch_size

    # first things first, we want a copy of our gradient sharded like our model, along with a loss value
    loss = jnp.zeros(())
    with jax.named_scope("zeros"):
        grad = jax.tree_util.tree_map(jnp.zeros_like, model)
        grad = hax.shard_with_axis_mapping(grad, parameter_axis_mapping)

    # second, we want to reshape our data to (num_micro_steps, micro_batch_size, ...), sharded along the data axis
    inputs = _reshape_for_microbatch(Batch, Microbatch, AccumStep, inputs, parameter_axis_mapping)

    # third, we want to do compute.
    def loop(acc, microbatch_key):
        loss, grad = acc
        microbatch, microbatch_kwargs, key = microbatch_key
        with jax.named_scope("grad"):
            microbatch_kwargs = microbatch_kwargs.copy()
            if key is not None:
                microbatch_kwargs["key"] = key
            this_loss, this_grad = f(model, *microbatch, **microbatch_kwargs)
            this_grad = hax.shard_with_axis_mapping(this_grad, parameter_axis_mapping)

        with jax.named_scope("accum"):
            loss += this_loss
            grad = jax.tree_map(jnp.add, grad, this_grad)
            grad = hax.shard_with_axis_mapping(grad, parameter_axis_mapping)

        return loss, grad

    loss, grad = hax.fold(loop, AccumStep)((loss, grad), (inputs, kwargs, key))

    return loss / num_micro_steps, jax.tree_map(lambda x: x / num_micro_steps, grad)


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
