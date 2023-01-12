from typing import Callable, Tuple, TypeVar

import jax
from jax import numpy as jnp
from jax.experimental.pjit import with_sharding_constraint
from jax.interpreters.pxla import PartitionSpec

import haliax as hax
from haliax import Axis, auto_sharded
from haliax.jax_utils import named_call
from haliax.partitioning import ResourceAxis, ResourceMapping
from levanter.jax_utils import reduce


M = TypeVar("M")
X = TypeVar("X")


@named_call
def accumulate_gradients(f: Callable[[M, X], Tuple[float, M]], model: M, *inputs: X) -> Tuple[float, M]:
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
    f: Callable[[M, X], Tuple[float, M]],
    model: M,
    *inputs: X,
    data_axis_size: int,
    per_device_parallelism: int,
    compute_axis_mapping: ResourceMapping,
    parameter_axis_mapping: ResourceMapping,
) -> Tuple[float, M]:
    """
    Accumulate gradients across a sharded dataset, keeping a local copy of the gradient on each row of the data
     parallel axis. (If the model is not sharded, then a copy of the gradient is on each individual device.)

     Parameters:
        f: a function that takes a model and a batch of inputs and returns a tuple of (loss, gradient)
        data_axis_size: the size of the data parallel axis
        per_device_parallelism: how many examples to process at once on each device
        inputs: inputs with a leading batch axis, which will be reshaped/split
        compute_axis_mapping: a ResourceMapping for doing compute. The model should be sharded this way
        parameter_axis_mapping: a ResourceMapping for doing parameter updates. The model should be sharded this way
    """
    # data comes in as (batch, ...), and we'll reshape to (data_axis_size, num_micro_steps, per_device_parallelism, ...)
    batch_size = jnp.shape(inputs[0])[0]
    microbatch_size = data_axis_size * per_device_parallelism
    num_micro_steps = batch_size // microbatch_size
    assert num_micro_steps * microbatch_size == batch_size

    # do gradient accumulation on the data parallel axis, with model partitioned according to compute_axis_mapping
    with hax.axis_mapping(compute_axis_mapping, merge=False):
        with jax.named_scope("mass reshape"):

            def _reshape(x):
                x = x.reshape((microbatch_size, num_micro_steps) + x.shape[1:])
                return with_sharding_constraint(x, PartitionSpec(ResourceAxis.DATA, *(None,) * (len(x.shape) - 1)))

            inputs = jax.tree_util.tree_map(_reshape, inputs)

        Microbatch = Axis("microbatch", microbatch_size)

        with jax.named_scope("accumulate grad vmap"), hax.axis_mapping(
            {Microbatch.name: ResourceAxis.DATA}, merge=True
        ):
            losses, grads = hax.vmap(accumulate_gradients, axis=Microbatch)(f, model, *inputs)
            grads = auto_sharded(grads)

    # compute means and shard according to the parameter_axis_mapping
    with jax.named_scope("reduce grads"), hax.axis_mapping(parameter_axis_mapping):
        # losses and grads have Data leading axis
        grads = hax.mean(grads, axis=Microbatch)
        grads = auto_sharded(grads)
        loss = jnp.mean(losses)

    return loss, grads
