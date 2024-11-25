import functools

import equinox
import jax
import jax.numpy as jnp
import numpy as np
from jax._src.partition_spec import PartitionSpec
from jax.experimental.shard_map import shard_map
from jaxtyping import ArrayLike, Scalar

import haliax as hax
from haliax import NamedArray


class Histogram(equinox.Module):
    """
    Has enough information to log to tensorboard and wandb
    """

    min: Scalar
    max: Scalar
    num: Scalar | int
    sum: Scalar
    sum_squares: Scalar
    bucket_limits: jax.Array
    bucket_counts: jax.Array

    @staticmethod
    def from_array(array: jax.Array, num_bins: int = 64) -> "Histogram":
        array = array.ravel()
        min = array.min()
        max = array.max()
        num = array.size
        sum = array.sum()
        sum_squares = (array**2).sum()
        counts, edges = jax.numpy.histogram(array, bins=num_bins)
        return Histogram(min, max, num, sum, sum_squares, edges, counts)

    @staticmethod
    def from_named_array(array: hax.NamedArray, num_bins: int = 64) -> "Histogram":
        raw_array = array.array
        min = raw_array.min()
        max = raw_array.max()
        num = array.size
        sum = raw_array.sum()
        sum_squares = (raw_array**2).sum()
        counts, edges = sharded_histogram(array, bins=num_bins)
        return Histogram(min, max, num, sum, sum_squares, edges, counts)

    def to_numpy_histogram(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array(self.bucket_counts), np.array(self.bucket_limits)


def sharded_histogram(a: NamedArray, bins: int | ArrayLike = 10) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    As [jax.numpy.histogram](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.histogram.html#jax.numpy.histogram),
    except:

    * It preserves sharding
    * It only works with NamedArrays
    * It is more performant on TPUs

    Credit to @aphoh for the original implementation, though that one crashes on TPUs due to some kind of driver bug
    """
    edges = jnp.histogram_bin_edges(a.array, bins=bins)
    return _shardmap_histogram(a, edges), edges


def _single_shard_histogram(a, bins, reduce_mesh):
    """Modified version of jax.numpy.histogram that returns integer counts instead of using the datatype of the input.
    Also avoids searchsorted, which is slow on TPUs.
    Args:
        a (Array): input array
        bins (Array): bins to use for histogram
    Returns:
        Array: counts. has length len(bins) - 1
    """
    a = a.flatten()

    bin_idx = (a[..., None] >= bins[:-1]).astype(jnp.int32) & (a[..., None] < bins[1:]).astype(jnp.int32)
    counts = bin_idx.sum(axis=0, dtype=jnp.int32)

    if len(reduce_mesh):
        counts = jax.lax.psum(counts, axis_name=reduce_mesh)
    return counts


def _shardmap_histogram(a: NamedArray, bins):
    mesh = hax.partitioning._get_mesh()
    spec = hax.partitioning.pspec_for_axis(a.axes)
    flattened_spec = _flattened_spec(spec)
    shard_h = shard_map(
        functools.partial(_single_shard_histogram, reduce_mesh=flattened_spec),
        mesh=mesh,
        in_specs=(spec, PartitionSpec(None)),
        out_specs=PartitionSpec(
            None,
        ),
    )
    res = shard_h(a.array, bins)

    # the filter misses the last bin, so we need to add it
    if res.size >= 1:
        res = res.at[-1].add(1)
    return res


def _flattened_spec(spec):
    out = []
    for s in spec:
        if isinstance(s, tuple):
            out.extend(s)
        elif s is None:
            pass
        else:
            out.append(s)

    return tuple(out)
