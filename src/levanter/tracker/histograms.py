import jax.numpy as jnp
import jax
from jax import Array
from haliax import Axis, NamedArray
from haliax.partitioning import ResourceAxis
import haliax
from jax.sharding import PartitionSpec
from jax.experimental.shard_map import shard_map

@jax.jit
def histogram(a: Array, bins: Array) -> Array:
  """Modified version of jax.numpy.histogram that returns integer counts instead of using the datatype of the input.
  Also avoids searchsorted, which is slow on TPUs.

  Args:
      a (Array): input array 
      bins (Array): bins to use for histogram 

  Returns:
      Array: counts. has length len(bins) - 1
  """
  a = a.flatten()
  prefix_sum = jnp.sum((a < bins[:, None]).astype(jnp.int32), axis=1)
  last_count = jnp.sum(a <= bins[-1])
  prefix_sum = prefix_sum.at[-1].set(last_count)
  return jnp.expand_dims(jnp.diff(prefix_sum), 0)

@jax.jit
def sharded_histogram(a: Array, bins: Array) -> Array:
    """Compute the histogram of an array a, assuming it's sharded across the `ResourceAxis.DATA` axis.

    Args:
        a (Array): The input array to compute the histogram of 
        bins (Array): The bins for the histogram

    Returns:
        Array: The resulting counts. Length is len(bins) - 1
    """
    P = PartitionSpec
    in_specs = (P(ResourceAxis.DATA, None), P(None))
    out_specs = (P(ResourceAxis.DATA, None))
    mesh = haliax.partitioning._get_mesh()
    a = a.reshape(a.shape[0], -1)
    shard_h = shard_map(histogram, mesh=mesh, in_specs=in_specs, out_specs=out_specs)
    res = shard_h(a, bins)
    res = res.sum(axis=0)
    return res

NSIDE = 254
NBINS = 2*NSIDE + 3
@jax.jit
def get_bins():
    bins = jnp.logspace(-16, 6, 254, base=2.0)
    inf = jnp.array([jnp.inf])
    zero = jnp.array([0.0])
    _BINS = jnp.concatenate([-inf, -bins[::-1], zero, bins, inf])
    return _BINS
    
BIN_AX = Axis("bins", NBINS-1)