import jax.numpy as jnp
import jax
from jax import Array
from haliax import Axis

@jax.jit
def histogram(a: Array, bins: Array) -> Array:
  """Modified version of jax.numpy.histogram that returns integer counts instead of using the datatype of the input.
  This lets us avoid errors with bfloat16.

  Args:
      a (Array): input array 
      bins (Array): bins to use for histogram 

  Returns:
      Array: _description_
  """
  a = a.flatten()
  bin_idx = jnp.searchsorted(bins, a, side='right')
  bin_idx = jnp.where(a == bins[-1], len(bins) - 1, bin_idx)
  counts = jnp.zeros(len(bins), jnp.int32).at[bin_idx].add(1)[1:]
  return counts


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