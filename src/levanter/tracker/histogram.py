import equinox
import jax
import numpy as np
from jaxtyping import Scalar


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

    def to_numpy_histogram(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array(self.bucket_counts), np.array(self.bucket_limits)
