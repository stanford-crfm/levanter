import jax
import numpy as np
from jax.random import PRNGKey
from jax.sharding import Mesh

import haliax as hax
from haliax.partitioning import ResourceAxis

import levanter.tracker.histogram
from test_utils import skip_if_not_enough_devices


def test_sharded_histogram_simple():
    mesh = Mesh((jax.devices()), (ResourceAxis.DATA,))

    Batch = hax.Axis("batch", 64)
    Feature = hax.Axis("feature", 128)

    with mesh, hax.axis_mapping({"batch": ResourceAxis.DATA}):
        a = hax.random.normal(PRNGKey(1), (Batch, Feature))
        a = hax.shard(a)
        hist, bins = levanter.tracker.histogram.sharded_histogram(a, bins=32)

    hist_normal, bins_normal = jax.numpy.histogram(a.array, bins=32)

    assert jax.numpy.allclose(hist, hist_normal)
    assert jax.numpy.allclose(bins, bins_normal)


@skip_if_not_enough_devices(2)
def test_sharded_histogram_tp():
    mesh = Mesh(np.array(jax.devices()).reshape(-1, 2), (ResourceAxis.DATA, ResourceAxis.MODEL))

    Batch = hax.Axis("batch", 64)
    Feature = hax.Axis("feature", 128)

    with mesh, hax.axis_mapping({"batch": ResourceAxis.DATA, "feature": ResourceAxis.MODEL}):
        a = hax.random.normal(PRNGKey(0), (Batch, Feature)) * 100
        a = hax.shard(a)
        hist, bins = levanter.tracker.histogram.sharded_histogram(a, bins=64)

    jnp_hist, jnp_bins = jax.numpy.histogram(a.array, bins=64)

    assert jax.numpy.allclose(hist, jnp_hist)
    assert jax.numpy.allclose(bins, jnp_bins)
