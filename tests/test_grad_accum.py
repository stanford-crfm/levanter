import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

import haliax as hax
import haliax.nn as hnn
from haliax.partitioning import ResourceAxis
from levanter.grad_accum import accumulate_gradients_sharded


class MLP(eqx.Module):
    p_up: hnn.Linear
    p_down: hnn.Linear
    dropout: hnn.Dropout

    def __init__(self, In, Mid, *, key):
        self.p_up = hnn.Linear.init(In, Mid, key=key)
        self.p_down = hnn.Linear.init(Mid, In, key=key)
        self.dropout = hnn.Dropout(0.5)

    def __call__(self, x, key):
        x = self.p_up(x)
        x = self.dropout(x, inference=key is None, key=key)
        x = self.p_down(x)
        return x


Batch = hax.Axis("batch", 64)
In = hax.Axis("in", 4)
Mid = hax.Axis("mid", 8)


@pytest.mark.parametrize("per_device_parallelism", (1, 2, 4))
def test_accumulate_gradients_sharded_simple(per_device_parallelism):
    def loss_fn(model, x):
        return model(x, key=None).mean().scalar()

    devices = jax.devices()
    axis_mapping = {"batch": ResourceAxis.DATA, "mid": ResourceAxis.MODEL}
    with Mesh(np.array(devices).reshape(-1, 1), (ResourceAxis.DATA, ResourceAxis.MODEL)):

        model = MLP(In, Mid, key=jax.random.PRNGKey(0))
        x = hax.ones((Batch, In))

        grad_loss = eqx.filter_value_and_grad(loss_fn)

        loss, grads = accumulate_gradients_sharded(
            grad_loss, Batch, per_device_parallelism, axis_mapping, axis_mapping
        )(model, x)

        no_accum_loss, no_accum_grads = grad_loss(model, x)

        assert loss.shape == ()
        assert jnp.isclose(loss, no_accum_loss).all()

        # jax.tree_map(lambda x: assert_sharded(x, axis_mapping), grads)
        # check gradients
        def check_is_close(x, y):
            assert jnp.isclose(x, y).all()

        jax.tree_map(check_is_close, grads, no_accum_grads)
