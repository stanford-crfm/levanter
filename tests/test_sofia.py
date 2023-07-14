import os

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import numpy as np

import levanter
from levanter.optim import stochastic_hessian_diagonal


def test_hessian_optimizer():
    key = jax.random.PRNGKey(0)
    model = nn.Linear(4, 4, use_bias=False, key=key)
    data = np.load(f"{os.path.dirname(__file__)}/data/hero_data.npy").astype("float32")
    optimizer = levanter.optim.sophia(lr=1, b1=0, b2=0.99, gamma=2)
    model = jax.tree_util.tree_map(lambda x: jnp.ones_like(x), model)

    opt_state = optimizer.init(model)

    def loss_fn(model, data):
        out = eqx.filter_vmap(model)(data)
        return jnp.mean(out**2) * 4

    jit_diag = eqx.filter_jit(stochastic_hessian_diagonal)
    for i, k_i in enumerate(jax.random.split(key, 10000)):
        hessian = jit_diag(loss_fn, model, data, g_key=k_i)
        opt_state = optimizer.hessian_update(hessian, opt_state)

    # print('Test-estimated hessian: most coordinates should be approximately 2')
    # print('Estimated hessian:', opt_state[0].h.weight)
    assert jnp.allclose(opt_state[0].h.weight, 2, rtol=0.2, atol=0.3)  # this is very approximate

    grad_loss_fn = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn))

    loss, grad = grad_loss_fn(model, data)
    model, opt_state = optimizer.update(grad, opt_state)

    print("Initial Loss:", loss.item())
    # loss should be 15.74834156036377
    assert jnp.allclose(loss, 15.74834156036377)

    # print("Test-model param after 1 step: most coordinates should be very loosely 0.5")
    assert jnp.allclose(model.weight, 0.5, rtol=0.2, atol=0.1)  # this is very approximate

    # print("Test-loss: loss should shrink by approximately 75% after each iteration")
    for i in range(10):
        loss, grad = grad_loss_fn(model, data)
        model, opt_state = optimizer.update(grad, opt_state)

        # print('Step:', i , "Loss:", loss.item())
        assert loss < 15.74834156036377 * 0.75 ** (i + 1)
