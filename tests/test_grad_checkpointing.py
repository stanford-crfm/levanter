import jax
import jax.numpy as jnp

from haliax import Axis
from levanter.grad_checkpointing import checkpointed_fold


def test_checkpointed_fold_basic():
    def fn(carry, x):
        return carry + x

    axis = Axis("a", 4)
    assert checkpointed_fold(fn, axis, checkpoint_block_size=2)(0.0, jnp.ones(axis.size)) == 4


def test_checkpointed_fold_gradient_match():
    def fn(carry, x):
        return carry + x

    axis = Axis("a", 4)

    grad_check = jax.grad(lambda x: checkpointed_fold(fn, axis, checkpoint_block_size=2)(0.0, x))(jnp.ones(axis.size))
    grad_plain = jax.grad(lambda x: jax.lax.scan(lambda q, r: (fn(q, r), None), 0.0, x)[0])(jnp.ones(axis.size))

    assert jnp.allclose(grad_check, grad_plain)


def test_checkpointed_fold_fancier_gradient():
    def fn(carry, x):
        return jnp.log1p(jnp.exp(carry + x))

    axis = Axis("a", 32)

    grad_check = jax.grad(lambda x: checkpointed_fold(fn, axis, checkpoint_block_size=2)(0.0, x))(jnp.ones(axis.size))
    grad_plain = jax.grad(lambda x: jax.lax.scan(lambda q, r: (fn(q, r), None), 0.0, x)[0])(jnp.ones(axis.size))

    assert jnp.allclose(grad_check, grad_plain)
