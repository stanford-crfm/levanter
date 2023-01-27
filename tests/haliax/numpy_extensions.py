import jax
import jax.numpy as jnp

from haliax.numpy_extensions import moving_window


def test_moving_window():
    a = jnp.arange(10)
    assert jnp.allclose(
        moving_window(a, 4),
        jnp.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]]),
    )


def test_moving_window_jit():
    a = jnp.arange(10)
    assert jnp.allclose(
        jax.jit(moving_window)(a, 4),
        jnp.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]]),
    )
