import jax
import jax.numpy as jnp

from haliax.numpy_extensions import moving_window, padded_moving_window


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


def test_moving_window_axis():
    a = jnp.arange(10).reshape((2, 5))
    moved = moving_window(a, 3, axis=1)
    assert moved.shape == (2, 3, 3)
    assert jnp.allclose(
        moved,
        jnp.array(
            [
                [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
                [[5, 6, 7], [6, 7, 8], [7, 8, 9]],
            ]
        ),
    )


def test_padded_moving_window():
    a = jnp.arange(10)
    assert jnp.allclose(
        padded_moving_window(a, 4, padding=100),
        jnp.array(
            [
                [100, 100, 100, 0],
                [100, 100, 0, 1],
                [100, 0, 1, 2],
                [0, 1, 2, 3],
                [1, 2, 3, 4],
                [2, 3, 4, 5],
                [3, 4, 5, 6],
                [4, 5, 6, 7],
                [5, 6, 7, 8],
                [6, 7, 8, 9],
            ]
        ),
    )


def test_padded_moving_window_axis():
    a = jnp.arange(10).reshape((2, 5))
    moved = padded_moving_window(a, 3, padding=100, axis=1)
    assert moved.shape == (2, 5, 3)
    assert jnp.allclose(
        moved,
        jnp.array(
            [
                [[100, 100, 0], [100, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4]],
                [[100, 100, 5], [100, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9]],
            ]
        ),
    )
