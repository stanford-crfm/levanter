import jax.numpy as jnp
import numpy as np

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
        moving_window(a, 4),
        jnp.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]]),
    )


def test_moving_window_axis():
    a = jnp.arange(10).reshape((2, 5))
    moved = moving_window(a, 3, axis=1)
    assert moved.shape == (3, 2, 3)

    np_a = jnp.arange(10).reshape((2, 5))
    np_sliding = np.lib.stride_tricks.sliding_window_view(np_a, 3, axis=1)
    # np_sliding.shape == (2, 3, 3)
    np_sliding = np_sliding.transpose((2, 0, 1))

    assert jnp.allclose(moved, np_sliding)

    a = jnp.arange(24).reshape((2, 3, 4))

    np_a = np.arange(24).reshape((2, 3, 4))

    moved = moving_window(a, 2, axis=1)
    np_sliding = np.lib.stride_tricks.sliding_window_view(np_a, 2, axis=1)
    # np_sliding.shape == (2, 2, 4, 2) because window dim comes last
    np_sliding = np_sliding.transpose((3, 0, 1, 2))
    assert moved.shape == (2, 2, 2, 4)
    assert jnp.allclose(moved, np_sliding)


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
    np_a = np.arange(10).reshape((2, 5))
    moved = padded_moving_window(a, 3, padding=100, axis=1)
    np_moved = np.lib.stride_tricks.sliding_window_view(
        np.pad(np_a, ((0, 0), (5 - 3, 0)), "constant", constant_values=100), 3, axis=1
    )

    np_moved = np_moved.transpose((1, 0, 2))

    assert moved.shape == (5, 2, 3)
    assert jnp.allclose(
        moved,
        np_moved,
        # jnp.array(
        #     [
        #         [[100, 100, 0], [100, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4]],
        #         [[100, 100, 5], [100, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9]],
        #     ]
        # ),
    )
