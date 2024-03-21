import jax
import numpy as np

from levanter.utils.jax_utils import best_effort_sharding
from test_utils import skip_if_not_enough_devices


def _assert_can_put_with_sharding(array, sharding):
    try:
        jax.device_put(array, sharding)
    except ValueError:
        assert False, f"Could not put array with shape {array.shape} with sharding {sharding}"


@skip_if_not_enough_devices(8)
def test_best_effort_sharding():
    # 1D array, 8 devices
    array = np.arange(8)
    sharding = best_effort_sharding(array.shape)
    _assert_can_put_with_sharding(array, sharding)

    array = array.reshape(2, 4)
    sharding = best_effort_sharding(array.shape)
    _assert_can_put_with_sharding(array, sharding)

    array = array.reshape(4, 2)
    sharding = best_effort_sharding(array.shape)
    _assert_can_put_with_sharding(array, sharding)

    array = array.reshape(8, 1)
    sharding = best_effort_sharding(array.shape)
    _assert_can_put_with_sharding(array, sharding)

    array = array.reshape(1, 8)
    sharding = best_effort_sharding(array.shape)
    _assert_can_put_with_sharding(array, sharding)

    array = array.reshape(2, 2, 2)
    sharding = best_effort_sharding(array.shape)
    _assert_can_put_with_sharding(array, sharding)

    devices = jax.devices()[:6]

    array = array.reshape(2, 2, 2)
    sharding = best_effort_sharding(array.shape, devices=devices)
    _assert_can_put_with_sharding(array, sharding)
