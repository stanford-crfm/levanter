import jax
import numpy as np
import pytest

from levanter.utils.jax_utils import best_effort_sharding, create_fsdp_mesh
from test_utils import skip_if_not_enough_devices


def _assert_can_put_with_sharding(array, sharding):
    try:
        jax.device_put(array, sharding)
    except ValueError:
        # assert False, f"Could not put array with shape {array.shape} with sharding {sharding}"
        raise AssertionError(f"Could not put array with shape {array.shape} with sharding {sharding}")


@skip_if_not_enough_devices(8)
def test_best_effort_sharding():
    if len(jax.devices()) % 8 != 0:
        pytest.skip("Not enough devices")
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


@pytest.mark.parametrize("fsdp_size", [1, 2, 4, 8])
def test_best_effort_sharding_with_mesh(fsdp_size):
    if fsdp_size > len(jax.devices()):
        pytest.skip("Not enough devices")
    elif len(jax.devices()) % fsdp_size != 0:
        pytest.skip("Number of devices is not a multiple of fsdp_size")

    mesh = create_fsdp_mesh(len(jax.devices()) // fsdp_size, fsdp_size, 1)

    array = np.arange(8)
    sharding = best_effort_sharding(array.shape, mesh=mesh)
    _assert_can_put_with_sharding(array, sharding)

    array = array.reshape(2, 4)
    sharding = best_effort_sharding(array.shape, mesh=mesh)
    _assert_can_put_with_sharding(array, sharding)

    array = array.reshape(4, 2)
    sharding = best_effort_sharding(array.shape, mesh=mesh)
    _assert_can_put_with_sharding(array, sharding)

    array = array.reshape(8, 1)
    sharding = best_effort_sharding(array.shape, mesh=mesh)
    _assert_can_put_with_sharding(array, sharding)

    array = array.reshape(1, 8)
    sharding = best_effort_sharding(array.shape, mesh=mesh)
    _assert_can_put_with_sharding(array, sharding)

    array = array.reshape(2, 2, 2)
    sharding = best_effort_sharding(array.shape, mesh=mesh)
    _assert_can_put_with_sharding(array, sharding)
