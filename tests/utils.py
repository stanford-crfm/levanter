import jax
import pytest


def skip_if_not_enough_devices(count: int):
    return pytest.mark.skipif(len(jax.devices()) < count, reason=f"Not enough devices ({len(jax.devices())})")
