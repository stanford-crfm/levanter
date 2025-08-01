import jax.numpy as jnp
from levanter.optim.model_averaging import EmaDecaySqrtModelAveraging


def _dummy(v):
    return {"p": jnp.array(v, jnp.float32)}


def test_ema_phase():
    ma = EmaDecaySqrtModelAveraging(model=_dummy(0.0), beta=0.5, switch_step=5, decay_steps=10)
    ma = ma.update(_dummy(1.0), step=0)
    assert jnp.allclose(ma.model["p"], 0.5)


def test_decay_phase():
    ma = EmaDecaySqrtModelAveraging(
        model=_dummy(0.0), beta=0.0, switch_step=0, decay_steps=4, epsilon=1e-5
    )
    ma = ma.update(_dummy(1.0), step=0)
    ma_end = ma.update(_dummy(10.0), step=4)
    expected = (1.0 - ma.epsilon) * 1.0 + ma.epsilon * 10.0
    assert jnp.allclose(ma_end.model["p"], expected)
