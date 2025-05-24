import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from levanter.optim.config import AdamConfig
from levanter.optim.skipstep import SkipStepConfig, SkipStepState


def test_no_stable_weirdness():
    optimizer = AdamConfig(
        learning_rate=2e-6,  # 2x10^-6
        weight_decay=0.0,
        warmup=0.03,
        min_lr_ratio=0.0,
        lr_schedule="linear",
        max_grad_norm=None,
        cycles=None,
        weight_decay_modules=None,
        default_weight_decay_mask=None,
    )

    sched_fn = optimizer.lr_scheduler(861)

    assert sched_fn(0) == 0.0
    assert np.isclose(sched_fn(int(861 * 0.03)), 2e-6)
    assert np.isclose(sched_fn(int(860)), 0.0)

    # get a middle value
    mid_cooldown = 0.03 + 0.97 / 2
    assert np.isclose(sched_fn(int(861 * mid_cooldown)), 2e-6 / 2)


def test_constant_schedule():
    optimizer = AdamConfig(
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup=0.0,
        min_lr_ratio=1.0,  # No decay
        lr_schedule="constant",
        cycles=None,
    )

    sched_fn = optimizer.lr_scheduler(1000)

    assert sched_fn(0) == 1e-3
    assert sched_fn(500) == 1e-3
    assert sched_fn(999) == 1e-3


def test_warmup_and_cosine_decay():
    optimizer = AdamConfig(
        learning_rate=1e-2,
        weight_decay=0.0,
        warmup=0.1,  # 10% of steps
        min_lr_ratio=0.1,
        lr_schedule="cosine",
        cycles=None,
    )

    sched_fn = optimizer.lr_scheduler(1000)

    # Warmup phase
    assert np.isclose(sched_fn(0), 0.0)
    assert np.isclose(sched_fn(50), 0.5e-2)
    assert np.isclose(sched_fn(100), 1e-2)

    # Decay phase
    assert np.isclose(sched_fn(999), 1e-3, atol=1e-5)


def test_linear_schedule_with_cycles():
    optimizer = AdamConfig(
        learning_rate=5e-4,
        weight_decay=0.0,
        warmup=50,
        min_lr_ratio=0.2,
        lr_schedule="linear",
        cycles=2,
    )

    sched_fn = optimizer.lr_scheduler(1000)

    # Warmup phase
    assert np.isclose(sched_fn(0), 0.0)
    assert np.isclose(sched_fn(50), 5e-4)

    num_main_steps = 1000

    first_nadir = num_main_steps // 2 - 1

    # First cycle decay
    assert np.isclose(sched_fn(first_nadir), 0.2 * 5e-4, atol=1e-5)

    # Second cycle starts
    assert np.isclose(sched_fn(first_nadir + 1), 5e-4)

    # midway through second cycle
    midpoint = first_nadir + num_main_steps // 4
    assert np.isclose(sched_fn(midpoint), (5e-4 + 0.2 * 5e-4) / 2, atol=1e-5)

    # Final value
    assert np.isclose(sched_fn(999), 0.2 * 5e-4, atol=1e-5)


def test_wsds_schedule():
    optimizer = AdamConfig(
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup=0.0,
        decay=0.1,
        min_lr_ratio=0.1,
        lr_schedule="cosine",
        cycles=[300, 700],
    )

    sched_fn = optimizer.lr_scheduler(1000)

    # First cycle
    assert np.isclose(sched_fn(0), 1e-3)
    assert np.isclose(sched_fn(269), 1e-3)
    assert sched_fn(271) < 1e-3

    # Second cycle
    assert np.isclose(sched_fn(300), 1e-3)
    assert np.isclose(sched_fn(659), 1e-3)
    assert sched_fn(661) < 1e-3

    # Third cycle
    assert np.isclose(sched_fn(701), 1e-3)
    assert np.isclose(sched_fn(969), 1e-3)
    assert sched_fn(971) < 1e-3


# Tests for SkipStepConfig


def test_skip_step_config_defaults_and_optimizer_config_field():
    # Test OptimizerConfig with default skip_step = None
    adam_config_default = AdamConfig()
    assert adam_config_default.skip_step is None

    # Test SkipStepConfig defaults
    skip_config_default = SkipStepConfig()
    assert skip_config_default.rolling_interval_length == 128
    assert skip_config_default.sigma_factor == 6.0

    # Test OptimizerConfig with SkipStepConfig instance
    skip_config_custom = SkipStepConfig(rolling_interval_length=64, sigma_factor=3.0)
    adam_config_with_skip = AdamConfig(skip_step=skip_config_custom)
    assert adam_config_with_skip.skip_step is skip_config_custom
    assert adam_config_with_skip.skip_step.rolling_interval_length == 64
    assert adam_config_with_skip.skip_step.sigma_factor == 3.0


# Mock optimizer for testing SkipStepConfig.wrap
# A simple optimizer that just scales gradients by -1 and maintains a count state
def mock_optimizer_transform() -> optax.GradientTransformation:
    def init_fn(params):
        return {"count": jnp.array(0, dtype=jnp.int32)}

    def update_fn(updates, state, params=None):
        new_state = {"count": state["count"] + 1}
        # simple transformation: negate gradients
        transformed_updates = jax.tree_util.tree_map(lambda g: -g, updates)
        return transformed_updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


def test_skip_step_config_wrap_basic_setup():
    mock_optimizer = mock_optimizer_transform()
    skip_config = SkipStepConfig()

    wrapped_optimizer = skip_config.wrap(mock_optimizer)

    assert isinstance(wrapped_optimizer, optax.GradientTransformation)
    # Check if it has init and update attributes
    assert hasattr(wrapped_optimizer, "init")
    assert hasattr(wrapped_optimizer, "update")


def test_skip_step_init_fn():
    mock_optimizer = mock_optimizer_transform()
    skip_config = SkipStepConfig(rolling_interval_length=10)  # Use a smaller interval for easier testing
    wrapped_optimizer = skip_config.wrap(mock_optimizer)

    dummy_params = {"w": jnp.array([1.0, 2.0])}
    state: SkipStepState = wrapped_optimizer.init(dummy_params)

    # Check inner_opt_state
    expected_inner_state = mock_optimizer.init(dummy_params)
    chex.assert_trees_all_close(state.inner_opt_state, expected_inner_state)

    # Check losses and grad_norms arrays
    assert state.losses.shape == (skip_config.rolling_interval_length + 1,)
    assert state.grad_norms.shape == (skip_config.rolling_interval_length + 1,)
    assert jnp.all(jnp.isnan(state.losses))
    assert jnp.all(jnp.isnan(state.grad_norms))

    # Check current_idx and count
    chex.assert_trees_all_close(state.current_idx, jnp.array(0, dtype=jnp.int32))
    chex.assert_trees_all_close(state.count, jnp.array(0, dtype=jnp.int32))


def test_skip_step_update_fn_loss_kwarg_requirement():
    mock_optimizer = mock_optimizer_transform()
    skip_config = SkipStepConfig()
    wrapped_optimizer = skip_config.wrap(mock_optimizer)

    dummy_params = {"w": jnp.array([1.0, 2.0])}
    dummy_grads = {"w": jnp.array([-0.1, 0.1])}
    initial_state = wrapped_optimizer.init(dummy_params)

    with pytest.raises(ValueError, match="Loss must be provided"):
        wrapped_optimizer.update(dummy_grads, initial_state, dummy_params)

    # Test that it works if loss is provided
    try:
        wrapped_optimizer.update(dummy_grads, initial_state, dummy_params, loss=jnp.array(1.0))
    except ValueError:
        pytest.fail("ValueError raised unexpectedly when loss was provided.")


@pytest.fixture
def initialized_wrapped_optimizer():
    mock_opt = mock_optimizer_transform()
    skip_conf = SkipStepConfig(rolling_interval_length=10)  # smaller interval for tests
    wrapped_opt = skip_conf.wrap(mock_opt)
    dummy_params = {"w": jnp.array([1.0, 2.0, 3.0]), "b": jnp.array([0.5])}
    initial_state = wrapped_opt.init(dummy_params)
    return wrapped_opt, initial_state, dummy_params, skip_conf


def test_update_fn_initial_phase_not_enough_data(initialized_wrapped_optimizer):
    wrapped_optimizer, state, params, config = initialized_wrapped_optimizer

    num_initial_steps = config.rolling_interval_length // 2 - 1
    if num_initial_steps < 1:  # ensure at least one step for the test logic
        num_initial_steps = 1

    dummy_updates = {"w": jnp.array([0.1, 0.2, 0.3]), "b": jnp.array([0.05])}
    loss_val = jnp.array(10.0)

    current_state = state
    for i in range(num_initial_steps):
        updates, new_state = wrapped_optimizer.update(dummy_updates, current_state, params, loss=loss_val + i)

        # Assert step_factor was 1.0 by checking if inner optimizer was applied
        # Mock optimizer negates gradients
        expected_inner_updates = jax.tree_util.tree_map(lambda g: -g, dummy_updates)
        chex.assert_trees_all_close(updates, expected_inner_updates, atol=1e-5)

        # Check state updates
        assert new_state.count == i + 1
        assert new_state.current_idx == (i + 1) % (config.rolling_interval_length + 1)
        assert np.isclose(new_state.losses[i], loss_val + i)
        expected_gnorm = optax.global_norm(dummy_updates)
        assert np.isclose(new_state.grad_norms[i], expected_gnorm)

        current_state = new_state

    # Check that after these initial steps, count is num_initial_steps
    assert current_state.count == num_initial_steps
    # And current_idx has advanced
    assert current_state.current_idx == num_initial_steps % (config.rolling_interval_length + 1)


def test_inv_sqrt_decay_schedule():
    optimizer = AdamConfig(
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup=0.1,
        min_lr_ratio=0.1,
        lr_schedule="inv_sqrt",
        cycles=None,
    )

    sched_fn = optimizer.lr_scheduler(100_000)

    # Warmup phase
    assert np.isclose(sched_fn(0), 0.0)
    assert np.isclose(sched_fn(5000), 0.5e-3)

    # Decay phase: our invsqrt has a non configurable, very long period
    assert sched_fn(50000) < sched_fn(30000)  # Decreasing after warmup


def test_rewarmup_schedule():
    optimizer = AdamConfig(
        learning_rate=1e-2,
        weight_decay=0.0,
        warmup=0.2,  # 20% of cycle
        min_lr_ratio=0.2,
        lr_schedule="linear",
        cycles=2,
        rewarmup=0.05,  # 5% of steps in each cycle
    )

    # cycle length is 500 steps
    sched_fn = optimizer.lr_scheduler(1000)

    # Warmup phase
    assert np.isclose(sched_fn(0), 0.0)
    assert np.isclose(sched_fn(100), 1e-2)  # Warmup reaches max LR

    # First decay phase
    assert np.isclose(sched_fn(300), 0.6e-2)  # Mid of first decay
    assert np.isclose(sched_fn(500), 0.2e-2)  # End of first decay

    # Rewarmup at start of second cycle
    rewarmup_start = 500
    rewarmup_end = rewarmup_start + int(0.05 * 500)
    assert np.isclose(sched_fn(rewarmup_start), 0.2e-2)  # End of previous decay
    assert np.isclose(sched_fn(rewarmup_end), 1e-2)  # Back to max LR after rewarmup
    # make sure this is the high point
    assert sched_fn(rewarmup_end - 1) < sched_fn(rewarmup_end)
    assert sched_fn(rewarmup_end + 1) < sched_fn(rewarmup_end)

    # Final decay phase
    assert sched_fn(999 - 1) > sched_fn(999)
    assert np.isclose(sched_fn(999), 0.2e-2, atol=1e-4)  # End of second decay


def test_linear_schedule_with_cycle_length():
    optimizer = AdamConfig(
        learning_rate=5e-4,
        weight_decay=0.0,
        warmup=50,
        min_lr_ratio=0.2,
        lr_schedule="linear",
        cycle_length=500,
    )

    sched_fn = optimizer.lr_scheduler(1000)

    # Warmup phase
    assert np.isclose(sched_fn(0), 0.0)
    assert np.isclose(sched_fn(50), 5e-4)

    num_main_steps = 1000

    # First cycle decay
    assert np.isclose(sched_fn(499), 0.2 * 5e-4, atol=1e-5)

    # Second cycle starts
    assert np.isclose(sched_fn(500), 5e-4)

    # midway through second cycle
    midpoint = 500 - 1 + num_main_steps // 4
    assert np.isclose(sched_fn(midpoint), (5e-4 + 0.2 * 5e-4) / 2, atol=1e-5)

    # Final value
    assert np.isclose(sched_fn(999), 0.2 * 5e-4, atol=1e-5)


def test_wsds_schedule_with_cycle_points():
    optimizer = AdamConfig(
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup=0.0,
        decay=0.1,
        min_lr_ratio=0.1,
        lr_schedule="cosine",
        cycle_length=[300, 400],
    )

    sched_fn = optimizer.lr_scheduler(1000)

    # First cycle
    assert np.isclose(sched_fn(0), 1e-3)
    assert np.isclose(sched_fn(269), 1e-3)
    assert sched_fn(271) < 1e-3

    # Second cycle
    assert np.isclose(sched_fn(300), 1e-3)
    assert np.isclose(sched_fn(659), 1e-3)
    assert sched_fn(661) < 1e-3

    # Third cycle
    assert np.isclose(sched_fn(701), 1e-3)
    assert np.isclose(sched_fn(969), 1e-3)
    assert sched_fn(971) < 1e-3
