import datetime
import pathlib
import tempfile
from datetime import timedelta

import equinox as eqx
import jax
import numpy as np
import optax
from chex import assert_trees_all_close
from jax import numpy as jnp

from levanter.checkpoint import (
    Checkpointer,
    CheckpointInterval,
    discover_latest_checkpoint,
    load_checkpoint,
    load_metadata,
    save_checkpoint,
)
from levanter.trainer import StepInfo, TrainerState
from test_utils import MLP, arrays_only, assert_trees_not_close


def _dummy_step_info(step):
    return StepInfo(
        state=TrainerState(
            # + 1 b/c step here is next step
            step=step + 1,
            model=None,
            opt_state=(),
            training_key=(),
        ),
        loss=0.0,
        step_duration=0.0,
    )


def _get_checkpoint_steps(checkpoint_dir):
    paths = list(pathlib.Path(checkpoint_dir).iterdir())
    return sorted([load_metadata(f)["step"] for f in paths])


def test_checkpointer_changing_policy():
    with tempfile.TemporaryDirectory(prefix="checkpoints") as tmpdir:
        checkpointer = Checkpointer(
            tmpdir,
            None,
            [
                CheckpointInterval(every=2, until=10),
                CheckpointInterval(every=5, until=20),
                CheckpointInterval(every=10, until=None),
            ],
        )

        for step in range(1, 50):
            checkpointer.on_step(_dummy_step_info(step))

        # ensure we saved the right checkpoints
        assert _get_checkpoint_steps(tmpdir) == [2, 4, 6, 8, 10, 15, 20, 30, 40]


def test_checkpointer_temporal_policy():
    fake_now = datetime.datetime(2021, 1, 1, 0, 0, 0)

    tick = 10

    def advance_time(delta_seconds):
        nonlocal fake_now
        fake_now += timedelta(seconds=delta_seconds)

    with tempfile.TemporaryDirectory(prefix="checkpoints") as tmpdir:
        checkpointer = Checkpointer(tmpdir, timedelta(seconds=tick), [], dt_now_injection=lambda: fake_now)

        checkpointer.on_step(_dummy_step_info(0))
        advance_time(tick)
        checkpointer.on_step(_dummy_step_info(1))
        assert _get_checkpoint_steps(tmpdir) == [1]

        advance_time(tick - 1)
        checkpointer.on_step(_dummy_step_info(2))
        assert _get_checkpoint_steps(tmpdir) == [1]
        advance_time(1)
        checkpointer.on_step(_dummy_step_info(3))
        assert _get_checkpoint_steps(tmpdir) == [3]


def test_checkpointer_mixed_policy():
    fake_now = datetime.datetime(2021, 1, 1, 0, 0, 0)

    tick = 10

    def advance_time(delta_seconds):
        nonlocal fake_now
        fake_now += timedelta(seconds=delta_seconds)

    with tempfile.TemporaryDirectory(prefix="checkpoints") as tmpdir:
        checkpointer = Checkpointer(
            tmpdir,
            timedelta(seconds=tick),
            [
                CheckpointInterval(every=2, until=10),
                CheckpointInterval(every=5, until=20),
                CheckpointInterval(every=10, until=None),
            ],
            dt_now_injection=lambda: fake_now,
        )

        checkpointer.on_step(_dummy_step_info(0))
        advance_time(tick)
        checkpointer.on_step(_dummy_step_info(1))
        assert _get_checkpoint_steps(tmpdir) == [1]

        advance_time(tick - 1)
        # time hasn't advanced enough, so we wouldn't save a checkpoint, but we do because of the interval
        checkpointer.on_step(_dummy_step_info(2))
        assert _get_checkpoint_steps(tmpdir) == [2]

        advance_time(1)
        # time has advanced enough now from last temporal save, but we don't save a checkpoint because we just saved one
        checkpointer.on_step(_dummy_step_info(3))
        assert _get_checkpoint_steps(tmpdir) == [2]

        for step in range(4, 11):
            advance_time(tick)
            checkpointer.on_step(_dummy_step_info(step))

        assert _get_checkpoint_steps(tmpdir) == [2, 4, 6, 8, 10]

        advance_time(tick - 1)
        checkpointer.on_step(_dummy_step_info(11))
        assert _get_checkpoint_steps(tmpdir) == [2, 4, 6, 8, 10]

        for step in range(12, 50):
            checkpointer.on_step(_dummy_step_info(step))
            advance_time(tick)

        # ensure we saved the right checkpoints
        assert _get_checkpoint_steps(tmpdir) == [2, 4, 6, 8, 10, 15, 20, 30, 40, 49]  # 49 is last temporary checkpoint


def test_checkpoint_simple():
    key0 = jax.random.PRNGKey(0)
    key1 = jax.random.PRNGKey(1)

    def make_state(key):
        model = MLP(in_size=2, out_size=1, width_size=2, depth=3, key=key)
        optim = optax.adam(1e-4)
        opt_state = optim.init(arrays_only(model))

        return model, opt_state, key

    initial_model, initial_opt_state, initial_key = make_state(key0)
    rep_model, rep_state, rep_key = make_state(key1)

    assert_trees_not_close(initial_model, rep_model)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_checkpoint(
            initial_model,
            (initial_opt_state, initial_key),
            step=10,
            checkpoint_path=tmpdir,
            exist_ok=True,
        )
        restored_model, (restored_optstate, rkey), step = load_checkpoint(
            rep_model,
            (rep_state, rep_key),
            checkpoint_path=tmpdir,
            discover_latest=False,
        )

        assert_trees_all_close(
            jax.tree_util.tree_leaves(arrays_only(restored_model)),
            jax.tree_util.tree_leaves(arrays_only(initial_model)),
        )
        assert all(np.isclose(rkey, initial_key))
        assert step == 10


def test_checkpoint_steps():
    key0 = jax.random.PRNGKey(0)
    key1 = jax.random.PRNGKey(1)

    optim = optax.adam(1e-4)

    def make_state(key):
        model = MLP(in_size=2, out_size=1, width_size=2, depth=3, key=key)
        opt_state = optim.init(arrays_only(model))

        return model, opt_state, key

    initial_model, initial_opt_state, initial_key = make_state(key0)
    data = jax.random.uniform(key0, (2, 2))

    @eqx.filter_grad
    def loss_fn(model, data):
        m = jax.vmap(model)
        return jnp.mean(jnp.square(m(data)))

    model, state = initial_model, initial_opt_state
    for i in range(3):
        grad = loss_fn(model, data)
        updates, state = optim.update(grad, state)
        model = eqx.apply_updates(model, updates)

    assert_trees_not_close(model, initial_model)
    assert_trees_not_close(state, initial_opt_state)

    rep_model, rep_state, rep_key = make_state(key1)
    assert_trees_not_close(model, rep_model)
    assert_trees_not_close(state, rep_state)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_checkpoint(model, state, step=3, checkpoint_path=tmpdir, exist_ok=True)
        restored_model, restored_optstate, step = load_checkpoint(
            rep_model, rep_state, checkpoint_path=tmpdir, discover_latest=False
        )

        assert_trees_all_close(
            jax.tree_util.tree_leaves(arrays_only(restored_model)),
            jax.tree_util.tree_leaves(arrays_only(model)),
        )
        assert_trees_all_close(
            jax.tree_util.tree_leaves(arrays_only(restored_optstate)),
            jax.tree_util.tree_leaves(arrays_only(state)),
        )
        assert step == 3


def test_checkpoint_discovery():
    with tempfile.TemporaryDirectory() as tempdir:
        save_checkpoint(model=1, training_state=2, step=10, checkpoint_path=f"{tempdir}/step-10")
        save_checkpoint(model=3, training_state=4, step=20, checkpoint_path=f"{tempdir}/step-20")
        save_checkpoint(model=5, training_state=6, step=30, checkpoint_path=f"{tempdir}/step-30")

        latest = discover_latest_checkpoint(tempdir)
        assert latest == f"{tempdir}/step-30"

        assert discover_latest_checkpoint("file:///tmp/does-not-exist") is None
