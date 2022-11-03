import datetime
import pathlib
import tempfile
from datetime import timedelta

from levanter.checkpoint import Checkpointer, CheckpointInterval, load_metadata
from levanter.trainer_hooks import StepInfo


def _dummy_step_info(step):
    return StepInfo(
        loss=0.0,
        step=step,
        model=None,
        opt_state=(),
        next_key=(),
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
