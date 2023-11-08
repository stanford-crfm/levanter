import logging
import os
from pathlib import Path
from typing import Any, Optional, Union

from levanter.tracker import Tracker


pylogger = logging.getLogger(__name__)


class TensorboardTracker(Tracker):
    def __init__(self, logdir: Union[str, Path]):
        self.logdir = logdir
        self.writer = None

    def init(self, run_id: Optional[str]):
        from tensorboardX import SummaryWriter

        dir_to_write = self.logdir
        if run_id is not None:
            dir_to_write = os.path.join(dir_to_write, run_id)
        self.writer = SummaryWriter(dir_to_write)

    def log_hyperparameters(self, hparams: dict[str, Any]):
        if self.writer is None:
            raise RuntimeError("Must call init before logging metrics")

        self.writer.add_hparams(hparams, {"dummy": 0})

    def log(self, metrics: dict[str, Any], *, step, commit=None):
        del commit
        if self.writer is None:
            raise RuntimeError("Must call init before logging metrics")

        for k, v in metrics.items():
            self.writer.add_scalar(k, v, step)

    def log_summary(self, metrics: dict[str, Any]):
        if self.writer is None:
            raise RuntimeError("Must call init before logging metrics")

        for k, v in metrics.items():
            self.writer.add_scalar(k, v, global_step=None)

    def log_artifact(self, artifact, *, name: Optional[str] = None, type: Optional[str] = None):
        pylogger.warning("TensorboardLogger does not support logging artifacts yet")
        pass
