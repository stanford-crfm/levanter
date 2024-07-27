import contextlib
import dataclasses
import logging
import sys
from dataclasses import dataclass
from typing import Optional

import ray
import tblib


@dataclass
class ExceptionInfo:
    ex: Optional[BaseException]
    tb: tblib.Traceback

    def restore(self):
        if self.ex is not None:
            exc_value = self.ex.with_traceback(self.tb.as_traceback())
            return (self.ex.__class__, exc_value, self.tb.as_traceback())
        else:
            return (Exception, Exception("Process failed with no exception"), self.tb.as_traceback())

    def reraise(self):
        if self.ex is not None:
            raise self.ex.with_traceback(self.tb.as_traceback())
        else:
            raise Exception("Process failed with no exception").with_traceback(self.tb.as_traceback())


@dataclass
class RayResources:
    num_cpus: int = 1
    num_gpus: int = 0
    resources: dict = dataclasses.field(default_factory=dict)

    def to_kwargs(self):
        return dict(num_cpus=self.num_cpus, num_gpus=self.num_gpus, resources=self.resources)

    def to_resource_dict(self):
        return dict(CPU=self.num_cpus, GPU=self.num_gpus, **self.resources)

    @staticmethod
    def from_resource_dict(resources: dict):
        return RayResources(num_cpus=resources.get("CPU", 0), num_gpus=resources.get("GPU", 0), resources=resources)


@dataclass
class RefBox:
    """Ray doesn't dereference ObjectRefs if they're nested in another object. So we use this to take advantage of that.
    https://docs.ray.io/en/latest/ray-core/objects.html#passing-object-arguments"""

    ref: ray.ObjectRef


class DoneSentinel:
    pass


DONE = DoneSentinel()


def ser_exc_info(exception=None) -> ExceptionInfo:
    if exception is None:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb = tblib.Traceback(exc_traceback)
        return ExceptionInfo(exc_value, tb)
    else:
        tb = exception.__traceback__
        tb = tblib.Traceback(tb)
        return ExceptionInfo(exception, tb)


def current_actor_handle() -> ray.actor.ActorHandle:
    return ray.runtime_context.get_runtime_context().current_actor


class SnitchRecipient:
    logger: logging.Logger

    def _child_failed(self, child: ray.actor.ActorHandle, exception: ExceptionInfo):
        info = exception.restore()
        self.logger.error(f"Child {child} failed with exception {info[1]}", exc_info=info)
        exception.reraise()


@contextlib.contextmanager
def log_failures_to(parent, suppress=False):
    # parent is actorref of SnitchRecipient
    try:
        yield
    except Exception as e:
        parent._child_failed.remote(current_actor_handle(), ser_exc_info(e))
        if not suppress:
            raise e
