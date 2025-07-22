import contextlib
import dataclasses
import logging
import logging as pylogging
import sys
from dataclasses import dataclass
from typing import Optional

import ray
import tblib
from ray.runtime_env import RuntimeEnv
import mergedeep


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


@dataclass(frozen=True)
class RayResources:
    """
    A dataclass that represents the resources for a ray task or actor. It's main use is to be
    fed to ray.remote() to specify the resources for a task.
    """

    num_cpus: int = 1
    num_gpus: int = 0
    resources: dict = dataclasses.field(default_factory=dict)
    runtime_env: RuntimeEnv = dataclasses.field(default_factory=RuntimeEnv)
    memory: int | None = None
    object_store_memory: int | None = None
    accelerator_type: Optional[str] = None

    def to_kwargs(self):
        """
        Returns a dictionary of kwargs that can be passed to ray.remote() to specify the resources for a task.
        """
        out = dict(
            num_cpus=self.num_cpus, num_gpus=self.num_gpus, resources=self.resources, runtime_env=self.runtime_env
        )

        if self.accelerator_type is not None:
            out["accelerator_type"] = self.accelerator_type

        return out

    def to_resource_dict(self):
        """
        Returns a dictionary of resources that describe the resources.

        Note that Ray for some reason doesn't like passing resources={"GPU": 1} to a remote so you can't use this
        for that purpose. Instead, use @ray.remote(**ray_resources.to_kwargs())
        """
        out = dict(CPU=self.num_cpus, GPU=self.num_gpus, **self.resources)
        # https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html#accelerator-types
        if self.accelerator_type is not None:
            out[f"accelerator_type:{self.accelerator_type}"] = 0.001
        return out

    @staticmethod
    def from_resource_dict(resources: dict):
        resources = dict(resources)  # Ensure we have a mutable copy
        num_cpus = resources.pop("CPU", 1)
        num_gpus = resources.pop("GPU", 0)
        memory = resources.pop("memory", None)
        object_store_memory = resources.pop("object_store_memory", None)
        return RayResources(num_cpus=num_cpus, num_gpus=num_gpus, memory=memory, object_store_memory=object_store_memory,
                            resources=resources)

    def merge_env_vars(self, env_vars: dict | None):
        """
        Return a new ``RayResources`` with the supplied ``env_vars`` merged into
        its ``runtime_env``.  The merge strategy is additive – existing
        environment variables are preserved unless explicitly overridden by the
        provided ``env_vars``.  If ``env_vars`` is ``None`` or empty, the
        original instance is returned unchanged.

        This is a convenience method that calls ``merge_runtime_env`` with
        a runtime environment containing only the specified environment variables.
        """

        # Short-circuit if there is nothing to merge.
        if not env_vars:
            return self

        # Create a runtime environment dict with just the env_vars
        runtime_env_with_vars = {"env_vars": env_vars}

        # Delegate to the more general merge_runtime_env method
        return self.merge_runtime_env(runtime_env_with_vars)

    def merge_runtime_env(self, runtime_env: dict | RuntimeEnv | None):
        """
        Return a new ``RayResources`` with the supplied ``runtime_env`` merged into
        its existing ``runtime_env``.  The merge strategy is additive – existing
        runtime environment fields are preserved unless explicitly overridden by the
        provided ``runtime_env``.  If ``runtime_env`` is ``None`` or empty, the
        original instance is returned unchanged.

        This handles merging of all runtime environment fields (env_vars, pip, conda, working_dir, etc.)
        not just environment variables.
        """

        # Short-circuit if there is nothing to merge.
        if not runtime_env:
            return self

        # Convert both runtime environments to plain dicts for merging.
        if isinstance(self.runtime_env, RuntimeEnv):
            base_runtime_env: dict = self.runtime_env.to_dict()
        else:
            base_runtime_env = dict(self.runtime_env or {})

        if isinstance(runtime_env, RuntimeEnv):
            merge_runtime_env: dict = runtime_env.to_dict()
        else:
            merge_runtime_env = dict(runtime_env or {})

        # Perform an additive deep merge of the runtime environments.
        merged_runtime_env = mergedeep.merge(
            {},
            base_runtime_env,
            merge_runtime_env,
            strategy=mergedeep.Strategy.ADDITIVE,
        )

        # Attempt to reconstruct a ``RuntimeEnv``; if this fails (e.g., because
        # unsupported fields were introduced) fall back to the raw dict.
        try:
            new_runtime_env = RuntimeEnv(**merged_runtime_env)
        except Exception:  # noqa: BLE001 – best-effort conversion only
            new_runtime_env = merged_runtime_env

        # Return a *new* RayResources instance with the updated runtime_env so
        # that the original remains unchanged/immutable.
        return dataclasses.replace(self, runtime_env=new_runtime_env)


@dataclass
class RefBox:
    """Ray doesn't dereference ObjectRefs if they're nested in another object. So we use this to take advantage of that.
    https://docs.ray.io/en/latest/ray-core/objects.html#passing-object-arguments"""

    ref: ray.ObjectRef

    def get(self):
        return ray.get(self.ref)


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

    def _child_failed(self, child: ray.actor.ActorHandle | str | None, exception: ExceptionInfo):
        info = exception.restore()
        self.logger.error(f"Child {child} failed with exception {info[1]}", exc_info=info)
        exception.reraise()


@contextlib.contextmanager
def log_failures_to(parent, suppress=False):
    # parent is actorref of SnitchRecipient
    try:
        yield
    except Exception as e:
        try:
            handle = current_actor_handle()
        except RuntimeError:
            handle = ray.runtime_context.get_runtime_context().get_task_id()

        parent._child_failed.remote(handle, ser_exc_info(e))
        if not suppress:
            raise e


DEFAULT_LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s %(levelname)s: %(message)s"


@ray.remote
class StopwatchActor:
    def __init__(self):
        pylogging.basicConfig(level=DEFAULT_LOG_LEVEL, format=LOG_FORMAT)
        self._logger = pylogging.getLogger("StopwatchActor")
        self._times_per = {}
        self._counts_per = {}
        self._total = 0

    def measure(self, name: str, time: float):
        self._times_per[name] = self._times_per.get(name, 0) + time
        self._counts_per[name] = self._counts_per.get(name, 0) + 1
        self._total += 1

        if self._total % 1000 == 0:
            for name, time in self._times_per.items():
                self._logger.info(f"{name}: {time / self._counts_per[name]}")

    def get(self, name: str):
        return self._times_per.get(name, 0), self._counts_per.get(name, 0)

    def average(self, name: str):
        return self._times_per.get(name, 0) / self._counts_per.get(name, 1)
