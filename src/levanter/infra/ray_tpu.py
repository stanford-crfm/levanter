import dataclasses
import functools
import logging
import multiprocessing
import os
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass
from queue import Empty as QueueEmpty
from typing import Callable, Optional, Sequence

import draccus
import mergedeep
import ray
from ray._private.accelerators import TPUAcceleratorManager
from ray.actor import ActorHandle
from ray.dag import FunctionNode
from ray.dashboard.modules.job.sdk import JobSubmissionClient
from ray.exceptions import (
    ActorDiedError,
    ActorUnavailableError,
    NodeDiedError,
    RayActorError,
    RayError,
    RaySystemError,
    RayTaskError,
    TaskCancelledError,
    WorkerCrashedError,
)
from ray.remote_function import RemoteFunction
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from levanter.infra.docker import make_docker_run_command
from levanter.utils.ray_utils import (
    ExceptionInfo,
    SnitchRecipient,
    current_actor_handle,
    log_failures_to,
    ser_exc_info,
)


# CF https://gist.github.com/allenwang28/e3400b9e9212b50aa1cda55ebeccea60
# CF: https://github.com/AI-Hypercomputer/ray-tpu/blob/main/src/ray_tpu.py

# Basic flow:
# 1. Create a number of nliceActors, each representing a slice of the TPU pod. SliceActor owns the tpu-XXX-head resource
# 2. Each SliceActor owns a set of HostActors, each representing a host in the slice. Each HostActor owns $tpu_name resource
# 3. Each HostActor owns a specific worker. It can run a workload on the TPU slice by calling a remote function.
# 4. The SliceActor can run a remote function on all its HostActors, which will run the workload on the TPU slice.
# (Mechanically, each HostActor schedules a remote function that uses 0.001 of the IP address of the host and the
# TPU resources available on that host.)
# 5. If a slice dies, the tasks will fail, and we'll try to recover by creating a new SliceActor.

# Challenges:
# * Ray doesn't free placement groups when actors are killed, so we need to manage them ourselves.
# * JAX doesn't always seem to crash when another slice dies(?!?)
# * Ray actor method calls **cannot** be canceled, so they cannot be heavy-weight. They should return quickly by forking
#   another ray process (or some other mechanism) to do the heavy lifting.
# * Alternatively, we can use max_concurrent_tasks and other threading mechanisms to let the actor handle multiple
#   tasks so we can cancel an in-progress task.
# * Even with max_calls=1, Ray doesn't reliably clean up and we can end up with libtpu_lockfiles.


# Tests to write (on TPU):

# Single Slice
# 1. Run a simple function on a single slice and verify it runs correctly.
# 2. Run a second function after the first one and verify it runs correctly.
# 3. Run a function that fails and verify it retries the correct number of times and doesn't have libtpu problems
# 4. Run a function that preempts and verify it retries the correct number of times and doesn't have libtpu problems

# Multislice
# 1. Run a simple function on a multislice and verify it runs correctly.
# 2. Run a second function after the first one and verify it runs correctly.
# 3. Run a function where one slice fails and verify it retries the correct number of times and doesn't have libtpu problems


#########################
# Flex Multislice notes #
#########################

# TODO: implement FlexWorkload and FlexRunner
# Flex multislice works vaguely similarly, with a pool of slice actors.
# Define K the current number of slice actors, L the minimum number of slices, and M the maximum number of slices.
# While K < M, we try to create M - K new SliceActors.
# When K >= L, we start executing.
# If we lose a SliceActor, the process will crash (b/c of multislice)
# Whenever we get a new Slice, we add it to a pending pool of SliceActors.
# FlexWorkloads are actors and they can expose a "good point for a checkpoint"
# cf: https://gist.github.com/allenwang28/d67f3e1cd75b5fc37e57e5f31946856d

# TODO: look into https://github.com/jax-ml/jax/blob/main/jax/experimental/transfer.py to see if we
#  can avoid crashing (probably doing some kind of diloco thing) Or we just go full parameter server

logger = logging.getLogger("ray")


@dataclass
class SliceInfo:
    """
    Information about a TPU slice.

    This is used to pass information about a TPU slice to the worker tasks.
    """

    slice_name: str
    num_hosts: int
    ip_address: str


@dataclass
class HostInfo:
    """
    Information about a TPU host.

    This is used to pass information about a TPU host to the worker tasks.
    """

    pod_name: str
    worker_id: int
    ip_address: str
    num_tpus: int


@dataclass
class _TpuInfo:
    """Internal class to hold information about a TPU pod."""

    name: str
    state: str
    kind: str


# My kingdom for ADTs
@dataclass
class _TpuRunResult:
    """Internal class to hold the result of a TPU job."""

    pass


@dataclass
class TpuSuccess(_TpuRunResult):
    result: object


@dataclass
class TpuPreempted(_TpuRunResult):
    error: Exception


@dataclass
class TpuFailed(_TpuRunResult):
    error: Exception


@dataclass
class TpuRunError(_TpuRunResult):
    error: Exception


@dataclass
class TpuCancelled(_TpuRunResult):
    error: Exception


@dataclass
class MultisliceInfo:
    """
    Information about a TPU multislice.

    This is used to pass information about a TPU multislice to the worker tasks.
    """

    coordinator_ip: str
    slice_id: int
    num_slices: int
    port: int = 8081


@ray.remote(max_concurrency=4)
class SliceActor(SnitchRecipient):
    """
    Actor that manages a single TPU slice.
    """

    def __init__(self):
        self.pod_name = ray.util.accelerators.tpu.get_current_pod_name()
        self.num_hosts = ray.util.accelerators.tpu.get_current_pod_worker_count()
        self.ip = socket.gethostbyname(socket.gethostname())
        self.pg = ray.util.placement_group(
            name=f"tpu-slice-{self.pod_name}",
            bundles=[{self.pod_name: 1, "CPU": 1} for _ in range(self.num_hosts)],
        )

        ray.get(self.pg.ready())

        self.workers = [
            _HostActor.options(
                name=f"tpu-host-{self.pod_name}-{i}",
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=self.pg, placement_group_bundle_index=i, placement_group_capture_child_tasks=False
                ),
            ).remote(current_actor_handle())
            for i in range(self.num_hosts)
        ]

        self._failed = False

    def get_pg(self):
        """
        Get the placement group for this SliceActor.
        We expose this b/c Ray apparently doesn't free PGs if the actor is killed, so we need to
        return it to the caller to remove it manually.
        """
        return self.pg

    def cancel_task(self):
        exc = None
        futures = []
        for worker in self.workers:
            try:
                futures.append(worker.kill_task.remote())
            except Exception as e:
                logger.warning(f"Failed to kill task on worker {worker}: {e}")
                exc = e

        try:
            ray.get(futures)
        except ray.exceptions.RayError as e:
            logger.exception(f"Failed to cancel tasks on SliceActor {self.pod_name}: {e}")
            exc = e

        if exc is not None:
            raise RuntimeError(f"Failed to cancel task on SliceActor {self.pod_name}") from exc

    def shutdown(self):
        """
        Shutdown the SliceActor and all its workers. This doesn't free this particular actor, so the actor
        should be killed or let die naturally.
        """
        self._failed = True
        logger.info(f"Shutting down SliceActor {self.pod_name} with {self.num_hosts} hosts")
        worker_shutdowns = []
        for worker in self.workers:
            try:
                worker_shutdowns.append(worker.shutdown.remote())
            except Exception:
                logger.exception(f"Failed to kill worker {worker}")

        try:
            ray.get(worker_shutdowns)
        except ray.exceptions.RayError as e:
            logger.exception(f"Failed to shutdown all workers for SliceActor {self.pod_name}: {e}")

        ray.util.remove_placement_group(self.pg)

    def healthy(self):
        return not self._failed and not self.is_being_preempted()

    def is_being_preempted(self):
        """
        Check if the TPU slice is being preempted.
        This is a workaround for the fact that Ray doesn't expose this information directly.
        """
        from levanter.infra.tpus import get_current_tpu_is_preempted

        return get_current_tpu_is_preempted()

    def get_slice_info(self):
        return SliceInfo(
            slice_name=self.pod_name,
            num_hosts=self.num_hosts,
            ip_address=self.ip,
        )

    def do_run(self, remote_fn, multislice: Optional[MultisliceInfo]) -> _TpuRunResult:
        if self._failed:
            raise RuntimeError("SliceActor has failed. Cannot run workload.")

        if multislice is not None:
            mxla_env = {
                "MEGASCALE_COORDINATOR_ADDRESS": f"{multislice.coordinator_ip}:{multislice.port}",
                "MEGASCALE_NUM_SLICES": str(multislice.num_slices),
                "MEGASCALE_PORT": f"{multislice.port}",
                "MEGASCALE_SLICE_ID": str(multislice.slice_id),
            }
        else:
            mxla_env = {}
        # objectrefs of objectrefs that represent the results of the remote function calls
        start_futures = [host.do_run.remote(remote_fn, mxla_env) for host in self.workers]
        try:
            result_futures = ray.get(start_futures)
        except Exception as e:
            logger.exception("Exception starting task")
            try:
                self.cancel_task()
            except Exception:
                logger.exception(f"Failed to cancel task on SliceActor {self.pod_name}")
            return TpuFailed(e)

        try:
            out = ray.get(result_futures)
            logger.info("TPU job finished")
        except TaskCancelledError as e:
            logger.info(f"SliceActor {self.pod_name} was cancelled. Cancelling its host actors' tasks.")
            self.cancel_task()
            return TpuCancelled(e)
        except Exception as e:
            logger.exception(f"Exception getting results from SliceActor {self.pod_name}")
            try:
                self.cancel_task()
            except Exception as e2:
                logger.exception(f"Failed to cancel task on SliceActor {self.pod_name}: {e2}")

            if isinstance(e, RayError):
                return _handle_ray_error(e)
            else:
                return TpuRunError(e)

        return TpuSuccess(out)

    def _child_failed(self, child: ray.actor.ActorHandle | str | None, exception: ExceptionInfo):
        """
        Override the SnitchRecipient method to log failures of child actors.
        """
        info = exception.restore()
        logger.exception(f"Child {child} failed with exception {info[1]}", exc_info=info)
        self._failed = True
        exception.reraise()


@ray.remote(max_concurrency=4)
class _HostActor:
    """
    Actor that manages a single TPU host. Typically, you shouldn't use this directly, but rather use the
    `SliceActor` which manages (potentially_ multiple hosts.
    """

    def __init__(self, owner: ActorHandle):
        # using private api but oh well
        with log_failures_to(owner):
            self.pod_name = ray.util.accelerators.tpu.get_current_pod_name()
            self.worker_id = TPUAcceleratorManager._get_current_node_tpu_worker_id()
            self.ip = ray.get_runtime_context().worker.node_ip_address
            self.num_tpus = TPUAcceleratorManager.get_current_node_num_accelerators()
            self._current_ref: ray.ObjectRef = None

    def get_host_info(self):
        return HostInfo(
            pod_name=self.pod_name,
            worker_id=self.worker_id,
            ip_address=self.ip,
            num_tpus=self.num_tpus,
        )

    def kill_task(self):
        if self._current_ref is not None:
            try:
                ray.cancel(self._current_ref, force=True)
            except ray.exceptions.RayError as e:
                logger.warning(f"Failed to cancel current task {self._current_ref}: {e}")

            _hacky_remove_tpu_lockfile()
            self._current_ref = None

    def shutdown(self):
        """
        Shutdown the HostActor and kill the current task if it exists.
        """
        self.kill_task()

    def do_run(self, remote_fn: RemoteFunction, env_vars: dict) -> ray.ObjectRef:
        """
        Run a remote function on this host. This is typically called by the `SliceActor` to run a workload on
        all hosts in the slice.

        Returns an ObjectRef to an ObjectRef that represents the result of the function.
        """
        _hacky_remove_tpu_lockfile()
        if self._current_ref is not None:
            logger.warning(f"HostActor {self.pod_name}-{self.worker_id} already has a running task. Cancelling it.")
            self.kill_task()
        current_resources = remote_fn._resources
        sources = [e for e in [remote_fn._runtime_env, dict(env_vars=env_vars)] if e is not None]
        runtime_env = mergedeep.merge({}, *sources, strategy=mergedeep.Strategy.ADDITIVE)

        current_ref = remote_fn.options(
            resources={**(current_resources or {}), f"node:{self.ip}": 0.001, "TPU": self.num_tpus},
            runtime_env=runtime_env,
        ).remote()
        # logger.info("TPU job finished")
        self._current_ref = current_ref
        return current_ref


def run_on_pod_new(
    remote_fn: RemoteFunction | Callable,
    tpu_type: str,
    *,
    num_slices: int = 1,
    max_retries_preemption=10000,
    max_retries_failure=10,
) -> ray.ObjectRef:
    """
    Repeatedly run a function on a TPU pod until it succeeds or a maximum number of retries is reached.

    Args:
        remote_fn: A remote function that takes no arguments
        tpu_type: The type of TPU to run on, e.g. "v4-32"
        max_retries_preemption: The maximum number of times to retry if the job is preempted
        max_retries_failure: The maximum number of times to retry if the job fails

    Returns:
        The result of the function (not an ObjectRef)
    """

    if num_slices <= 0:
        raise ValueError("num_slices must be greater than 0")

    return _run_on_pod_ray.remote(remote_fn, tpu_type, num_slices, max_retries_preemption, max_retries_failure)


@ray.remote(num_cpus=0.01)
def _run_on_pod_ray(
    remote_fn: RemoteFunction, tpu_type: str, num_slices: int, max_retries_preemption: int, max_retries_failure: int
):

    num_failures = 0
    num_preemptions = 0
    attempt = 0

    pool: list[ActorHandle] = []  # pool of SliceActors
    # Ray doesn't free PGs automatically, so we need to keep track of them
    actors_to_pgs: dict[ActorHandle, PlacementGroup] = {}
    problems: list[Exception] = []
    problem: Exception | None

    if isinstance(remote_fn, FunctionNode):
        raise ValueError(
            "Remote function must be a Ray remote function or a plain function, not a FunctionNode. Don't use bind."
        )
    elif not isinstance(remote_fn, RemoteFunction):
        remote_fn = ray.remote(max_calls=1)(remote_fn)
    elif remote_fn._default_options.get("max_calls") is None:
        raise ValueError("Remote function must have max_calls set to 1 for TPU workloads.")

    try:
        while num_failures <= max_retries_failure and num_preemptions <= max_retries_preemption:
            logger.info(f"Running on {num_slices} x TPU {tpu_type}. Attempt {attempt}")
            attempt += 1
            problems.clear()

            # prune all bad actors from pool
            if len(pool):
                pool = _prune_dead_slices(pool, actors_to_pgs)
                if len(pool) < num_slices:
                    logger.info(f"Pool has {len(pool)} actors, but we need {num_slices}. Creating more actors.")

            # if we don't have enough actors, create more
            new_actors_to_pgs: dict[ActorHandle, ray.ObjectRef] = {}
            while len(pool) < num_slices:
                a = SliceActor.options(resources={f"TPU-{tpu_type}-head": 1}).remote()  # type: ignore
                new_actors_to_pgs[a] = a.get_pg.remote()
                pool.append(a)

            # wait for all new actors to be ready
            if new_actors_to_pgs:
                logger.info(f"Waiting for {len(new_actors_to_pgs)} new actors to be ready...")
                for a, pg_ref in new_actors_to_pgs.items():
                    try:
                        actors_to_pgs[a] = ray.get(pg_ref)
                    except ray.exceptions.RayError as e:
                        logger.exception("Failed to get new actors ready", exc_info=e)
                        continue
                logger.info(f"Pool ready with {len(pool)} actors.")

            # If we're doing multislice, we need to get the slice info from the first actor
            try:
                if len(pool) > 1:
                    head_slice_info: SliceInfo | None = ray.get(pool[0].get_slice_info.remote())
                else:
                    head_slice_info = None
            except ray.exceptions.RayError as e:
                logger.exception("Failed to get slice info from head actor", exc_info=e)
                problems.append(e)
                error_type = _handle_ray_error(e)
                if isinstance(error_type, TpuPreempted):
                    num_preemptions += 1
                    if num_preemptions >= max_retries_preemption:
                        logger.warning(f"Preempted {num_preemptions} times. Continuing to retry.", exc_info=e)
                elif isinstance(error_type, TpuRunError | TpuFailed):
                    num_failures += 1
                    if num_failures <= max_retries_failure:
                        logger.warning(f"Failed {num_failures} times. Continuing to retry.", exc_info=e)
                continue

            futures = []
            future_to_actor = {}
            future_to_index = {}
            for i, actor in enumerate(pool):
                if head_slice_info is not None:
                    mxla_env = MultisliceInfo(
                        slice_id=i,
                        num_slices=len(pool),
                        coordinator_ip=head_slice_info.ip_address,
                        port=8081,  # default port for megascale
                    )
                else:
                    mxla_env = None

                f = actor.do_run.remote(remote_fn, mxla_env)
                futures.append(f)
                future_to_actor[f] = actor
                future_to_index[f] = i

            slice_actor_results: list[_TpuRunResult | None] = [None] * len(futures)

            # Proactively cancel jobs if one fails.
            # We wait for one job to finish at a time. If it fails, we cancel the rest.
            pending_futures = set(futures)
            had_a_failure = False

            while pending_futures and not had_a_failure:
                # wait for one to finish
                finished_set, pending_futures_set = ray.wait(list(pending_futures), num_returns=1)
                pending_futures = pending_futures_set

                f = finished_set[0]

                try:
                    result = ray.get(f)
                    slice_actor_results[future_to_index[f]] = result
                    if not isinstance(result, TpuSuccess):
                        logger.warning(f"SliceActor for future {f} returned non-success result: {result}")
                        had_a_failure = True
                except RayError as e:
                    logger.warning(f"Task {f} failed with error {e}. Will retry.")
                    had_a_failure = True
                    slice_actor_results[future_to_index[f]] = _handle_ray_error(e)
                except Exception as e:
                    logger.warning(f"Task {f} failed with unexpected error {e}. Will retry.")
                    had_a_failure = True
                    slice_actor_results[future_to_index[f]] = TpuRunError(e)

            if had_a_failure and pending_futures:
                logger.info(
                    f"Failure detected. Telling {len(pending_futures)} pending slice actors to cancel their tasks."
                )
                for f_pending in pending_futures:
                    # We can't ray.cancel an actor method future. We have to ask the actor to cancel its own work.
                    actor_to_cancel = pool[future_to_index[f_pending]]
                    actor_to_cancel.cancel_task.remote()

                # Now, wait for them to finish cancelling and fill in their results.
                # The do_run method will get a TaskCancelledError and return a TpuCancelled object.
                for f_pending in pending_futures:
                    try:
                        # this will get the result from the actor, which should be TpuCancelled
                        result = ray.get(f_pending)
                        slice_actor_results[future_to_index[f_pending]] = result
                    except RayError as e:
                        # it might have failed for another reason before the cancel took effect
                        slice_actor_results[future_to_index[f_pending]] = _handle_ray_error(e)
                    except Exception as e:
                        slice_actor_results[future_to_index[f_pending]] = TpuRunError(e)

            # at this point, all futures are accounted for in slice_actor_results
            # so we can process them as a complete set

            # Process results, figure out if we succeeded or failed or preempted
            out_results: list = []
            any_preempted = False
            any_failed = False
            any_cancelled = False

            for result in slice_actor_results:
                logger.info(f"Result {result}")
                if isinstance(result, TpuSuccess):
                    out_results.extend(result.result)  # type: ignore
                elif isinstance(result, TpuPreempted):
                    problems.append(result.error)
                    any_preempted = True
                elif isinstance(result, TpuFailed):
                    any_preempted = True
                    problems.append(result.error)
                    logger.warning(f"TPU node failure. Treating as preempted: {num_preemptions} times")
                elif isinstance(result, TpuRunError):
                    problems.append(result.error)
                    any_failed = True
                elif isinstance(result, TpuCancelled):
                    logger.info("TPU job was cancelled, probably because something else failed.")
                    any_cancelled = True
                elif result is None:
                    continue
                else:
                    raise RuntimeError(f"Unexpected result: {result}")

            if any_preempted:
                problem = problems[0] if problems else RuntimeError("TPU job was preempted")
                num_preemptions += 1
                if any_failed:
                    logger.exception(
                        f"Preempted {num_preemptions} times. "
                        "Got some failures, but assuming they are due to preemption.",
                        exc_info=problem,
                    )
                else:
                    logger.warning(f"Preempted {num_preemptions} times. Continuing to retry.", exc_info=problem)
                continue
            elif any_failed:
                problem = problems[0] if problems else RuntimeError("TPU job failed")
                num_failures += 1
                logger.warning(f"Failed {num_failures} times. Continuing to retry.", exc_info=problem)
                continue
            elif any_cancelled:
                logger.info("A slice's task was cancelled, probably due to another slice's failure. Retrying.")
                continue
            else:
                assert not any(result is None for result in slice_actor_results), (
                    "Some results were None, but we didn't have any failures or preemptions. "
                    f"Results: {slice_actor_results}"
                )
                logger.info("All slices succeeded. Returning results.")
                return out_results

    finally:
        # Cleanup actors and placement groups
        logger.info("Cleaning up actors and placement groups")
        futures = []
        for actor in pool:
            try:
                futures.append(actor.shutdown.remote())
            except Exception as e:
                logger.warning(f"Failed to shutdown actor {actor}: {e}")

        if futures:
            try:
                ray.get(futures)
            except ray.exceptions.RayError as e:
                logger.warning(f"Failed to shutdown all actors: {e}")

        for pg in actors_to_pgs.values():
            try:
                ray.util.remove_placement_group(pg)
            except Exception as e:
                logger.warning(f"Failed to remove placement group {pg}: {e}")

        pool.clear()
        actors_to_pgs.clear()

    # Note: PyCharm flags this as unreachable code, but it is reachable if the loop exits without returning.
    problem = problems[0] if problems else None

    if num_preemptions > max_retries_preemption:
        logger.exception("Preempted too many times", exc_info=problem)
        raise RuntimeError("TPU job was preempted too many times") from problem
    elif num_failures >= max_retries_failure:
        logger.exception("Failed too many times", exc_info=problem)
        raise problem or RuntimeError("TPU job failed too many times")
    else:
        raise RuntimeError("Unknown error occurred during TPU job") from problem


def _prune_dead_slices(pool: list[ActorHandle], pgs: dict[ActorHandle, PlacementGroup]) -> list[ActorHandle]:
    """
    Prune dead or unhealthy actors from the pool. Takes and returns a list of ActorHandles to SliceActors.

    mutates pgs
    """
    new_pool = []
    to_kill_futures = []
    actors_and_healthy = [(actor, actor.healthy.remote()) for actor in pool]
    for actor, healthy in actors_and_healthy:
        try:
            if ray.get(healthy):
                new_pool.append(actor)
            else:
                logger.warning(f"Actor {actor} is unhealthy. Removing from pool.")
                to_kill_futures.append(actor.shutdown.remote())
                try:
                    pg = pgs.pop(actor, None)
                    if pg is not None:
                        ray.util.remove_placement_group(pg)
                except Exception:  # noqa: E722
                    logger.exception(f"Failed to remove placement group for actor {actor}")
        except (RayActorError, RayTaskError, ActorDiedError, ActorUnavailableError) as e:
            logger.warning(f"Actor {actor} is dead or unavailable. Removing from pool. Error: {e}")
            to_kill_futures.append(actor.shutdown.remote())
            pg = pgs.pop(actor, None)
            if pg is not None:
                ray.util.remove_placement_group(pg)

    if to_kill_futures:
        try:
            ray.get(to_kill_futures)
        except ray.exceptions.RayError as e:
            logger.warning(f"Failed to kill some actors: {e}")

    return new_pool


def _cancel_all_futures(futures):
    for f in futures:
        try:
            ray.cancel(f)
        except Exception:
            logger.exception("Failed to kill job after primary failure")


def _handle_ray_error(e: RayError):
    """
    Handle a Ray error that occurred on a TPU pod. Tries to determine if the error was due to a
    node failure or preemption or just an application error.
    """
    # treat node failures as preemptions
    if isinstance(e, NodeDiedError):
        logger.exception("Node died", exc_info=e)
        return TpuPreempted(e)
    elif isinstance(e, ray.exceptions.ActorUnavailableError | ray.exceptions.ActorDiedError):
        logger.exception("Actor died", exc_info=e)
        return TpuPreempted(e)
    elif isinstance(e, WorkerCrashedError):
        logger.exception("Worker crashed", exc_info=e)
        return TpuPreempted(e)
    elif isinstance(e, RaySystemError):
        logger.exception("System error", exc_info=e)
        return TpuRunError(e)
    elif isinstance(e, RayTaskError):
        # node preemptions don't always show up as one of the above errors and can just be a RayTaskError. We have
        # to try to sniff out the TPU's status.
        from levanter.infra.tpus import get_current_tpu_is_preempted

        if get_current_tpu_is_preempted():
            logger.exception("Preempted", exc_info=e)
            return TpuPreempted(e)

        logger.exception(f"Task error {e}", exc_info=e)
        if isinstance(e.cause, TimeoutError) or "timed out" in str(e):
            logger.exception("Timeout error. Assuming preempted", exc_info=e)
            return TpuPreempted(e)
        return TpuRunError(e)

    else:
        logger.exception("Unknown error", exc_info=e)
        return TpuRunError(e)


# @ray.remote
# class FlexsliceActor:
#     def __init__(self,
#                  workload: Callable[[int], None],
#                  slice_type: str,
#                  valid_slice_counts: Sequence[int]
#                  ):
#         self.workload = workload
#         self.slice_type = slice_type
#         self.valid_slice_counts = valid_slice_counts
#
#
#
# def run_on_pod_flex(workload: RemoteFunction | Callable, tpu_type: str, valid_slice_counts: Sequence[int]):
#     # First query how many we think have. We'll try to be greedy and grab them all (as makes sense)
#     for


def run_on_pod(remote_fn: RemoteFunction | Callable, tpu_type: str) -> ray.ObjectRef:
    """
    Run a remote function on a TPU pod.

    Args:
        remote_fn: A remote function that takes no arguments
        tpu_type: The type of TPU to run on, e.g. "v4-32"

    Returns:
        A Ray ObjectRef that represents the result of the function
    """

    @ray.remote(resources={f"TPU-{tpu_type}-head": 1})
    def do_run(remote_fn) -> _TpuRunResult:
        logging.basicConfig(level=logging.INFO)
        num_hosts = ray.util.accelerators.tpu.get_current_pod_worker_count()  # -> 4
        remote_fn, tpu_name = _redecorate_remote_fn_for_tpu(remote_fn, num_hosts)

        futures = [remote_fn.remote() for _ in range(num_hosts)]
        try:
            out = ray.get(futures)
            logger.info("TPU job finished")
            return TpuSuccess(out)
        except RayError as e:
            _cancel_all_futures(futures)
            result = _handle_ray_error(e)

            return result

        except Exception as e:
            _cancel_all_futures(futures)
            return TpuFailed(e)

    return do_run.remote(remote_fn)


def run_on_pod_multislice(remote_fn: RemoteFunction | Callable, tpu_type: str, num_slices: int) -> list[ray.ObjectRef]:
    """
    Run a remote function on multiple TPU slices.

    Args:
        remote_fn: A remote function that takes no arguments
        tpu_type: The type of TPU to run on, e.g. "v4-32"
        num_slices: The number of slices to run

    Returns:
        A Ray ObjectRef that represents the result of the function
    """

    @ray.remote(resources={f"TPU-{tpu_type}-head": 1})
    class MultisliceActor:
        def __init__(self):
            self.pod_name = ray.util.accelerators.tpu.get_current_pod_name()
            self.num_hosts = ray.util.accelerators.tpu.get_current_pod_worker_count()
            self.ip = socket.gethostbyname(socket.gethostname())

        def get_slice_info(self):
            return self.pod_name, self.num_hosts, self.ip

        def do_run(self, remote_fn, coordinator_ip, slice_id, num_slices) -> _TpuRunResult:
            port = 8081
            mxla_env = {
                "MEGASCALE_COORDINATOR_ADDRESS": f"{coordinator_ip}:{port}",
                "MEGASCALE_NUM_SLICES": str(num_slices),
                "MEGASCALE_PORT": f"{port}",
                "MEGASCALE_SLICE_ID": str(slice_id),
            }

            remote_fn, tpu_name = _redecorate_remote_fn_for_tpu(remote_fn, self.num_hosts, env_vars=mxla_env)

            futures = [remote_fn.remote() for _ in range(self.num_hosts)]
            try:
                out = ray.get(futures)
                logger.info("TPU job finished")
                return TpuSuccess(out)
            except RayError as e:
                logger.exception(f"Ray error {e}. Killing futures for this slice")
                _cancel_all_futures(futures)
                return _handle_ray_error(e)
            except Exception as e:
                logger.exception(f"Exception {e}")
                _cancel_all_futures(futures)
                return TpuFailed(e)

    actors = [MultisliceActor.remote() for _ in range(num_slices)]  # type: ignore
    futures = [actor.get_slice_info.remote() for actor in actors]
    try:
        logger.info("Getting slice infos...")
        # also act as a sync step
        slice_infos = ray.get(futures)
        logger.info(f"TPU slice infos {slice_infos}")
    except RayError as e:
        logger.exception(e)
        for actor in actors:
            try:
                ray.kill(actor)
            except Exception:
                logger.exception("Failed to kill actor after primary failure")
        return futures

    coordinator_ip = slice_infos[0][2]

    return [actor.do_run.remote(remote_fn, coordinator_ip, i, num_slices) for i, actor in enumerate(actors)]


def _redecorate_remote_fn_for_tpu(remote_fn, num_hosts, **runtime_env):
    """
    Redecorate a remote function to run on a TPU pod.

    Specifically, this function:

    * Adds the TPU resources to the function
    * forces the function to run in its own process to remove the TPU lockfile (and shutdown jax distributed)

    """
    remote_fn = _forkify_remote_fn(remote_fn)
    if not isinstance(remote_fn, RemoteFunction):
        remote_fn = ray.remote(remote_fn)

    tpu_name = ray.util.accelerators.tpu.get_current_pod_name()  # -> my-tpu
    num_tpus_per_host = TPUAcceleratorManager.get_current_node_num_accelerators()  # -> 8

    # ray doesn't merge the runtime envs properly, so we have to do it ourselves
    # we need to do a deep merge
    sources = [e for e in [remote_fn._runtime_env, runtime_env] if e is not None]
    runtime_env = mergedeep.merge({}, *sources, strategy=mergedeep.Strategy.ADDITIVE)

    remote_fn = remote_fn.options(
        runtime_env=runtime_env,
        resources={tpu_name: 1, "TPU": num_tpus_per_host},
    )

    logger.info(f"Running on TPU {tpu_name} with {num_hosts} hosts and {num_tpus_per_host} TPUs per host")
    return remote_fn, tpu_name


def run_on_pod_resumable(remote_fn, tpu_type, max_retries_preemption=1e6, max_retries_failure=10):
    """
    Repeatedly run a function on a TPU pod until it succeeds or a maximum number of retries is reached.

    Args:
        remote_fn: A remote function that takes no arguments
        tpu_type: The type of TPU to run on, e.g. "v4-32"
        max_retries_preemption: The maximum number of times to retry if the job is preempted
        max_retries_failure: The maximum number of times to retry if the job fails

    Returns:
        The result of the function (not an ObjectRef)

    """
    num_failures = 0
    num_preemptions = 0
    attempt = 0
    problem: Exception | None = None

    while num_failures < max_retries_failure and num_preemptions < max_retries_preemption:
        logger.info(f"Running on TPU {tpu_type}. Attempt {attempt}")
        attempt += 1
        problem = None
        try:
            out = ray.get(run_on_pod(remote_fn, tpu_type))
        except ray.exceptions.RayTaskError as e:
            problem = e
            if "preempted" in str(e).lower():
                num_preemptions += 1
                logger.warning(f"Preempted {num_preemptions} times, {e}")
            else:
                num_failures += 1
                logger.warning(f"Failed {num_failures} times", exc_info=e)
            continue
        except Exception as e:
            problem = e
            num_failures += 1
            if num_failures >= max_retries_failure:
                logger.exception("Failed too many times", exc_info=e)
                raise e
            else:
                logger.warning(f"Failed {num_failures} times", exc_info=e)
                continue

        if isinstance(out, TpuSuccess):
            result = out.result
            logger.info("Success")
            return result
        elif isinstance(out, TpuPreempted):
            problem = out.error
            num_preemptions += 1
            logger.warning(f"Preempted {num_preemptions} times. {problem}", exc_info=problem)
        elif isinstance(out, TpuFailed):
            num_preemptions += 1
            logger.warning(f"TPU node failure. Treating as preempted: {num_preemptions} times")
        elif isinstance(out, TpuRunError):
            problem = out.error
            num_failures += 1
            logger.warning(f"Failed {num_failures} times", exc_info=problem)
        else:
            raise RuntimeError(f"Unexpected result: {out}")

    if num_preemptions >= max_retries_preemption:
        raise RuntimeError("Preempted too many times") from problem
    elif num_failures >= max_retries_failure:
        raise RuntimeError("Failed too many times") from problem


def run_on_pod_multislice_resumable(
    remote_fn, tpu_type, num_slices, max_retries_preemption=1e6, max_retries_failure=10
):
    """
    Repeatedly run a function on a TPU pod until it succeeds or a maximum number of retries is reached.

    Args:
        remote_fn: A remote function that takes no arguments
        tpu_type: The type of TPU to run on, e.g. "v4-32"
        num_slices: The number of slices to run
        max_retries_preemption: The maximum number of times to retry if the job is preempted
        max_retries_failure: The maximum number of times to retry if the job fails

    Returns:
        The result of the function (not an ObjectRef)

    """
    num_failures = 0
    num_preemptions = 0
    attempt = 0
    problem: Exception | None = None

    while num_failures < max_retries_failure and num_preemptions < max_retries_preemption:
        logger.info(f"Running on TPU {tpu_type}. Attempt {attempt}")
        attempt += 1
        problem = None
        futures = run_on_pod_multislice(remote_fn, tpu_type, num_slices)
        try:
            outs = ray.get(futures)
        except ray.exceptions.ActorUnavailableError as e:
            problem = e
            num_preemptions += 1
            logger.warning(f"Preempted {num_preemptions} times, {e}")
            continue
        except ray.exceptions.ActorDiedError as e:
            problem = e
            num_preemptions += 1
            logger.warning(f"Preempted {num_preemptions} times, {e}")
            continue
        except ray.exceptions.RayTaskError as e:
            for f in futures:
                try:
                    ray.cancel(f)
                except Exception:
                    logger.exception("Failed to kill job after primary failure")
            problem = e
            if "preempted" in str(e).lower():
                num_preemptions += 1
                logger.warning(f"Preempted {num_preemptions} times, {e}")
            else:
                num_failures += 1
                logger.warning(f"Failed {num_failures} times", exc_info=e)
            continue
        except Exception as e:
            for f in futures:
                try:
                    ray.cancel(f)
                except Exception:
                    logger.exception("Failed to kill job after primary failure")
            problem = e
            num_failures += 1
            if num_failures >= max_retries_failure:
                logger.exception("Failed too many times", exc_info=e)
                raise e
            else:
                logger.warning(f"Failed {num_failures} times", exc_info=e)
                continue

        if all(isinstance(out, TpuSuccess) for out in outs):
            results = [out.result for out in outs]
            logger.info("Success")
            return results
        elif any(isinstance(out, TpuPreempted) for out in outs):
            out = None
            for o in outs:
                if isinstance(o, TpuPreempted):
                    out = o
            assert out is not None
            problem = out.error
            num_preemptions += 1
            logger.warning(f"Preempted {num_preemptions} times. {problem}", exc_info=problem)
        elif any(isinstance(out, TpuFailed) for out in outs):
            num_preemptions += 1
            logger.warning(f"TPU node failure. Treating as preempted: {num_preemptions} times")
        elif any(isinstance(out, TpuRunError) for out in outs):
            out = None
            for o in outs:
                if isinstance(o, TpuRunError):
                    out = o
            assert out is not None
            problem = out.error
            num_failures += 1
            logger.warning(f"Failed {num_failures} times", exc_info=problem)
        else:
            raise RuntimeError(f"Unexpected result: {out}")

    if num_preemptions >= max_retries_preemption:
        raise RuntimeError("Preempted too many times") from problem
    elif num_failures >= max_retries_failure:
        raise RuntimeError("Failed too many times") from problem


def _run_command(*args, **kwargs):
    return subprocess.check_call(args, **kwargs)


def run_docker_on_pod(
    image_id: str, command: Sequence[str], *, tpu_type: str, num_slices: int, env: dict, name="levanter", retries=10
):
    env = _massage_env(env)

    docker_cmd = make_docker_run_command(image_id, command, env=env, foreground=True, name=name)

    def run_docker():
        _kill_old_container(name)
        try:
            return _run_command(*docker_cmd)
        except subprocess.CalledProcessError as e:
            logger.exception("Failed to run docker command")
            raise e

    run_on_pod_new(
        ray.remote(max_calls=1)(run_docker),
        tpu_type=tpu_type,
        num_slices=num_slices,
        max_retries_failure=retries,
        max_retries_preemption=10000,
    )


def _kill_old_container(name):
    try:
        logger.info(f"Killing old container {name}")
        _run_command("sudo", "docker", "rm", "-f", name)
    except subprocess.CalledProcessError:
        pass


def _forkify_remote_fn(remote_fn: RemoteFunction | Callable):
    """
    This is a bit of a hacky way to force a remote function to run in its own process, using multiprocessing.

    There are a few issues we're trying to cover:

    * libtpu only allows one process to access the TPU at a time, and it uses a lockfile to enforce this.
    * Ray runs tasks in a long-running daemon, so the lockfile persists across tasks.
    * jax.distributed likes to only be called once per process, even if you call shutdown

    """
    if isinstance(remote_fn, RemoteFunction):
        fn = remote_fn._function

        @functools.wraps(fn)
        def wrapped_fn(*args, **kwargs):
            return _separate_process_fn(fn, args, kwargs)

        # We need these arguments to be able to reconstruct the remote function
        # def __init__(
        #         self,
        #         language,
        #         function,
        #         function_descriptor,
        #         task_options,
        # ):
        remote_fn = RemoteFunction(
            language=remote_fn._language,
            function=wrapped_fn,
            function_descriptor=remote_fn._function_descriptor,
            task_options=remote_fn._default_options,
        )
        return remote_fn
    else:
        return functools.partial(_separate_process_fn, remote_fn)


def _separate_process_fn(underlying_function, args, kwargs):
    """
    Helper function for _forkify_remote_fn. This runs the function in a separate process.
    """

    def target_fn(queue, args, kwargs):
        try:
            # Call the original function
            result = underlying_function(*args, **kwargs)
            queue.put((True, result))  # Success, put the result
        except Exception as e:
            # Capture and return the full traceback in case of an exception
            info = ser_exc_info(e)
            queue.put((False, info))

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=target_fn, args=(queue, args, kwargs))
    process.start()
    process.join()

    # Retrieve the result or error from the queue
    logger.info("Process finished")
    try:
        success, value = queue.get(timeout=5)
    except QueueEmpty:
        logger.error("Process timed out")
        process.terminate()
        raise TimeoutError("Process timed out")

    if success:
        return value
    else:
        value.reraise()


def _hacky_remove_tpu_lockfile():
    """
    This is a hack to remove the lockfile that TPU pods create on the host filesystem.

    libtpu only allows one process to access the TPU at a time, and it uses a lockfile to enforce this.
    Ordinarily a lockfile would be removed when the process exits, but in the case of Ray, the process is
    a long-running daemon that doesn't typically exit until the node is shut down. This means that the lockfile
    persists across Ray tasks. This doesn't apply to our docker-based workloads, but it does apply to other
    tasks that use JAX directly.
    """
    if os.path.exists("/tmp/libtpu_lockfile"):
        try:
            os.unlink("/tmp/libtpu_lockfile")
        except FileNotFoundError:
            pass
        except PermissionError:
            logger.warning("Failed to remove lockfile")
            try:
                os.system("sudo rm /tmp/libtpu_lockfile")
            except Exception:  # noqa
                pass


@dataclass
class RunDockerOnPodConfig:
    image_id: str
    command: list[str] | str
    tpu_type: str
    env: dict = dataclasses.field(default_factory=dict)
    name: str = "levanter"
    retries: int = 10
    node_count: int = 1


def submit_tpu_job_on_ray(config: RunDockerOnPodConfig, ray_address: str, run_id: Optional[str] = None):
    """
    Submit a job to run on a TPU pod on a Ray cluster. This programmatically submits a job to the Ray cluster.
    This should be run on your local machine, not on the Ray cluster itself.

    If run_id is not provided, a default run ID will be generated.
    """

    with tempfile.NamedTemporaryFile(suffix=".yaml", prefix=f"launch-{run_id}-", dir=".") as f:
        yaml = draccus.dump(config)
        f.write(yaml.encode("utf-8"))
        f.flush()

        f_name = os.path.relpath(f.name)
        logger.info(f"Submitting job with config path {f_name}")

        client = JobSubmissionClient(ray_address)

        job_id = _make_unique_job_id(client, run_id) if run_id is not None else None

        job_id = client.submit_job(
            entrypoint=f"python -m levanter.infra.ray_tpu --config_path {f_name}",
            runtime_env={"working_dir": ".", "env_vars": {"PYTHONPATH": "src:."}},
            submission_id=job_id,
        )

        return job_id


# try to make the job id be the same as the run id, but if it already exists, just make it unique
def _make_unique_job_id(client, run_id):
    job_id = run_id
    try:
        while client.get_job_status(job_id) is not None:
            job_id = f"{run_id}-{time.time_ns()}"
    except Exception as e:  # noqa
        if "does not exist" in str(e):
            pass
        else:
            raise
    return job_id


@draccus.wrap()
def main(args: RunDockerOnPodConfig):
    """
    *This command is designed to run on a Ray cluster, not on your local machine. You probably want submit_tpu_job_on_ray.*

    Run a command on a TPU pod. This is a wrapper around `run_docker_on_pod` that takes a config object as a CLI.

    We use this via infra/launch_on_ray.py to run docker containers on TPUs.
    """

    import shlex

    if isinstance(args.command, str):
        command = shlex.split(args.command)
    else:
        command = args.command

    run_docker_on_pod(
        args.image_id,
        command,
        tpu_type=args.tpu_type,
        env=args.env,
        name=args.name,
        retries=args.retries,
        num_slices=args.node_count,
    )


def _massage_env(env):
    # Ray pretends it's running in a TTY, which leads to a ton of log spam from tqdm.
    # Levanter uses tqdm_loggable, which tries to sniff out the TTY, but it doesn't work with Ray.
    # So we force it
    if "TERM" not in env:
        env = {**env, "TERM": "dumb"}

    return env


if __name__ == "__main__":
    # main()

    # leaving this here for testing purposes
    ray.init()
    tpu_type = "v4-8"
    num_slices = 2

    @ray.remote(max_calls=1)
    def fn():
        import jax
        import jax.random as jrandom
        from jax.lax import with_sharding_constraint
        from jax.sharding import Mesh
        from jax.sharding import PartitionSpec as P

        mesh = Mesh(jax.devices("tpu"), ("x",))
        print(jax.devices())

        @jax.jit
        def init():
            with mesh:
                x = jrandom.normal(jrandom.PRNGKey(0), (32,))
                weights = jrandom.normal(jrandom.PRNGKey(1), (32, 4))
                bias = jrandom.normal(jrandom.PRNGKey(2), (4,))

                x_sharded = with_sharding_constraint(x, P("x"))
                weights_sharded = with_sharding_constraint(weights, P("x"))
                return x_sharded, weights_sharded, bias

        x, weights, bias = init()

        @jax.jit
        def layer(x, weights, bias):
            with mesh:
                return with_sharding_constraint(jax.nn.sigmoid(x @ weights + bias), P())

        out = layer(x, weights, bias)

        import numpy

        return numpy.array(out)

    results = ray.get(run_on_pod_new(fn, tpu_type, num_slices=num_slices))

    print(f"Results: {results}")
