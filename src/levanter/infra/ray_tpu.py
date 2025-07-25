import dataclasses
import logging
import multiprocessing
import os
import socket
import subprocess
import tempfile
import time
from asyncio import QueueEmpty
from dataclasses import dataclass
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
    GetTimeoutError,
    NodeDiedError,
    RayActorError,
    RayError,
    RaySystemError,
    RayTaskError,
    WorkerCrashedError,
)
from ray.remote_function import RemoteFunction
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from levanter.infra.docker import make_docker_run_command
from levanter.utils.ray_utils import ser_exc_info


# CF https://gist.github.com/allenwang28/e3400b9e9212b50aa1cda55ebeccea60
# CF: https://github.com/AI-Hypercomputer/ray-tpu/blob/main/src/ray_tpu.py

# Basic flow:
# 1. The controller creates a pool of SliceActors, each representing a slice of the TPU pod. SliceActor owns the
#     tpu-XXX-head resource
# 2. Each SliceActor creates a placement group with one bundle per host in the slice, each with 1 CPU and N
#    TPUs (N=4 typically)
# 3. Controller allocates tasks using the placement groups onto all slices
# 4. If a slice fails, the controller gets a new slice

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
class HostInfo:
    """
    Information about a TPU host.

    This is used to pass information about a TPU host to the worker tasks.
    """

    pod_name: str
    worker_id: int
    ip_address: str
    num_tpus: int


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


@dataclass
class SliceInfo:
    """
    Information about a TPU slice.

    This is used to pass information about a TPU slice to the worker tasks.
    """

    slice_name: str
    num_hosts: int
    ip_address: str
    num_tpus_per_host: int

    def resources_per_job(self) -> dict:
        """
        Returns the resources required for a job on this slice.
        This is used to set the resources for the remote function.
        """
        return {
            self.slice_name: 1,
            "CPU": 8,  # default number of CPUs per host
            "TPU": self.num_tpus_per_host,
            "memory": 20e9,  # default memory per host
        }

    def as_ray_resources_kwargs(self):
        """Ray obnoxiously uses these resource dicts but you can't pass them to ray.remote"""
        return dict(
            resources={
                self.slice_name: 1,
                "TPU": self.num_tpus_per_host,
            },
            num_cpus=8,
            num_gpus=0,
            memory=20e9,
        )


# Timeouts (in seconds)
_HEALTH_CHECK_TIMEOUT = 60
_TERMINATE_ACTOR_TIMEOUT = 300
_START_ACTOR_TIMEOUT = 300
_PLACEMENT_GROUP_READY_TIMEOUT = 300


def _multislice_info_from_head(head: SliceInfo, slice_id: int, num_slices: int) -> MultisliceInfo:
    """
    Create a MultisliceInfo object from the head slice info and the slice ID and number of slices.
    """
    return MultisliceInfo(
        coordinator_ip=head.ip_address,
        slice_id=slice_id,
        num_slices=num_slices,
        port=8081,  # default port for megascale
    )


def _multislice_info_to_env_vars(multislice: MultisliceInfo) -> dict[str, str]:
    if multislice is not None:
        mxla_env = {
            "MEGASCALE_COORDINATOR_ADDRESS": f"{multislice.coordinator_ip}:{multislice.port}",
            "MEGASCALE_NUM_SLICES": str(multislice.num_slices),
            "MEGASCALE_PORT": f"{multislice.port}",
            "MEGASCALE_SLICE_ID": str(multislice.slice_id),
        }
    else:
        mxla_env = {}
    return mxla_env


def _create_placement_group(pod_name: str, num_hosts: int, num_tpus: int) -> PlacementGroup:
    pg_name = f"tpu-slice-{pod_name}"

    bundles = [
        {
            "CPU": 8,
            "TPU": num_tpus,
            "memory": 20e9,
            pod_name: 1,  # your custom resource key
        }
        for _ in range(num_hosts)
    ]

    try:
        pg = ray.util.placement_group(bundles=bundles, name=pg_name)
    except RaySystemError as e:
        if "already exists" in str(e):
            logger.warning(f"Placement group {pg_name} already exists. Reusing it.")
            # In theory this is fine since in order to use this PG we have to have the lock on the slice
            pg = ray.util.get_placement_group(pg_name)
        else:
            try:
                logger.exception(f"Failed to create placement group {pg_name}. Removing it. Error: {e}")
                ray.util.remove_placement_group(pg)  # don’t leave a half-ready PG behind
            except Exception as remove_pg_exception:
                logger.exception(f"Failed to remove placement group {pg_name} after failure: {remove_pg_exception}")
            raise RuntimeError(f"Failed to create placement group {pg_name}") from e

    return pg


@ray.remote
class SliceActor:
    """
    Actor that manages a single TPU slice.
    """
    def __init__(self):
        self._failed = False

    def healthy(self) -> bool:
        return not self._failed and not self.is_being_preempted()

    def is_being_preempted(self) -> bool:
        """
        Check if the TPU slice is being preempted.
        This is a workaround for the fact that Ray doesn't expose this information directly.
        """
        from levanter.infra.tpus import get_current_tpu_is_preempted

        return get_current_tpu_is_preempted()

    def get_slice_info(self):
        pod_name = ray.util.accelerators.tpu.get_current_pod_name()
        num_hosts = ray.util.accelerators.tpu.get_current_pod_worker_count()
        num_tpus_per_host = TPUAcceleratorManager.get_current_node_num_accelerators()
        tpe = TPUAcceleratorManager._get_current_node_tpu_pod_type()
        # there seems to be a bug with some version of ray here
        if tpe.startswith("v4") or tpe.startswith("v5"):
            num_cores = int(tpe.split("-")[1])
            num_tpus_per_host = 4
            num_hosts = num_cores // 8
        ip_address = socket.gethostbyname(socket.gethostname())

        return SliceInfo(
            slice_name=pod_name,
            num_hosts=num_hosts,
            num_tpus_per_host=num_tpus_per_host,
            ip_address=ip_address,
        )


@dataclass(frozen=True)
class SliceResource:
    """A collection of all resources associated with a single TPU slice."""

    actor: ActorHandle
    """Actor handle for the SliceActor"""

    slice_info: SliceInfo
    """Information about the slice"""

    placement_group: PlacementGroup
    """Placement group of hosts"""


def run_on_pod(
    remote_fn: RemoteFunction | Callable,
    tpu_type: str,
    *,
    num_slices: int = 1,
    max_retries_preemption=10000,
    max_retries_failure=10,
):
    """
    Repeatedly run a function on a TPU pod until it succeeds or a maximum number of retries is reached.

    Note: This function will block until the function completes or fails too many times. If you want to run it asynchronously,
    use `run_on_pod_ray` instead.

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

    return ray.get(run_on_pod_ray.remote(remote_fn, tpu_type, num_slices, max_retries_preemption, max_retries_failure))


@ray.remote(num_cpus=0.01)
def run_on_pod_ray(
    remote_fn: RemoteFunction,
    tpu_type: str,
    num_slices: int = 1,
    max_retries_preemption: int = 10000,
    max_retries_failure: int = 10,
):
    """
    Repeatedly run a function on a TPU pod until it succeeds or a maximum number of retries is reached.

    This function is a Ray remote function that can be called from anywhere in the Ray cluster.
    """
    if num_slices <= 0:
        raise ValueError("num_slices must be greater than 0")

    # failures here means the job failed due to an error in the remote function, not a preemption
    num_failures = 0
    # we include any kind of non-`remote_fn` failure in this count, including preemptions
    num_preemptions = 0
    attempt = 0

    slice_pool: list[SliceResource] = []
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
            try:
                slice_pool = _scale_slice_pool(slice_pool, tpu_type, num_slices)
            except Exception as e:
                logger.exception("Failed to prune dead slices or create new actors", exc_info=e)
                problems.append(e)
                num_preemptions += 1
                continue

            # If we're doing multislice, we need to get the slice info from the first actor
            head_slice_info = slice_pool[0].slice_info if len(slice_pool) > 1 else None

            # Ok finally time to run the remote function on all slices
            futures: list[ray.ObjectRef] = []  # one per host in each slice
            future_to_index: dict[ray.ObjectRef, int] = {}  # maps futures to their index in the results list
            global_index = 0  # index into results list

            for i, tpu_slice in enumerate(slice_pool):
                if head_slice_info is not None:
                    multislice_info = _multislice_info_from_head(head_slice_info, i, len(slice_pool))
                    mxla_env = _multislice_info_to_env_vars(multislice_info)
                else:
                    mxla_env = {}

                futures_for_slice = _start_fn_on_slice(tpu_slice, remote_fn, mxla_env)

                futures.extend(futures_for_slice)
                for future in futures_for_slice:
                    future_to_index[future] = global_index
                    global_index += 1

            tpu_results: list[_TpuRunResult | None] = [None] * len(futures)

            # We wait for jobs to finish one at a time. If a preemption or failure occurs, we cancel all
            pending_futures = list(futures)
            had_a_failure = False

            # check health of actors in the loop too
            actor_health_futures = [tpu_slice.actor.healthy.remote() for actor in slice_pool]

            while pending_futures and not had_a_failure:
                finished, pending_futures = ray.wait(pending_futures, num_returns=1, timeout=10.0)

                for f in finished:
                    try:
                        tpu_results[future_to_index[f]] = TpuSuccess(ray.get(f))
                    except RayError as e:
                        had_a_failure = True
                        problems.append(e)
                        tpu_results[future_to_index[f]] = _handle_ray_error(e)
                    except Exception as e:
                        logger.warning(f"Task {f} failed with unexpected error {e}. Will retry.")
                        had_a_failure = True
                        tpu_results[future_to_index[f]] = TpuRunError(e)

                if had_a_failure:
                    # skip health checks if we already had a failure
                    break

                # Check if any actors are unhealthy. We hit this if it's been 10 seconds or we got a result
                try:
                    actor_healths = ray.get(actor_health_futures)
                except RayError as e:
                    logger.warning("Failed to get actor healths", exc_info=e)
                    # assume things are bad
                    had_a_failure = True
                else:
                    for i, healthy in enumerate(actor_healths):
                        if not healthy:
                            logger.warning(f"Actor {slice_pool[i]} is unhealthy. Will retry.")
                            had_a_failure = True

            # Proactively cancel jobs if one fails.
            if had_a_failure and pending_futures:
                logger.info(f"Failure detected. Cancelling {len(pending_futures)} futures.")
                try:
                    for f in pending_futures:
                        ray.cancel(f, force=True)
                except Exception:
                    logger.exception("Failed to cancel pending futures")

                # Now, fill in the cancellations
                for f in pending_futures:
                    index = future_to_index.get(f)
                    if index is not None:
                        tpu_results[index] = TpuCancelled(
                            RuntimeError("Task was cancelled due to a failure in another task")
                        )
                    else:
                        logger.warning(f"Future {f} was not found in future_to_index. Skipping.")

            # Process results, figure out if we succeeded or failed or preempted
            out_results: list = []
            any_preempted = False
            any_failed = False
            any_cancelled = False

            for result in tpu_results:
                if isinstance(result, TpuSuccess):
                    out_results.append(result.result)
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
                    assert False, "We should never have None results here. "
                else:
                    raise RuntimeError(f"Unexpected result: {result}")

            # try to cleanup lockfiles
            cleanup_futs = []
            for tpu_slice in slice_pool:
                cleanup_futs.extend(_cleanup_lockfiles(tpu_slice))

            if cleanup_futs:
                try:
                    ray.get(cleanup_futs)
                except Exception as e:
                    logger.warning("Failed to cleanup lockfiles", exc_info=e)

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
                logger.info("All slices succeeded. Returning results.")
                return out_results
    except Exception as e:
        logger.exception("Unexpected error. This is a bug in Levanter. Please report it.", exc_info=e)
        raise
    finally:
        # Cleanup actors and placement groups
        logger.info("Cleaning up actors and placement groups")
        for tpu_slice in slice_pool:
            logger.info(f"Removing {tpu_slice.slice_info.slice_name} from pool.")
            _release_slice_resource(tpu_slice)
        slice_pool.clear()

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


def _stop_actor(actor: ActorHandle) -> None:
    try:
        # This is recommended by https://docs.ray.io/en/latest/ray-core/api/doc/ray.kill.html
        #
        # > If you want to kill the actor but let pending tasks finish, you can call actor.__ray_terminate__.remote()
        # > instead to queue a termination task. Any atexit handlers installed in the actor will be run in this case.
        #
        # NOTE: Not sure if this always returns an exception (because the actor will terminate before finishing)
        # but it doesn't really matter
        ray.get(actor.__ray_terminate__.remote(), timeout=_TERMINATE_ACTOR_TIMEOUT)
    except ActorDiedError:
        # This is expected because the actor will terminate within  __ray_terminate__() task,
        # so the task will never succeed.
        pass
    except GetTimeoutError as e:
        logger.warning(f"Failed to gracefully shut down actor in {_TERMINATE_ACTOR_TIMEOUT} seconds; killing it instead: {e}")
    finally:
        ray.kill(actor)


def _release_slice_resource(slice_resource: SliceResource) -> None:
    _stop_actor(slice_resource.actor)
    try:
        ray.util.remove_placement_group(slice_resource.placement_group)
    except Exception as e:
        logger.warning(f"Failed to remove placement group: {e}")


def _scale_slice_pool(slice_pool: list[SliceResource], tpu_type: str, num_slices: int) -> list[SliceResource]:
    """Scale the slice pool to the desired number of slices.

    Terminate unhealth slices, then allocate new slices to fill up the shortfall.

    This function expects to be called repeatedly in an outer retry loop until it succeeds.

    Returns a new pool. Does not mutate `slice_pool`."""
    # NOTE: Do not add a retry loop, as this function will be run in an outer retry loop.
    healthy_slices = _prune_dead_slices(slice_pool)
    del slice_pool  # Defensively prevent mutations to slice_pool
    if len(healthy_slices) >= num_slices:
        return healthy_slices

    # if we don't have enough slices, create more
    logger.info(f"Pool slices: {[tpu_slice.slice_info.slice_name for tpu_slice in healthy_slices]}")
    logger.info(f"Pool has {len(healthy_slices)} slices, but we want {num_slices}. Creating more slices.")
    actors = [SliceActor.options(resources={f"TPU-{tpu_type}-head": 1}).remote() for _ in range(num_slices - len(healthy_slices))]  # type: ignore

    actors_and_slice_info_awaitables = [(actor, actor.get_slice_info.remote()) for actor in actors]
    started_slices: list[SliceResource] = []
    logger.info(f"Waiting for {len(actors)} new actors to start...")
    for actor, slice_info_awaitable in actors_and_slice_info_awaitables:
        try:
            slice_info = ray.get(slice_info_awaitable, timeout=_START_ACTOR_TIMEOUT)
        except Exception as e:
            logger.exception(f"Actor {actor} failed to start: {e}")
            _stop_actor(actor)
            continue
        try:
            placement_group = _create_placement_group(slice_info.slice_name, slice_info.num_hosts, slice_info.num_tpus_per_host)
        except Exception as e:
            logger.exception(f"Could not create placement group for slice {slice_info.slice_name}: {e}")
            _stop_actor(actor)
            continue
        logger.info(f"Actor {actor} for slice {slice_info.slice_name} started.")
        started_slices.append(SliceResource(actor, slice_info, placement_group))

    slices_and_placement_group_ready_awaitables = [(tpu_slice, tpu_slice.placement_group.ready()) for tpu_slice in started_slices]
    logger.info(f"Waiting for {len(slices_and_placement_group_ready_awaitables)} placement groups to be ready...")
    for tpu_slice, placement_group_ready_awaitable in slices_and_placement_group_ready_awaitables:
        try:
            ray.get(placement_group_ready_awaitable, timeout=_PLACEMENT_GROUP_READY_TIMEOUT)
        except Exception as e:
            logger.exception(f"Placement group for {tpu_slice.slice_info.slice_name} failed to become ready: {e}")
            _release_slice_resource(tpu_slice)
            continue
        logger.info(f"Placement group for {tpu_slice.slice_info.slice_name} ready.")
        healthy_slices.append(tpu_slice)
        logger.info(f"Added {tpu_slice.slice_info.slice_name} to pool.")

    logger.info(f"Pool slices: {[tpu_slice.slice_info.slice_name for tpu_slice in healthy_slices]}")
    logger.info(f"Pool ready with {len(healthy_slices)} actors.")

    if len(healthy_slices) < num_slices:
        raise Exception(f"Wanted {num_slices} slices but only acquired {len(healthy_slices)} slices.")
    return healthy_slices


def _start_fn_on_slice(tpu_slice: SliceResource, remote_fn: RemoteFunction, mxla_env: dict | None) -> list[ray.ObjectRef]:
    """
    Start the remote function on a slice of the TPU pod.
    """
    runtime_env = remote_fn._runtime_env or {}
    if mxla_env is not None:
        mxla_env = dict(env_vars=mxla_env)
        runtime_env = mergedeep.merge({}, runtime_env, mxla_env, strategy=mergedeep.Strategy.ADDITIVE)
    futures_for_slice = [
        remote_fn.options(
            runtime_env=runtime_env,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=tpu_slice.placement_group,
                placement_group_bundle_index=host,
                placement_group_capture_child_tasks=False,
            ),
            **tpu_slice.slice_info.as_ray_resources_kwargs(),
        ).remote()
        for host in range(tpu_slice.slice_info.num_hosts)
    ]
    return futures_for_slice


def _cleanup_lockfiles(tpu_slice: SliceResource) -> list[ray.ObjectRef]:
    return _start_fn_on_slice(
        tpu_slice,
        _hacky_remove_tpu_lockfile_ray,
        mxla_env=None,  # no need for MXLA env vars here
    )


def _prune_dead_slices(pool: list[SliceResource]) -> list[SliceResource]:
    """Prune dead or unhealthy slices from the pool.

    Returns a new pool. Does not mutate `pool`."""
    healthy_slices = []
    unhealthy_slices = []
    slices_and_health = [(tpu_slice, tpu_slice.actor.healthy.remote()) for tpu_slice in pool]
    del pool  # Defensively prevent pool from being mutated
    for tpu_slice, health in slices_and_health:
        try:
            if ray.get(health, timeout=_HEALTH_CHECK_TIMEOUT):
                healthy_slices.append(tpu_slice)
            else:
                logger.warning(f"Slice {tpu_slice.slice_info.slice_name} is unhealthy. Removing from pool.")
                unhealthy_slices.append(tpu_slice)
        except (RayActorError, RayTaskError, ActorDiedError, ActorUnavailableError, GetTimeoutError) as e:
            logger.warning(f"Slice {tpu_slice.slice_info.slice_name} is dead or unavailable. Removing from pool. Error: {e}")
            unhealthy_slices.append(tpu_slice)

    # NOTE: For simplicity, we serially process the unhealthy slices, rather than doing it in parallel.
    for unhealthy_slice in unhealthy_slices:
        # This is a synchronous blocking call.
        _release_slice_resource(unhealthy_slice)

    return healthy_slices


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
    return ray.get(
        run_on_pod(remote_fn, tpu_type, num_slices=num_slices, max_retries_failure=0, max_retries_preemption=0)
    )


def run_on_pod_resumable(remote_fn: RemoteFunction | Callable, tpu_type: str, max_retries_preemption: int = 1_000_000, max_retries_failure: int = 10):
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
    return run_on_pod(
        remote_fn,
        tpu_type,
        num_slices=1,
        max_retries_preemption=max_retries_preemption,
        max_retries_failure=max_retries_failure,
    )


def run_on_pod_multislice_resumable(
    remote_fn: RemoteFunction | Callable, tpu_type: str, num_slices: int, max_retries_preemption: int = 1_000_000, max_retries_failure: int = 10
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
    return run_on_pod(
        remote_fn,
        tpu_type,
        num_slices=num_slices,
        max_retries_preemption=max_retries_preemption,
        max_retries_failure=max_retries_failure,
    )


def _run_command(*args, **kwargs):
    return subprocess.check_call(args, **kwargs)


def run_docker_on_pod(
    image_id: str, command: Sequence[str], *, tpu_type: str, num_slices: int, env: dict, name: str = "levanter", retries: int = 10
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

    run_on_pod(
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


@ray.remote(num_cpus=0.0)
def _hacky_remove_tpu_lockfile_ray():
    _hacky_remove_tpu_lockfile()


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
        success, value = queue.get(timeout=1)
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
    env = dict(env)
    if "TERM" not in env:
        env["TERM"] = "dumb"

    if "TF_CPP_MIN_LOG_LEVEL" not in env:
        # Suppress TensorFlow logs, which can be very verbose
        env["TF_CPP_MIN_LOG_LEVEL"] = "3"

    return env


if __name__ == "__main__":
    main()

    # leaving this here for testing purposes
    # ray.init()
    # tpu_type = "v4-8"
    # num_slices = 2
    #
    # @ray.remote(max_calls=1)
    # def fn():
    #     import jax
    #     import jax.random as jrandom
    #     from jax.lax import with_sharding_constraint
    #     from jax.sharding import Mesh
    #     from jax.sharding import PartitionSpec as P
    #
    #     mesh = Mesh(jax.devices("tpu"), ("x",))
    #     print(jax.devices())
    #
    #     @jax.jit
    #     def init():
    #         with mesh:
    #             x = jrandom.normal(jrandom.PRNGKey(0), (32,))
    #             weights = jrandom.normal(jrandom.PRNGKey(1), (32, 4))
    #             bias = jrandom.normal(jrandom.PRNGKey(2), (4,))
    #
    #             x_sharded = with_sharding_constraint(x, P("x"))
    #             weights_sharded = with_sharding_constraint(weights, P("x"))
    #             return x_sharded, weights_sharded, bias
    #
    #     x, weights, bias = init()
    #
    #     @jax.jit
    #     def layer(x, weights, bias):
    #         with mesh:
    #             return with_sharding_constraint(jax.nn.sigmoid(x @ weights + bias), P())
    #
    #     out = layer(x, weights, bias)
    #
    #     import numpy
    #
    #     return numpy.array(out)
    #
    # results = ray.get(run_on_pod_new(fn, tpu_type, num_slices=num_slices))
    #
    # print(f"Results: {results}")
