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
from ray.dashboard.modules.job.sdk import JobSubmissionClient
from ray.exceptions import NodeDiedError, RayError, RaySystemError, RayTaskError, WorkerCrashedError
from ray.remote_function import RemoteFunction

from levanter.infra.docker import make_docker_run_command
from levanter.utils.ray_utils import ser_exc_info


# CF https://gist.github.com/allenwang28/e3400b9e9212b50aa1cda55ebeccea60

logger = logging.getLogger("ray")


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

    info: _TpuInfo


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

        info = _TpuInfo(tpu_name, "ACTIVE", "TPU")
        futures = [remote_fn.remote() for _ in range(num_hosts)]
        try:
            out = ray.get(futures)
            logger.info("TPU job finished")
            return TpuSuccess(info, out)
        except RayError as e:
            _cancel_all_futures(futures)
            return _handle_ray_error(info, e)
        except Exception as e:
            _cancel_all_futures(futures)
            return TpuFailed(info, e)

    return do_run.remote(remote_fn)


def _cancel_all_futures(futures):
    for f in futures:
        try:
            ray.cancel(f)
        except Exception:
            logger.exception("Failed to kill job after primary failure")


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

            info = _TpuInfo(tpu_name, "ACTIVE", "TPU")
            futures = [remote_fn.remote() for _ in range(self.num_hosts)]
            try:
                out = ray.get(futures)
                logger.info("TPU job finished")
                return TpuSuccess(info, out)
            except RayError as e:
                logger.exception(f"Ray error {e}. Killing futures for this slice")
                _cancel_all_futures(futures)
                return _handle_ray_error(info, e)
            except Exception as e:
                logger.exception(f"Exception {e}")
                _cancel_all_futures(futures)
                return TpuFailed(info, e)

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
            num_preemptions += 1
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

    if num_slices == 1:
        run_on_pod_resumable(
            ray.remote(run_docker), tpu_type=tpu_type, max_retries_failure=retries, max_retries_preemption=10000
        )
    else:
        run_on_pod_multislice_resumable(
            ray.remote(run_docker),
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


def _handle_ray_error(tpu_info: _TpuInfo, e: RayError):
    """
    Handle a Ray error that occurred on a TPU pod. Tries to determine if the error was due to a
    node failure or preemption or just an application error.
    """
    # treat node failures as preemptions
    if isinstance(e, NodeDiedError):
        logger.exception("Node died", exc_info=e)
        return TpuPreempted(tpu_info, e)
    elif isinstance(e, ray.exceptions.ActorUnavailableError | ray.exceptions.ActorDiedError):
        logger.exception("Actor died", exc_info=e)
        return TpuPreempted(tpu_info, e)
    elif isinstance(e, WorkerCrashedError):
        logger.exception("Worker crashed", exc_info=e)
        return TpuPreempted(tpu_info, e)
    elif isinstance(e, RaySystemError):
        logger.exception("System error", exc_info=e)
        return TpuRunError(tpu_info, e)
    elif isinstance(e, RayTaskError):
        # node preemptions don't always show up as one of the above errors and can just be a RayTaskError. We have
        # to try to sniff out the TPU's status.
        from levanter.infra.tpus import get_current_tpu_is_preempted

        if get_current_tpu_is_preempted():
            logger.exception("Preempted", exc_info=e)
            return TpuPreempted(tpu_info, e)

        logger.exception(f"Task error {e}", exc_info=e)
        return TpuRunError(tpu_info, e)

    else:
        logger.exception("Unknown error", exc_info=e)
        return TpuRunError(tpu_info, e)


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
        success, value = queue.get(timeout=10)
    except QueueEmpty:
        logger.error("Process timed out")
        process.terminate()
        raise RuntimeError("Process timed out")

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
    main()

    # leaving this here for testing purposes
    # ray.init()
    # tpu_type = "v4-64"
    # @ray.remote
    # def fn():
    #     import jax
    #     import jax.random as jrandom
    #     from jax.lax import with_sharding_constraint
    #     from jax.sharding import PartitionSpec as P, Mesh
    #     mesh = Mesh(jax.devices("tpu"), ("x",))
    #     sharding = jax.sharding.NamedSharding(mesh, P('x'))
    #     print(jax.devices())
    #
    #     @jax.jit
    #     def init():
    #         x = jrandom.normal(jrandom.PRNGKey(0), (32,))
    #         weights = jrandom.normal(jrandom.PRNGKey(1), (32, 4))
    #         bias = jrandom.normal(jrandom.PRNGKey(2), (4,))
    #
    #         x_sharded = jax.device_put(x, sharding)
    #         weights_sharded = jax.device_put(weights, sharding)
    #         return x_sharded, weights_sharded, bias
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
    #     print(out)
    #     import numpy
    #     return numpy.array(out)
    # results = ray.get(run_on_pod(fn, tpu_type))
    # print(results)
