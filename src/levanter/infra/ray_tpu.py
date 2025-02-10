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

@ray.remote
class TPUHeadNodeActor:
    def __init__(self):
        self.pod_name = ray.util.accelerators.tpu.get_current_pod_name()
        self.num_hosts = ray.util.accelerators.tpu.get_current_pod_worker_count()
        self.ip = socket.gethostbyname(socket.gethostname())
        self.worker_actors = None

    def get_info(self):
        return self.pod_name, self.num_hosts, self.ip
    
    def run(self, remote_fn) -> _TpuRunResult:
        if self.worker_actors is not None:
            raise RuntimeError("Actors already created")
        self.worker_actors = [TPUWorkerActor.remote() for _ in range(self.num_hosts)]
        futures = [actor.get_info.remote() for actor in self.worker_actors]
        try:
            worker_infos = ray.get(futures)
            for i, (actor, info) in enumerate(zip(self.worker_actors, worker_infos)):
                if info[0] is None:
                    raise RuntimeError(f"Worker actor {i} returned invalid info: {info}")
        except Exception as e:
            self.cleanup()
            raise RuntimeError("Failed to initialize worker actors") from e
        
        # Start process on all workers
        try:
            futures = [
                actor.run.remote(remote_fn, self.ip, i, self.num_hosts) 
                for i, actor in enumerate(self.worker_actors)
            ]
            run_infos = ray.get(futures)

            for actor, info in zip(self.worker_actors, run_infos):
                if not isinstance(info, _TpuInfo):
                    raise RuntimeError(f"Worker actor {actor} returned invalid info: {info}")

            wait_futures = [actor.wait.remote() for actor in self.worker_actors]
            results = ray.get(wait_futures)

            logger.info("TPU job finished, with info", run_infos[0])
            return TpuSuccess(run_infos[0], results)
        except RayError as e:
            logger.exception(f"Failed to run job on {self.pod_name}")
            return _handle_ray_error(run_infos[0], e)
        except Exception as e:
            logger.exception(f"Failed to run job on {self.pod_name}")
            raise RuntimeError(f"Failed to run job on {self.pod_name}") from e
        
    def cleanup(self):
        if self.worker_actors is not None:
            for actor in self.worker_actors:
                try:
                    ray.get(actor.cleanup.remote())
                    ray.kill(actor)
                except Exception:
                    logger.exception(f"Failed to kill worker actor {actor}")
            self.worker_actors = None

@ray.remote
class TPUWorkerActor:
    def __init__(self):
        self.pod_name = ray.util.accelerators.tpu.get_current_pod_name()
        self.num_hosts = ray.util.accelerators.tpu.get_current_pod_worker_count()
        self.ip = socket.gethostbyname(socket.gethostname())
        self.process: multiprocessing.Process | None = None
        self.queue: multiprocessing.Queue | None = None

    def get_info(self):
        return self.pod_name, self.num_hosts, self.ip
    
    def run(self, remote_fn, coordinator_ip, slice_id, num_slices) -> _TpuRunResult:
        if self.process is not None:
            raise RuntimeError("Process already started")
        
        port = 8081
        mxla_env = {
            "MEGASCALE_COORDINATOR_ADDRESS": f"{coordinator_ip}:{port}",
            "MEGASCALE_NUM_SLICES": str(num_slices),
            "MEGASCALE_PORT": f"{port}",
            "MEGASCALE_SLICE_ID": str(slice_id),
        }

        # redecorate remote function
        remote_fn, tpu_name = _redecorate_remote_fn_for_tpu(remote_fn, self.num_hosts, env_vars=mxla_env)
        info = _TpuInfo(tpu_name, "ACTIVE", "TPU")

        # Create queue for process communication
        self.queue = multiprocessing.Queue()
        self.process = multiprocessing.Process(
            target=self._run_in_process, 
            args=(remote_fn, self.queue)
        )
        self.process.start()
        return info

    def _run_in_process(self, fn, queue):
        try:
            future = fn.remote()
            result = ray.get(future)
            queue.put((True, result))
        except Exception as e:
            info = ser_exc_info(e)
            queue.put((False, info))

    def wait(self) -> object:
        if self.process is None or self.queue is None:
            raise RuntimeError("No process running")
        
        self.process.join()
        try:
            success, value = self.queue.get()
            self.process = None
            self.queue = None

            if success:
                return value
            else:
                value.reraise()  # Reraise the exception with original traceback
        except multiprocessing.queues.Empty:
            logger.log("Process timed out")
            self.cleanup()
                    
    def cleanup(self):
        logger.info(f"Cleaning up worker actor {self.pod_name}")
        if self.process is not None:
            self.process.terminate()
            self.process.join()
            self.process = None
            self.queue = None


def _redecorate_remote_fn_for_tpu(remote_fn, num_hosts, **runtime_env):
    if not isinstance(remote_fn, RemoteFunction):
        logger.info("CATHY log: decorating non remote function")
        remote_fn = ray.remote(remote_fn)
    else:
        logger.info("CATHY log: decorating remote function")

    tpu_name = ray.util.accelerators.tpu.get_current_pod_name()  # -> my-tpu
    num_tpus_per_host = TPUAcceleratorManager.get_current_node_num_accelerators()  # -> 8

    # ray doesn't merge the runtime envs properly, so we have to do it ourselves
    # we need to do a deep merge
    sources = [e for e in [remote_fn._runtime_env, runtime_env] if e is not None]
    runtime_env = mergedeep.merge({}, *sources, strategy=mergedeep.Strategy.ADDITIVE)

    new_options = dict(remote_fn._default_options)
    new_options.update(
        runtime_env=runtime_env,
        resources={tpu_name: 1, "TPU": num_tpus_per_host},
    )
    remote_fn = RemoteFunction(
        language=remote_fn._language,
        function=remote_fn._function,
        function_descriptor=remote_fn._function_descriptor,
        task_options=new_options,
    )

    logger.info(f"Running on TPU {tpu_name} with {num_hosts} hosts and {num_tpus_per_host} TPUs per host")
    return remote_fn, tpu_name

def run_on_pod_resumable(
    remote_fn, 
    tpu_type: str, 
    num_slices: int = 1,
    max_retries_preemption: int = int(1e6), 
    max_retries_failure: int = 10
) -> object | list[object]:
    """
    Repeatedly run a function on a TPU pod until it succeeds or a maximum number of retries is reached.
    Handles both single-slice and multi-slice cases.

    Args:
        remote_fn: A remote function that takes no arguments
        tpu_type: The type of TPU to run on, e.g. "v4-32"
        num_slices: The number of slices to run (default=1 for single-slice mode)
        max_retries_preemption: The maximum number of times to retry if the job is preempted
        max_retries_failure: The maximum number of times to retry if the job fails

    Returns:
        For single-slice: The result of the function
        For multi-slice: List of results from each slice
        TODO(cathy): refactor to always return a list of results
    """
    num_failures = 0
    num_preemptions = 0
    attempt = 0
    problem: Exception | None = None
    HeadNodeActor = TPUHeadNodeActor.options(resources={f"TPU-{tpu_type}-head": 1})

    while num_failures < max_retries_failure and num_preemptions < max_retries_preemption:
        logger.info(f"Running on TPU {tpu_type}. Attempt {attempt}")
        attempt += 1
        problem = None
        
        # Create head node actors, one per slice
        head_actors = [HeadNodeActor.remote() for _ in range(num_slices)]
        infos_futures = [actor.get_info.remote() for actor in head_actors]
        try:
            infos = ray.get(infos_futures)
            for info in infos:
                if info[0] is None:
                    raise RuntimeError(f"Worker actor {actor} returned invalid info: {info}")

            # Run the job on all slices
            futures = [actor.run.remote(remote_fn) for actor in head_actors]
            outs = ray.get(futures)
            
            # Check results from all slices
            results = []
            all_succeeded = True
            for out in outs:
                if isinstance(out, TpuSuccess):
                    results.append(out.result)
                elif isinstance(out, TpuPreempted):
                    problem = out.error
                    num_preemptions += 1
                    logger.warning(f"Preempted {num_preemptions} times. {problem}", exc_info=problem)
                    all_succeeded = False
                    break
                elif isinstance(out, TpuFailed):
                    num_preemptions += 1
                    logger.warning(f"TPU node failure. Treating as preempted: {num_preemptions} times")
                    all_succeeded = False
                    break
                elif isinstance(out, TpuRunError):
                    problem = out.error
                    num_failures += 1
                    logger.warning(f"Failed {num_failures} times", exc_info=problem)
                    all_succeeded = False
                    break
                else:
                    raise RuntimeError(f"Unexpected result: {out}")
            
            if all_succeeded:
                logger.info("Success")
                return results[0] if num_slices == 1 else results
                
        except ray.exceptions.RayTaskError as e:
            problem = e
            if "preempted" in str(e).lower():
                num_preemptions += 1
                logger.warning(f"Preempted {num_preemptions} times, {e}")
            else:
                num_failures += 1
                logger.warning(f"Failed {num_failures} times", exc_info=e)
        except Exception as e:
            problem = e
            num_failures += 1
            if num_failures >= max_retries_failure:
                logger.exception("Failed too many times", exc_info=e)
                raise e
            else:
                logger.warning(f"Failed {num_failures} times", exc_info=e)
        finally:
            try:
                # Kill all head actors
                for actor in head_actors:
                    ray.get(actor.cleanup.remote())
                    ray.kill(actor)
            except Exception:
                logger.exception("Failed to kill head actors during cleanup")

    if num_preemptions >= max_retries_preemption:
        raise RuntimeError("Preempted too many times") from problem
    elif num_failures >= max_retries_failure:
        raise RuntimeError("Failed too many times") from problem


def run_on_pod_multislice_resumable(
    remote_fn, 
    tpu_type: str, 
    num_slices,
    max_retries_preemption: int = int(1e6),
    max_retries_failure: int = 10
) -> list[object]:
    """
    TODO(cathy): deprecated?
    """
    return run_on_pod_resumable(remote_fn, tpu_type, num_slices, max_retries_preemption, max_retries_failure)

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

    run_on_pod_resumable(
        ray.remote(run_docker), tpu_type=tpu_type, num_slices=num_slices, max_retries_failure=retries, max_retries_preemption=10000
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
