import dataclasses
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Sequence

import draccus
import ray
from ray.exceptions import NodeDiedError, RayError, RaySystemError, RayTaskError, WorkerCrashedError
from ray.remote_function import RemoteFunction

from levanter.infra.cli_helpers import make_docker_run_command


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


def run_on_pod(remote_fn: RemoteFunction, tpu_type: str):
    """
    Run a remote function on a TPU pod.

    Args:
        remote_fn: A remote function that takes no arguments
        tpu_type: The type of TPU to run on, e.g. "v4-32"
    """

    @ray.remote(resources={f"TPU-{tpu_type}-head": 1})
    def do_run(remote_fn) -> _TpuRunResult:
        tpu_name = ray.util.accelerators.tpu.get_current_pod_name()  # -> my-tpu
        num_hosts = ray.util.accelerators.tpu.get_current_pod_worker_count()  # -> 4
        remote_fn = remote_fn.options(resources={tpu_name: 1, "TPU": 1})

        info = _TpuInfo(tpu_name, "ACTIVE", "TPU")
        try:
            try:
                out = ray.get([remote_fn.remote() for _ in range(num_hosts)])
                logger.info("TPU job finished")
                return TpuSuccess(info, out)
            except RayError as e:
                return _handle_ray_error(info, e)
        finally:
            # remove the tpu lockfile on each host
            logger.debug("Removing lockfiles")
            _rm_lockfile = ray.remote(resources={tpu_name: 1, "TPU": 1})(_hacky_remove_tpu_lockfile)
            try:
                ray.get([_rm_lockfile.remote() for _ in range(num_hosts)])
            except Exception:
                logger.exception("Failed to remove lockfile")
                # swallow the exception

    return do_run.remote(remote_fn)


def run_on_pod_resumable(remote_fn, tpu_type, max_retries_preemption=1e6, max_retries_failure=10):
    """
    Repeatedly run a function on a TPU pod until it succeeds or a maximum number of retries is reached.

    Args:
        remote_fn: A remote function that takes no arguments
        tpu_type: The type of TPU to run on, e.g. "v4-32"
        max_retries_preemption: The maximum number of times to retry if the job is preempted
        max_retries_failure: The maximum number of times to retry if the job fails
    """
    num_failures = 0
    num_preemptions = 0

    while num_failures < max_retries_failure and num_preemptions < max_retries_preemption:
        try:
            out = ray.get(run_on_pod(remote_fn, tpu_type))
            if isinstance(out, TpuSuccess):
                result = out.result
                logger.info("Success")
                return result
            elif isinstance(out, TpuPreempted):
                e = out.error
                num_preemptions += 1
                print(f"Preempted {num_preemptions} times. {e}")
                logger.warning(f"Preempted {num_preemptions} times. {e}", exc_info=e)
            elif isinstance(out, TpuFailed):
                num_preemptions += 1
                logger.warning(f"TPU node failure. Treating as preempted: {num_preemptions} times")
            elif isinstance(out, TpuRunError):
                e = out.error
                num_failures += 1
                logger.warning(f"Failed {num_failures} times")
                logger.exception(e)
            else:
                raise RuntimeError(f"Unexpected result: {out}")
        except ray.exceptions.RayTaskError as e:
            if "preempted" in str(e):
                num_preemptions += 1
                logger.warning(f"Preempted {num_preemptions} times, {e}")
            else:
                num_failures += 1
                logger.warning(f"Failed {num_failures} times")
        except Exception as e:
            num_failures += 1
            logger.warning(f"Failed {num_failures} times")
            logger.exception(e)
            if num_failures >= max_retries_failure:
                raise e

        if num_preemptions >= max_retries_preemption:
            raise RuntimeError("Preempted too many times")
        elif num_failures >= max_retries_failure:
            raise RuntimeError("Failed too many times")


def _run_command(*args, **kwargs):
    return subprocess.check_call(args, **kwargs)


def run_docker_on_pod(image_id: str, command: Sequence[str], *, tpu_type: str, env: dict, name="levanter", retries=10):
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
        ray.remote(run_docker), tpu_type=tpu_type, max_retries_failure=retries, max_retries_preemption=10000
    )


def _kill_old_container(name):
    try:
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
        print("Node died")
        logger.exception("Node died", exc_info=e)
        return TpuPreempted(tpu_info, e)
    elif isinstance(e, WorkerCrashedError):
        print("Worker crashed")
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
            print("Preempted")
            logger.exception("Preempted", exc_info=e)
            return TpuPreempted(tpu_info, e)

        logger.exception(f"Task error {e}", exc_info=e)
        return TpuRunError(tpu_info, e)

    else:
        logger.exception("Unknown error", exc_info=e)
        return TpuRunError(tpu_info, e)


@dataclass
class RunOnPodConfig:
    image_id: str
    command: list[str] | str
    tpu_type: str
    env: dict = dataclasses.field(default_factory=dict)
    name: str = "levanter"
    retries: int = 10


@draccus.wrap()
def main(args: RunOnPodConfig):
    """
    Run a command on a TPU pod. This is a wrapper around `run_docker_on_pod` that takes a config object as a CLI.

    We use this via infra/launch_on_ray.py to run docker containers on TPUs.
    """
    ray.init()

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
    )


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
