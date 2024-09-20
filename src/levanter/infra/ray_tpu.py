import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Sequence

import ray
from ray.exceptions import NodeDiedError, RayError, RaySystemError, RayTaskError, WorkerCrashedError

from levanter.infra.cli_helpers import make_docker_run_command


# CF https://gist.github.com/allenwang28/e3400b9e9212b50aa1cda55ebeccea60

logger = logging.getLogger(__name__)

tpu_type_to_resources = {
    "v2": {"TPU": 8},
    "v3": {"TPU": 8},
    "v4": {"TPU": 4},
    "v5litepod": {"TPU": 8},
    "v5 lite": {"TPU": 8},
}


def _hacky_remove_tpu_lockfile():
    try:
        os.unlink("/tmp/libtpu_lockfile")
    except FileNotFoundError:
        pass


@dataclass
class _TpuInfo:
    name: str
    state: str
    kind: str


# My kingdom for ADTs
@dataclass
class _TpuRunResult:
    info: _TpuInfo


@dataclass
class TpuSuccess(_TpuRunResult):
    result: object


@dataclass
class TpuPreempted(_TpuRunResult):
    error: Exception

    def __post_init__(self):
        raise "No."


@dataclass
class TpuFailed(_TpuRunResult):
    error: Exception


@dataclass
class TpuRunError(_TpuRunResult):
    error: Exception


def run_on_pod(remote_fn, tpu_type):
    """
    Args:
        remote_fn: A remote function that takes no arguments
        tpu_type: The type of TPU to run on, e.g. "v4-32"
    """

    @ray.remote(resources={f"TPU-{tpu_type}-head": 1})
    def do_run(remote_fn) -> _TpuRunResult:
        tpu_name = ray.util.accelerators.tpu.get_current_pod_name()  # -> my-tpu
        num_hosts = ray.util.accelerators.tpu.get_current_pod_worker_count()  # -> 4
        tpu_version = tpu_type.split("-")[0]
        resource_spec = tpu_type_to_resources[tpu_version]
        remote_fn = remote_fn.options(resources={tpu_name: 1, **resource_spec})

        info = _TpuInfo(tpu_name, "ACTIVE", "TPU")
        try:
            try:
                out = ray.get([remote_fn.remote() for _ in range(num_hosts)])
                logger.info("tpu job finished")
                return TpuSuccess(info, out)
            except RayError as e:
                return _handle_ray_error(info, e)
        finally:
            # remove the tpu lockfile on each host
            logger.info("Removing lockfiles")
            _rm_lockfile = ray.remote(resources={tpu_name: 1, **resource_spec})(_hacky_remove_tpu_lockfile)
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
            match out:
                case TpuSuccess(result):
                    logger.info("Success")
                    return result
                case TpuPreempted(error=e):
                    num_preemptions += 1
                    print(f"Preempted {num_preemptions} times. {e}")
                    logger.warning(f"Preempted {num_preemptions} times. {e}", exc_info=e)
                case TpuFailed():
                    num_preemptions += 1
                    logger.warning(f"TPU node failure. Treating as preempted: {num_preemptions} times")
                case TpuRunError(e):
                    num_failures += 1
                    logger.warning(f"Failed {num_failures} times")
                    logger.exception(e)
                case _:
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


def run_docker_on_pod(image_id: str, command: Sequence[str], *, tpu_type: str, env: dict, name="levanter"):
    command = make_docker_run_command(image_id, command, env=env, foreground=True, name=name)

    def try_to_run():
        try:
            try:
                _run_command("sudo", "docker", "rm", "-f", name)
            except subprocess.CalledProcessError:
                pass
            return _run_command(*command)
        except subprocess.CalledProcessError as e:
            logger.exception("Failed to run docker command")
            raise e

    run_on_pod_resumable(ray.remote(try_to_run), tpu_type=tpu_type)


def _handle_ray_error(tpu_info: _TpuInfo, e: RayError):
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


if __name__ == "__main__":
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
    ray.init()

    run_docker_on_pod(
        "ghcr.io/stanford-crfm/levanter-tpu:latest",
        # ["python", "-c", "while True: pass"],
        "python -m levanter.main.train_lm --config_path config/gpt2_small_fast.yaml --trainer.checkpointer.base_path"
        " gs://levanter-checkpoints/gpt-itest/ --trainer.checkpointer.save_interval 30m".split(" "),
        tpu_type="v4-64",
        env={"WANDB_MODE": "offline"},
        name="levanter",
    )
