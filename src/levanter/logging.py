import contextlib
import logging as pylogging
import time
from pathlib import Path
from typing import List, Optional, Union

import equinox
import jax
import jax.numpy as jnp
from equinox import is_array
from optax import MultiStepsState

import wandb

from levanter.compat.torch_serialization import _is_named_tuple
from levanter.utils.jax_utils import jnp_to_python


logger = pylogging.getLogger(__name__)


def log_optimizer_hyperparams(opt_state, prefix: Optional[str] = None, *, step=None):
    if isinstance(opt_state, MultiStepsState):
        opt_state = opt_state.inner_opt_state

    def wrap_key(key):
        if prefix:
            return f"{prefix}/{key}"
        return key

    if hasattr(opt_state, "hyperparams"):
        params = {wrap_key(k): jnp_to_python(v) for k, v in opt_state.hyperparams.items()}
        # print(params)
        wandb.log(params, step=step)


def log_optimizer_state(opt_state, prefix: Optional[str] = None, *, step=None):
    if hasattr(opt_state, "inner_opt_state"):
        return log_optimizer_state(opt_state.inner_opt_state, prefix=prefix, step=step)
    elif hasattr(opt_state, "inner_state"):
        return log_optimizer_state(opt_state.inner_state, prefix=prefix, step=step)
    elif isinstance(opt_state, tuple) and not _is_named_tuple(opt_state):
        for substate in opt_state:
            log_optimizer_state(substate, prefix=prefix, step=step)
    else:
        _log_opt_state(opt_state, prefix=prefix, step=step)


@equinox.filter_jit
def _log_opt_state(opt_state, prefix: Optional[str] = None, *, step=None):
    def wrap_key(key):
        if prefix:
            return f"{prefix}/{key.replace('.', '/')}"
        return key

    from levanter.compat.torch_serialization import jax_tree_to_state_dict
    state_dict = jax_tree_to_state_dict(opt_state)

    log_dict = {}

    for k, v in state_dict.items():
        if k.endswith("count"):
            continue
        if is_array(v):
            v = jnp.sum(v ** 2)
            log_dict[wrap_key(f"norms/{k}")] = v
        else:
            log_dict[wrap_key(k)] = v

    jittable_wandb_log(log_dict, step=step)



def init_logger(path: Union[str, Path], level: int = pylogging.INFO) -> None:
    """
    Initialize logging.Logger with the appropriate name, console, and file handlers.

    :param path: Path for writing log file
    :param level: Default logging level
    """
    process_index = jax.process_index()
    log_format = f"%(asctime)s - {process_index} - %(name)s - %(filename)s:%(lineno)d - %(levelname)s :: %(message)s"
    # use ISO 8601 format for timestamps, except no TZ, because who cares
    date_format = "%Y-%m-%dT%H:%M:%S"

    handlers: List[pylogging.Handler] = [pylogging.FileHandler(path, mode="a"), pylogging.StreamHandler()]

    # Create Root Logger w/ Base Formatting
    pylogging.basicConfig(level=level, format=log_format, datefmt=date_format, handlers=handlers, force=True)


def save_xla_dumps_to_wandb(initial_time: float):
    import os

    # attempt to parse xla_flags to see if we're dumping assembly files
    flags = os.getenv("XLA_FLAGS", None)
    if flags is not None and "xla_dump_to" in flags:
        # parse the path
        # this isn't robust to quotes
        path = flags.split("xla_dump_to=")[1].split(" ")[0]
        logger.info(f"Found xla_dump_to={path}, logging to wandb")
        if wandb.run:
            # only want to save the files that were generated during this run
            # XLA_FLAGS has to be set before the first jax call, so we can't just set it in the middle of the run
            # which means it's a pain to control where the files are saved
            # so we just save all the files that were generated during this run
            # this is a bit hacky, but it works
            def include_file(path: str):
                return os.path.getmtime(path) > initial_time

            wandb.run.log_code(root=path, name="xla_dumps", include_fn=include_file)
    else:
        logger.warning("XLA_FLAGS is not set to dump to a path, so we can't save the dumps to wandb")


@contextlib.contextmanager
def capture_time():
    start = time.perf_counter()
    done = False

    def fn():
        if done:
            return end - start
        else:
            return time.perf_counter() - start

    yield fn
    end = time.time()


@contextlib.contextmanager
def log_time_to_wandb(name: str, *, step=None):
    with capture_time() as fn:
        yield fn
    wandb.log({name: fn()}, step=step)


def jittable_wandb_log(data, *, step=None):
    """uses jax effect callback to log to wandb from the host"""
    if is_wandb_available():
        jax.debug.callback(wandb.log, data, step=step)


def is_wandb_available():
    return wandb is not None and wandb.run is not None
