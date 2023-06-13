import atexit
import inspect
import logging
import os
import sys
import tempfile
import urllib.parse
from dataclasses import dataclass
from datetime import timedelta
from functools import wraps
from typing import List, Optional, Type, Union

import fsspec
import jax
import jmp
import pyrallis
from fsspec import AbstractFileSystem
from jax._src.clusters import SlurmCluster, TpuCluster
from pyrallis import parse

from levanter.distributed import LevanterSlurmCluster, auto_ray_cluster
from levanter.utils.datetime_utils import encode_timedelta, parse_timedelta


logger = logging.getLogger(__name__)

JsonAtom = Union[str, int, float, bool, None]


@dataclass(frozen=True)
class DistributedConfig:
    coordinator_address: Optional[str] = None  # if None, we'll use the default coordinator address (for TPU or GPU)
    num_processes: Optional[int] = None
    process_id: Optional[int] = None
    local_device_ids: Optional[Union[int, List[int]]] = None

    def _is_distributed(self):
        if (
            (self.coordinator_address is not None)
            or (self.num_processes is not None)
            or (self.process_id is not None)
            or (self.local_device_ids is not None)
        ):
            return True

        # jax will automatically detect slurm or tpu, so we check those too. This is a bit fragile
        # since it depends on the jax internals, but it's the best we can do
        if SlurmCluster.is_env_present() or TpuCluster.is_env_present():
            return True

        return False

    def initialize(self):
        if self._is_distributed():
            device_ids = self.local_device_ids
            coordinator_address = self.coordinator_address

            if LevanterSlurmCluster.is_env_present():
                if device_ids is None:
                    device_ids = LevanterSlurmCluster.get_local_device_ids_for_process()

                if coordinator_address is None:
                    coordinator_address = LevanterSlurmCluster.get_coordinator_address()

            jax.distributed.initialize(coordinator_address, self.num_processes, self.process_id, device_ids)
            logger.info(
                f"Initialized jax.distributed with {jax.device_count()} devices, {jax.process_count()} hosts"
                f", coordinator_address={coordinator_address}, process_id={self.process_id}"
            )


@dataclass
class RayConfig:
    address: Optional[str] = None
    start_workers: bool = True
    auto_start_cluster: bool = True

    def initialize(self):
        if self.auto_start_cluster:
            auto_ray_cluster(address=self.address, start_workers=self.start_workers)


def register_codecs():
    # pyrallis.encode.register(jnp.dtype, lambda dtype: dtype.name)
    # pyrallis.encode.register(type(jnp.float32), lambda meta: meta.dtype.name)
    # pyrallis.decode.register(jnp.dtype, lambda dtype_name: jnp.dtype(dtype_name))

    def policy_encode(policy: jmp.Policy):
        def name(dtype):
            if hasattr(dtype, "name"):
                return dtype.name
            elif hasattr(dtype, "dtype"):
                return name(dtype.dtype)

        out = f"compute={name(policy.compute_dtype)},params={name(policy.param_dtype)},output={name(policy.output_dtype)}"
        assert jmp.get_policy(out) == policy
        return out

    pyrallis.decode.register(jmp.Policy, lambda policy_str: jmp.get_policy(policy_str))
    pyrallis.encode.register(jmp.Policy, policy_encode)

    pyrallis.decode.register(timedelta, parse_timedelta)
    pyrallis.encode.register(timedelta, encode_timedelta)


register_codecs()


def config_registry(cls: Type):
    """
    A decorator to register a config class with a registry that we can use to find the config class for a given
    config. This is used for abstract classes/interfaces where we want to select a concrete implementation based on
    the config.

    the syntax for a yaml file would be:
    ```yaml
    model:
      gpt:
         <config for gpt>
    ```

    Subclasses can be added with the `register_subclass` method
    :param cls:
    :return: the decorated classes.
    """

    # add the registry to the class if it doesn't exist
    if not hasattr(cls, "_config_registry"):
        cls._config_registry = {}

    # add register_subclass_config to the class if it doesn't exist
    if not hasattr(cls, "register_subclass"):

        def register_subclass(name: str, subcls):
            cls._config_registry[name] = subcls
            return subcls

        cls.register_subclass = register_subclass

    # now register the cls with pyrallis
    def encode_config(config):
        for name, subcls in cls._config_registry.items():
            if isinstance(config, subcls):
                return {name: pyrallis.encode(config)}

        raise ValueError(f"Could not find a registered subclass for {config}")

    def decode_config(config):
        if len(config) != 1:
            raise ValueError(f"Expected exactly one key in config, got {config}")

        name, config = config.popitem()
        try:
            subcls = cls._config_registry[name]
            return pyrallis.decode(subcls, config)
        except KeyError:
            raise ValueError(f"Could not find a registered subclass for {name}")

    pyrallis.encode.register(cls, encode_config)
    pyrallis.decode.register(cls, decode_config)

    return cls


def main(args: list = None):
    """
    Like levanter.config.main_decorator but can handle config paths that are urls loadable by fsspec.
    This isn't documented in levanter.config.main_decorator, but only the first arg can be config-ified.
    """
    _cmdline_args = args
    if args is None:
        _cmdline_args = sys.argv[1:]

    def wrapper_outer(fn):
        @wraps(fn)
        def wrapper_inner(*args, **kwargs):
            config_path, cmdline_args = _maybe_get_config_path_and_cmdline_args(_cmdline_args)
            argspec = inspect.getfullargspec(fn)
            argtype = argspec.annotations[argspec.args[0]]
            cfg = parse(config_class=argtype, config_path=config_path, args=cmdline_args)
            response = fn(cfg, *args, **kwargs)
            return response

        return wrapper_inner

    return wrapper_outer


def _maybe_get_config_path_and_cmdline_args(args):
    """
    We want to accept ... --config_path <config> ... where config could be a path or url.
    If URL, we need to download it and save it to a temp file. We then want to remove --config_path
    from the cmdline args so that pyrallis doesn't try to load it as a config path and return it separately here
    along with the modified cmdline args.
    """
    if "--config_path" not in args:
        return None, args
    else:
        config_path_index = args.index("--config_path")
        config_path = args[config_path_index + 1]

        if urllib.parse.urlparse(config_path).scheme:
            fs: AbstractFileSystem
            fs, fs_path = fsspec.core.url_to_fs(config_path)
            temp_file = tempfile.NamedTemporaryFile(prefix="config", suffix=".yaml", delete=False)
            atexit.register(lambda: os.unlink(temp_file.name))
            fs.get(fs_path, temp_file.name)
            config_path = temp_file.name

        args = args.copy()
        del args[config_path_index]
        del args[config_path_index]
        return config_path, args
