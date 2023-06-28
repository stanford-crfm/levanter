import atexit
import dataclasses
import functools
import importlib
import inspect
import os
import pkgutil
import sys
import tempfile
import urllib.parse
from dataclasses import is_dataclass
from datetime import timedelta
from functools import wraps
from typing import Any, Dict, List, Optional, Type, Union

import draccus
import fsspec
import jmp
from draccus import parse
from fsspec import AbstractFileSystem

from levanter.utils.datetime_utils import encode_timedelta, parse_timedelta


JsonAtom = Union[str, int, float, bool, None]


def register_codecs():
    # draccus.encode.register(jnp.dtype, lambda dtype: dtype.name)
    # draccus.encode.register(type(jnp.float32), lambda meta: meta.dtype.name)
    # draccus.decode.register(jnp.dtype, lambda dtype_name: jnp.dtype(dtype_name))

    def policy_encode(policy: jmp.Policy):
        def name(dtype):
            if hasattr(dtype, "name"):
                return dtype.name
            elif hasattr(dtype, "dtype"):
                return name(dtype.dtype)

        out = f"compute={name(policy.compute_dtype)},params={name(policy.param_dtype)},output={name(policy.output_dtype)}"
        assert jmp.get_policy(out) == policy
        return out

    draccus.decode.register(jmp.Policy, lambda policy_str: jmp.get_policy(policy_str))
    draccus.encode.register(jmp.Policy, policy_encode)

    draccus.decode.register(timedelta, parse_timedelta)
    draccus.encode.register(timedelta, encode_timedelta)

    # draccus' decode function for bool accepts anything truthy (it uses bool(x)), so we need to override it
    # we need to raise if it's not a bool, because anything can be converted to a bool
    truthy = {"true", "t", "yes", "y", "True", "T", "Yes", "Y", "TRUE", True}
    falsy = {"false", "f", "no", "n", "False", "F", "No", "N", "FALSE", False}

    def bool_decode(x):
        if x in truthy:
            return True
        elif x in falsy:
            return False
        else:
            raise ValueError(f"Could not convert {x} to bool")

    draccus.decode.register(bool, bool_decode)


register_codecs()


def config_registry(cls: Optional[Type] = None, *, discover_packages: Optional[str] = None):
    """
    A decorator to register a config class with a registry that we can use to find the config class for a given
    config. This is used for abstract classes/interfaces where we want to select a concrete implementation based on
    the config. Subclasses can be added with the `register_subclass` method.

    We have a rudimentary package discovery system.
    If discover_packages is not None, then we will attempt to identify subclasses by importing
    packages from discover_packages. They should still be registered with register_subclass.
    If you use discover_packages, you should register a class with the same name as the package.
    For example, if you make a new LmConfig, which has discover_packages="levanter.models" and your
    model is defined in my_transformer.py, then you should register your config with
    `LmConfig.register_subclass("my_transformer", MyTransformerConfig)`

    Usage:
    ```python
    @config_registry
    @dataclasses.dataclass
    class ModelConfig:
        pass

    @dataclasses.dataclass
    class GPTConfig(ModelConfig):
        pass

    ModelConfig.register_subclass("gpt", GPTConfig)

    the syntax for a yaml file would be:
    ```yaml
    model:
      gpt:
         <config for gpt>
    ```

    As a special case we also allow just using a string if you want defaults:
    ```yaml
    model: gpt
    ```

    :param cls:
    :return: the decorated classes.
    """

    if cls is None:
        return functools.partial(config_registry, discover_packages=discover_packages)

    if not hasattr(cls, "_config_registry"):
        cls._config_registry = {}

    if not hasattr(cls, "register_subclass"):

        def register_subclass(name: str, subcls=None):
            assert cls is not None
            if subcls is None:
                return functools.partial(register_subclass, name)
            if name in cls._config_registry:
                raise ValueError(f"Config class {name} already registered with {cls._config_registry[name]}")
            cls._config_registry[name] = subcls
            return subcls

        cls.register_subclass = register_subclass

    # now register the cls with draccus
    def encode_config(config):
        for name, subcls in cls._config_registry.items():
            if isinstance(config, subcls):
                # singledispatch means that draccus.encode(config) will call this function even if config is a subclass of cls
                return {name: _default_encode(config)}

        raise ValueError(f"Could not find a registered subclass for {config}")

    def decode_config(config):
        if type(config) is str:
            config = {config: {}}

        if len(config) != 1:
            raise ValueError(f"Expected exactly one key in config, got {config}")

        name, config = config.popitem()
        subcls = _try_discover_packages(cls, name)

        try:
            subcls = cls._config_registry[name]
            return draccus.decode(subcls, config)
        except KeyError:
            if discover_packages:
                if subcls is not None:
                    return draccus.decode(subcls, config)

            raise ValueError(f"Could not find a registered subclass for {name}")

    def _try_discover_packages(cls, subcls_name):
        if discover_packages is None:
            return None
        # from https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/
        # resolve the package path
        package_module = importlib.import_module(discover_packages)

        def iter_namespace(ns_pkg):
            # Specifying the second argument (prefix) to iter_modules makes the
            # returned name an absolute name instead of a relative one. This allows
            # import_module to work without having to do additional modification to
            # the name.
            return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")

        for finder, pkg_name, ispkg in iter_namespace(package_module):
            # if pkg_name == f"{discover_packages}.{subcls_name}":
            _ = importlib.import_module(pkg_name)
            # registration should happen in the __init__.py of the package
            # cls.register_subclass(name, subcls)

        return cls._config_registry[subcls_name]

    draccus.encode.register(cls, encode_config)
    draccus.decode.register(cls, decode_config)

    return cls


def _default_encode(obj):
    if is_dataclass(obj):
        d: Dict[str, Any] = dict()
        for field in dataclasses.fields(obj):
            value = getattr(obj, field.name)
            try:
                d[field.name] = draccus.encode(value)
            except TypeError as e:
                raise ValueError(f"Could not encode field {field.name} of type {field.type} of {obj}") from e
        return d
    else:
        raise ValueError(f"Could not encode {obj}")


DEFAULT_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")


def main(*, args: list = None, config_dir: Optional[str] = DEFAULT_CONFIG_DIR):
    """
    Like draccus.wrap but can handle config paths that are urls loadable by fsspec.
    This isn't documented in levanter.config.main_decorator, but only the first arg can be config-ified.

    :param args: the args to parse. If None, will use sys.argv[1:]
    :param config_dir: the directory to look for configs in (if the path does not exist already). If None, will only use the current working directory
    """
    _cmdline_args = args
    if args is None:
        _cmdline_args = sys.argv[1:]

    def wrapper_outer(fn):
        @wraps(fn)
        def wrapper_inner(*args, **kwargs):
            config_path, cmdline_args = _maybe_get_config_path_and_cmdline_args(_cmdline_args)
            paths_to_check = [config_path, f"{config_path}.yaml", f"{config_path}.yml"]
            if config_path is not None and config_dir is not None:
                paths_to_check.extend([os.path.join(config_dir, p) for p in paths_to_check])

            for path in paths_to_check:
                if path is not None and os.path.exists(path):
                    config_path = path
                    break

            argspec = inspect.getfullargspec(fn)
            argtype = argspec.annotations[argspec.args[0]]
            cfg = parse(config_class=argtype, config_path=config_path, args=cmdline_args)
            response = fn(cfg, *args, **kwargs)
            return response

        return wrapper_inner

    return wrapper_outer


def _maybe_get_config_path_and_cmdline_args(args: List[str]):
    """
    We want to accept ... --config_path <config> ... where config could be a path or url.
    If URL, we need to download it and save it to a temp file. We then want to remove --config_path
    from the cmdline args so that draccus doesn't try to load it as a config path and return it separately here
    along with the modified cmdline args.
    """
    if "--config_path" not in args and "--config" not in args:
        return None, args
    else:
        try:
            config_path_index = args.index("--config_path")
        except ValueError:
            config_path_index = args.index("--config")

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
