"""Experimental utilities for running Ray with Cloud TPU."""
from collections import defaultdict
from enum import Enum
import re
import logging
from typing import Any, Callable, List, Mapping, Optional, Union
import inspect
from functools import partial
import socket
import ray
from ray.util.placement_group import (
    placement_group,
)
from ray.util import placement_group_table
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


class TpuStatus(Enum):
    AVAILABLE = "available"
    BUSY = "busy"


class RayTpuPod:
    def __init__(self, tpu_name: str, num_hosts: int, head_ip: str):
        self.tpu_name = tpu_name
        self.head_ip = head_ip
        self.num_hosts = num_hosts

    def initialize(self):
        logging.info(
            "Creating %d placement groups for %s",
            self.num_hosts, self.tpu_name)
        pgs = [
            placement_group(
                [{"CPU": 200, "TPU": 4, self.tpu_name: 1}],
            )
            for i in range(self.num_hosts)
        ]
        ray.get([pg.ready() for pg in pgs])
        logging.info("Placement groups are ready.")

        @ray.remote(resources={"TPU": 4})
        def get_host_name():
            import socket
            return socket.gethostname()

        host_names = ray.get([
            get_host_name.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg)).remote() for pg in pgs])
        self.pgs = {}
        for host_name, pg in zip(host_names, pgs):
            self.pgs[host_name] = pg

    def get_status(self, available_resources: Optional[Mapping[str, Any]] = None):
        """Gets the status of the TPU pod.

        The key insight here is that ray.available_resources() shows the resources
        available within placement groups, i.e. in the format of:

        {tpu_name}_group_{placement_group_id}: 1

        Since remote calls within TpuPod will consume the "TPU name resource", we can
        leverage this to check the status.

        Args:
            available_resources: Optionally provided `ray.available_resources()` dict
                to reduce the overhead of calling it multiple times.

        """
        logging.debug("Checking status of %s", self.tpu_name)
        if not available_resources:
            available_resources = ray.available_resources()

        # Pick a random placement group, as we assume that users will not
        # be targeting placement groups manually.
        pg = next(iter(self.pgs.values()))
        pg_id = placement_group_table(pg)["placement_group_id"]
        pg_resource_string = f"{self.tpu_name}_group_{pg_id}"
        logging.debug("Checking for resource %s", pg_resource_string)
        if pg_resource_string in available_resources:
            logging.info("TPU is available")
            return TpuStatus.AVAILABLE
        else:
            logging.info("TPU is not available")
            return TpuStatus.BUSY

    def run(
        self,
        f: Union[Callable[[Any], Any], type],
        env: Optional[Mapping[str, Any]] = None,
        include_worker_name: bool = False,
        *args, **kwargs):
        logging.debug("Running on TPU pod %s", self.tpu_name)
        f = ray.remote(resources={"TPU": 4})(f)
        handles = []
        if env is None:
            env = {}
        for worker_name, pg in self.pgs.items():
            if include_worker_name:
                handles.append(f.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg),
                    runtime_env={"env_vars": env},
                    num_cpus=200,
                    resources={"TPU": 4, self.tpu_name: 1},
                ).remote(*args, worker_name=worker_name, **kwargs))
            else:
                handles.append(f.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg),
                    runtime_env={"env_vars": env},
                    num_cpus=200,
                    resources={"TPU": 4, self.tpu_name: 1},
                ).remote(*args, **kwargs))
        return handles


class RayTpuPodManager:

    def initialize(self):
        tpu_pattern = re.compile(r"TPU-(.+)-head")
        self.tpu_pods = defaultdict(list)

        @ray.remote
        def _get_tpu_pod_metadata():
            import time
            # avoid race conditions
            time.sleep(3)
            tpu_name = ray.util.accelerators.tpu.get_current_pod_name()
            num_hosts = ray.util.accelerators.tpu.get_current_pod_worker_count()
            ip = socket.gethostbyname(socket.gethostname())
            return tpu_name, num_hosts, ip

        for key, value in ray.available_resources().items():
            logging.debug(f"Checking key: {key} with value: {value}")
            match = tpu_pattern.match(key)
            if match:
                accelerator_type = f"{match.group(1)}"
                accelerator_type_key = key
                num_tpu_pods = int(value)
                logging.info(
                    "Found %d TPU pods of type: %s",
                    num_tpu_pods, accelerator_type)
                metadata_handles = []
                for pod_index in range(num_tpu_pods):
                    metadata_handles.append(_get_tpu_pod_metadata.options(
                        resources={accelerator_type_key: 1}
                    ).remote())
                logging.debug("Gathering TPU pod metadata")
                metadata = ray.get(metadata_handles)
                for tpu_name, num_hosts, head_ip in metadata:
                    self.tpu_pods[accelerator_type].append(RayTpuPod(
                        tpu_name=tpu_name,
                        num_hosts=num_hosts,
                        head_ip=head_ip,
                    ))
        for _, pods in self.tpu_pods.items():
            for pod in pods:
                pod.initialize()

    def list_status(self):
        logging.info("Printing accelerator_type, status tuples...")
        available_resources = ray.available_resources()
        for accelerator_type, pods in self.tpu_pods.items():
            for pod in pods:
                logging.info(
                    "%s (%s, %s): %s",
                    accelerator_type, pod.head_ip, pod.tpu_name,
                    pod.get_status(
                        available_resources=available_resources
                    ))

    def cluster_resources(self):
        cluster_resources = {}
        for accelerator_type, pods in self.tpu_pods.items():
            cluster_resources[accelerator_type] = len(pods)
        return cluster_resources

    def _get_available_pods(self) -> Mapping[str, RayTpuPod]:
        available_pods = defaultdict(list)
        available_resources = ray.available_resources()
        for accelerator_type, pods in self.tpu_pods.items():
            for pod in pods:
                if pod.get_status(
                    available_resources=available_resources) == TpuStatus.AVAILABLE:
                    available_pods[accelerator_type].append(pod)
        return available_pods

    def available_resources(self):
        available_resources = defaultdict(int)
        for accelerator_type, pods in self._get_available_pods().items():
            available_resources[accelerator_type] = len(pods)
        return dict(available_resources)

    def run(
        self,
        f: Union[Callable[[Any], Any], type],
        accelerator_type: str,
        num_slices: int = 1,
        with_mxla: bool = True,
        include_worker_name: bool = False,
        env: Optional[Mapping[str, Any]] = None,
        *args, **kwargs):
        logging.debug(
            "Running a workload on %d slices of %s", num_slices, accelerator_type)
        # Validate that we have enough resources
        num_available_tpu_slices = self.available_resources().get(accelerator_type, 0)
        if num_available_tpu_slices < num_slices:
            logging.warning(
                "Requested to schedule %d slices, but only %d are available.",
                num_slices, num_available_tpu_slices)

        # Reserve those resources
        available_slices = self._get_available_pods()[accelerator_type]
        pods = available_slices[:num_slices]
        if env is None:
            env = {}

        if with_mxla:
            # From any of the nodes we reserved, grab an IP address
            # Build the MXLA environment variables
            # Execute iteratively for the number of pods.
            # build and return the list of futures
            coordinator_port = 8081
            handles = []
            for pod_id, pod in enumerate(pods):
                mxla_env = {
                    "MEGASCALE_COORDINATOR_ADDRESS": f"{pods[0].head_ip}:{coordinator_port}",
                    "MEGASCALE_NUM_SLICES": str(num_slices),
                    "MEGASCALE_PORT": f"{coordinator_port}",
                    "MEGASCALE_SLICE_ID": str(pod_id),
                }
                new_env = {**env, **mxla_env}
                handles.extend(pod.run(
                    f, include_worker_name=include_worker_name, env=new_env,
                    *args, **kwargs))
            return handles
        else:
            handles = []
            for pod in pods:
                handles.extend(pod.run(
                    f, include_worker_name=include_worker_name,
                    env=env, *args, **kwargs))
            return handles

_manager = RayTpuPodManager()


def init():
    logging.info("Initializing ray_tpu!")
    if not ray.is_initialized():
        ray.init()
    _manager.initialize()


def cluster_resources() -> Mapping[str, Any]:
    """Returns a dict of the registered TPU pod slices."""
    return _manager.cluster_resources()


def available_resources() -> Mapping[str, Any]:
    """Returns a dict of the available TPU pod slices."""
    return _manager.available_resources()


def _remote_func_wrapper(
    f: Callable[[Any], Any],
    accelerator_type: str,
    num_slices: int = 1,
    with_mxla: bool = False,
    env: Optional[Mapping[str, Any]] = None,
    *f_args, **f_kwargs):
    return _manager.run(
        f=f,
        accelerator_type=accelerator_type,
        num_slices=num_slices,
        with_mxla=with_mxla,
        env=env,
        *f_args, **f_kwargs
    )


class _RemoteClassWrapper:
    def __init__(
        self, cls: type, accelerator_type: str,
        num_slices: int, with_mxla: bool,
        env: Optional[Mapping[str, Any]] = None):
        self.cls = cls
        self.accelerator_type = accelerator_type
        self.num_slices = num_slices
        self.with_mxla = with_mxla
        self.env = env

    def __call__(self, *args, **kwargs):
        # Instantiate all actors
        class_name = self.cls.__name__
        class _LabeledCls(self.cls):
            """A wrapper to cls with additional information."""
            def __init__(self, worker_name: str, *args, **kwargs):
                self._worker_name = worker_name
                self._class_name = class_name
                super().__init__(*args, **kwargs)

            def __repr__(self):
                return f"{self._class_name}:{self._worker_name}"

        self.instances = _manager.run(
            f=_LabeledCls,
            accelerator_type=self.accelerator_type,
            num_slices=self.num_slices,
            with_mxla=self.with_mxla,
            include_worker_name=True,
            env=self.env,
            *args, **kwargs)
        return self

    def __getattr__(self, key):
        assert len(self.instances) > 0, "Instances not initialized."
        all_values = [
            getattr(inst, key) for inst in self.instances
        ]
        if callable(all_values[0]):
            def _wrapper(*args, **kwargs):
                return [
                    func.remote(*args, **kwargs) for func in all_values
                ]
            return _wrapper
        return all_values


def remote(
    accelerator_type: str,
    num_slices: int = 1,
    with_mxla: bool = False,
    env: Optional[Mapping[str, Any]] = None,
):
    def decorator(f_or_c: Union[Callable[Any, Any], type]):
        if inspect.isfunction(f_or_c):
            return partial(
                _remote_func_wrapper,
                f=f_or_c,
                accelerator_type=accelerator_type,
                num_slices=num_slices,
                with_mxla=with_mxla,
                env=env)
        elif inspect.isclass(f_or_c):
            return _RemoteClassWrapper(
                f_or_c,
                accelerator_type=accelerator_type,
                num_slices=num_slices,
                with_mxla=with_mxla,
                env=env)
        else:
            raise ValueError(
                "Expected input to `ray_tpu.remote` to be a function or a class."
            )
    return decorator
