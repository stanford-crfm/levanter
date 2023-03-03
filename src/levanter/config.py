# Various Pyrallis configs
import atexit
import dataclasses
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import timedelta
from functools import cached_property
from pathlib import Path
from typing import List, Mapping, Optional, Union

import jax
import jmp
import numpy as np
import optax
import pyrallis
from git import InvalidGitRepositoryError, NoSuchPathError, Repo
from jax._src.clusters import SlurmCluster, TpuCluster
from jax.experimental.maps import Mesh
from pyrallis import field

import levanter.logging
from haliax.partitioning import ResourceAxis, ResourceMapping
from levanter.checkpoint import Checkpointer, CheckpointInterval
from levanter.distributed import LevanterSlurmCluster
from levanter.utils import cloud_utils, jax_utils
from levanter.utils.datetime_utils import encode_timedelta, parse_timedelta


logger = logging.getLogger(__name__)


@dataclass
class WandbConfig:
    """
    Configuration for wandb.
    """

    entity: Optional[str] = None
    project: Optional[str] = None
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    id: Optional[str] = None
    group: Optional[str] = None
    mode: Optional[str] = None
    resume: Optional[Union[bool, str]] = None

    save_code: Union[bool, str] = True
    """If string, will save code from that directory. If True, will attempt to sniff out the main directory (since we
    typically don't run from the root of the repo)."""

    save_xla_dumps: bool = False
    """If True, will save the XLA code to wandb (as configured by XLA_FLAGS). This is useful for debugging."""

    def init(self, hparams=None, **extra_hparams):
        import wandb

        if hparams is None:
            hparams_to_save = {}
        elif dataclasses.is_dataclass(hparams):
            hparams_to_save = dataclasses.asdict(hparams)
        else:
            hparams_to_save = dict(hparams)

        if extra_hparams:
            hparams_to_save.update(extra_hparams)

        # for distributed runs, we only want the primary worker to use wandb, so we make everyone else be disabled
        # however, we do share information about the run id, so that we can link to it from the other workers
        mode = self.mode
        if jax.process_index() != 0:
            mode = "disabled"

        if isinstance(self.save_code, str):
            code_dir = self.save_code
        elif self.save_code:
            code_dir = WandbConfig._infer_experiment_git_root() or "."
        else:
            code_dir = None

        other_settings = dict()
        if code_dir is not None:
            logger.info(f"Setting wandb code_dir to {code_dir}")
            other_settings["code_dir"] = code_dir
            other_settings["git_root"] = code_dir

        r = wandb.init(
            entity=self.entity,
            project=self.project,
            name=self.name,
            tags=self.tags,
            id=self.id,
            group=self.group,
            resume=self.resume,
            mode=mode,
            config=hparams_to_save,
            settings=other_settings,
        )

        if jax.process_count() > 1:
            # we need to share wandb run information across all hosts, because we use it for checkpoint paths and things
            metadata_to_share = dict(
                entity=r.entity,
                project=r.project,
                name=r.name,
                tags=r.tags,
                id=r.id,
                group=r.group,
            )
            metadata_to_share = jax_utils.multihost_broadcast_sync(
                metadata_to_share, is_source=jax.process_index() == 0
            )

            if jax.process_index() != 0:
                assert r.mode == "disabled"
                for k, v in metadata_to_share.items():
                    setattr(r, k, v)

            logger.info(f"Synced wandb run information from process 0: {r.name} {r.id}")

        if dataclasses.is_dataclass(hparams):
            with tempfile.TemporaryDirectory() as tmpdir:
                config_path = f"{tmpdir}/config.yaml"
                with open(config_path, "w") as f:
                    pyrallis.dump(hparams, f, encoding="utf-8")
                wandb.run.log_artifact(str(config_path), name="config.yaml", type="config")

        wandb.summary["num_devices"] = jax.device_count()
        wandb.summary["num_hosts"] = jax.process_count()
        wandb.summary["backend"] = jax.default_backend()

    @staticmethod
    def _infer_experiment_git_root() -> Optional[str]:
        # sniff out the main directory (since we typically don't run from the root of the repo)
        # we'll walk the stack and directories for the files in the stack the until we're at a git root
        import os
        import traceback

        stack = traceback.extract_stack()
        # start from the top of the stack and work our way down since we want to hit the main file first
        top_git_root = None
        for frame in stack:
            dirname = os.path.dirname(frame.filename)
            # bit hacky but we want to skip anything that's in the python env
            if any(x in dirname for x in ["site-packages", "dist-packages", "venv", "opt/homebrew", "conda"]):
                continue
            # see if it's under a git root
            try:
                repo = Repo(dirname, search_parent_directories=True)
                top_git_root = repo.working_dir
                break
            except (NoSuchPathError, InvalidGitRepositoryError):
                logger.debug(f"Skipping {dirname} since it's not a git root")
                pass
        return top_git_root


@dataclass
class CheckpointerConfig:
    base_path: str = "checkpoints/"
    save_interval: timedelta = timedelta(hours=6)
    # TODO: I'd like to write this, but it's not supported by pyrallis
    # keep: List[CheckpointInterval] = field(default_factory=lambda: [CheckpointInterval(every=1000)])
    keep: List[dict] = field(
        default_factory=lambda: [dict(every=10000)]
    )  # list of dicts with two keys: every and until

    def create(self, run_name) -> Checkpointer:
        keeps = [CheckpointInterval(**k) for k in self.keep]
        return Checkpointer(
            base_path=os.path.join(self.base_path, run_name),
            save_interval=self.save_interval,
            step_policies=keeps,
        )

    def __post_init__(self):
        self.base_path = os.path.expanduser(self.base_path)

        # validate the checkpoint intervals.
        # we want to make sure that the intervals are monotonic. only the last one can be None
        prev_interval = None
        for interval in self.keep:
            if prev_interval is not None:
                assert prev_interval["until"] is not None, "Only the last checkpoint interval can be None"
                assert (
                    interval["until"] is None or interval["until"] > prev_interval["until"]
                ), "Checkpoint intervals must be monotonic"
            prev_interval = interval


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
class TrainerConfig:
    seed: int = 0
    mp: jmp.Policy = jmp.get_policy("f32")

    wandb: WandbConfig = WandbConfig()
    log_dir: Path = Path("logs/")
    run_base_dir: Path = Path("runs/")

    # config related to partitioning
    # TODO: in theory we can support tuples of physical axis names, but I don't think anyone actually uses that.
    model_axis_size: int = 1  # how many devices to shard each model over. Data axis is the other axis
    axis_resources: Mapping[str, str] = field(default_factory=dict)  # mapping from logical axis to physical axis
    parameter_axis_resources: Mapping[str, str] = field(default_factory=dict)  # overrides axis_mapping for parameter
    # and optimizer sharding

    # Config related to batch sizes
    train_batch_size: int = 512
    per_device_parallelism: int = -1

    per_device_eval_parallelism: int = -1

    # Config related to duration
    num_train_steps: int = 400_000
    steps_per_eval: int = 1_000
    max_eval_batches: Optional[int] = None  # max number of batches to evaluate on. None means all batches

    checkpointer: CheckpointerConfig = CheckpointerConfig()
    load_last_checkpoint: bool = True
    load_checkpoint_path: Optional[str] = None

    # Config related to optimizer (always adam for now)
    learning_rate: float = 6e-4
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    max_grad_norm: Optional[float] = 1.0

    warmup_ratio: float = 0.01  # fraction of training steps to use as warmup
    lr_schedule: str = "cosine"  # constant, cosine, linear

    use_hardware_rng: bool = False  # whether to use less-reproducible but faster rng
    use_jax_array: bool = True  # whether or not to use the new jax.Array for pjitted models.

    distributed: DistributedConfig = DistributedConfig()

    # whether or not to require an accelerator (e.g. TPU or GPU).
    # default depends on the platform: on macos False, else True
    require_accelerator: Optional[bool] = None

    # whether or not to shutdown the tpu at exit. If a float, shutdown after that many seconds. True = 5 minutes
    shutdown_at_exit: Union[bool, float] = False

    @property
    def run_name(self) -> str:
        import wandb

        return wandb.run.name or wandb.run.id

    @property
    def run_dir(self) -> Path:
        return self.run_base_dir / self.run_name

    def initialize(self, all_config):
        """Initializes jax, wandb, logging, setting the run name in the process"""
        self.distributed.initialize()
        self._initialize_jax_config()
        self.wandb.init(all_config)
        self._initialize_logging()
        self._validate()

        if self.require_accelerator is None:
            self.require_accelerator = not sys.platform.startswith("darwin")

        if self.require_accelerator:
            assert jax.default_backend() != "cpu", "Accelerator required but not found"

        if self.shutdown_at_exit is not False:
            if isinstance(self.shutdown_at_exit, bool):
                self.shutdown_at_exit = 5.0 * 60
            logger.info(f"At end of run, shutting down TPU VM in {self.shutdown_at_exit} seconds")
            atexit.register(cloud_utils.shutdown_tpu_vm, self.shutdown_at_exit)

    @cached_property
    def device_mesh(self) -> Mesh:
        print("device mesh")
        devices = jax.devices()
        devices = np.array(devices).reshape(self.data_axis_size, self.model_axis_size)
        return Mesh(devices, (ResourceAxis.DATA, ResourceAxis.MODEL))

    @property
    def eval_batch_size(self):
        return self.per_device_eval_parallelism * self.data_axis_size

    @property
    def data_axis_size(self):
        """size of the data parallel/batch parallel axis."""
        print("data axis size")

        assert jax.device_count() % self.model_axis_size == 0
        return jax.device_count() // self.model_axis_size

    @property
    def compute_axis_mapping(self) -> ResourceMapping:
        return self.axis_resources

    @property
    def parameter_axis_mapping(self) -> ResourceMapping:
        mapping = dict(self.axis_resources)
        if self.parameter_axis_resources:
            mapping.update(self.parameter_axis_resources)

        return mapping

    def _initialize_jax_config(self):
        """Initialize global jax config with settings we like, based on config"""
        jax_utils.set_hardware_rng_ops(self.use_hardware_rng)
        jax.config.update("jax_array", self.use_jax_array)

    def _initialize_logging(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        levanter.logging.init_logger(self.log_dir / f"{self.run_name}.log")

    def optimizer(self):
        """Creates the optimizer"""

        # indirection makes it work with optax.inject_hyperparams so we can log the learning rate
        def _optimizer(learning_rate):
            components = []

            if self.max_grad_norm:
                components.append(optax.clip_by_global_norm(self.max_grad_norm))

            components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))

            if self.weight_decay > 0:
                # TODO: add weight decay masking??
                components.append(optax.add_decayed_weights(self.weight_decay))

            # - learning rate for descent
            components.append(optax.scale(-learning_rate))

            optimizer = optax.chain(*components)

            return optimizer

        optimizer = optax.inject_hyperparams(_optimizer)(learning_rate=self.lr_scheduler())

        return optimizer

    def lr_scheduler(self):
        warmup_steps = int(self.warmup_ratio * self.num_train_steps)
        lr_decay_steps = self.num_train_steps - warmup_steps
        if warmup_steps == 0 and self.lr_schedule == "constant":
            schedule = optax.constant_schedule(self.learning_rate)
        else:
            if self.lr_schedule == "constant":
                schedule = optax.constant_schedule(self.learning_rate)
            elif self.lr_schedule == "cosine":
                schedule = optax.cosine_decay_schedule(self.learning_rate, lr_decay_steps - warmup_steps)
            elif self.lr_schedule == "linear":
                schedule = optax.linear_schedule(self.learning_rate, 0.0, lr_decay_steps - warmup_steps)
            else:
                raise ValueError(f"Unknown lr_schedule: {self.lr_schedule}")

            if warmup_steps != 0:
                warmup = optax.linear_schedule(0.0, self.learning_rate, warmup_steps)
                schedule = optax.join_schedules([warmup, schedule], [warmup_steps])
        return schedule

    # we can't do this in post_init because we don't want to call jax.device_count before calling distributed.initialize
    def _validate(self):
        if jax.device_count() % self.model_axis_size != 0:
            raise ValueError(
                f"num_devices ({jax.device_count()}) is not divisible by model_axis_size ({self.model_axis_size})"
            )

        if (
            jax.local_device_count() % self.model_axis_size != 0
            and self.model_axis_size % jax.local_device_count() != 0
        ):
            raise ValueError("either model_axis_size or local_device_count must be divisible by the other")

        if self.per_device_parallelism == -1:
            self.per_device_parallelism = self.train_batch_size // jax.device_count()

        # validate size of per_device_parallelism
        if self.train_batch_size % (self.per_device_parallelism * self.data_axis_size) != 0:
            raise ValueError(
                f"train_batch_size ({self.train_batch_size}) must be divisible by per_device_parallelism *"
                f" data_axis_size ({self.per_device_parallelism}, {self.data_axis_size})"
            )

        if self.per_device_eval_parallelism == -1:
            self.per_device_eval_parallelism = self.per_device_parallelism


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
