# Various Pyrallis configs
import dataclasses
import logging
import tempfile
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import List, Mapping, Optional, Union

import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pyrallis
from furl import furl
from git import InvalidGitRepositoryError, NoSuchPathError, Repo
from jax.experimental.maps import Mesh
from pyrallis import field

import levanter.logging
from haliax.partitioning import ResourceAxis
from levanter import jax_utils
from levanter.mesh import MeshInfo


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
            metadata_to_share = jax_utils.multihost_broadcast_obj(
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
class TrainerConfig:
    seed: int = 0
    mp: jmp.Policy = jmp.get_policy("f32")

    wandb: WandbConfig = WandbConfig()
    log_dir: Path = Path("logs/")
    run_base_dir: furl = furl("runs/")
    checkpoint_dir: furl = furl("checkpoints/")

    # config related to partitioning
    # TODO: in theory we can support tuples of physical axis names, but I don't think anyone actually uses that.
    model_axis_size: int = 1  # how many devices to shard each model over. Data axis is the other axis
    axis_resources: Mapping[str, str] = field(default_factory=dict)  # mapping from logical axis to physical axis
    parameter_axis_resources: Mapping[str, str] = field(default_factory=dict)  # overrides axis_mapping for parameter
    # and optimizer sharding

    # Config related to batch sizes
    train_batch_size: int = 512
    per_device_train_batch_size: int = -1

    per_device_eval_batch_size: int = -1

    # Config related to duration
    num_train_steps: int = 400_000
    steps_per_eval: int = 1_000

    steps_per_save: int = 20_000
    load_last_checkpoint: bool = True
    load_checkpoint_path: Optional[furl] = None

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
    use_gda: bool = True  # whether or not to use GlobalDeviceArrays for pjitted models.

    @property
    def run_name(self) -> str:
        import wandb

        return wandb.run.name or wandb.run.id

    @property
    def run_dir(self) -> furl:
        return self.run_base_dir / self.run_name

    @property
    def checkpoint_path(self) -> furl:
        return self.checkpoint_dir / self.run_name

    def initialize(self, all_config):
        """Initializes jax, wandb, logging, setting the run name in the process"""
        self._initialize_jax_config()
        self.wandb.init(all_config)
        self._initialize_logging()

    @cached_property
    def device_mesh(self) -> Mesh:
        devices = jax.devices()
        devices = np.array(devices).reshape(self.data_axis_size, self.model_axis_size)
        return Mesh(devices, (ResourceAxis.DATA, ResourceAxis.MODEL))

    @cached_property
    def train_mesh_info(self):
        return MeshInfo(self.device_mesh, self.train_batch_size, self.per_device_train_batch_size)

    @cached_property
    def eval_mesh_info(self):
        return MeshInfo(
            self.device_mesh,
            self.per_device_eval_batch_size * self.data_axis_size,
            self.per_device_eval_batch_size,
        )

    @property
    def data_axis_size(self):
        """size of the data parallel/batch parallel axis."""
        assert jax.device_count() % self.model_axis_size == 0
        return jax.device_count() // self.model_axis_size

    def _initialize_jax_config(self):
        """Initialize global jax config with settings we like, based on config"""
        jax_utils.set_hardware_rng_ops(self.use_hardware_rng)
        jax.config.update("jax_parallel_functions_output_gda", self.use_gda)

    def _initialize_logging(self):
        log_dir = self.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        levanter.logging.init_logger(log_dir / f"{self.run_name}.log")

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

    # post init
    def __post_init__(self):
        if jax.device_count() % self.model_axis_size != 0:
            raise ValueError(
                f"num_devices ({jax.device_count()}) is not divisible by model_axis_size ({self.model_axis_size})"
            )

        if (
            jax.local_device_count() % self.model_axis_size != 0
            and self.model_axis_size % jax.local_device_count() != 0
        ):
            raise ValueError("either model_axis_size or local_device_count must be divisible by the other")

        if self.per_device_train_batch_size == -1:
            self.per_device_train_batch_size = self.train_batch_size // jax.device_count()

        # validate size of per_device_train_batch_size
        if self.train_batch_size % (self.per_device_train_batch_size * self.data_axis_size) != 0:
            raise ValueError(
                f"train_batch_size ({self.train_batch_size}) must be divisible by per_device_train_batch_size *"
                f" data_axis_size ({self.per_device_train_batch_size}, {self.data_axis_size})"
            )

        if self.per_device_eval_batch_size == -1:
            self.per_device_eval_batch_size = self.per_device_train_batch_size


def register_codecs():
    pyrallis.encode.register(jnp.dtype, lambda dtype: dtype.name)
    pyrallis.encode.register(type(jnp.float32), lambda meta: meta.dtype.name)
    pyrallis.decode.register(jnp.dtype, lambda dtype_name: jnp.dtype(dtype_name))
    pyrallis.decode.register(furl, furl)
    pyrallis.encode.register(furl, str)

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


register_codecs()
