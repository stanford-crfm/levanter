# In-Depth Configuration

This page gives an overview of the various settings you can use to customize a training run.

We use [Draccus](https://github.com/dlwh/draccus) for configuration. Draccus is yet-another yaml-to-dataclass library
that uses both dataclasses to generate yaml and argparse to parse command line arguments.

Typically, your config data class will look something like this:

```python
@dataclass
class TrainLmConfig:
    data: LMDatasetConfig = field(default_factory=LMDatasetConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LmConfig = field(default_factory=Gpt2Config)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
```

Your training run will typically be associated with a single config file. For instance, you might have a file
`my-run.yaml` that looks like this:

```yaml
data:
    train_urls:
      - "gs://my_bucket/openwebtext-sharded/openwebtext_train.{1..128}-of-128.jsonl.gz"
    validation_urls:
      - "gs://my_bucket/openwebtext-sharded/openwebtext_val.{1..8}-of-8.jsonl.gz"
    cache_dir: "gs://my_bucket/tokenized/openwebtext_2/"
model:
  type: gpt2
  hidden_dim: 768
  num_heads: 12
  num_layers: 12
  seq_len: 1024
  gradient_checkpointing: true
  scale_attn_by_inverse_layer_idx: true
trainer:
  wandb:
    project: "levanter"
    tags: [ "openwebtext", "gpt2"]

  mp: p=f32,c=bfloat16
  model_axis_size: 1
  per_device_parallelism: 4

  train_batch_size: 512
optimizer:
  learning_rate: 6E-4
  weight_decay: 0.1
  min_lr_ratio: 0.1
```

### Including Other Config Files

Draccus supports inclusion of config files via the `!include` special syntax. For instance, this:

```yaml
# my-run.yaml
data: !include data.yaml
trainer:
    num_train_steps: 1000000

# data.yaml
train_urls:
    - "gs://my_bucket/openwebtext-sharded/openwebtext_train.{1..128}-of-128.jsonl.gz"
validation_urls:
    -  "gs://my_bucket/openwebtext-sharded/openwebtext_val.{1..8}-of-8.jsonl.gz"
cache_dir: "gs://my_bucket/tokenized/openwebtext_2/"
```

will expand to:

```yaml
data:
    train_urls:
        - "gs://my_bucket/openwebtext-sharded/openwebtext_train.{1..128}-of-128.jsonl.gz"
    validation_urls:
        - "gs://my_bucket/openwebtext-sharded/openwebtext_val.{1..8}-of-8.jsonl.gz"
    cache_dir: "gs://my_bucket/tokenized/openwebtext_2/"
trainer:
    num_train_steps: 1000000
```

The inclusion path is always relative to the config file. Unfortunately, we don't (can't) support inclusion
at the top level.

## Trainer and TrainerConfig

The [levanter.trainer.Trainer][] class is governed by the [levanter.trainer.TrainerConfig][] dataclass.

Trainer has a lot of stuff in it. We highlight some of them in the following sections.

The following table lists some of the parameters that you might want to change.

### Core Training Loop Configuration

| Parameter                      | Description                                                         | Default                                                   |
|--------------------------------|---------------------------------------------------------------------|-----------------------------------------------------------|
| `seed`                         | The random seed                                                     | 0                                                         |
| `num_train_steps`              | The number of training steps to run                                 | 400,000                                                   |
| `train_batch_size`             | The batch size                                                      | 32                                                        |
| `per_device_train_parallelism` | Number of examples to process on each device during training        | `train_batch_size / (num_accelerators * model_axis_size)` |
| `per_device_eval_parallelism`  | Number of examples to process on each device during eval            | `per_device_train_parallelism`                            |
| `steps_per_eval`               | How often to evaluate the model during training                     | 1,000                                                     |
| `max_eval_batches`             | How many batches to evaluate during each evaluation                 | `None` (meaning all)                                      |
| `mp`                           | Mixed Precision policy using [jmp](https://github.com/deepmind/jmp) | `f32` (full precision)                                    |

### Logging and Reporting

| Parameter      | Description                                                                   | Default |
|----------------|-------------------------------------------------------------------------------|---------|
| `log_dir`      | Where to save logs (python logger). `$run_id` will be appended                | `logs/` |
| `run_base_dir` | where to save run artifacts. not really used much. `$run_id` will be appended | `runs/` |



### Partitioning / FSDP

Sharding in Levanter is done with axis mappings, which specify how to map logical axes (e.g. "batch") to physical axes in the JAX device mesh.
(See the [Haliax Scaling Tutorial](https://colab.research.google.com/drive/1QX4yH3zRFF3Xiibf1aahETcSQ5nbcUMz?authuser=1#scrollTo=lFZOnJD7QtZm&uniqifier=2)
for a more detailed explanation of axis mappings.) Levanter's Trainer uses two axis mappings:  `parameter_axis_resources` and `compute_axis_resources`.
`parameter_axis_resources` specifies how to shard the model parameters and optimizer state: basically how the model is sharded "at rest",
while the `compute_axis_resources` specifies how to shard the model during computation.

`TrainerConfig` allows you to specify these axis mappings in two ways, with a "basic" mode that has
reasonable defaults and an "advanced" mode that gives you more control.

#### Basic Mode

| Parameter              | Description                                                                  | Default   |
|------------------------|------------------------------------------------------------------------------|-----------|
| `batch_axis`           | The axis to shard the batch over, for distributed data parallelism           | `"batch"` |
| `fsdp_axis`            | The axis or axes to shard the model over, for Fully Sharded Data Parallelism | `"embed"` |
| `tensor_parallel_axes` | The axis or axes to shard the model over, for Tensor Parallelism             | `None`    |
| `model_axis_size`      | How many devices for tensor parallelism                                      | `1`       |

#### Advanced Mode

| Parameter                  | Description                                                          | Default |
|----------------------------|----------------------------------------------------------------------|---------|
| `axis_resources`           | Mapping from logical axis to physical axis shared by both mappings   | --      |
| `parameter_axis_resources` | Mapping from logical axis to physical axis for the parameter mapping | --      |
| `compute_axis_resources`   | Mapping from logical axis to physical axis for the compute mapping   | --      |
| `model_axis_size`          | How many devices for tensor parallelism                              | `1`     |

### Checkpointing

See also [Checkpointer](#checkpointer).

| Parameter                    | Description                                       | Default                                    |
|------------------------------|---------------------------------------------------|--------------------------------------------|
| `load_checkpoint`            | Whether to load checkpoint from `base_path`       | `None`: load if possible, but don't error. |
| `load_checkpoint_path`       | Path to load checkpoint from. May be a parent     | `checkpointer.base_path`                   |
| `checkpointer.base_path`     | Base path to save checkpoints to                  | `checkpoints/${run_id}`                    |
| `checkpointer.save_interval` | How often to save checkpoints (time)              | 15 minutes                                 |
| `checkpointer.keep`          | How often to keep checkpoints (steps). See below. | 10000 steps                                |

#### Checkpointer Save Policy

The checkpointer logic has two kinds of checkpoints:

* **time-based checkpoints**: **temporary** checkpoints that are saved every `save_interval` minutes. The previous time-based checkpoint is deleted when a new one is saved.
* **step-based checkpoints**: **permanent** checkpoints that are saved according to a policy. These checkpoints are never deleted.

Step-based checkpoint configuration looks like this:
```yaml
checkpointer:
  keep:
    - every: 1000  # steps
      until: 10000 # step
    - every: 5000  # steps
      until: 40000 # step
    - every: 10000
```

This policy will save permanent checkpoints every 1,000 steps until 10,000 steps, then every 5,000 steps until 40,000 steps, then every 10,000 steps.
The default step-based checkpoint policy is to save a checkpoint every 10,000 steps.



## WandB

We mostly use wandb for logging, including using wandb for allocating the run id. We may change this.

These all live in a nested object `wandb` inside `trainer`. Most of these are the same as the corresponding `wandb.init`
parameters.


| Parameter      | Description                                    | Default                    |
|----------------|------------------------------------------------|----------------------------|
| entity         | The wandb entity to use.                       | your default entity        |
| project        | The wandb project to use.                      | wandb's default            |
| tags           | Tags to add to the run.                        | `[]`                       |
| id             | Unique run id                                  | wandb's autogenerated id   |
| name           | The name of the run.                           | wandb's autogenerated name |
| save_code      | Whether to save the code to wandb.             | `True`                     |
| save_xla_dumps | Whether to save XLA compiler outputs to wandb. | `False`                    |


Notes:

* WandB's code saving logic isn't very good for our use case, so we have our own. We automatically sniff out the git repo
of your main script.
* `save_xla_dumps` is useful for debugging XLA compilation issues. It tends to dump a lot of stuff, so we don't save it by default.
To use it, you must also set the right environment variables. Something like `XLA_FLAGS="--xla_dump_to=/tmp/output_folder/xla_dumps --xla_dump_hlo_pass_re=.*`.
We will automatically parse out the env variable.

## Ray Config

Levanter will by default automatically start a Ray cluster with all
the machines being used for training. This is useful for distributed
preprocessing. You can disable this behavior using `auto_start_cluster: false`.


| Parameter           | Description                                                                 | Default |
|---------------------|-----------------------------------------------------------------------------|---------|
| `address`           | The address of the Ray cluster to connect to.                                | `None`  |
| `start_workers`     | Whether to start Ray workers. If `False`, you must start them yourself.      | `True`  |
| `auto_start_cluster`| Whether to start a Ray cluster automatically.                                | `True`  |


## Distributed Config

JAX can automatically sniff out clusters in SLURM and TPU environments.
If you're not using SLURM or TPUs, you can specify the cluster manually using this config.

**Don't use this on TPU, and possibly not on SLURM either.**

| Parameter           | Description                                                                 | Default                 |
|---------------------|-----------------------------------------------------------------------------|-------------------------|
| `coordinator_address`| The address of the coordinator. If `None`, we'll use the default address.   | `None`                  |
| `num_processes`     | The number of processes in the cluster.                                     | `None`                  |
| `process_id`        | The process id of this process.                                             | `None`                  |
| `local_device_ids`  | The local device ids of this process.                                       | ${CUDA_VISIBLE_DEVICES} |



## Optimizer

[levanter.trainer.OptimizerConfig][] is a dataclass that specifies the optimizer configuration. It has the following fields:

| Parameter       | Description                                                       | Default  |
|-----------------|-------------------------------------------------------------------|----------|
| `learning_rate` | The learning rate.                                                | `1e-4`   |
| `weight_decay`  | The weight decay.                                                 | `0.0`    |
| `beta1`         | The beta1 parameter for Adam.                                     | `0.9`    |
| `beta2`         | The beta2 parameter for Adam.                                     | `0.999`  |
| `epsilon`       | The epsilon parameter for Adam.                                   | `1e-8`   |
| `max_grad_norm` | The maximum gradient norm (for clipping).                         | `1.0`    |
| `min_lr_ratio`  | The minimum learning rate ratio.                                  | `0.0`    |
| `warmup_ratio`  | The warmup ratio. Fraction of total steps to warmup               | `0.01`   |
| `lr_schedule`   | The learning rate schedule. One of `constant`, `cosine`, `linear` | `cosine` |


## LM Model Config

[levanter.models.lm_model.LmConfig][] is a Draccus "choice class" that acts as a base class for all autoregressive
language models in Levanter. You typically will specify a kind of model by using the `type` field, which is a string
that specifies the kind of model. For instance, `type: gpt2` will use the [levanter.models.gpt2.Gpt2Config][] class,
while `type: llama` will use the [levanter.models.llama.LlamaConfig][] class.

We won't go into detail here. You can see the auto-generated docs below.


## Auto-generated Documentation

### Trainer

::: levanter.trainer.TrainerConfig

::: levanter.trainer.Trainer

### Checkpointer

::: levanter.checkpoint.CheckpointerConfig

::: levanter.checkpoint.Checkpointer

### Wandb
::: levanter.logging.WandbConfig

### Distributed and Ray

::: levanter.distributed.DistributedConfig

::: levanter.distributed.RayConfig

### Optimizer

::: levanter.trainer.OptimizerConfig

### LM Model

::: levanter.models.lm_model.LmConfig

::: levanter.models.gpt2.Gpt2Config

::: levanter.models.llama.LlamaConfig

::: levanter.models.mpt.MptConfig
