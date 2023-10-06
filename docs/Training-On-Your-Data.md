# Training on Your Own Data

This guide is meant to be a detailed walkthrough of training a model on your own data using Levanter.

The basic steps are:

- [ ] [Configure your environment/cloud](#environment-setup)
- [ ] [Prepare your data and upload to cloud](#data-preparation)
- [ ] [Configure your training run](#configuration)
- [ ] [Upload the training configuration file](#upload-config-to-gcs)
- [ ] [Launch training](#launching-training)
- [ ] [Evaluate](#evaluation)
- [ ] [Export your model to Huggingface](#huggingface-export)


## Environment Setup

### TPU Setup

See the [TPU guide](./Getting-Started-TPU-VM.md) for instructions on setting up a TPU VM instance. You should go through
the installation steps in that guide before continuing. Don't spin up a TPU VM instance yet, though.

### CUDA Setup

See the [CUDA guide](./Getting-Started-CUDA.md) for instructions on setting up a CUDA machine.

### WandB Setup

Levanter mainly uses [WandB](https://wandb.ai) for logging. You should create a WandB account and [get an API key](https://wandb.ai/authorize).


## Data Preparation

Currently, our data preprocessing pipeline assumes that you have already:

1. Turned your data into one or more JSONL files of *plain text*.
2. Split your data into train and validation sets.
3. Done any randomization of your data.

Instead of (1), you may instead use a Huggingface Dataset, such as [The Pile](https://huggingface.co/datasets/EleutherAI/pile).
If you have, and if your data is already split and randomized, you may skip to the [Machine Setup](#machine-setup) section.

If you have a sequence-to-sequence task, as of September 2023, you should turn each example into a single text
by e.g. using a templating mechanism with a prompt (a la Alpaca).

(2) is something we're looking to relax. And we'd like to support automatically handling shuffling.

### Data Format: JSONL

The canonical format for training data in Levanter is (compressed) JSONL, or JSON Lines.
Each line of the file is a JSON object, which is a dictionary of key-value pairs.
The only required key is `"text"`, which should map to a string of plain text.

Once you have done so, you can create the `data` section of your training configuration:

```yaml
data:
    train_urls:
      - "gs://path/to/train.{1..32}.jsonl.gz"
    validation_urls:
      - "gs://path/to/valid.{1..4}.jsonl.gz"
    cache_dir: "gs://path/to/cache"
    tokenizer: "gpt2"  # any HF tokenizer path, or GCS path to an HF tokenizer
```

Levanter uses [fsspec](https://filesystem-spec.readthedocs.io/en/latest/) to read data from files,
so it can transparently handle compressed files and files in cloud storage (like Google Cloud Storage or AWS S3).
Levanter uses [braceexpand](https://pypi.org/project/braceexpand/) to expand the `{1..32}` syntax.
You can also use more than one entry if you have urls that don't follow a naming scheme:

```yaml
data:
    train_urls:
      - "gs://path/to/train_web.jsonl.gz"
      - "gs://path/to/train_news.jsonl.gz"
      - "gs://path/to/train_wiki.jsonl.gz"
    validation_urls: # etc.
```

**Note**: Levanter's preprocessing pipeline works best if you split your data into at least 1 shard for every machine
(i.e. 8 TPUs). This isn't a big deal, but it helps.

### Data Format: Huggingface Datasets

If you have a Huggingface Dataset, you can use it directly in Levanter. It must
have a `"text"` column, and it must be split into train and validation sets. To use it,
you can specify the dataset name in the `data` section of your training configuration:

```yaml
data:
    id: "EleutherAI/pile"
    # if needed:
    # name: "subset"
```

This will be passed to `datasets.load_dataset`. If the dataset supports
streaming, you can use `stream: true` to stream the data instead of loading it all into memory.
If a streaming dataset is sharded, we will attempt to exploit the sharded structure to preprocess more efficiently.

## Data Preprocessing

Levanter supports both online and offline preprocessing. Online preprocessing is done on-the-fly
during training. With online preprocessing, you don't need to think about preprocessing your data.

Our data loading pipeline will automatically break and concatenate documents into chunks equal
to the model's `seq_len` parameter. It will also automatically add special tokens to the
end of documents.

We don't yet handle sequence-to-sequence tasks, but we plan to.

### Online Preprocessing

We have a sophisticated caching mechanism using [Ray](https://docs.ray.io/en/latest/)
that builds a cache of preprocessed data on the fly. Online caching happens transparently
in the background, using the mostly-idle CPU-cores of the machine(s) you are training on.

The cache that is built is fully reproducible, and can be used for future training runs.
Training will start as soon as each training machine has its first shard of data cached
and once the validation data is cached.

### Offline Preprocessing

If you want, you can also preprocess your data offline, and then upload the preprocessed data to cloud storage.

Levanter has a script that basically runs the same online preprocessing code, but doesn't do any training.
You can run it like this:

```bash
python -m levanter.main.cache_dataset --config_path my_config.yaml
```

You can actually connect this to a Ray cluster, and use the cluster to do the preprocessing. This lets you
use any of Ray's autoscaling features to scale up the preprocessing job.

To do so:

```bash
python -m levanter.main.cache_dataset \
    --config_path my_config.yaml \
    --address <ray-cluster-address> \
    --start_workers false \
    --auto_start_cluster false
```

## Configuration

Levanter uses [Draccus](https://github.com/dlwh/draccus) to configure training runs. It's a YAML-to-dataclass
library that also supports argument parsing via argparse. A detailed guide to configuring Levanter is available
in the [Configuration Guide](./Configuration-Guide.md).

This section will cover the basics of configuring a training run.

### TL;DR

Here's a configuration for a 1.4B parameter model with reasonable values for everything:

```yaml
data:
    train_urls:
      - "gs://path/to/train.{1..32}.jsonl.gz" # TODO
    validation_urls:
      - "gs://path/to/valid.{1..4}.jsonl.gz" # TODO
    cache_dir: "gs://path/to/cache"  # TODO
    tokenizer: "gpt2"  # any HF tokenizer path, or GCS path to an HF tokenizer
model:
  type: gpt2
  hidden_dim: 1536
  num_heads: 24
  num_layers: 48
  seq_len: 1024
  gradient_checkpointing: true
  scale_attn_by_inverse_layer_idx: true
trainer:
  wandb:
    project: "levanter" # TODO
    tags: ["gpt2"]

  mp: p=f32,c=bfloat16
  num_train_steps: 100000  # TODO
  train_batch_size: 512  # you may need to tune this or per_device_parallelism
  per_device_parallelism: -1
  per_device_eval_parallelism: 8

  max_eval_batches: null # set to a number to limit eval batches. useful if your eval set is enormous

  checkpointer:
    base_path: "gs://path/to/checkpoints"  # TODO
    save_interval: 15m
    keep:
      - every: 10000
optimizer:
  learning_rate: 1E-4
  weight_decay: 0.1
  min_lr_ratio: 0.1
# if you want:
hf_save_steps: 10000
hf_save_path: "gs://path/to/hf/checkpoints"  # TODO
hf_upload: null # set to an hf repo if you want to upload automatically. You need to have logged in to hf-cli
```

If you want a different model size or architecture, you can look at the config files in
[levanter/config](https://github.com/stanford-crfm/levanter/tree/main/config).

### Continued Pretraining

Levanter supports starting from an HF pretrained model. To do so, you should set your config like this:

```yaml
model:
  type: mpt
initialize_from_hf: "mosaicml/mpt-7b" # can also reference a version, e.g. "mosaicml/mpt-7b@deadbeef"
use_hf_model_config: true
```

You should probably reduce the learning rate by a factor of 10 or so. TODO: figure out best practices here.

#### Llama 2

For Llama 2 specifically (or other gated models), you'll need a few extra steps:

If you haven't already, go to [Llama 2's Hugging Face page](https://huggingface.co/meta-llama/Llama-2-7b-hf) and request access to the model.

Once you have access, go to [Hugging Face's Tokens page](https://huggingface.co/settings/tokens) to get an API token.
Then, pass in the token as an environment variable:

```bash
HUGGING_FACE_HUB_TOKEN=hf...
```

Pass that in anywhere you're passing in a `WANDB_API_KEY`.

Then, you can use the model like so:

```yaml
model:
  type: llama
initialize_from_hf: "meta-llama/Llama-2-7b-hf"
use_hf_model_config: true
```

### Checkpointing

See also the [Checkpointing section of the Configuration Guide](./Configuration-Guide.md#checkpointing).

Levanter supports checkpointing to both local and Google Cloud Storage, backed by [TensorStore](https://google.github.io/tensorstore/).
If you're using multiple machines, you should probably use cloud storage or NFS.

Levanter saves two kinds of checkpoints:

* **time-based checkpoints**: **temporary** checkpoints that are saved every `save_interval` minutes. The previous
time-based checkpoint is deleted when a new one is saved.
* **step-based checkpoints**: **permanent** checkpoints that are saved according to a policy. These checkpoints are never deleted.

At a minimum, you should set a `base_path` for your checkpoints. This can be a local path or a GCS path.

### Determining Batch Size

We don't have a mechanism for tuning batch size automatically, so you need to do this by hand. Memory usage is controlled
by `per_device_parallelism`, `train_batch_size`, gradient checkpointing, model size, and number of accelerators, and
`model_axis_size`. (For most models `model_axis_size` should be 1, so you can ignore it.)

`per_device_parallelism` is analogous to `per_device_batch_size` in other frameworks. It controls how many examples
are processed at once on a single accelerator, but the name is a bit more "correct" in the presence of tensor or pipeline
parallelism.  A `per_device_parallelism` of `-1` means "use as many as possible to not perform gradient accumulation."

Gradient accumulation is performed whenever `num_accelerators * per_device_parallelism / model_axis_size < train_batch_size`.
Gradient checkpointing is enabled by default and highly recommended.

So, to find your batch size, you should modify either `per_device_parallelism` or `train_batch_size` until
your job runs. Note that, due to FSDP, as you add more TPUs, you can increase the effective parallelism because
you will use less memory per accelerator to store parameters and optimizer states.

### Number of Training Steps

Levanter does not support epochs or number of tokens/examples, so if you want to train for a certain number of epochs or
tokens, you'll need to compute the number of steps yourself. You can use the following formula:

```
num_train_steps = num_epochs * num_tokens_per_epoch / train_batch_size / seq_len
```

Note however that epoch boundaries aren't really respected: our implementation of sharded data loading restarts
from the beginning as soon as any machine finishes its shards.

## Launching Training

### TPU

First, we assume you've gone through the setup steps in [the TPU guide](./Getting-Started-TPU-VM.md), at least through setting up your gcloud account.
We also strongly recommend setting up ssh keys and ssh-agent.

#### Upload Config To GCS

Once you have your config built, you should upload it to GCS. You could also `scp` it to all workers, but this is easier
and works with the TPU babysitting script.

```bash
gsutil cp my_config.yaml gs://path/to/config.yaml
```

#### Using the Babysitting Script with a Preemptible or TRC TPU VM

If you are using a preemptible TPU VM, or a TRC TPU VM, you should use the babysitting script to automatically restart
your VM if it gets preempted. A detailed guide to babysitting is available in the
[babysitting section of the TPU guide](./Getting-Started-TPU-VM.md#using-the-babysitting-script-with-a-preemptible-or-trc-tpu-vm).
Here is the upshot:

```bash
infra/babysit-tpu-vm my-tpu -z us-east1-d -t v3-128 -- \
    WANDB_API_KEY=... levanter/infra/run.sh python levanter/src/levanter/main/train_lm.py --config_path gs://path/to/config.yaml
```

#### Spin up and manual launch

You should probably use the automated setup script, as described in the [relevant section of the TPU guide](./Getting-Started-TPU-VM.md#automatic-setup).
Here's what that looks like:

```bash
bash infra/spin-up-tpu-vm.sh my-tpu -z us-east1-d -t v3-128
```

This will spin up a TPU VM instance and install Levanter on it. You can then run a command like so:

```bash
gcloud compute tpus tpu-vm ssh my-tpu   --zone us-east1-d --worker=all --command="WANDB_API_KEY=... levanter/infra/launch.sh python levanter/src/levanter/main/train_lm.py --config_path gs://path/to/config.yaml"
```

### GPU

TODO, but you can mostly follow the guide for [TPU](#tpu) above.

## Monitoring

Levanter integrates with WandB for logging. You can view your run on the WandB website. Levanter will also log
to the console, and write logs to `logs/$RUN_ID.log` on each machine. Logs can be pretty verbose.

We recommend monitoring `train/loss` and `eval/loss` in log/log scale. You should be seeing roughly a linear decrease
in loss followed by a gradual flattening. You can also monitor `throughput`.

## Evaluation

Levanter will run evaluation every `trainer.steps_per_eval` steps.

You can also run evaluation manually by running the `levanter/main/eval_lm.py` script:

```bash
python -m levanter.main.eval_lm --config_path gs://path/to/config.yaml --checkpoint_path gs://path/to/checkpoint
```

You can also use this script to evaluate on other datasets by modifying the config.


## Huggingface Export

### Exporting during Training

You can export to HF during training using the `hf_save_steps` and `hf_save_path` options in your config. You can
also set `hf_upload` to an HF repo to automatically upload your model to HF. See the [config above](#tldr) for an example.

Typically, you will have saved checkpoints in a directory like `gs://path/to/checkpoints/hf/my_run/step_10000/`.
Hugging Face Transformers doesn't know how to read these. So, you'll need to copy the files to a local directory:

```bash
gsutil -m cp gs://path/to/checkpoints/hf/my_run/step_10000/* /tmp/my_exported_model
```

Then you can use the model as you would expect:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("/tmp/my_exported_model")
tokenizer = AutoTokenizer.from_pretrained("/tmp/my_exported_model")
```

### Exporting after Training

After training, you can run a separate script to export levanter checkpoints to Huggingface:

```bash
python -m levanter.main.export_to_hf --config_path my_config.yaml --output_dir gs://path/to/output
```
