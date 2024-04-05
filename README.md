# Levanter

<a href="https://github.com/stanford-crfm/levanter/actions?query=branch%3Amain++">
    <img alt="Build Status" src="https://img.shields.io/github/actions/workflow/status/stanford-crfm/levanter/run_tests.yaml?branch=main">
</a>
<a href="https://levanter.readthedocs.io/en/latest/?badge=latest">
    <img alt="Documentation Status" src="https://readthedocs.org/projects/levanter/badge/?version=latest">
</a>
<a href="">
<img alt="License" src="https://img.shields.io/github/license/stanford-crfm/levanter?color=blue" />
</a>
<a href="https://https://pypi.org/project/levanter/">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/levanter?color=blue" />
</a>


## Getting Started

Here is a small set of examples to get you started. For more information about the various configuration options,
please see the [Getting Started](./docs/Getting-Started-Training.md) guide or the [In-Depth Configuration Guide](docs/Configuration-Guide.md).
You can also use `--help` or poke around other configs to see all the options available to you.


### Installing Levanter

<!--levanter-installation-start-->

After [installing JAX](https://github.com/google/jax/blob/main/README.md#installation) with the appropriate configuration
for your platform, you can install Levanter with:

```bash
pip install levanter
```

or using the latest version from GitHub:

```bash
git clone https://github.com/stanford-crfm/levanter.git
cd levanter
pip install -e .
wandb login  # optional, we use wandb for logging
```



<!--levanter-installation-end-->

Please refer to the [Installation Guide](docs/Installation.md) for more information on how to install Levanter.

If you're using a TPU, more complete documentation for setting that up is available [here](docs/Getting-Started-TPU-VM.md). GPU support is still in-progress; documentation is available [here](docs/Getting-Started-GPU.md).

<!--levanter-user-guide-start-->

### Training a GPT2-nano

As a kind of hello world, here's how you can train a GPT-2 "nano"-sized model on a small dataset.

```bash
python -m levanter.main.train_lm --config_path config/gpt2_nano.yaml

# alternatively, if you didn't use -e and are in a different directory
python -m levanter.main.train_lm --config_path gpt2_nano
```

This will train a GPT2-nano model on the [WikiText-103](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) dataset.

### Training a GPT2-small on your own data

You can also change the dataset by changing the `dataset` field in the config file.
If your dataset is a [Hugging Face dataset](https://huggingface.co/docs/datasets/loading_datasets.html), you can use the `data.id` field to specify it:

```bash
python -m levanter.main.train_lm --config_path config/gpt2_small.yaml --data.id openwebtext

# optionally, you may specify a tokenizer and/or a cache directory, which may be local or on gcs
python -m levanter.main.train_lm --config_path config/gpt2_small.yaml --data.id openwebtext --data.tokenizer "EleutherAI/gpt-neox-20b" --data.cache_dir "gs://path/to/cache/dir"
```

If instead your data is a list of URLs, you can use the `data.train_urls` and `data.validation_urls` fields to specify them.
Data URLS can be local files, gcs files, or http(s) URLs, or anything that [fsspec](https://filesystem-spec.readthedocs.io/en/latest/) supports.
Levanter (really, fsspec) will automatically uncompress `.gz` and `.zstd` files, and probably other formats too.

```bash
python -m levanter.main.train_lm --config_path config/gpt2_small.yaml --data.train_urls ["https://path/to/train/data_*.jsonl.gz"] --data.validation_urls ["https://path/to/val/data_*.jsonl.gz"]
```

### Customizing a Config File

You can modify the config file to change the model, the dataset, the training parameters, and more. Here's
the `gpt2_small.yaml` file:

```yaml
data:
  train_urls:
      - "gs://pubmed-mosaic/openwebtext-sharded/openwebtext_train.{1..128}-of-128.jsonl.gz"
  validation_urls:
      - "gs://pubmed-mosaic/openwebtext-sharded/openwebtext_val.{1..8}-of-8.jsonl.gz"
  cache_dir: "gs://pubmed-mosaic/tokenized/openwebtext/"
model:
  gpt2:
    hidden_dim: 768
    num_heads: 12
    num_layers: 12
    seq_len: 1024
    gradient_checkpointing: true
    scale_attn_by_inverse_layer_idx: true
trainer:
  tracker:
    type: wandb
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

### Other Architectures

Currently, we support the following architectures:
* GPT-2
* [LLama 1 or 2](https://ai.meta.com/llama/)
* [Backpacks](http://backpackmodels.science/)
* MosaicML's [MPT](https://www.mosaicml.com/blog/mpt-7b)

We plan to add more in the future.

#### Continued Pretraining with Llama 1 or Llama 2

Here's an example of how to continue pretraining a Llama 1 or Llama 2 model on the OpenWebText dataset:

```bash
python -m levanter.main.train_lm --config_path config/llama2_7b_continued.yaml
```


## Distributed and Cloud Training

### Training on a TPU Cloud VM

Please see the [TPU Getting Started](docs/Getting-Started-TPU-VM.md) guide for more information on how to set up a TPU Cloud VM and run Levanter there.

### Training with CUDA

Please see the [CUDA Getting Started](docs/Getting-Started-GPU.md) guide for more information on how to set up a CUDA environment and run Levanter there.

<!--levanter-user-guide-end-->

## Model Tracing

To run a mode connectivity experiment testing Llama2 against Lemma (for example) run the following
```bash
python -m levanter.main.eval_lm --config_path config/for_llama2_7b.yaml --trainer.wandb.name trace_llama2_lemma
```

Currently the config will look like so

```yaml
data:
  id: dlwh/wikitext_103_detokenized
  cache_dir: "gs://levanter-data/tokenized/wiki-llama2/"
  tokenizer: "meta-llama/Llama-2-70b-hf"
  
model:
  type: llama
# TODO: uncomment this once we resolve the resource exhaustion issue
hf_checkpoint: "meta-llama/Llama-2-7b-hf"
second_hf_checkpoint: "openlm-research/open_llama_7b"

trainer:
  wandb:
    project: "levanter"
    name: "llama2_7b--openllama-github"
    tags: ["openwebtext", "github", "openllama"]

  mp: p=f32,c=bfloat16
  train_batch_size: 64  # set for v4-64 TPU
  num_train_steps: 2
  steps_per_eval: 1
  tensor_parallel_axes: ["mlp", "heads"]
  fsdp_axis: "embed"
  batch_axis: "batch"
  per_device_eval_parallelism: -1
  max_eval_batches: 1
```

