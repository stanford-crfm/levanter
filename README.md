# Levanter

<!--levanter-intro-start-->
> *You could not prevent a thunderstorm, but you could use the electricity; you could not direct the wind, but you could trim your sail so as to propel your vessel as you pleased, no matter which way the wind blew.* <br/>
> — Cora L. V. Hatch

Levanter is a framework for training large language models (LLMs) and other foundation models that strives for legibility, scalability, and reproducibility:

1. **Legible**: Levanter uses our named tensor library [Haliax](https://github.com/stanford-crfm/haliax) to write easy-to-follow, composable deep learning code, while still being high performance.
2. **Scalable**: Levanter scales to large models, and to be able to train on a variety of hardware, including GPUs and TPUs.
3. **Reproducible**: Levanter is bitwise deterministic, meaning that the same configuration will always produce the same results, even in the face of preemption and resumption.

We built Levanter with [JAX](https:://github.com/google/jax), [Equinox](https://github.com/patrick-kidger/equinox),
and [Haliax](https://github.com/stanford-crfm/haliax).

[//]: # (Levanter supports distributed training on TPUs &#40;and, soon, GPUs&#41;, including FSDP, tensor parallelism, distributed checkpointing, distributed data loading, and more.)
[//]: # (Levanter integrates with WandB for logging and with the Hugging Face ecosystem for tokenizers, datasets, and model import and export.)

## Features

* **Distributed Training**: We support distributed training on TPUs (and soon, GPUs), including FSDP and tensor parallelism.
* **Cached On-Demand Data Preprocessing**: We preprocess corpora online, but we cache the results of preprocessing so
that resumes are much faster and so that subsequent runs are even faster. As soon as the first part of the cache is complete, Levanter will start training.
* **Online Visualization**: Levanter also provides a feature for visualizing the probability of each token in the validation set during training. This is useful for debugging and for understanding how the model is learning.
* **Export**: We support exporting models to the Hugging Face Hub, with export compatible with Pytorch and Transformers via [SafeTensors](https://github.com/huggingface/safetensors).
* **Logging**: Logging is done with [WandB](https://wandb.ai/), complete with a fancy online visualization of the validation set during training.
* **Distributed Checkpointing**: Distributed checkpointing is supported via Google's [TensorStore](https://google.github.io/tensorstore/) library. Training can even be resumed on a different number of hosts, though this breaks reproducibility for now.
* **Optimization**: Levanter uses [Optax](https://github.com/deepmind/optax) for optimization,
  though our new optimizer, [Sofia](https://arxiv.org/abs/2305.14342), is coming to Levanter soon!
* **Stability**: The GPT-2 implementation uses the [Mistral stability trick](https://crfm.stanford.edu/2021/08/26/mistral.html) to improve stability during training.

<!--levanter-intro-end-->

Levanter was created by [Stanford's Center for Research on Foundation Models (CRFM)](https://crfm.stanford.edu/)'s research engineering team. ([We're hiring!](https://crfm.stanford.edu/apply.html)) You can also find us in the #levanter channel on the unofficial [Jax LLM Discord](https://discord.gg/FkRGNX3ND)

## Getting Started

Here is a small set of examples to get you started. For more information about the various configuration options, please see the [Training Getting Started](docs/Getting-Started-Training.md) guide.
You can also use `--help` or poke around other configs to see all the options available to you.

### Installing Levanter

<!--levanter-installation-start-->

After [installing JAX]((https://github.com/google/jax/blob/main/README.md#installation)) with the appropriate configuration
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

If you're using a TPU, more complete documentation for setting that up is available [here](docs/Getting-Started-TPU-VM.md). GPU support is still in-progress; documentation is available [here](docs/Getting-Started-CUDA.md).

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

### Other Architectures

Currently, we support GPT-2, [Backpacks](http://backpackmodels.science/) and MosaicML's [MPT](https://www.mosaicml.com/blog/mpt-7b)
architectures. We plan to add more in the future.

#### A Tiny Backpack Model

```bash
python -m levanter.main.train_lm --config_path config/backpack_nano.yaml
```

#### Continued Pretraining with MPT

```bash
python -m levanter.main.train_lm --config_path config/mpt_7b_continued.yaml
```


## Distributed and Cloud Training

### Training on a TPU Cloud VM

Please see the [TPU Getting Started](docs/Getting-Started-TPU-VM.md) guide for more information on how to set up a TPU Cloud VM and run Levanter there.

### Training with CUDA

Please see the [CUDA Getting Started](docs/Getting-Started-CUDA.md) guide for more information on how to set up a CUDA environment and run Levanter there.

<!--levanter-user-guide-end-->

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

## License

Levanter is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full license text.
