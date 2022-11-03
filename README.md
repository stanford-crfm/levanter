# Levanter

> You could not prevent a thunderstorm, but you could use the electricity; you could not direct the wind, but you could trim your sail so as to propel your vessel as you pleased, no matter which way the wind blew. <br/>
> — Cora L. V. Hatch


Levanter is a library based on [Jax](https:://github.com/google/jax) and [Equinox](https://github.com/patrick-kidger/equinox)
for training [foundation models](https://en.wikipedia.org/wiki/Foundation_models) created by [Stanford's Center for Research
on Foundation Models (CRFM)](https://crfm.stanford.edu/).

## Haliax

> Though you don’t seem to be much for listening, it’s best to be careful. If you managed to catch hold of even just a piece of my name, you’d have all manner of power over me.<br/>
> — Patrick Rothfuss, *The Name of the Wind*

Haliax is a module (currently) inside Levanter for named tensors, modeled on Alexander Rush's [Tensor Considered Harmful](https://arxiv.org/abs/1803.09868).
It's designed to work with Jax and Equinox to make constructing distributed models easier.


## Getting Started with Levanter

### Installation

First install the appropriate version of Jax for your system. See [Jax's installation instructions](https://github.com/google/jax/blob/main/README.md#installation)
as it varies from platform to platform.

If you're using a TPU, more complete documentation for setting that up is available [here](docs/Getting-Started-TPU-VM.md).

Now clone this repository and install it with pip:

```bash
git clone https://github.com/stanford-crfm/levanter.git
cd levanter
pip install -e .
wandb login  # optional, we use wandb for logging
```

TODO: put things on pypi, etc


### Training a GPT2-nano

As a kind of hello world, here's how you can train a GPT2-nano model on a small dataset.

```bash
python examples/gpt2_example.py --config_path config/gpt2_nano.yaml
```

This will train a GPT2-nano model on the [WikiText-2](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) dataset.
You can change the dataset by changing the `dataset` field in the config file.

The config file is a [Pyrallis](https://github.com/eladrich/pyrallis) config file. Pyrallis is yet-another yaml-to-dataclass library.
You can use `--help` or poke around other configs to see all the options available to you.


## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

## License

Levanter is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full license text.
