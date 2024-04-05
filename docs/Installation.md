# Installation

If you are using GPUs, please follow the installation steps [here instead](Getting-Started-GPU.md).
If you're using a TPU, more complete documentation for setting that up is available [here](Getting-Started-TPU-VM.md).

If you're on an M1 or later Mac with macOS Sonoma, you can follow the Metal instructions [at the bottom of this page](#metal).


## Create a virtual environment

It is recommended to install Levanter using a virtual environment with Python version 3.10 to avoid dependency conflicts. Levanter requires Python version 3.10. To create, a Python virtual environment with Python version >= 3.10 and activate it, follow the instructions below.

Using [Anaconda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or
[Miniconda](https://docs.conda.io/en/latest/miniconda.html):

```
# Create a virtual environment.
# Only run this the first time.
conda create -n levanter python=3.10 pip

# Activate the virtual environment.
conda activate levanter
```

Using [Virtualenv](https://docs.python.org/3/library/venv.html#creating-virtual-environments):

```
# Create a virtual environment.
# Only run this the first time.
python3 -m pip install virtualenv
python3 -m virtualenv -p python3.10 levanter-venv

# Activate the virtual environment.
source levanter-venv/bin/activate
```

## Setting up a development environment

For development, please follow these steps:

First, install the appropriate version of Jax for your system.
See [JAX's installation instructions](https://github.com/google/jax/blob/main/README.md#installation)
as it varies from platform to platform.

## Install Levanter

{%
   include-markdown "../README.md"
   start="<!--levanter-installation-start-->"
   end="<!--levanter-installation-end-->"
%}


## Metal

If you are using an M1 or later Mac with macOS Sonoma, you can use Metal for GPU acceleration.
To do so, you will need to install the `jax-metal` package, which is [available on PyPI](https://pypi.org/project/jax-metal/).

We've tested Levanter with `jax-metal` version 0.0.5 on macOS Sonoma 14.3.1. Note that `jax-metal` is still in
development and is definitely not feature-complete or bug free. `train_lm` works fine though. Haliax tests do not pass,
though they mostly should by 0.0.6, I think.

```bash
conda create -n levanter-metal python=3.10 pip
conda activate levanter-metal
# this will also install a compatible version of jax and jaxlib
pip install jax-metal==0.0.5
# We recommend installing Levanter from source to get the latest updates
git clone https//github.com/stanford-crfm/levanter.git
cd levanter
pip install -e .
```

Then you can run a test:

```bash
python -m levanter.main.train_lm --config config/gpt2_nano.yaml
```

### Various Limitations

There are many limitations to using Metal. You can see [bugs in the JAX issue tracker](https://github.com/google/jax/labels/Apple%20GPU%20%28Metal%29%20plugin).
Some of the limitations are:

* `with_sharding_constraint` is not supported. Haliax's `shard` operation will just ignore these on Metal, so you don't need to worry about it.
* `bfloat16` is not currently supported. You can use `f32` instead. (Change your configs so that it's `mp: f32`.)
* Argument donation in `jit` is not supported. This is a JAX feature that allows you to donate the memory of an argument to the output of a function. This is not supported on Metal,
and you'll get a warning if you try to use it. It's not as critical as it is on TPU/GPU because Metal has a lot of memory for the amount of FLOPs it can do.
* Reductions with more than 4 dimensions are not supported.
