# Installation

If you're using a TPU, more complete documentation for setting that up is available [here](docs/Getting-Started-TPU-VM.md).
If you're using CUDA, more complete documentation for setting that up is available [here](docs/Getting-Started-CUDA.md).

## Setting up a development environment

<!--levanter-installation-start-->
For development, please follow these steps:

First, install the appropriate version of Jax for your system. 
See [Jax's installation instructions](https://github.com/google/jax/blob/main/README.md#installation)
as it varies from platform to platform.

Now clone this repository and install it with pip:

```bash
git clone https://github.com/stanford-crfm/levanter.git
cd levanter
pip install -e .
wandb login  # optional, we use wandb for logging
```

<!--levanter-installation-end-->

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
python3 -m virtualenv -p python3.8 levanter-venv

# Activate the virtual environment.
source levanter-venv/bin/activate
```
