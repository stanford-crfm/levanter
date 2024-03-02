# Getting Started on GPU

**Note** that right now Levanter does not work on a multi-machine setup with Infiniband. (We think this is just a dependency/environment issue.)

We only test on Ampere (e.g. A100s or 30xx series)  GPUs. If it works with JAX it should work, though.


## Installation

### TL;DR

```bash
virtualenv -p python3.10 levanter
source levanter/bin/activate
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
git clone https://github.com/stanford-crfm/levanter.git
cd levanter
pip install -e .
```

### Setting up a Virtual Environment

We recommend using a virtual environment to install Levanter.
You can use either `virtualenv` or `conda` to create a virtual environment.

#### Setting up a Virtualenv

Here are the steps for creating a virtual environment with `virtualenv`

```bash
virtualenv -p python3.10 levanter
source levanter/bin/activate
```

#### Setting up a Conda Environment

```bash
conda create --name levanter python=3.10 pip
conda activate levanter
```
### Install Jax with CUDA

Please take a look at the [JAX Installation Guide](https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier). Here are two options that work circa June 2023:

```bash
# CUDA 12 installation
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# CUDA 11 installation
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Install Levanter

You can either install Levanter from PyPI or from source. We recommend installing from source if you plan to make changes to the codebase.

#### Install from PyPI

```bash
pip install levanter
```

#### Install from Source

```bash
git clone https://github.com/stanford-crfm/levanter.git
cd levanter
pip install -e .
```

### WandB login

To use WandB, you con log in to your WandB account on the command line as follows:
```bash
wandb login ${YOUR TOKEN HERE}
```
You can find more information on getting setup with Weights and Biases here: https://wandb.ai/site


## Running a Job

For more details on how to configure training runs, please see the [Getting Started Training](Getting-Started-Training.md) guide.
Here are some examples of running a job.

### Running a job locally

```bash
python -m levanter.main.train_lm --config config/gpt2_small
```

### Running a job on Slurm

#### Single Node: 1 Process per Node

Here's a simple example of running a job on a single node. This example assumes you have cloned the Levanter repository
and are in the root directory of the repository.

```bash
srun --account=nlp --cpus-per-task=128 --gpus-per-node=8 --job-name=levanter-multi-1 --mem=1000G  --open-mode=append --partition=sphinx --time=14-0 infra/run-slurm.sh python src/levanter/main/train_lm.py --config_path config/gpt2_small.yaml
```

#### Single Node: One Process Per GPU

This example uses `sbatch` to submit a job to Slurm. This example assumes you have cloned the Levanter repository

```bash
#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --job-name=levanter-test
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --output=levanter_%j.log
#SBATCH --mem=16G

## On the Stanford NLP cluster you might need this:
export PATH=$(echo $PATH | sed 's|:/usr/local/cuda/bin||')

## Activate your virtual environment
source levanter/bin/activate

srun python -m levanter.main.train_lm --config config/gpt2_small_fast --trainer.per_device_parallelism -1
```

Then run the job with `sbatch`:

```bash
sbatch my-job.sh
```

#### Multinode

Something is wrong and this doesn't work right now on the NLP cluster. In theory this should work:

```bash
#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --job-name=levanter-test
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --output=levanter_%j.log
#SBATCH --mem=16G
#SBATCH --nodes=2

## On the Stanford NLP cluster you might need this:
export PATH=$(echo $PATH | sed 's|:/usr/local/cuda/bin||')

## Activate your virtual environment
source levanter/bin/activate

srun --nodes=2 python -m levanter.main.train_lm --config config/gpt2_small --trainer.per_device_parallelism -1
```

Then run the job with `sbatch`:

```bash
sbatch my-job.sh
```


## Miscellaneous Problems

Please see the [FAQ](faq.md) for solutions to common problems.

###  CUDA: `XLA requires ptxas version 11.8 or higher`

See FAQ entry, but some variant of this should work:

```bash
export PATH=$(echo $PATH | sed 's|:/usr/local/cuda/bin||')
```
