# Getting Started on GPU

**Note** that right now GPU support is very preliminary and Levanter does not work on a multi-machine setup. (We think this is just a dependency/environment issue.)

We only test on Ampere (e.g. A100s or 30xx series)  GPUs. If it works with JAX it should work, though.

## Installation
### Setting up Conda Environment
```bash
# 3.11 is too new for tensorstore
conda create --name levanter python=3.10
conda activate levanter
```
### Install Jax with CUDA

This is a bit in flux in JAX, but please take a look at the [JAX Installation Guide](https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier). Here are two options that work circa June 2023:

```bash
# CUDA 12 installation
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# CUDA 11 installation
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Install Levanter

```bash
git clone https://github.com/stanford-crfm/levanter.git
cd levanter
pip install -e .
```

### WandB login

To use WandB to track your learning curves and other experiment metrics, you con log in to your WandB account on the command line as follows:
```bash
wandb login ${YOUR TOKEN HERE}
```
You can find more information on getting setup with Weights and Biases here: https://wandb.ai/site


## Running a job on Slurm

### Single Node

A simple srun example within your levanter conda environment and cloned levanter directory. Customize to your needs:

```bash
srun --account=nlp --cpus-per-task=128 --gpus-per-node=8 --job-name=levanter-multi-1 --mem=1000G  --open-mode=append --partition=sphinx --time=14-0 infra/run-slurm.sh python src/levanter/main/train_lm.py --config_path config/gpt2_small.yaml
```

### Multinode

Something is wrong and this doesn't work right now.
