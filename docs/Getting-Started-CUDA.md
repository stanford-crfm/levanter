# Getting Started on CUDA

**Note** that right now CUDA support is very preliminary and Levanter does not work on a multi-machine setup. (We think this is just a dependency/environment issue.)

We only test on Ampere (e.g. A100s or 30xx series)  GPUs. If it works with Jax it should work, though.

## Installation
### Setting up Conda Environment
```bash
# 3.11 is too new for tensorstore
# this doesn't actually install pytorch, but it bundles cuda dependencies nicely
conda create --name levanter -c pytorch -c nvidia pytorch-cuda=11.7 python~=3.10
conda activate levanter
```
### Install Jax with CUDA

This is a bit in flux in Jax, but please take a look at the [Jax Installation Guide](https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier). Here are two options that work circa June 2023:

```bash
# CUDA 12 installation
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# CUDA 11 installation
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Install Levanter

```bash
cd levanter
pip install -e .
```

## Running a job on Slurm

### Single Node

Simple example, customize to your needs:

```bash
srun --cpus-per-task=128 --gres=gpu:8 --job-name=levanter-multi-1 --mem=1000G --nodelist=sphinx7 --open-mode=append --partition=sphinx --time=14-0 ~/src/levanter/scripts/run-slurm.sh python src/levanter/main/train_lm.py --config_path config/gpt2_small.yaml
```

### Multinode

Something is wrong and this doesn't work right now.
