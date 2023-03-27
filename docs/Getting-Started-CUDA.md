# Getting Started with CUDA

**Note** that right now CUDA support is very preliminary and Levanter does not work on a multi-machine setup.

## Setting up dependencies
```bash
# 3.11 is too new for tensorstore
# this doesn't actually install pytorch, but it bundles cuda dependencies nicely
conda create --name levanter -c pytorch -c nvidia pytorch-cuda=11.7 python=3.10
conda activate levanter
pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt
```

```bash
# optional: install go to get gpu heap utilization
# nvidia-smi doesn't work when using Jax since it preallocates all the memory
sudo apt-get install golang
# or do this if you don't have sudo:
conda install -c conda-forge go
```

## Running a job on Slurm

### Single Node

```bash
srun --cpus-per-task=128 --gres=gpu:8 --job-name=levanter-multi-1 --mem=1000G --nodelist=sphinx[7-8] --open-mode=append --partition=sphinx --time=14-0 ~/src/levanter/scripts/run-slurm.sh python examples/gpt2_example.py --config_path config/gpt2_small.yaml
```

### Multinode

Something is wrong and this doesn't work right now.
