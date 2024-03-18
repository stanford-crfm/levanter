# Getting Started on GPU

**Note** We only test on Ampere (e.g. A100s or 30xx series)  GPUs. If it works with JAX it should work, though.

We have two installation options for Levanter.

1. [Using a Virtual Environment](#using-a-virtual-environment). This is the simplest way if you don't have root access to your machine (and don't have rootless docker installed).
2. [Using a Docker Container](#using-a-docker-container). This is the best way to get the fastest training speeds, because the Docker container has [TransformerEngine](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html),
and Levanter users TransformerEngine's FusedAttention implementation to accelerate training.

## Using a Virtual Environment

### TL;DR

```bash
virtualenv -p python3.10 levanter
source levanter/bin/activate
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
git clone https://github.com/stanford-crfm/levanter.git
cd levanter
pip install -e .
```

### Step 1: Setting up a Virtual Environment

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
### Step 2: Install Jax with CUDA

Please take a look at the [JAX Installation Guide](https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier). Here are two options that work circa June 2023:

```bash
# CUDA 12 installation
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# CUDA 11 installation
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Step 3: Install Levanter

You can either install Levanter from PyPI or from source. We recommend installing from source if you plan to make changes to the codebase.


#### Install from Source

```bash
git clone https://github.com/stanford-crfm/levanter.git
cd levanter
pip install -e .
```

#### Install from PyPI

This is frequently out of date, so we recommend installing from source.

```bash
pip install levanter
```

### Step 4: WandB login

By default, Levanter logs training runs to Weights and Biases. You can sign up for a free WandB account at https://wandb.ai/site.

You can get an API token from [Weights and Biases](https://wandb.ai/authorize) and use it to log in to your WandB account on the command line as follows:

To use WandB, you con log in to your WandB account on the command line as follows:
```bash
wandb login ${YOUR TOKEN HERE}
```
You can find more information on getting setup with Weights and Biases here: https://wandb.ai/site

If you don't want to use WandB, you can disable it by running:

```bash
wandb offline
```

#### Using a Different Tracker

You can also use TensorBoard for logging. See the [Tracker](./dev/Trackers.md) documentation for more information.

## Using a Docker Container

To take advantage of the fastest training speeds Levanter has to offer, we recommend training in a Docker container.
Training speeds are accelerated by [TransformerEngine's](https://github.com/NVIDIA/TransformerEngine) [FusedAttention](https://arxiv.org/abs/2205.14135) implementation, which requires a TransformerEngine installation in your environment. Luckily, we can use a Docker container that already has Levanter and TransformerEngine installed for us.

This Docker image is built by NVIDIA's [JAX Toolbox](https://github.com/NVIDIA/JAX-Toolbox) effort
and is continuously updated with the latest versions of JAX, CUDA, TransformerEngine, and Levanter.

### Ensure You Have Docker Installed
To check if you have Docker installed, run

```bash
sudo docker --version
```

If it is not installed, you can follow the [installation instructions on their website](https://docs.docker.com/engine/install/).

You'll also need to have the `nvidia-container-toolkit` installed. You can follow the [installation instructions on their website](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

### Download the Docker Container

Technically optional, since the first time you run the container it will be downloaded, but you can download the container ahead of time with the following command:

```bash
sudo docker pull ghcr.io/nvidia/jax:levanter
```

### Running the Docker Container For Levanter Users

If you just want to use Leventer out of the box to train models, these are the Docker setup steps you should follow.

If you're interested in actively developing Levanter while using a Docker container, see the [Developing in a GPU Docker Container](./dev/GPU-Docker-Dev.md) guide.

#### Running an Interactive Docker Shell
To run a docker container interactively, you can use the following command:

```bash
sudo docker run -it --gpus=all --shm-size=16g ghcr.io/nvidia/jax:levanter
```

Then you can run training commands from within your docker container as follows:

```bash
python -m levanter.main.train_lm \
    --config_path /opt/levanter/config/gpt2_small.yaml
```

#### Running A Job in a Docker Container
You can also run a job in a Docker container with the following command:

```bash
sudo docker run \
    --gpus=all \
    --shm-size=16g \
    -i ghcr.io/nvidia/jax:levanter \
    python -m levanter.main.train_lm \
    --config_path /opt/levanter/config/gpt2_small.yaml
```

For more information on how to train models in Levanter see our [User Guide](Getting-Started-Training.md).

### Mounting A Local Fork of Levanter Inside the Docker Container

If you are going to be adding to or extending Levanter for your own use case, these are the Docker setup steps you should follow.

Clone the Levanter repository:


```bash
git clone https://github.com/stanford-crfm/levanter.git
```

Then run an interactive docker container with your levanter directory mounted as a volume. In this example, let's say your Levanter
repo is located at `/nlp/src/username/levanter`, then you would run the command below to make that directory accessible to the docker container.

```bash
sudo docker run -it --gpus=all -v /nlp/src/username/levanter:/levanter --shm-size=16g ghcr.io/nvidia/jax:levanter
```

When your container starts, the Levanter repo you cloned will be available at `/levanter`.
You should `cd` into the levanter directory and run the install command for levanter from that directory.

```bash
cd /levanter
pip install -e .
```

Now you should be able to run training jobs in this container and it will use the Levanter version you have in your mounted directory:

```bash
python src/levanter/main/train_lm.py \
    --config_path config/gpt2_small.yaml
```


### Things to Watch Out For When Using Docker + Levanter

1. To use the Levanter datasets available on google cloud within a Docker container, you need to install gcloud and login inside the docker container. See Google Cloud Setup instructions at the top of [Getting Started on TPU VMs](./Getting-Started-TPU-VM.md).

2. If you are using a Docker container on the Stanford NLP cluster, you need to check which GPUs have been allocated to you within your slurm job. Run `nvidia-smi` before you start your docker container and note the `Bus-Id` for each GPU. Then, after starting your docker container, run `nvidia-smi` again to discover the indices of the GPUs you've been allocated within the full node. The GPU index is listed to the left of the GPU name in the left most column. Run `export CUDA_VISIBLE_DEVICES=[YOUR GPU INDICES]` so the container will only use your allocated GPUs and not all the GPUs on the node. For example, if you are using GPUs `[2, 3, 4, 5]` you would run `export CUDA_VISIBLE_DEVICES=2,3,4,5`.

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

This example uses `sbatch` to submit a job to Slurm. This example assumes you have cloned and installed the Levanter repository.

Nvidia recommends using this method (rather than one process per node) for best performance.

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
