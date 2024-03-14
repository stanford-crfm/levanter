# Getting Started on GPU

**Note** We only test on Ampere (e.g. A100s or 30xx series)  GPUs. If it works with JAX it should work, though.

## Installation Option 1: Using a Virtual Environment

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

### Step 4: WandB login

To use WandB, you con log in to your WandB account on the command line as follows:
```bash
wandb login ${YOUR TOKEN HERE}
```
You can find more information on getting setup with Weights and Biases here: https://wandb.ai/site

## Installation Option 2: Using a Docker Container
To take advantage of the fastest training speeds levanter has to offer, we recommend training in a Docker container. Training speeds are accelerated by [TransformerEngine's](https://github.com/NVIDIA/TransformerEngine) [FusedAttention](https://arxiv.org/abs/2205.14135) implementation, which requires a TransformerEngine installation in your environment. Luckily, we can use a docker container that already has Levanter and TransformerEngine installed for us.

### Step 1: Ensure You Have Docker Installed
To check if you have Docker installed, run
```
sudo docker --version
```
If it is not installed, you can follow the [installation instructions on their website]().

### Step 2: Download the Docker Container
```
sudo docker pull ghcr.io/nvidia/jax:levanter
```

### Step 3: Running the Docker Container For Levanter Users

If you just want to use Leventer out of the box to train models, these are the docker setup steps you should follow.

#### Running an Interactive Docker Shell
To run a docker container interactively, you can use the following command:
```
sudo docker run -it --gpus=all --shm-size=16g ghcr.io/nvidia/jax:levanter
```

Then you can run training commands from within your docker container as follows:
```
python -m levanter.main.train_lm \
    --config_path /opt/levanter/config/gpt2_small.yaml
```

#### Running a Docker Container
You can also run your docker container by passing it a command and letting it run without an interactive shell:
```
sudo docker run \
    --gpus=all \
    --shm-size=16g \
    -i ghcr.io/nvidia/jax:levanter \
    python -m levanter.main.train_lm \
    --config_path /opt/levanter/config/gpt2_small.yaml
```

For more information on how to train models in Levanter see our [User Guide](Getting-Started-Training.md).

### Step 3: Running the Docker Container For Levanter Developers
If you are going to be adding to or extending Levanter for your own use case, these are the docker setup steps you should follow.

Clone the Levanter repository:

```
git clone https://github.com/stanford-crfm/levanter.git
```

Then run an interactive docker container with your levanter directory mounted as a volume. In this example, let's say your Levanter
repo is located at `/nlp/src/username/levanter`, then you would run the command below to make that directory accessible to the docker container.

```
sudo docker run -it --gpus=all -v /nlp/src/username/levanter:/levanter --shm-size=16g ghcr.io/nvidia/jax:levanter
```

When your container starts, the Levanter repo you cloned will be available at `/levanter`.
You should `cd` into the levanter directory and run the install command for levanter from that directory.

```
cd /levanter
pip install -e .
```

Now you should be able to run training jobs in this container and it will use the Levanter version you have in your mounted directory:

```
python src/levanter/main/train_lm.py \
    --config_path config/gpt2_small.yaml
```

For more information on how to train models in Levanter see our [User Guide](Getting-Started-Training.md).

### Saving Your Updated Container

After you make updates to your docker container, you may want to preserve your changes by creating a new docker image. To do so, you can detach or exit from your running container and commit these changes to a new docker images.

#### Exiting the Container
To detach from your container, but leave it running use Ctrl + P -> Ctrl + Q keys.
To stop and exit from your container, just type `exit`

#### Getting the Container ID
Next you need to get the ID for your container. If you run
```
sudo docker ps -a
```
All running and stopped containers will be listed, it should look something like this:
```
CONTAINER ID   IMAGE                         COMMAND                  CREATED             STATUS                         PORTS     NAMES
6ab837e447fd   ghcr.io/nvidia/jax:levanter   "/opt/nvidia/nvidia_…"   26 minutes ago      Exited (1) 6 seconds ago                 gifted_ganguly
f5f5b36634f0   ghcr.io/nvidia/jax:levanter   "/opt/nvidia/nvidia_…"   30 minutes ago      Exited (127) 26 minutes ago              great_mendel
a602487cb169   ghcr.io/nvidia/jax:levanter   "/opt/nvidia/nvidia_…"   39 minutes ago      Exited (0) 31 minutes ago                practical_lumiere
```

#### Committing the Container
You should select the `CONTAINER ID` for the container you want to preserve and run

```
sudo docker container commit [CONTAINER ID] [image-Name]:[image-tag]
```

If I wanted to create a new docker image called `levanter:latest` from the container `6ab837e447fd`, I would run

```
sudo docker container commit 6ab837e447fd levanter:latest
```

#### Using the Committed Container
Now you can start a new container using your new image and all the changes you made to the original container should still be there:

```
sudo docker run -it --gpus=all --shm-size=16g levanter:latest
```

### Things to Watch Out For When Using Docker + Levanter

1. To use the Levanter datasets available on google cloud within a docker container, you need to install gcloud and login. See Google Cloud Setup instructions at the top of [Getting Started on TPU VMs](Getting-Started-TPU-VM).

2. If you are using a docker container on the Stanford NLP cluster, you need to check which GPUs have been allocated to you within your slurm job. Run `nvidia-smi` before you start your docker container and note the `Bus-Id` for each GPU. Then, after starting your docker container, run `nvidia-smi` again to discover the indices of the GPUs you've been allocated within the full node. The GPU index is listed to the left of the GPU name in the left most column. Run `export CUDA_VISIBLE_DEVICES=[YOUR GPU INDICES]` so the container will only use your allocated GPUs and not all the GPUs on the node. For example, if you are using GPUs `[2, 3, 4, 5]` you would run `export CUDA_VISIBLE_DEVICES=2,3,4,5`.



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
