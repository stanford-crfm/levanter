# Getting Started on TPU VMs

This guide will walk you through the steps to get started with Levanter on TPU VMs.

## Google Cloud Setup

First you need gcloud installed and configured. You can find instructions for that [here](https://cloud.google.com/sdk/docs/quickstarts)
or if you're a conda person you can just run `conda install -c conda-forge google-cloud-sdk`.

Second, you need to follow some steps to enable Cloud TPU VM. You can follow Google's directions [here](https://cloud.google.com/tpu/docs/users-guide-tpu-vm)
but the gist of it is you need to enable the TPU API and the Compute Engine API. You can do this by running:

```bash
gcloud auth login  # if you haven't already
gcloud auth application-default login  # on your local machine
gcloud components install alpha
gcloud services enable tpu.googleapis.com
gcloud config set account your-email-account
gcloud config set project your-project
```

You can follow more steps there to get used to things like creating instances and such, but we'll only discuss the
most important details here.

Google recommends not running those first two commands on a VM, and instead using tokens and service accounts. You can
find more information about that [here](https://cloud.google.com/docs/authentication/production#auth-cloud-implicit-python).
Honestly, if you're working outside of a corp environment and not dealing with private data, I don't bother...

You may also need to create an SSH key and add it to your Google Cloud account. Consider using
[GCloud's guide on ssh keys](https://cloud.google.com/compute/docs/connect/add-ssh-keys#metadata) (or OS Login if you do that)
to set up ssh keys and [using `ssh-agent`](https://kb.iu.edu/d/aeww) to make executing the SSH commands easier.

## Creating a TPU VM Instance

An important thing to know about TPU VMs is that they are not a single machine (for more than a v3-8). Instead, they
are a collection of workers that are all connected to the same TPU pod. Each worker manages a set of 8 TPUs.
This means that you can't just run a single process on a TPU VM instance, you need to run a distributed process,
and you can't just set up one machine, but a whole cluster. We have some scripts to help with this.

### Automatic Setup

You can use `infra/spin-up-vm.sh` to create a TPU VM instance. In addition to creating the instance, it will set up
the venv on each worker, and it will clone the repo to `~/levanter/`.

**For Public Users**:

```bash
bash infra/spin-up-vm.sh <name> -z <zone> -t <type> -n <subnetwork> [--preemptible] [--use-alpha]
```

Defaults are:
- `zone`: `us-east1-d`
- `type`: `v3-32`
- `subnetwork`: `default` (set to custom VPC subnet, useful for tpuv4 configs)
- `preemptible`: `false`
- `use-alpha`: `false` (mark `true` for tpuv4s in alpha zones like `us-central2`)

**Notes**:

* This uploads setup scripts via scp. If the ssh-key that you used for Google Cloud requires passphrase or your ssh key
path is not `~/.ssh/google_compute_engine`, you will need to modify the script.
* The command will spam you with a lot of output, sorry.
* If you use a preemptible instance, you probably want to use the ["babysitting" script](#babysitting-script) to
the VM. That's explained down below in the [Running Levanter GPT-2](#running-levanter-gpt-2) section.


## Useful commands

### SSHing into one TPU VM worker

`gcloud compute tpus tpu-vm ssh $name   --zone us-east1-d --worker=0`

### Running a command on all workers (in parallel)
`gcloud compute tpus tpu-vm ssh $name   --zone us-east1-d --worker=all --command="echo hello"`

### SCPing a file to all workers
`gcloud compute tpus tpu-vm scp my_file $name:path/to/file --zone us-east1-d --worker=all`

### SCPing a file to one worker
`gcloud compute tpus tpu-vm scp my_file $name:path/to/file --zone us-east1-d --worker=0`

### SCPing a file from one worker
`gcloud compute tpus tpu-vm scp $name:path/to/file my_file --zone us-east1-d --worker=0`

## Running Levanter GPT-2
Now that you have a TPU VM instance, you can follow the [Getting Started](Getting-Started-Training.md) steps, but here are a few shortcuts:

### Launch a GPT-2 Small in unattended mode

You will need a [Docker installation](https://docs.docker.com/engine/install/)
on your development machine to build and run images on TPUs.

First create a configuration file for future launches in your Levanter directory:

```
cat > .config <<EOF
env:
    WANDB_API_KEY:
    WANDB_ENTITY:
    WANDB_PROJECT:
    HF_TOKEN:
    TPU_STDERR_LOG_LEVEL: 0
    TPU_MIN_LOG_LEVEL: 0
    LIBTPU_INIT_ARGS: <extra args to libtpu>

docker_repository: levanter
zone: us-west4-a
tpu_name: test-spin-up-32
tpu_type: "v5litepod-16"
vm_image: "tpu-ubuntu2204-base"
capacity_type: "preemptible"
autodelete: false
subnetwork: "default"

EOF
```

If you want to customize the docker image that is created and uploaded to GCP Artifact Registry, you can add config `image_name: "YOUR-DOCKER-NAME"`.

Note that you can also configure docker to push to GHCR by setting
```
docker_registry: ghcr
github_user: <YOUR USERNAME>
github_token: <YOUR TOKEN>
```
By default, the tpu instance won't be able to access the docker image, so you may need to make it public.

Now run `launch.py`. This will package your current directory into a Docker image and run it on your workers. Everything after the `--` is run on each worker.

```bash
python infra/launch.py -- python src/levanter/main/train_lm.py --config_path config/gpt2_small.yaml --trainer.checkpointer.base_path gs://<somewhere>'
```

### Launch a GPT-2 Small in interactive mode

To run in the foreground, use `--foreground` with the `launch.py` script. You should use tmux or something for long running jobs for this version. It's mostly for debugging.
```bash
python infra/launch.py -- python src/levanter/main/train_lm.py --config_path config/gpt2_small.yaml --trainer.checkpointer.base_path gs://<somewhere>'
```

### Babysitting Script

If you are using a preemptible TPU VM, you probably want to use the "babysitting" script that automatically re-creates
the VM. This is because preemptible instances can be preempted and will always be killed every 24 hours. You can run `launch.py` with the `--retries` and `--foreground` parameter to accomplish this. If `--retries` is greater than 1, `launch.py` will automatically attempt to re-create the VM and re-run the command if it fails. (`--foreground` is necessary to keep the script from returning immediately.)

```bash
    python infra/launch.py --retries=100 --foreground --tpu_name=my_tpu -- python src/levanter/main/train_lm.py --config_path config/my_config.yaml \
    --trainer.checkpointer.base_path gs://path/to/checkpoints/
```

That `--` is important! It separates the spin up args from the running args.
Also you should always use `--foregrouund` with `babysit-tpu-vm`, as the
background mode will always return immediately.

### Running your own config

If you want to run your own config, we suggest you start from one of the existing configs. Just copy it to
a new file:

`cp config/gpt2_small.yaml config/my_config.yaml`

If you're using `launch.py`, the config will be automatically uploaded as part of your Docker image, so you
can just reference the local config path in your command line:

```

Afterward, you can use the config directly from the TPU VM instance, e.g.:

```bash
    python infra/launch.py -- python src/levanter/main/train_lm.py --config_path config/my_config.yaml \
    --trainer.checkpointer.base_path gs://path/to/checkpoints/
```

With this configuration (unless `trainer.load_checkpoint` is false), Levanter will automatically
try to load the latest checkpoint if it exists.

Tokenizers and configuration files are loaded via `fsspec` which supports remote
filesystems , so you can also copy your tokenizer or config file to GCS and use
a `gs://` path to access it.

## Common Issues
### (CRFM) Permission denied on `/files`

If you get a permission denied error on `/files`, you probably need to run `sudo chmod -R a+rw /files/whatever` on the
TPU VM instance. This is because the TPU VM instance sets different UID/GID for the user on each and every worker, so
you need to make sure that the permissions are set correctly. These periodically get messed up. A umask would probably
fix this. (TODO!)

### (CRFM) Git permissions issues

Git doesn't like doing operations in a directory that is owned by root or that has too funky of permissions. If you get a git error, you probably need to
add a safe directory on your workers:

```bash
gcloud compute tpus tpu-vm ssh $NAME --zone $ZONE --worker=all --command 'git config --global --add safe.directory /files/<wherever>'
```

### Can't find TPUs

If you get the warning `No GPU/TPU found, falling back to CPU.` then something else might be using the TPU, like a zombie python
process. You can kill all python processes with `gcloud compute tpus tpu-vm ssh $NAME --zone $ZONE --worker=all --command 'pkill python'`

If that fails, you can try rebooting the TPU VM instance. This is harder than it sounds, because you can't just reboot the instance
from the console or with a special command. You have to do it with ssh:

```bash
gcloud compute tpus tpu-vm ssh $NAME --zone $ZONE --worker=all --command 'sudo reboot'
```

**and then you have to ctrl-c this after about 10 seconds**. Otherwise, gcloud will think the command failed and will
try again, and get stuck in a loop forever. (You can ctrl-c it at any point after 10 seconds.)

### Docker-related Issues

If you get a `permission denied` error with your `docker.sock`,
you should be able to fix it by running the following script in the launching machine and then restarting your shell:

```shell
sudo usermod -aG docker $USER
```

If you get an error like `denied: Unauthenticated request. Unauthenticated requests do not have permission "artifactregistry.repositories.uploadArtifacts"`,
You should run the following authentication script:

```shell
# change based on your GCP zone
gcloud auth configure-docker ${GCP_ZONE}.pkg.dev
# for example:
gcloud auth configure-docker us-central2-docker.pkg.dev
```
