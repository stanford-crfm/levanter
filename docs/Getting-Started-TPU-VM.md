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
bash infra/spin-up-vm.sh <name> -z <zone> -t <type> [--preemptible]
```

Defaults are:
- `zone`: `us-east1-d`
- `type`: `v3-32`
- `preemptible`: `false`

**Notes**:

* This uploads setup scripts via scp. If the ssh-key that you used for Google Cloud requires passphrase or your ssh key
path is not `~/.ssh/google_compute_engine`, you will need to modify the script.
* The command will spam you with a lot of output, sorry.
* If you use a preemptible instance, you probably want to use the ["babysitting" script](#babysitting-script) to
the VM. That's explained down below in the [Running Levanter GPT-2](#running-levanter-gpt-2) section.



**For Stanford CRFM Developers**:

Stanford CRFM folks who are developing Levanter can pass a different setup script to `infra/spin-up-vm.sh` to get our NFS automounted:

```bash
bash infra/spin-up-vm.sh <name> -z <zone> -t <type> [--preemptible] -s infra/helpers/setup-tpu-vm-nfs.sh
```

In addition to creating the instance, it will also mount the `/files/` nfs share to all workers, which has a good
venv and a copy of the repo.


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

### Launch a GPT-2 Small in unattended mode (using nohup)
```bash
gcloud compute tpus tpu-vm ssh $NAME --zone $ZONE --worker=all --command 'WANDB_API_KEY=... levanter/infra/launch.sh python levanter/src/levanter/main/train_lm.py --config_path levanter/config/gpt2_small.yaml --trainer.checkpointer.base_path gs://<somewhere>'
```

`launch.sh` will run the command in the background and redirect stdout and stderr to a log file in the home directory
on each worker.

### Launch a GPT-2 Small in interactive mode
This version writes to the terminal, you should use tmux or something for long running jobs for this version. It's mostly for debugging.
```bash
gcloud compute tpus tpu-vm ssh $NAME --zone $ZONE --worker=all --command 'WANDB_API_KEY=... levanter/infra/run.sh python levanter/src/levanter/main/train_lm.py --config_path levanter/config/gpt2_small.yaml --trainer.checkpointer.base_path gs://<somewhere>'
```

### Babysitting Script

If you are using a preemptible TPU VM, you probably want to use the "babysitting" script that automatically re-creates
the VM. This is because preemptible instances can be preempted and will always be killed every 24 hours. The babysitting
script handles both the creation of the node and the running of a job, and also relaunches the TPU VM if it gets preempted.
It keeps running the command (and relaunching) until the command exits successfully.

Note that the babysitting-script will automatically set the `RUN_ID` environment variable if not set, and pass it to the
training command. This ensures that restarted jobs have the same run id, which is important for resumes to work.

You can run it like this:

```bash
infra/babysit-tpu-vm <name> -z <zone> -t <type> [--preemptible]  -- \
    WANDB_API_KEY=... levanter/infra/run.sh python levanter/src/levanter/main/train_lm.py --config_path levanter/config/gpt2_small.yaml
```

That `--` is important! It separates the spin up args from the running args. Also, you should never use `launch.sh`
with `babysit`, because nohup exits immediately with exit code 0.

### Running your own config

If you want to run your own config, we suggest you start from one of the existing configs. Then, if you're not using
an NFS server or similar, you should upload your config to GCS:

```bash
gsutil cp my_config.yaml gs://my_bucket//my_config.yaml
```

Afterward, you can use the config directly from the TPU VM instance, e.g.:

```bash
infra/babysit-tpu-vm <name> -z <zone> -t <type> [--preemptible] -- \
    WANDB_API_KEY=... levanter/infra/run.sh python levanter/src/levanter/main/train_lm.py --config_path gs://my_bucket/my_config.yaml \
    --trainer.checkpointer.base_path gs://path/to/checkpoints/
```

The `--config_path` argument can be a local path, a GCS path, or any URL loadable by fsspec.
With this configuration (unless `trainer.load_checkpoint` is false), Levanter will automatically
try to load the latest checkpoint if it exists.

Tokenizers are also loaded via fsspec, so you can use the same trick to load them from GCS if you have a custom
tokenizer, or you can use an HF tokenizer.

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


## Random Tricks

I (@dlwh) personally like to use pdsh instead of gcloud to run commands on all workers. It doesn't have the reboot
issue, and seems to work better for long-lived jobs and such. You can install it with `sudo apt-get install pdsh`.
You can then get the ips for your machines like so:

```bash
gcloud compute tpus tpu-vm describe --zone us-east1-d $name | awk '/externalIp: (.*)/ {print $2}'  > my-hosts
```

Then you can run a command on all workers like so:

```bash
pdsh -R ssh -w ^my-hosts 'echo hello'
```
