# Getting Started with Levanter on TPU VMs

This guide will walk you through the steps to get started with Levanter on TPU VMs.

## Google Cloud Setup

First you need gcloud installed and configured. You can find instructions for that [here](https://cloud.google.com/sdk/docs/quickstarts)
or if you're a conda person you can just run `conda install -c conda-forge google-cloud-sdk`.

Second, you need to follow some steps to enable Cloud TPU VM. You can follow Google's directions [here](https://cloud.google.com/tpu/docs/users-guide-tpu-vm)
but the gist of it is you need to enable the TPU API and the Compute Engine API. You can do this by running:

```bash
gcloud components install alpha
gcloud services enable tpu.googleapis.com
gcloud config set account your-email-account
gcloud config set project your-project
```

You can follow more steps there to get used to things like creating instances and such, but we'll only discuss the
most important details here.

You may also need to create an SSH key and add it to your Google Cloud account. TODO

## Creating a TPU VM Instance

An important thing to know about TPU VMs is that they are not a single machine (for more than a v3-8). Instead, they
are a collection of workers that are all connected to the same TPU pod, though each worker manages a set of 8 TPUs.
This means that you can't just run a single process on a TPU VM instance, you need to run a distributed process,
and you can't just set up one machine. You need to set up a cluster of machines.

### Stanford CRFM

Stanford CRFM folks can use `scripts/spin-up-tpu-vm.sh` to create a TPU VM instance:
```bash
bash scripts/spin-up-tpu-vm.sh <name> -z <zone> -t <type> [--preemptible]
```

Defaults are:
- `zone`: `us-east1-d`
- `type`: `v3-32`
- `preemptible`: `false`

The command will spam you with a lot of output, sorry.

In addition to creating the instance, it will also mount the `/files/` nfs share to all workers, which has a good
venv and a copy of the repo.

### Other Folks

TODO, but you can follow the script above to get an idea of what you need to do.

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


## Next Steps

Now that you have a TPU VM instance, you can follow the [Running Levanter] steps, but here are a few shortcuts:

### Running Levanter GPT-2


#### Launch a GPT-2 Small in unattended mode (using nohup)
```bash
gcloud compute tpus tpu-vm ssh $NAME --zone $ZONE --worker=all --command 'WANDB_API_KEY=... bash /files/levanter/scripts/launch.sh python /files/levanter/examples/gpt2_example.py --config_path /files/levanter/config/gpt2_small.yaml --trainer.checkpoint_dir gs://<somewhere>'
```

launch.sh will run the command in the background and redirect stdout and stderr to a log file in the home directory
on each worker.

#### Launch a GPT-2 Small in interactive mode
This version writes to the terminal, you should use tmux or something for long running jobs for this version. It's mostly for debugging.
```bash
gcloud compute tpus tpu-vm ssh $NAME --zone $ZONE --worker=all --command 'WANDB_API_KEY=... bash /files/levanter/scripts/run.sh python /files/levanter/examples/gpt2_example.py --config_path /files/levanter/config/gpt2_small.yaml --trainer.checkpoint_dir gs://<somewhere>'
```

## Common Issues
### (CRFM) Permission denied on `/files`

If you get a permission denied error on `/files`, you probably need to run `sudo chmod -R a+rw /files/whatever` on the
TPU VM instance. This is because the TPU VM instance sets different UID/GID for the user on each and every worker, so
you need to make sure that the permissions are set correctly. These periodically get messed up. A umask would probably
fix this. (TODO!)

### Git permissions issues

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
