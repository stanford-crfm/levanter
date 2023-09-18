# Replicating Alpaca

In this tutorial, we will replicate [Stanford Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)
using the new [Llama 2](https://ai.meta.com/llama/) model and [Levanter](https://github.com/stanford-crfm/levanter).
We'll use a TPU V3-32 VM, though this same tutorial should work on an A100 box as well.

## Setup

### Cloning Levanter

First, we'll clone Levanter:

```bash
git clone https://github.com/stanford-crfm/levanter.git
cd levanter
pip install -e .
```

### Setting up a TPU VM

First, we'll spin up a TPU VM using the [Getting Started with TPUs](./Getting-Started-TPU-VM.md) guide.
If you haven't gone through that guide before, you should do so now. If you have, you can just run, e.g.:

```bash
bash infra/spin-up-vm.sh llama-32 -z us-east1-d -t v3-32 --preemptible
```

## Choosing a Llama

### Llama 1

If you want, you can use [Decapoda's repo for Llama 1](https://huggingface.co/decapoda-research/llama-7b-hf),
you'll just need to pass in `--model_name_or_path decapoda-research/llama-7b-hf` instead of `meta-llama/Llama-2-7b-hf`.

### Getting Llama 2

If you haven't already, go to [Llama 2's Hugging Face page](https://huggingface.co/meta-llama/Llama-2-7b-hf) and request access to the model.

Once you have access, go to [Hugging Face's Tokens page](https://huggingface.co/settings/tokens) to get an API token.

## The Alpaca script

We have a [Levanter version](https://github.com/stanford-crfm/levanter/blob/main/examples/alpaca.py) of the [original Alpaca script](https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py)

There's a bit of ceremony in both versions, but the broad strokes of the script are the same. The main differences
are highlighted in the Levanter version.

We also need a config file, which we paste here:

```yaml
# cf https://github.com/tatsu-lab/stanford_alpaca#fine-tuning
#model_name_or_path: decapoda-research/llama-7b-hf
model_name_or_path: meta-llama/Llama-2-7b-hf
trainer:
  mp: p=f32,c=bfloat16
  wandb:
    project: "levanter-alpaca"
  num_train_steps: 1218  # 128 * 1218 = 155904, which is almost but not quite 3 epochs, which is what alpaca did
  train_batch_size: 128
  per_device_parallelism: 1  # TPUS have fairly limited memory, so we can't do too much parallelism
                             # If using Llama 1 you can probably do 4 here
                             # or a TPU v3-64 with LLama 2 can probably do 2
  # if using model parallelism, this is useful:
  tensor_parallel_axes: ["mlp", "heads"]
optimizer:
  learning_rate: 2e-5
  weight_decay: 0.0
```

## Launching the job

Now we can launch the job. We need just a tiny bit of ceremony to get the Hugging Face and WANDB API tokens in the environment:
(If you're using Llama 1, you don't need the `HUGGING_FACE_HUB_TOKEN` line.)

```bash
gcloud compute tpus tpu-vm ssh llama-32 -z us-east1-d --worker=all \
--command="WANDB_API_KEY=${YOUR TOKEN HERE} \
HUGGING_FACE_HUB_TOKEN=${YOUR TOKEN HERE} \
bash levanter/infra/run.sh python \
--config_path levanter/examples/train-alpaca.yaml \
--trainer.checkpointer.base_path gs://<somewhere> \
--hf_save_path gs://<somewhere> \
--trainer.wandb.id <some id>"  # optional, but useful if using preemption
```

If you're using preemptible or TRC TPUs, you'll want to add `--trainer.wandb.id <some id>` to the command line,
and probably use the [babysitting script](./Getting-Started-TPU-VM.md#babysitting-script) to automatically restart the
vm and job if it gets preempted. That would look like this:

```bash
infra/babysit-tpu-vm.sh llama-32 -z us-east1-d -t v3-32 --preemptible -- \
WANDB_API_KEY=${YOUR TOKEN HERE} \
HUGGING_FACE_HUB_TOKEN=${YOUR TOKEN HERE} \
bash levanter/infra/run.sh python \
--config_path levanter/examples/train-alpaca.yaml \
--trainer.checkpointer.base_path gs://<somewhere> \
--hf_save_path gs://<somewhere> \
--trainer.wandb.id <some id>  # optional, but useful if using preemption
```
