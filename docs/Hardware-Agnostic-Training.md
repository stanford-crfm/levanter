# Switching Hardware Mid Training Run

One of the standout features of Levanter is its ability to seamlessly operate on both GPU and TPU platforms. This guide will walk you through the steps and best practices for switching hardware configurations mid-training, ensuring a smooth and efficient training process for your models.

### Starting a Training Run

After getting setup up in your [GPU](Getting-Started-GPU.md) or [TPU](Getting-Started-TPU-VM.md) environment (please see [our tutorials](Installation.md) on how to do so if you haven't already), the following command will kick off a training run.

```bash
python src/levanter/main/train_lm.py \
    --config_path config/gpt2_small.yaml \
    --trainer.checkpointer.base_path output/gpt2_small_webtext \
    --trainer.train_batch_size 256 \
    --trainer.per_device_parallelism 128
```
This example command assumes you're starting your training run on 2 GPUs in a [FSDP (Fully Sharded Data Parallel)](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) setup. The `trainer.per_device_parallelism` parameter instructs Levanter to distribute the processing of each training batch across multiple accelerators, in this case, across 2 GPUs. With this setting, each GPU handles a subset of the training batch, processing 128 examples each, and their combined efforts contribute to the effective handling of the entire 256 batch size.

We have comprehensive documentation on [getting started with training](Getting-Started-Training.md) in levanter, the [configuration file](Configuration-Guide.md), and [training on your own dataset](Training-On-Your-Data.md) that you should check out for more details. We also have a notebook tutorial on [how to add FSDP to custom architectures](https://colab.research.google.com/drive/1QX4yH3zRFF3Xiibf1aahETcSQ5nbcUMz) implemented in Levanter with Haliax.


### How the Model Checkpoint is Saved
The frequency of checkpoint saves can be adjusted by configuring the `checkpointer.save_interval` parameter in your training command. This parameter requires an integer value, which specifies the interval (in minutes) between each checkpoint save. By default, checkpoints are saved every 15 minutes.

Whenever a checkpoint is saved, Levanter creates a new directory named after the training run ID within the path specified by `trainer.checkpointer.base_path`. This directory stores each checkpoint file. Every checkpoint file containes:

* the model's parameters
* the current optimizer and trainer states
* metadata on the current training step and the time the training step was saved

```
user@DN0a2335e6 rj96a79n/step-1591 % ls
metadata.json	model		training_state	worker-0.cert
```

### Moving the Checkpoint File to Google Storage
To switch to a new hardware configuration mid training run, you need to move your checkpoint file and your training data over to the new hardware environment. If you started your training run on a GPU server and want to move it to a TPU VM instance, you can use the following command to upload your checkpoint file to a Google Cloud Storage bucket that your TPU VM can access.

```bash
gsutil -m cp -r <GPU MODEL CHECKPT PATH> gs://<SOMEWHERE>
```
You need to be logged into your GCP account in your current environment for this command to run properly. Information on how to set this up can be found [here](Getting-Started-TPU-VM.md).

### Re-Configuring `per_device_parallelism`
When you switch to a new hardware setup, whether its from GPU to TPU or simply adjusting the number of GPUs, it is important you keep your training batch size (`trainer.train_batch_size`) the same. You should however, adjust your `trainer.per_device_parallelism` size to match the new number of accelerators you will be using. For example, if you are moving to a v3-32 TPU VM instances, this means you are using v3 TPUs with 32 TensorCores (I found [this page](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm) helpful for understanding TPU terminology). Each tensor core can handle a portion of your total batch, so you might set your `trainer.per_device_parallelism` size to be 256 / 32 = 8. If you were to set `trainer.per_device_parallelism` to be some suitable value smaller than 8 (like 1, 2, or 4), then Levanter would perform gradient accumulation until 8 examples have been processed by the model on this accelerator.

## Resuming A Training Run
Once your new hardware environment is setup and you've moved your training checkpoint over, you can resume your training run with this command:

```bash
python levanter/src/levanter/main/train_lm.py \
	--config_path levanter/config/gpt2_small.yaml \
	--trainer.load_checkpoint_path gs://<SOMEWHERE> \
	--trainer.wandb.resume true \
	--trainer.id rj96a79n
```
Use `trainer.load_checkpoint_path` to specify where the model checkpoint you moved is located. `trainer.wandb.resume true` will resume all your learning curves where the stopped training run left off, and `trainer.id` is the trainer ID used by WandB to identify your stopped training run.

## Picking Up Where You Left Off
When your training run resumes, the trainer will first fast-forward through your training data to continue from the exact point in the epoch where your checkpoint was saved. You should see a progress bar tracking the fast-forward through your data.

Resuming your training run restores your checkpoint's optimizer states, so your WandB learning curves should resume smoothly from where they left off.

Loss curve when training was stopped on 2 GPUs:
![stopped-loss](figures/stopped_curve.png "loss curve when training stopped")

Loss curve after resuming training on a TPU v3-32 VM instance:
![resumed-loss](figures/resumed_curve.png "loss curve after training resumed")
