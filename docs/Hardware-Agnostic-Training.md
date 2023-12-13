# Switching Hardware Mid Training Run

Welcome to our tutorial on how to leverage the flexible and dynamic capabilities of Levanter for your machine learning projects. One of the standout features of Levanter is its ability to seamlessly operate on both GPU and TPU platforms. This guide will walk you through the steps and best practices for switching hardware configurations mid-training, ensuring a smooth and efficient training process for your models.

## Starting a Training Run

After getting setup up in your [GPU](Getting-Started-GPU.md) or [TPU](Getting-Started-TPU-VM.md) environment (please see [our tutorials](Installation.md) on how to do so if you haven't already), the following command will kick off a training run.

```bash
python src/levanter/main/train_lm.py \
    --config_path config/gpt2_small.yaml \
    --trainer.checkpointer.base_path output/gpt2_small_webtext \
    --trainer.train_batch_size 256 \
    --trainer.per_device_parallelism 128
```
This example command assumes you're starting your training run on 2 GPUs in a [FSDP (Fully Sharded Data Parallel)](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) setup. The `trainer.per_device_parallelism` argument tells Levanter how to split your overall training batch size across the different accelerators, in this case, across 2 GPUs. Here, 128 training examples will be passed into the model on each GPU, and the outputs will be accumulated to form the full 256 batch size.

We have thorough documentation on [getting started with training](Getting-Started-Training) in levanter, the [configuration file](Configuration-Guide.md), and [training on your own dataset](Training-On-Your-Data.md) that you should check out for more details. We also have a notebook tutorial on [how to add FSDP to custom architectures](https://colab.research.google.com/drive/1QX4yH3zRFF3Xiibf1aahETcSQ5nbcUMz?authuser=1#scrollTo=lFZOnJD7QtZm&uniqifier=2) implemented in Levanter.

## How the Model Checkpoint is Saved
## Re-Configuring Batch Size
## Resuming A Training Run
## Fastfowarding Through the Data
## Picking Up Where You Left Off
