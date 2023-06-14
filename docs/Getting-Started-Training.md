# Training

This document provides a guide on how to launch model training and configure it according to your specific needs.

## Launch Model Training

To launch the training of a GPT2 model, run the following command:
```bash
python levanter/examples/gpt2_example.py --config_path config/gpt2_small.yaml
```

This will execute the training pipeline pre-defined in the [gp2_example.py](../examples/gpt2_example.py) and set model and training configuration
set in [gpt2_small.yaml](../config/gpt2_small.yaml). You can find more template configurations in the [config](../config/) directory.

Configuration files are processed using [Pyrallis](https://github.com/eladrich/pyrallis). Pyrallis is yet-another yaml-to-dataclass library.

## Set Custom Training Configuration
In machine learning experiments, it is common to adjust model hyperparameters. In this section, we will provide examples of different use cases
and explain the corresponding parameters that you can change.

### Change Model Parameters
To change the dimensions of your GPT2 model and increase the number of training steps to 10,000, I can use the following command:

```
python levanter/examples/gpt2_example.py \
    --config_path config/gpt2_small.yaml \
    --model.num_heads 20 \
    --model.num_layers 36 \
    --model.hidden_dim 1280 \
    --trainer.num_train_steps 10000
```

This will overwrite the default model and training configurations and set the following parameters:
- `model.num_heads`: The number of heads in the multi-head attention layers of the transformer encoder and decoder.
- `model.num_layers`: The number of layers in the transformer encoder and decoder.
- `model.hidden_dim`: The hidden dimension of the model. This is the dimension of the hidden states of the transformer encoder and decoder. Note
that the hidden dimension must be divisible by the number of heads.
- `trainer.num_train_steps`: The number of training steps to run.

You can find a complete list of parameters to change from the `TrainerConfig` in [config.py](src/levanter/config.py) and `Gpt2Config` in
[gpt2.py](src/levanter/models/gpt2.py).

### Change Checkpoint Settings
To change the frequency of saving checkpoints, you can use the following command:

```
python levanter/examples/gpt2_example.py \
    --config_path config/gpt2_small.yaml \
    --trainer.load_checkpoint_path checkpoints/gpt2/wandb_run_name  \
    --trainer.checkpointer.base_path checkpoints/gpt2/ \
    --trainer.checkpointer.save_interval 20m
```

This will overwrite the default checkpoint settings from the `TrainerConfig` and `CheckpointerConfig` in [config.py](src/levanter/config.py) to
save checkpoints every 20 minutes. The checkpoint will be saved to the directory `checkpoints/gpt2/` with the WandB name `wandb_run_name`.

Note that:
- The `--trainer.load_checkpoint_path` argument is optional. You only need to specify it if you want to load a checkpoint from a previous
run. If it is specified, the trainer will recursively search for the latest checkpoint in the directory and load it.
If you do not specify it, the trainer will start training from scratch.
- Both `--trainer.load_checkpoint_path` and `--trainer.checkpointer.base_path` supports local path and cloud storage path (e.g. S3, GCS, etc.), as
long as the path is accessible from the machine that you are running the training script on.
- The `--trainer.checkpointer.save_interval` argument supports the following units: `s` (seconds), `m` (minutes), `h` (hours), `d` (days).
- We will soon change to use wandb_run_id instead of wandb_run_name as the checkpoint directory name.

### Change Evaluation Parameters
To change how often the model is evaluated during training, you can use the following command:

```
python levanter/examples/gpt2_example.py \
    --config_path config/gpt2_small.yaml \
    --trainer.steps_per_eval 500
```

This will overwrite the default eval frequency (every 1,000) from the `TrainerConfig` in [config.py](src/levanter/config.py) to every 500 steps.

### Change Parallelism Settings
By default, Levanter will split the number of examples in `train_batch_size` equally across all available GPUs.
To set explicit number of examples to process on each device during training and evaluation, you can use the following command:

```
python levanter/examples/gpt2_example.py \
    --config_path config/gpt2_small.yaml \
    --trainer.batch_size 256 \
    --trainer.per_device_parallelism 64 \
    --trainer.eval_per_device_parallelism 64
```

Note that:
- `batch_size` must be divisible by `per_device_parallelism` and `eval_per_device_parallelism`.
- If `num_devices * per_device_parallelism` is smaller than `batch_size`, we use gradient accumulation to accumulate gradients from multiple
batches before performing a parameter update. This is useful when you have a large batch size but do not have enough memory to fit the
entire batch.

### Change WandB Configuration
Levanter supports extensive logging, metrics tracking, and artifact saving on WandB. Once you have created a WandB account and finished setting up
your WandB API key, all of your training runs will be automatically logged to WandB, without any additional configuration in your training commands.

Suppose you want to set more control on your WandB logging, you can use the following command:

```
python levanter/examples/gpt2_example.py \
    --config_path config/gpt2_small.yaml \
    --trainer.wandb.project my_project \
    --trainer.wandb.name my_run \
    --trainer.wandb.id asdf1234 \
    --trainer.wandb,group my_new_exp_group
```

This will overwrite the default WandB configuration from the `TrainerConfig` in [config.py](src/levanter/config.py).
We pass all these arguments to the `wandb.init()` function at the same verbatim.
For more information on the WandB configuration, please refer to the [WandB documentation](https://docs.wandb.ai/ref/python/init).

### Resume Training Runs
When you resume a training run, you may like to restart from a previously saved checking and resume the corresponding WandB run, as well.
To do so, you can use the following command:

```
python levanter/examples/gpt2_example.py \
    --config_path config/gpt2_small.yaml \
    --trainer.load_checkpoint_path checkpoints/gpt2/wandb_run_name  \
    --trainer.wandb.resume True \
    --trainer.wandb.id asdf1234
```

There are two things to note here:
1. Resume from a checkpoint: as we stated in a section above, the `--trainer.load_checkpoint_path` argument is optional. If it is specified,
the trainer will recursively search for the latest checkpoint in the directory and load it. If you do not specify it, the trainer will
start training from scratch.
2. Resume from a WandB run: if you set `--trainer.wandb.resume`, it will resume the corresponding WandB run with the ID `asdf1234`. You can
find the WandB run ID in the URL of your WandB run page. For more information, please refer to the
[WandB documentation](https://docs.wandb.ai/guides/runs/resuming).
