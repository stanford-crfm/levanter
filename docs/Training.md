# Training

This document provides a guide on how to launch model training and configure it according to your specific needs.

## Launch Model Training

To launch the training of a GPT2 model, run the following command:
```bash
python levanter/examples/gpt2_example.py --config_path config/gpt2_small.yaml
```

This will execute the training pipeline pre-defined in the [gp2_example.py](../examples/gpt2_example.py) and set model and training configuration 
set in [gpt2_small.yaml](../config/gpt2_small.yaml). You can find more template configurations in the [config](../config/) directory.

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
run. If you do not specify it, the trainer will start training from scratch.
- Both `--trainer.load_checkpoint_path` and `--trainer.checkpointer.base_path` supports local path and cloud storage path (e.g. S3, GCS, etc.), as
long as the path is accessible from the machine that you are running the training script on.
- The `--trainer.checkpointer.save_interval` argument supports the following units: `s` (seconds), `m` (minutes), `h` (hours), `d` (days).

### Change Evaluation Parameters
To change how often the model is evaluated during training, you can use the following command:

```
python levanter/examples/gpt2_example.py \
    --config_path config/gpt2_small.yaml \
    --trainer.steps_per_eval 500
```

This will overwrite the default eval frequency (every 1,000) from the `TrainerConfig` in [config.py](src/levanter/config.py) to every 500 steps.

