# Training

This document provides a guide on how to launch model training and configure it according to your specific needs.

## Launch Model Training

To launch the training of a GPT2 model, run the following command:
```bash
python levanter/examples/gpt2_example.py --config_path config/gpt2_small.yaml
```

This will execute the training pipeline pre-defined in the [gp2_example.py](../examples/gpt2_example.py) and set model and training configuration defined in [gpt2_small.yaml](../config/gpt2_small.yaml). You can find more template configurations in the [config](../config/) directory.

## Set Custom Training Configuration
In machine learning experiments, it is common to adjust model hyperparameters. In this section, we will provide examples of different use cases and explain the corresponding parameters that you can change.

### Change Model Parameters
Assuming that I want to set different dimensions for my GPT2 model and increase the number of training steps to 10,000, I can use the following command:

```
python levanter/examples/gpt2_example.py \
    --config_path config/gpt2_small.yaml \
    --model.num_heads 20 \
    --model.num_layers 36 \
    --model.hidden_dim 1280 \
    --trainer.num_train_steps 10000
```


## Training Configurations

You can find an example training configuration [here](../config/gpt2_small.yaml)


### Data
Dataset configurations are managed by the class `LMDatasetConfig` in [src/levanter/data/text.py](src/levanter/data/text.py).

### Model
GPT2's configurations are managed by the class `GPT2Config` in [src/levanter/models/gpt2.py](src/levanter/models/gpt2.py).
- `seq_len`: The sequence length of the model. This is the maximum number of tokens that can be processed in a single forward pass.
- `hidden_dim`: The hidden dimension of the model. This is the dimension of the hidden states of the transformer encoder and decoder. The hidden dimension must be divisible by the number of heads.
- `num_heads`: The number of heads in the multi-head attention layers of the transformer encoder and decoder.
- `num_layers`: The number of layers in the transformer encoder and decoder.
- `gradient_checkpointing`: Whether to use gradient checkpointing. This is a memory optimization that trades compute for memory. It is recommended to use this for large models and/or large batch sizes.
- `scale_attn_by_inverse_layer_idx`: Whether to scale the attention weights by the inverse of the layer index. This is a memory optimization that trades compute for memory. It is recommended to use this for large models and/or large batch sizes.

### Trainer
