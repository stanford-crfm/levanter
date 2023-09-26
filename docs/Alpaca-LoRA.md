In the [Replicating Alpaca](./Replicating-Alpaca.md) tutorial, we reproduced Alpaca using Levanter and Llama 1 or Llama 2. 

In this guide, we'll use [LoRA](https://arxiv.org/abs/2106.09685) to do a lighter-weight
version of Alpaca, similar to [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora). We'll borrow heavily from 
the our "vanilla" Alpaca script, and only change the parts that are necessary to use LoRA.

The LoRA model we create will be compatible with [Hugging Face's PEFT](https://github.com/huggingface/peft) library,
so that you can use it with their inference scripts or anywhere else you might want to use a PEFT model.

## Changes to the Alpaca script

There are three things we need to do differently:
1. Apply the lora transform to the model.
2. Tell the trainer to only train the lora params.
3. Serialize a PEFT-compatible checkpoint.

### 1. Apply the lora transform to the model

We'll use the same model as in the Alpaca tutorial, but we'll apply the lora transform to it. Levanter's implementation
of LoRA is in `levanter.lora.loraize`. It takes a model and a `LoraConfig` and returns a new model with the LoRA transform
applied. The `LoraConfig` is a dataclass with the following fields:
```python
@dataclass(frozen=True)
class LoraConfig:
    target_modules: Optional[Union[List[str], str]] = None
    """modules to loraize. can either be a regex or a list of strings of module names, or None, meaning all linear modules"""
    r: int = 8  # rank of LoRA transform
    alpha: float = 8.0  # scaling factor for LoRA transform
    dropout: float = 0.0  # dropout probability for LoRA layers
```

By default, we loraize all linear modules in the model, which we recommend. This was found to be better than the other
options: https://twitter.com/Tim_Dettmers/status/1689375417189412864, https://arxiv.org/pdf/2305.14314.pdf Section 4.

In our modifications below, we apply `loraize` inside of a `haliax.named_jit` function. This ensures that the 
parameters are sharded correctly.

```python
@dataclass
class TrainArgs(alpaca.TrainArgs):
    lora: LoraConfig = LoraConfig()
    
...

def train(config: TrainArgs):
    ...
    with config.trainer.device_mesh:
        # Major difference from Alpaca: we loraize the model.

        @hax.named_jit(axis_resources=parameter_axis_mapping, donate_args=(True))
        def loraize_hf_model(model):
            return loraize(model, config.lora, key=lora_key)

        model = loraize_hf_model(model)
        
        # we only want to train on the lora params. The way to do this in Equinox is generally with
        # a filter tree (cf https://docs.kidger.site/equinox/examples/frozen_layer/),
        # which is a tree with the same structure (or a "tree prefix" thereof) as the model, but with
        # bools or Callable[..., bool] at the leaves. We can then pass this tree to the trainer and it
        # will only train the parameters that are True in the tree.
        # Levanter defines `is_lora_param` for this purpose, but we need to be careful about how we use it.
        # Equinox's primitives don't really have a "match all tree nodes matching a predicate" function (just
        # a "match all tree leaves matching a predicate" function), so we need to be just a bit careful.
        # Basically, we want to halt recursion in the tree whenever we hit a node that is a lora param.

        # Functionally, this filter is the same as the model, except every lora param is replaced with True
        # and every other leaf (really, every array) is replaced with False
        lora_param_filter = jax.tree_util.tree_map(is_lora_param, model, is_leaf=is_lora_param)

        def compute_loss(model: LmHeadModel, example: LmExample, key=None):
            return model.compute_loss(example, key=key).scalar()

        trainer = Trainer(config.trainer, optimizer, compute_loss, is_trainable=lora_param_filter)
```

