# LoRA Tutorial: Alpaca-LoRA

In the [Fine-Tuning tutorial](./Fine-Tuning.md), we reproduced Alpaca using Levanter and Llama 1 or Llama 2.

In this guide, we'll use Levanter's implementation of [LoRA](https://arxiv.org/abs/2106.09685) to do a lighter-weight
version of Alpaca, similar to [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora). We'll borrow heavily from
our "vanilla" Alpaca script, and only change the parts that are necessary to use LoRA.
The point of this tutorial is to show you how to use LoRA with Levanter, so that you can use it with your
own data.

The LoRA model we create will be compatible with [Hugging Face's PEFT](https://github.com/huggingface/peft) library,
so that you can use it with their inference scripts or anywhere else you might want to use a PEFT model.


## Changes to the Alpaca script

There are four things we need to do differently:
1. Apply the LoRA transform to the model.
2. Tell the trainer to only train the lora params.
3. Serialize a PEFT-compatible checkpoint.
4. (Nitpicky) we shouldn't add tokens to the vocabulary, since we're adapting a model.

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

By default, we LoRA-ize all linear modules in the model, which we recommend. This was found to be better than the other
options: https://twitter.com/Tim_Dettmers/status/1689375417189412864, https://arxiv.org/pdf/2305.14314.pdf Section 4.
(As with all config in Levanter, [these can be changed in the config file or via command line flags](./Configuration-Guide.md).)

In our modifications below, we apply `loraize` inside of a [`haliax.named_jit`](https://haliax.readthedocs.io/en/latest/partitioning/#haliax.named_jit) function. This ensures that the
parameters are sharded correctly.

```python
@dataclass
class TrainArgs(alpaca.TrainArgs):
    lora: LoraConfig = LoraConfig()

    # should we save merged (i.e. not peft) checkpoints?
    merged_hf_save_path: Optional[str] = None  # path to save merged hf checkpoints
    merged_hf_upload: Optional[str] = None
...

def train(config: TrainArgs):
    ...
    with config.trainer.device_mesh:
        ...

        # Major difference from Alpaca: we loraize the model.
        @hax.named_jit(axis_resources=parameter_axis_mapping, donate_args=(True))
        def loraize_hf_model(model):
            return loraize(model, config.lora, key=lora_key)

        model = loraize_hf_model(model)


```

### 2. Tell the trainer to only train the LoRA params

`Trainer` takes an optional `is_trainable` argument, which is a [Equinox filter_spec](https://docs.kidger.site/equinox/examples/frozen_layer/).
You don't need to worry about the internals, but the gist is that it's a "tree of functions" that has the same
shape as the model's tree, except that instead of arrays there are boolean values for whether or not to train that part
of the model.

```python
def train(config: TrainArgs):
    ...
    with config.trainer.device_mesh:
        ...

        lora_param_filter = lora_trainable_params_filter(model)

        def compute_loss(model: LmHeadModel, example: LmExample, key=None):
            return model.compute_loss(example, key=key).scalar()

        trainer = Trainer(config.trainer, optimizer, compute_loss, is_trainable=lora_param_filter)
```

### 3. Serialize a PEFT-compatible checkpoint

Levanter's LoRA module has a function for saving a PEFT-compatible checkpoint, `levanter.lora.save_peft_pretrained`,
which is analogous to PEFT's `model.save_pretrained`.

```python
        # Save HF PEFT checkpoints periodically (and at the end of training), which is just the lora weights
if config.hf_save_path is not None:
    full_save_path = os.path.join(config.hf_save_path, trainer.id)
    trainer.add_hook(
        save_peft_checkpoint_callback(
            full_save_path, config.lora, config.model_name_or_path, config.hf_upload
        ),
        every=config.hf_save_steps,
    )
```

For good measure, we'll also save the merged HF checkpoint, which is the full model with the LoRA parameters
merged back in. This is useful if you want to use the model with Hugging Face's inference scripts without PEFT,
or want to use [llama.cpp](https://github.com/ggerganov/llama.cpp) or something.

```python

# Save merged HF checkpoints if requested
if config.merged_hf_save_path is not None:
    full_save_path = os.path.join(config.merged_hf_save_path, trainer.id)
    trainer.add_hook(
        save_merged_hf_checkpoint_callback(full_save_path, converter, config.merged_hf_upload),
        every=config.hf_save_steps,
    )
```


#### Using the checkpoints in Hugging Face's PEFT library

You can use the checkpoints in Hugging Face's PEFT library by doing something like this:

```python
peft_config = PeftConfig.from_pretrained(path)
base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path)
lora_model = PeftModel.from_pretrained(base_model, path)
```

#### Using the merged checkpoints

You can use the merged checkpoints with Hugging Face's inference scripts by doing something like this:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(path)
```


## The Configuration File

The configuration files are almost identical to the non-LoRA versions, with two differences:

1. We should dial up the learning rate. We use 3e-4, which is pretty high relative to what's usually recommended (1e-4),
but it seems to work.
2. For the LLama-2 version, we set `per_device_parallelism` to 4 (The max for a v3-32 with batch size 128 is 4 since 32 * 4 = 128).

We also add WandB tags, but that's not important.

You can change the LoRA parameters by adding a `lora` section, or providing appropriate command line flags:

```yaml
lora:
  r: 16
  dropout: 0.1
```

The default configs are available as [`alpaca-lora.yaml`](https://github.com/stanford-crfm/levanter/blob/main/examples/alpaca-lora/alpaca-lora.yaml)
and [`alpaca-lora-llama2.yaml`](https://github.com/stanford-crfm/levanter/blob/main/examples/alpaca-lora/alpaca-lora-llama2.yaml)


## Running Alpaca-Lora

Running this modified script is basically identical to running the original script.
For example:

```bash
infra/babysit-tpu-vm.sh llama-32 -z us-east1-d -t v3-32 --preemptible -- \
WANDB_API_KEY=${YOUR TOKEN HERE} \
HUGGING_FACE_HUB_TOKEN=${YOUR TOKEN HERE} \
bash levanter/infra/run.sh python \
levanter/examples/alpaca-lora/alpaca_lora.py \
--config_path levanter/examples/alpaca-lora/alpaca-lora-llama2.yaml \
--trainer.checkpointer.base_path gs://<somewhere> \
--hf_save_path gs://<somewhere>
```


## Using the model

The model should work out-of-the-box as a Hugging Face PEFT model. First, copy the checkpoint to a local directory:

```bash
gsutil cp gs://<somewhere>/<run id>/step-<something> ./my-alpaca
```

Then, you can use it like this:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel

peft_model_id = "./my-alpaca"

config = PeftConfig.from_pretrained(peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map="auto")
model = PeftModel.from_pretrained(model, peft_model_id, device_map="auto")

instruction = "Translate the following phrase into French."
input = "I love you."

input = ("Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n {instruction}\n### Input:\n{input}\n### Response:\n")

input_ids = tokenizer(input, return_tensors="pt").input_ids.to(model.device)
output_ids = model.generate(input_ids, do_sample=True, max_length=100, num_beams=5, num_return_sequences=5)

for output_id in output_ids:
    print(tokenizer.decode(output_id, skip_special_tokens=True))
```
