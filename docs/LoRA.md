# LoRA Tutorial: GSM8K on Llama-2 7b

In the [Fine-Tuning tutorial](./Fine-Tuning.md), we reproduced Alpaca using Levanter and Llama 1 or Llama 2.
The Alpaca methodology is a good way to make a pretty good general-purpose instruction-tuned model, but what
if we want to make a model that's good at a specific task?

In this tutorial, we're going to do that: we're going to use Levanter's implementation of [LoRA](https://arxiv.org/abs/2106.09685) to adapt Llama 2 to
[GSM8K](https://arxiv.org/abs/2110.14168v2), which is a dataset of grade-school math problems.
To evaluate the model, we'll see [CRFM's HELM](https://crfm.stanford.edu/helm/), comparing it to the baseline Llama 2 model.
The goal with this tutorial isn't to get to SoTA on GSM8K, but rather to show you how to use LoRA with Levanter.

LoRA is a technique for adapting a model to a new task by adding a low-rank linear layer to the model. It's mostly
nice because it's highly memory efficient, and it results in adapters that are only a few megabytes on disk, rather than the
many gigabytes that a fully fine-tuned Llama 2 model would be.

The LoRA model we create will be compatible with [Hugging Face's PEFT](https://github.com/huggingface/peft) library,
so that you can use it with their inference scripts or anywhere else you might want to use a PEFT model.
Levanter also supports saving "merged" checkpoints, which are compatible with Hugging Face's inference scripts,
as well as any library that doesn't support PEFT checkpoints.

## The Data

Let's talk a little bit about the data. GSM8K is a dataset of grade-school math problems, with the goal of
evaluating models' ability to do arithmetic. Only recently (Claude, GPT-4, Gemini) have LLMs been able to perform
well on this dataset. Llama 2 7B by itself only gets about 13% accuracy on the dataset.

Here's an example problem from the dataset (You can see the whole dataset on the
[Hugging Face Datasets page](https://huggingface.co/datasets/gsm8k/viewer/main/train?row=0))

```
Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
A: Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72
```

You are only actually evaluated on the answer, not the explanation,
which comes after the `####` in the answer.


## LORA-izing the Model

### 0. Quick LoRA Overview

There are now many tutorials on what LoRA is and how it works, so we won't go into too much detail here.

This is a diagram taken from the LoRA paper, which shows the basic idea:

![Lora schematic diagram. The text gives a detailed description.](figures/lora-diagram.png)

The idea is that we add an extra low-rank linear layer (or rank $r$) to each linear layer in the model, and
the output of the new layer is the sum of the outputs of the two layers. The low-rank layer
can be represented as a pair of matrices $A \in \mathbb{R}^{d \times r}$ and $B \in \mathbb{R}^{r \times d}$.

Because the low-rank layer is small, it's much more space efficient than "full finetuning" like we did for Alpaca.
For example, in this tutorial with default settings, the LoRA adapter will have about 20M parameters, compared to the
more than 6.7B parameters in the base model for a reduction of about 99.7% in parameters.


### 1. Apply the LoRA transform to the model

Thus, to "LoRA-ize" a model, we need to do some surgery on the model's linear layers. This is what Levanter's
`loraize` function does. It takes a model and a `LoraConfig` and returns a new model with the LoRA transform
 The `LoraConfig` is a dataclass with the following fields:
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
class TrainArgs:
    lora: LoraConfig = LoraConfig()

    # ... some other stuff

    # should we save merged (i.e. not peft) checkpoints?
    merged_hf_save_path: Optional[str] = None  # path to save merged hf checkpoints
    merged_hf_upload: Optional[str] = None
...

def train(config: TrainArgs):
    ...
    with config.trainer.device_mesh:
        ...

        @hax.named_jit(axis_resources=parameter_axis_mapping, donate_args=(True))
        def loraize_hf_model(model):
            return loraize(model, config.lora, key=lora_key)

        model = loraize_hf_model(model)


```

### 2. Tell the trainer to only train the LoRA params

`Trainer` takes an optional `is_trainable` argument, which is an [Equinox `filter_spec`](https://docs.kidger.site/equinox/examples/frozen_layer/).
You don't need to worry about the internals, but the gist is that it's a "tree of functions" that has the same
shape as the model's tree, except that instead of arrays there are boolean values for whether or not to train that part
of the model. For LoRA, we want to train the LoRA parameters, but not the base model parameters. We can do this by
using the `lora_trainable_params_filter` function, which takes a model and returns an `is_trainable` `filter_spec`.

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

### 4. Serialize a merged checkpoint

HELM doesn't (yet) support PEFT checkpoints, so we also need to save a merged checkpoint, which is the full model
with the LoRA parameters merged back in. This gives up a lot of the space efficiency benefits of PEFT, but it's
just for evaluation. This is also useful if you want to use the model with Hugging Face's inference scripts without PEFT,
or want to use another inference server that doesn't support PEFT.

```python

# Save merged HF checkpoints if requested
if config.merged_hf_save_path is not None:
    full_save_path = os.path.join(config.merged_hf_save_path, trainer.id)
    trainer.add_hook(
        save_merged_hf_checkpoint_callback(full_save_path, converter, config.merged_hf_upload),
        every=config.hf_save_steps,
    )
```


## The Configuration File

Here's the complete configuration file for the model.

```yaml
model_name_or_path: "meta-llama/Llama-2-7b-hf"
data: gsm8k
trainer:
  mp: p=f32,c=bfloat16
  wandb:
    project: "levanter-gsm8k"
    tags: ["gsm8k", "lora", "llama2"]
  num_train_steps: 550  # 64 * 550 = 35200, which is a bit more than 4 epochs
  train_batch_size: 64

  # if using model parallelism, this is useful:
  tensor_parallel_axes: ["mlp", "heads"]
optimizer:
  # values in qlora paper
  learning_rate: 2e-4
  weight_decay: 0.0
  lr_schedule: "constant"
lora:
  # These are the defaults, but just so you can see them
  r: 8  # rank of LoRA transform
  alpha: 8.0  # scaling factor for LoRA transform
  dropout: 0.0  # dropout probability for LoRA layers
```


The default config is available at [`gsm8k-llama2.yaml`](https://github.com/stanford-crfm/levanter/blob/main/examples/gsm8k-lora/gsm8k-llama2.yaml).

## Running Training


Here's how to run the command on a TPUv3-32, but you can use any TPU or GPU you want. We'll save a merged
checkpoint to the Hugging Face Hub, so that we can use it with HELM. As we typically do, we leave all
the paths specified on the command line, so that you can easily change them. They can be either regular
paths or GCS paths.

```bash
infra/babysit-tpu-vm.sh llama-32 -z us-east1-d -t v3-32 --preemptible -- \
WANDB_API_KEY=${YOUR TOKEN HERE} \
HUGGING_FACE_HUB_TOKEN=${YOUR TOKEN HERE} \
bash levanter/infra/run.sh python \
levanter/examples/gsm8k-lora/gsm8k_lora.py \
--config_path levanter/examples/gsm8k-lora/gsm8k-llama2.yaml \
--trainer.checkpointer.base_path gs://<somewhere>/ckpts \
--data_cache_dir gs://<somewhere>/cache \
--hf_save_path gs://<somewhere>/lora \
--merged_hf_save_path gs://<somewhere>/merged \
--merged_hf_upload <somewhere>/gsm8k-llama2-merged \
```

You'll want to replace the placeholders with your own paths. You'll also want to replace the `--merged_hf_upload`
with your own Hugging Face Hub username and repo name. (You can also just remove it if you don't want to upload
the merged checkpoint.)

## Evaluating the Model with HELM

As we said before, we're going to use HELM to evaluate the model. HELM makes it easy to evaluate models on the
(many) benchmarks that the HELM team has added to the system. It also makes it easy to compare models to each other,
and to see the individual results for error analysis.

### 1. Install HELM

We recommend installing HELM in a separate virtual environment or conda environment, to minimize the chance of
dependency conflicts.

```bash
virtualenv -p python3.8 helm
source helm/bin/activate
pip install crfm-helm
```

### 2. Evaluate the model

```bash
export MYMODEL="<somewhere>/gsm8k-llama2-merged"
helm-run --run-specs gsm:model=$MYMODEL     --enable-huggingface-models $MYMODEL  --suite v1  --max-eval-instances 1000
```

This will run the evaluation.
If you want to also evaluate the baseline Llama 2 model, you can do that by replacing `$MYMODEL` with
`meta-llama/Llama-2-7b-hf`:

```bash
export HUGGING_FACE_HUB_TOKEN=${YOUR TOKEN HERE}
export MYMODEL="meta-llama/Llama-2-7b-hf"
helm-run --run-specs gsm:model=$MYMODEL     --enable-huggingface-models $MYMODEL  --suite v1  --max-eval-instances 1000
```

### 3. Summarize the results

This command will summarize the results and produce some nice tables.

```bash
helm-summarize --suite v1
```

If you want, you can now look at the LaTeX table:

```bash
cat benchmark_output/runs/v1/groups/latex/reasoning_accuracy.tex
\begin{table*}[htp]
\resizebox{\textwidth}{!}{
\begin{tabular}{lrr}
\toprule
Model/adapter & Mean win rate & GSM8K - EM \\
\midrule
meta-llama/Llama-2-7b-hf &  & 0.133 \\
stanford-crfm/gsm8k-lora-merged & 1.0 & 0.252 \\
\bottomrule
\end{tabular}}
\caption{Results for accuracy (reasoning)}
\label{fig:accuracy (reasoning)}
\end{table*}
```

From the table, we can see that the LoRA model is much better than the baseline Llama 2 model, almost doubling the accuracy! If you compare it to [the
official HELM results](https://crfm.stanford.edu/helm/latest/#/groups/gsm), you'll see that it's a bit better than
Llama 13B, which is pretty good for a first try!

### 4. Look at the individual results with the HELM UI

You can also look at the individual results with the HELM UI. To do this, you'll need to run the HELM server:

```bash
helm-server --suite v1
```

Then, you can go to `http://0.0.0.0:8000/?group=gsm` to see the results. That will show a table of the results like this:

![results on gsm8k. Our lora model gets 0.252 exact match to Llama 2's 0.133](figures/helm-gsm8k-results.png)

If you click through to the individual results, you can see the model's predictions and the reference answers:

![Example question that the LoRA model happens to get right](figures/helm-instance-example.png)

## Using the Model

The model should work out-of-the-box as a Hugging Face PEFT model. First, copy the checkpoint to a local directory:

```bash
gsutil cp -r gs://<somewhere>/lora/<run id>/step-<something> ./gsm8k-lora
```

Then, you can use it like this:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel

peft_model_id = "./gsm8k-lora"

config = PeftConfig.from_pretrained(peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map="auto")
model = PeftModel.from_pretrained(model, peft_model_id, device_map="auto")

question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"

input = f"Q: {question}\nA: "

input_ids = tokenizer(input, return_tensors="pt").input_ids.to(model.device)
output_ids = model.generate(input_ids, do_sample=True, max_length=100, num_beams=5, num_return_sequences=5)

for output_id in output_ids:
    print(tokenizer.decode(output_id, skip_special_tokens=True))
```

(You probably want to use a dedicated inference server, rather than the naive generation built into Hugging Face's
inference scripts, but this is just an example.)

#### Using the merged checkpoints

You can use the merged checkpoints with Hugging Face's inference scripts by doing something like this:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(path)
```
