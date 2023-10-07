# Replicating Alpaca

In this tutorial, we will replicate [Stanford Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)
using either Llama 1 or the new [Llama 2](https://ai.meta.com/llama/) model and [Levanter](https://github.com/stanford-crfm/levanter).
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

## The Alpaca script

We have a [Levanter version](https://github.com/stanford-crfm/levanter/blob/main/examples/alpaca.py) of the [original Alpaca script](https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py)

There's a bit of ceremony in both versions, but the broad strokes of the script are the same. The main differences
are highlighted in the Levanter version.

We also need a config file. We provide two versions: [an "original" Alpaca config](https://github.com/stanford-crfm/levanter/blob/main/examples/alpaca.yaml) that uses LLaMA and [Llama 2 config](https://github.com/stanford-crfm/levanter/blob/main/examples/alpaca-llama2.yaml) that uses
Llama 2.

### Original Alpaca Config

```yaml
# cf https://github.com/tatsu-lab/stanford_alpaca#fine-tuning
model_name_or_path: huggyllama/llama-7b
trainer:
  mp: p=f32,c=bfloat16
  wandb:
    project: "levanter-alpaca"
  num_train_steps: 1218  # 128 * 1218 = 155904, which is almost but not quite 3 epochs, which is what alpaca did
  train_batch_size: 128
  per_device_parallelism: 4
  # if using model parallelism, this is useful:
  tensor_parallel_axes: ["mlp", "heads"]
optimizer:
  learning_rate: 2e-5
  weight_decay: 0.0
```

This config uses mixed fp32/bf16 precision and sets the number of training steps to be roughly 3 epochs. It sets up the optimizer
to use a learning rate of 2e-5 and no weight decay. `trainer.per_device_parallelism` is roughly equivalent to HF's
`per_device_train_batch_size`. If you want to use model parallelism, you can set `trainer.model_axis_size` to something
like 2. (This will split the model across two devices. This might be useful if you're using a v3-64 or something similar and
want to maintain the same batch size.)

### Llama 2 Config

The [Llama 2 config](https://github.com/stanford-crfm/levanter/blob/main/examples/alpaca-llama2.yaml) is identical,
except for the HF model name and `per_device_parallelism`. The reason it's different
is that Llama 2's width is 4096 tokens instead, and it pushes us over the line for the number of examples we can fit
on a single TPU.

If you haven't already, go to [Llama 2's Hugging Face page](https://huggingface.co/meta-llama/Llama-2-7b-hf) and request access to the model.

Once you have access, go to [Hugging Face's Tokens page](https://huggingface.co/settings/tokens) to get an API token. You'll need to provide this
to the TPU VM as an environment variable. (We'll show you how to do this later.)


### Changing the config

If you make changes to the config, you'll need to get the config file to all the workers. The best way to do this
is to copy it to Google Cloud Storage so that it persists when the machine is preempted. You can do this with:

```bash
gsutil cp levanter/examples/alpaca.yaml gs://<somewhere>/train-alpaca.yaml
```

If using Llama 2:

```bash
gsutil cp levanter/examples/alpaca-llama2.yaml gs://<somewhere>/train-alpaca.yaml
```

And then using `--config_path gs://<somewhere>/alpaca.yaml` instead of `--config_path levanter/examples/train-alpaca.yaml`
in the command line below.

## Launching the job

Now we can launch the job. We need just a tiny bit of ceremony to get the Hugging Face and WANDB API tokens in the environment:
(If you're using Llama 1, you don't need the `HUGGING_FACE_HUB_TOKEN` line.)

```bash
gcloud compute tpus tpu-vm ssh llama-32 --zone us-east1-d --worker=all \
--command="WANDB_API_KEY=${YOUR TOKEN HERE} \
HUGGING_FACE_HUB_TOKEN=${YOUR TOKEN HERE} \
bash levanter/infra/run.sh python \
levanter/examples/alpaca.py \
--config_path levanter/examples/alpaca.yaml \
--trainer.checkpointer.base_path gs://<somewhere> \
--hf_save_path gs://<somewhere> \
--trainer.id <some id>"  # optional, but useful if using preemption
```

If you're using preemptible or TRC TPUs, you'll want to add `--trainer.id <some id>` to the command line,
and probably use the [babysitting script](./Getting-Started-TPU-VM.md#babysitting-script) to automatically restart the
vm and job if it gets preempted. That would look like this:

```bash
infra/babysit-tpu-vm.sh llama-32 -z us-east1-d -t v3-32 --preemptible -- \
WANDB_API_KEY=${YOUR TOKEN HERE} \
HUGGING_FACE_HUB_TOKEN=${YOUR TOKEN HERE} \
bash levanter/infra/run.sh python \
levanter/examples/alpaca.py \
--config_path levanter/examples/alpaca-llama2.yaml \
--trainer.checkpointer.base_path gs://<somewhere> \
--hf_save_path gs://<somewhere> \
--trainer.id <some id>  # optional, but useful if using preemption
```


## Waiting

At some point it will spit out a Wandb link. You can click on that to see the training progress. There's
not a ton to see here (yet), but you can see the training loss go down over time.

Llama 1 should take about ~3.5 hours on a v3-32 (which is more or less in line with A100 times). Unfortunately, LLama 2
is much slower because of the much longer max sequence length of 4096 and the resulting requirement to do gradient
accumulation to fit on the TPU. It should take about ~9 hours on a v3-32.

## Code Walkthrough

While you're waiting, we'll step through the code in the Alpaca script, highlighting the differences between the
original and the Levanter version. We'll skip over boring bits like imports and the like.

### The Config

We use [Draccus](https://github.com/dlwh/draccus), which is yet another yaml-to-dataclass library. It should
mostly not surprise you.

```python
@dataclass
class TrainArgs:
    optimizer: OptimizerConfig
    trainer: TrainerConfig

    data: str = "tatsu-lab/alpaca"  # Path to the training data, or huggingface dataset name.
    data_cache_dir: str = "cache/"  # Path to cache the tokenized data. can be gcs

    model_name_or_path: str = "meta-llama/Llama-2-7b-hf"
    trust_remote_code: bool = False  # Trust remote code when loading from HuggingFace checkpoints.

    model_cache_dir: Optional[str] = None  # Path to cache the model. must be local.

    hf_save_path: Optional[str] = None  # Path to save the HuggingFace checkpoint.
    hf_upload: Union[bool, str] = False  # Name of the HuggingFace repo to upload to (if any).
    hf_save_steps: int = 1000  # How often to save the HuggingFace checkpoint.
```


### The Dataset

Next up is the dataset. Alpaca basically takes input/output pairs from the dataset and interpolates them into a
single sequence, taking care to mask out the prompt and the input from the loss computation.

Levanter's input for language models is the `LmExample`, which is an Equinox module that looks like this:
(Don't let the module thing scare you. Think of it as a frozen dataclass that JAX understands.)

```python
class LmExample(eqx.Module):
    tokens: hax.NamedArray
    targets: hax.NamedArray
    attn_mask: AttnMask
    loss_mask: hax.NamedArray
```

We just need to populate this with the right values. To do so, we'll preprocess the dataset, following the recipe
from the original Alpaca script, but adapted to Levanter's preprocessing.

Levanter has distributed, cached preprocessing. The caching is useful for us because we're going to be running
on preemptible TPUs, which means we won't have to re-preprocess the data if the TPU gets preempted. (Preprocessing
the original Alpaca dataset is pretty slow with the original script. This has the virtue of being a bit faster.)

Here we go. Don't worry about the `EncoderDecoderProcessor` for now. We'll get to that next.

```python
class SupervisedDataset(Dataset[LmExample]):
    def __init__(
        self, cache_dir, Pos: hax.Axis, KeyPos: hax.Axis, data: str, tokenizer: transformers.PreTrainedTokenizer
    ):
        super(SupervisedDataset, self).__init__()
        self.Pos = Pos
        self.KeyPos = KeyPos
        self.pad_token_id = tokenizer.pad_token_id

        # Levanter's preprocessing will automatically cache the preprocessed data. This is a bit overkill for this
        # dataset, but it's a good example of how to use it. It's also useful if you're using preemptible nodes.
        logging.warning(f"Checking for cached preprocessed data in {cache_dir}")
        source = _get_data_source(data)
        cache = levanter.data.build_cache(
            cache_dir=cache_dir,
            input_shards=source,
            processor=EncoderDecoderProcessor(tokenizer),
        )

        # This converts the on-disk cache into a dataset that we can iterate over. It's functionally an iterable
        # over dicts for each example.
        self.batch_encoding_dataset = BatchEncodingDataset(cache)

    def __iter__(self):
        for ex in self.batch_encoding_dataset:
            input_ids = hax.named(ex["input_ids"], self.Pos)
            targets = hax.roll(input_ids, -1, self.Pos)

            # mask out padding and anything before the start of the target
            loss_mask = hax.arange(self.Pos) >= ex["input_ids_lens"]
            loss_mask = loss_mask & (targets != self.pad_token_id)
            attn_mask = CausalMask(self.Pos, self.KeyPos)

            yield LmExample(input_ids, targets, attn_mask, loss_mask)
```

Here's the `EncoderDecoderProcessor`. It basically replicates the [original logic from the alpaca script](https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py#L88-L124).
The important bits are:

* the prompt interpolation logic
* the `input_ids_lens` field, which is used to mask out the loss
* the `num_cpus` field, which tells Ray how many cpus the tokenizer will use. (HF tokenizers are sometimes multithreaded.)

```python
class EncoderDecoderProcessor(BatchProcessor[dict]):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, input_key: str = "input", output_key: str = "output"):
        self.tokenizer = tokenizer
        self.input_key = input_key
        self.output_key = output_key

    def __call__(self, batch: Sequence[dict]) -> dict:
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in batch
        ]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in batch]
        # TODO: this seems pretty wasteful since you end up tokenizing twice, but it's how the original code does it.
        examples = [s + t for s, t in zip(sources, targets)]
        sources_tokenized = self.tokenizer(sources, return_tensors="np", padding="max_length", truncation=True)
        examples_tokenized = self.tokenizer(examples, return_tensors="np", padding="max_length", truncation=True)

        # We want to modify our examples with an extra field for the length of the input.
        # this will turn into a loss mask later.
        input_ids_lens = (sources_tokenized["input_ids"] != self.tokenizer.pad_token_id).sum(axis=-1)

        return {
            "input_ids": examples_tokenized["input_ids"],
            "input_ids_lens": input_ids_lens,
        }

    @property
    def num_cpus(self) -> int:
        # HF tokenizers are (sometimes) multithreaded, so we tell Ray how many cpus the tokenizer will use.
        return num_cpus_used_by_tokenizer(self.tokenizer)
```

### The Main Function

Now there's just a bunch of ceremony to get everything ready.
First we do initialization and setup:

```python
def train(config: TrainArgs):
    config.trainer.initialize(config)

    # Since Levanter has different implementations of models from HF, we need to convert the HF checkpoint.
    # This class is a wrapper around the HF checkpoint converter that also downloads the checkpoint if necessary.
    converter = HFCheckpointConverter.from_hf(config.model_name_or_path, trust_remote_code=config.trust_remote_code)
    model_config = converter.default_config

    # Randomness in JAX is tightly controlled. We pass around a key that is used to generate random numbers.
    training_key = jrandom.PRNGKey(config.trainer.seed)

    # This is largely the same as in Alpaca. Only change is we use the fast tokenizer.
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        cache_dir=config.model_cache_dir,
        model_max_length=model_config.Pos.size,
        padding_side="right",
    )

    optimizer = config.optimizer.build(config.trainer.num_train_steps)

    def compute_loss(model: LmHeadModel, example: LmExample, key=None):
        return model.compute_loss(example, key=key).scalar()

    trainer = Trainer(config.trainer, optimizer, compute_loss)

```

Next, we load the model. This is where the magic happens. We use the `converter` to load the HF checkpoint,
automatically sharding it across our devices for FSDP. We then resize the model's vocabulary to match the tokenizer's.
Levanter already has the "smart resize" logic built in, so we don't need to be careful here.

```python
def train(config: TrainArgs):
    ...

    with trainer.device_mesh:
        # how we shard parameters across devices
        parameter_axis_mapping = trainer.parameter_axis_mapping

        # load the underlying hf model
        logger.info(f"Loading pretrained model from {converter.reference_checkpoint}")
        model: LmHeadModel = converter.load_pretrained(model_config, axis_mapping=parameter_axis_mapping)

        special_tokens_dict = dict()
        if tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

        # this is smart_token_embeddings_resize in the original
        num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
        logger.info(f"Added {num_new_tokens} new tokens")
        # this must be in jit b/c it uses arrays across accelerators (b/c of FSDP)
        model = hax.named_jit(lambda m: m.resize_vocab(len(tokenizer)))(model)
```

Next, we set up the training loop. This is mostly the same as in Alpaca, except we use Levanter's `replicated_loader`
We also save HF-compatible checkpoints periodically (along with Levanter checkpoints), and at the end of training.

```python
def train(config: TrainArgs):
    ...

    with trainer.device_mesh:
        ...

        train_dataset = SupervisedDataset(config.data_cache_dir, model.Pos, model.KeyPos, config.data, tokenizer)
        # Levanter has two kinds of data loaders: sharded and replicated. Replicated is simpler and allows for
        # single pass training. Sharded only loads a subset of the data on each device, and is more efficient for large
        # datasets. We use replicated here since the dataset is small.
        loader = trainer.replicated_loader(train_dataset, trainer.TrainBatch)
        loader = non_caching_cycle(loader)

        trainer.add_default_hooks()
        state = trainer.initial_state(training_key, model=model)

        if state.step != 0:
            logger.info(f"Resuming training from step {state.step}")
            for i in range(state.step):
                next(loader)  # type: ignore

        # We also save HF checkpoints periodically (and at the end of training).
        if config.hf_save_path is not None:
            full_save_path = os.path.join(config.hf_save_path, trainer.id)

            trainer.add_hook(
                save_hf_checkpoint_callback(full_save_path, converter, upload_to_hf=config.hf_upload),
                every=config.hf_save_steps,
            )

        trainer.train(state, loader)
```

That's it! Now you probably just need to wait another 8 hours or so. (Or 3 hours if you're using Llama 1.)
