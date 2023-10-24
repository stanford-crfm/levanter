# levanter version of https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional, Union

import fsspec
import jax.random as jrandom
import numpy as np
import transformers

import haliax as hax

import levanter
from levanter.compat.hf_checkpoints import HFCheckpointConverter, save_hf_checkpoint_callback
from levanter.data import Dataset
from levanter.data.sharded_dataset import JsonDataset, WrappedHFDataset
from levanter.models.lm_model import LmExample, LmHeadModel
from levanter.trainer import OptimizerConfig, Trainer, TrainerConfig
from levanter.utils import fsspec_utils
from levanter.utils.hf_utils import num_cpus_used_by_tokenizer
from levanter.utils.py_utils import non_caching_cycle


# Differences:
# - We use the huggingface dataset version of alpaca rather than checking it in
# - Levanter doesn't (currently) do epochs, just steps.
# - We use Levanter's distributed preprocessing, which is a bit overkill for this dataset but is a good example.
#   (The original's preprocessing is very slow, which is usually fine, but not good for preemptible nodes.)
# - We use fast tokenizers. I don't know why the original code doesn't use them.
# - We produce Levanter's LmExample class instead of a dict, and loss masks are used instead of the -100 sentinel value.

# Ways this script could be improved:
# * Could tune hparams more for throughput

#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


# TODO
# * make a pad_stack function that can pad and stack arrays in one go
# * make batch loader support pad_stack


logger = logging.getLogger(__name__)

# copy paste from alpaca

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


@dataclass
class TrainArgs:
    optimizer: OptimizerConfig
    trainer: TrainerConfig

    max_tune_length: int = 2048  # maximum length of the input to the model during tuning
    data: str = "tatsu-lab/alpaca"  # Path to the training data, or huggingface dataset name.
    data_cache_dir: str = "cache/"  # Path to cache the tokenized data. can be gcs
    prompts: Optional[str] = None  # Path to the prompts file. can be gcs

    model_name_or_path: str = "meta-llama/Llama-2-7b-hf"
    trust_remote_code: bool = False  # Trust remote code when loading from HuggingFace checkpoints.

    model_cache_dir: Optional[str] = None  # Path to cache the model. must be local.

    hf_save_path: Optional[str] = None  # Path to save the HuggingFace checkpoint.
    hf_upload: Union[bool, str] = False  # Name of the HuggingFace repo to upload to (if any).
    hf_save_steps: int = 1000  # How often to save the HuggingFace checkpoint.


# Encoder/Decoder dataset for Alpaca.
# We basically do string interpolation of the (instruction, input, output) triples with the prompt,
# and mask out the prompt and padding.
class SupervisedDataset(Dataset[LmExample]):
    def __init__(self, preproc_dataset, tokenizer):
        self.preproc_dataset = preproc_dataset
        self.tokenizer = tokenizer

    def __iter__(self):
        for ex in self.preproc_dataset:
            # annoyingly, pad expects things to be batched so we have to prepend a batch axis
            ex = self.tokenizer.pad(
                {k: np.expand_dims(v, 0) for k, v in ex.items()}, return_tensors="np", padding="max_length"
            )
            ex = {k: v[0] for k, v in ex.items()}
            input_ids = hax.named(ex["input_ids"], "position")

            # mask out padding and anything before the start of the target
            Pos = input_ids.resolve_axis("position")
            loss_mask = hax.arange(Pos) >= ex["source_lens"]

            # don't predict the padding
            targets = hax.roll(input_ids, -1, Pos)
            loss_mask = loss_mask & (targets != self.tokenizer.pad_token_id)

            yield LmExample(input_ids, loss_mask)


def _get_data_source(path_or_id):
    """The original alpaca.py used a json file, but it's since been moved to the HF dataset hub. You can use any
    dataset that's compatible with the structure of the alpaca dataset."""
    if fsspec_utils.exists(path_or_id):
        return JsonDataset([path_or_id])
    else:
        return WrappedHFDataset(path_or_id, split="train")


def mk_dataset(config: TrainArgs, tokenizer: transformers.PreTrainedTokenizerBase):
    dataset = _get_data_source(config.data)

    prompt_input, prompt_no_input = get_prompts(config.prompts)

    def preprocess(batch):
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in batch
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in batch]
        # TODO: this seems pretty wasteful since you end up tokenizing twice, but it's how the original code does it.
        examples = [s + t for s, t in zip(sources, targets)]
        sources_tokenized = tokenizer(sources, return_tensors="np", padding=False, truncation=True)
        examples_tokenized = tokenizer(examples, return_tensors="np", padding=False, truncation=True)

        source_lens = [len(s) for s in sources_tokenized["input_ids"]]

        return {
            "input_ids": examples_tokenized["input_ids"],
            "source_lens": source_lens,
        }

    dataset = dataset.map_batches(preprocess, batch_size=128, num_cpus=num_cpus_used_by_tokenizer(tokenizer))
    dataset = dataset.build_cache(config.data_cache_dir, await_finished=True)

    dataset = SupervisedDataset(dataset, tokenizer)

    return dataset


def get_prompts(prompt_path):
    prompts = DEFAULT_PROMPT_DICT
    if prompt_path is not None:
        with fsspec.open(prompt_path) as f:
            prompts = json.load(f)
    return (prompts["prompt_input"]), (prompts["prompt_no_input"])


def train(config: TrainArgs):
    config.trainer.initialize(config)

    # Since Levanter has different implementations of models from HF, we need to convert the HF checkpoint.
    # This class is a wrapper around the HF checkpoint converter that also downloads the checkpoint if necessary.
    converter = HFCheckpointConverter.from_hf(config.model_name_or_path, trust_remote_code=config.trust_remote_code)
    model_config = converter.default_config

    if config.max_tune_length > model_config.Pos.size:
        logger.warning(
            f"max_tune_length ({config.max_tune_length}) is greater than the model's maximum length"
            f" ({model_config.Pos.size}). "
        )

    # Randomness in JAX is tightly controlled. We pass around a key that is used to generate random numbers.
    training_key = jrandom.PRNGKey(config.trainer.seed)

    # This is largely the same as in Alpaca. Only change is we use the fast tokenizer.
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        cache_dir=config.model_cache_dir,
        model_max_length=config.max_tune_length,
        padding_side="right",
    )
    num_new_tokens = add_special_tokens(tokenizer)
    logger.info(f"Added {num_new_tokens} new tokens")

    # modify converter to use our tokenizer, mostly so it saves the right vocab
    converter = converter.replaced(tokenizer=tokenizer)

    train_dataset = mk_dataset(config, tokenizer)

    optimizer = config.optimizer.build(config.trainer.num_train_steps)

    def compute_loss(model: LmHeadModel, example: LmExample, key=None):
        return model.compute_loss(example, key=key).scalar()

    trainer = Trainer(config.trainer, optimizer, compute_loss)

    with trainer.device_mesh:
        # how we shard parameters across devices
        parameter_axis_mapping = trainer.parameter_axis_mapping

        # load the underlying hf model
        logger.info(f"Loading pretrained model from {converter.reference_checkpoint}")
        model: LmHeadModel = converter.load_pretrained(model_config, axis_mapping=parameter_axis_mapping)

        # this must be in jit b/c it uses arrays across accelerators (b/c of FSDP)
        model = hax.named_jit(lambda m: m.resize_vocab(len(tokenizer)))(model)

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
            full_save_path = os.path.join(config.hf_save_path, trainer.run_id)

            trainer.add_hook(
                save_hf_checkpoint_callback(full_save_path, converter, upload_to_hf=config.hf_upload),
                every=config.hf_save_steps,
            )

        trainer.train(state, loader)


def add_special_tokens(tokenizer, use_unk_instead_of_adding=False):
    special_tokens_dict = dict()
    if use_unk_instead_of_adding:
        if tokenizer.unk_token is None:
            raise ValueError("use_unk_instead_of_add is True but tokenizer doesn't have an unk token")

    unk = tokenizer.unk_token if use_unk_instead_of_adding else None

    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN if not use_unk_instead_of_adding else unk
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN if not use_unk_instead_of_adding else unk
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN if not use_unk_instead_of_adding else unk
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    return tokenizer.add_special_tokens(special_tokens_dict)


if __name__ == "__main__":
    levanter.config.main(train)()
