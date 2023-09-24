# levanter version of https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py

# Differences:
# - We use the huggingface dataset version of alpaca rather than checking it in
# - Levanter doesn't do epochs, just steps.
# - We use Levanter's distributed preprocessing, which is a bit overkill for this dataset but is a good example.
#   (The original's preprocessing is very slow, which is usually fine, but not good for preemptible nodes.)
# - We use the fast tokenizers. I don't know why the original code doesn't use them.
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

import logging
import os
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import jax.random as jrandom
import transformers
from transformers import PreTrainedTokenizerBase

import haliax as hax

import levanter
from levanter.compat.hf_checkpoints import HFCheckpointConverter, save_hf_checkpoint_callback
from levanter.data import Dataset
from levanter.data.shard_cache import BatchProcessor
from levanter.data.shard_source import HFDatasetDataSource, JsonDataSource
from levanter.data.text import BatchEncodingDataset
from levanter.models.attention import CausalMask
from levanter.models.lm_model import LmExample, LmHeadModel
from levanter.trainer import OptimizerConfig, Trainer, TrainerConfig
from levanter.utils import fsspec_utils
from levanter.utils.hf_utils import num_cpus_used_by_tokenizer
from levanter.utils.py_utils import non_caching_cycle


logger = logging.getLogger(__name__)

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
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

    data: str = "tatsu-lab/alpaca"  # Path to the training data, or huggingface dataset name.
    data_cache_dir: str = "cache/"  # Path to cache the data.

    model_name_or_path: str = "meta-llama/Llama-2-7b-hf"
    trust_remote_code: bool = False  # Trust remote code when loading from HuggingFace checkpoints.

    cache_dir: Optional[str] = None

    hf_save_path: Optional[str] = None  # Path to save the HuggingFace checkpoint.
    hf_upload: Union[bool, str] = False  # Name of the HuggingFace repo to upload to (if any).
    hf_save_steps: int = 1000  # How often to save the HuggingFace checkpoint.


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
        return num_cpus_used_by_tokenizer(self.tokenizer)


class SupervisedDataset(Dataset[LmExample]):
    def __init__(self, Pos: hax.Axis, KeyPos: hax.Axis, data: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        self.Pos = Pos
        self.KeyPos = KeyPos
        self.pad_token_id = tokenizer.pad_token_id

        logging.warning("Preprocesing data...")
        source = _get_data_source(data)
        cache = levanter.data.build_cache(
            cache_dir="cache/",
            input_shards=source,
            processor=EncoderDecoderProcessor(tokenizer),
        )

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


def _get_data_source(path_or_id):
    if fsspec_utils.exists(path_or_id):
        return JsonDataSource([path_or_id])
    else:
        return HFDatasetDataSource(path_or_id, split="train")


def train(config: TrainArgs):
    config.trainer.initialize(config)

    # DIFFERENCE: We have to import models from HF's checkpoint
    converter = HFCheckpointConverter.from_hf(config.model_name_or_path, trust_remote_code=config.trust_remote_code)
    model_config = converter.default_config

    training_key = jrandom.PRNGKey(config.trainer.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        cache_dir=config.cache_dir,
        model_max_length=model_config.Pos.size,
        padding_side="right",
        # DIFFERENCE: we use the fast tokenizer
        # TODO: why was this necessary?
        # use_fast=False,
    )

    # DIFFERENCE: we are a bit more explicit about the optimizer
    optimizer = config.optimizer.build(config.trainer.num_train_steps)

    # DIFFERENCE: The loss function needs to be manually defined
    def compute_loss(model: LmHeadModel, example: LmExample, key=None):
        return model.compute_loss(example, key=key).scalar()

    trainer = Trainer(config.trainer, optimizer, compute_loss)

    # DIFFERENCE: we set context managers that tell JAX/Haliax which devices to use and how to shard
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

        # DIFFERENCE: we don't need a collator but we do need to specify the data loader
        train_dataset = SupervisedDataset(model.Pos, model.KeyPos, config.data, tokenizer)
        loader = trainer.replicated_loader(train_dataset, trainer.TrainBatch)
        # loop the loader as long as we want:
        loader = non_caching_cycle(loader)

        trainer.add_default_hooks()
        state = trainer.initial_state(training_key, model=model)

        if state.step != 0:
            logger.info(f"Resuming training from step {state.step}")
            for i in range(state.step):
                next(loader)  # type: ignore

        # DIFFERENCE: we save HF checkpoints periodically
        if config.hf_save_path is not None:
            full_save_path = os.path.join(config.hf_save_path, trainer.config.run_id)

            trainer.add_hook(
                save_hf_checkpoint_callback(full_save_path, converter, upload_to_hf=config.hf_upload),
                every=config.hf_save_steps,
            )

        trainer.train(state, loader)


if __name__ == "__main__":
    levanter.config.main(train)()
