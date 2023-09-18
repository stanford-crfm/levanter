# levanter version of https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py

# We attempt to stick fairly close the original code, but there are some differences:
# - We use the huggingface dataset version of alpaca rather than checking it in
# - Levanter doesn't do epochs, just steps.
# - We produce Levanter's LmExample class instead of a dict, and loss masks are used instead of the -100 sentinel value.
# - We use the fast tokenizers. I don't know why the original code doesn't use them.

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

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import jax.random as jrandom
import transformers

import haliax as hax

import levanter
from levanter.compat.hf_checkpoints import HFCheckpointConverter, save_hf_checkpoint_callback
from levanter.data import Dataset
from levanter.models.attention import CausalMask
from levanter.models.lm_model import LmExample, LmHeadModel
from levanter.trainer import OptimizerConfig, Trainer, TrainerConfig
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

    model_name_or_path: str = "meta-llama/Llama-2-7b-hf"
    trust_remote_code: bool = False  # Trust remote code when loading from HuggingFace checkpoints.

    cache_dir: Optional[str] = None

    hf_save_path: Optional[str] = None  # Path to save the HuggingFace checkpoint.
    hf_upload: Optional[str] = None  # Name of the HuggingFace repo to upload to (if any).
    hf_save_steps: int = 1000  # How often to save the HuggingFace checkpoint.


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        (tokenized.input_ids != tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    Pos: hax.Axis,
    KeyPos: hax.Axis,
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> List[LmExample]:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]

    out_examples = []
    for input_ids, source_len in zip(examples_tokenized["input_ids"], sources_tokenized["input_ids_lens"]):
        input_ids = hax.named(input_ids, Pos)

        targets = hax.roll(input_ids, -1, Pos)

        loss_mask = hax.arange(Pos) < source_len
        loss_mask = loss_mask & (targets != tokenizer.pad_token_id)
        # TODO: do we want to use prefixlm?
        attn_mask = CausalMask(Pos, KeyPos)

        out_examples.append(LmExample(input_ids, targets, attn_mask, loss_mask))

    return out_examples


def _load_data(path: str) -> List[Dict[str, str]]:
    """Load the data from a json file, or, if it's a huggingface dataset, open it that way."""
    if os.path.exists(path):
        return json.load(open(path, "r"))
    else:
        import datasets

        return list(datasets.load_dataset(path, split="train"))


class SupervisedDataset(Dataset[LmExample]):
    """Dataset for supervised fine-tuning."""

    def __init__(self, Pos: hax.Axis, KeyPos: hax.Axis, data: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = _load_data(data)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        self.examples = preprocess(Pos, KeyPos, sources, targets, tokenizer)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> LmExample:
        return self.examples[i]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


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
        parameter_axis_mapping = config.trainer.parameter_axis_mapping

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

            upload_to_hf: bool | str
            if config.hf_upload:
                upload_to_hf = config.hf_upload
            else:
                upload_to_hf = False

            trainer.add_hook(
                save_hf_checkpoint_callback(full_save_path, converter, upload_to_hf=upload_to_hf),
                every=config.hf_save_steps,
            )

        trainer.train(state, loader)


if __name__ == "__main__":
    levanter.config.main(train)()
