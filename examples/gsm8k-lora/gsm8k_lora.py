# Modified version of [alpaca.py] to use LoRA instead of full finetuning.
import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Union

import equinox as eqx
import jax
import jax.random as jrandom
import numpy as np
import transformers

import haliax as hax

import levanter
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.data import Dataset
from levanter.data.dataset import ShuffleDataset
from levanter.data.sharded_dataset import WrappedHFDataset
from levanter.lora import (
    LoraConfig,
    lora_trainable_params_filter,
    loraize,
    save_merged_hf_checkpoint_callback,
    save_peft_checkpoint_callback,
)
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel
from levanter.optim import OptimizerConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.hf_utils import num_cpus_used_by_tokenizer
from levanter.utils.jax_utils import parameter_count
from levanter.utils.py_utils import non_caching_cycle


logger = logging.getLogger(__name__)


@dataclass
class TrainArgs:
    optimizer: OptimizerConfig
    trainer: TrainerConfig
    lora: LoraConfig = LoraConfig()
    model: LmConfig = field(default_factory=LlamaConfig)

    max_tune_length: int = 2048  # maximum length of the input to the model during tuning

    data: str = "gsm8k"  # name of the dataset to use
    data_seed: Optional[int] = None
    data_cache_dir: str = "cache/"  # Path to cache the tokenized data. can be gcs

    input_key: str = "question"  # key in the dataset for the input
    output_key: str = "answer"  # key in the dataset for the output
    mask_inputs: bool = True  # if True, mask out the input and prompt for loss calculation

    model_name_or_path: str = "meta-llama/Llama-2-7b-hf"
    trust_remote_code: bool = False  # Trust remote code when loading from HuggingFace checkpoints.

    model_cache_dir: Optional[str] = None  # Path to cache the model. must be local.

    hf_save_path: Optional[str] = None  # Path to save the HuggingFace checkpoint.
    hf_upload: Union[bool, str] = False  # Name of the HuggingFace repo to upload to (if any).
    hf_save_steps: int = 1000  # How often to save the HuggingFace checkpoint.

    # should we save merged (i.e. not peft) checkpoints?
    merged_hf_save_path: Optional[str] = None  # path to save merged hf checkpoints
    merged_hf_upload: Optional[str] = None


class SupervisedDataset(Dataset[LmExample]):
    def __init__(self, preproc_dataset, tokenizer, mask_inputs):
        self.preproc_dataset = preproc_dataset
        self.tokenizer = tokenizer
        self.mask_inputs = mask_inputs

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
            if self.mask_inputs:
                loss_mask = hax.arange(Pos) >= ex["source_lens"]

                # don't predict the padding
                targets = hax.roll(input_ids, -1, Pos)
                loss_mask = loss_mask & (targets != self.tokenizer.pad_token_id)
            else:
                loss_mask = 1 - hax.nn.one_hot(-1, Pos, dtype=jax.numpy.float32)

            yield LmExample.causal(input_ids, loss_mask=loss_mask)


def mk_dataset(config: TrainArgs, tokenizer: transformers.PreTrainedTokenizerBase):
    dataset = WrappedHFDataset("gsm8k", split="train", name="main")

    def preprocess(batch):
        def format_example(ex):
            return f"Q: {ex[config.input_key]}\nA: "

        def format_output(ex):
            # this is what helm does
            answer_text = ex[config.output_key].replace("####", "The answer is").replace("\n", " ") + "."
            return f"{answer_text}\n{tokenizer.eos_token}"

        sources = [format_example(example) for example in batch]
        targets = [format_output(example) for example in batch]

        examples = [s + t for s, t in zip(sources, targets)]
        sources_tokenized = tokenizer(sources, return_tensors="np", padding=False, truncation=True)
        examples_tokenized = tokenizer(examples, return_tensors="np", padding=False, truncation=True)

        source_lens = [len(s) for s in sources_tokenized["input_ids"]]

        return {
            "input_ids": examples_tokenized["input_ids"],
            "source_lens": source_lens,
        }

    dataset = dataset.map_batches(preprocess, batch_size=128, num_cpus=num_cpus_used_by_tokenizer(tokenizer))  # type: ignore
    dataset = dataset.build_or_load_cache(config.data_cache_dir, await_finished=True)  # type: ignore

    dataset = SupervisedDataset(dataset, tokenizer, mask_inputs=config.mask_inputs)  # type: ignore

    return dataset


def train(config: TrainArgs):
    levanter.initialize(config)

    # Since Levanter has different implementations of models from HF, we need to convert the HF checkpoint.
    # This class is a wrapper around the HF checkpoint converter that also downloads the checkpoint if necessary.
    converter = config.model.default_hf_checkpoint_converter
    model_config = converter.default_config

    tokenizer = copy.deepcopy(converter.tokenizer)
    # if we don't have a pad token, just use the first token in the vocab, which is usually <unk>
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(0)

    tokenizer.model_max_length = config.max_tune_length

    # Randomness in JAX is tightly controlled. We pass around a key that is used to generate random numbers.
    training_key, data_key, lora_key = jrandom.split(jrandom.PRNGKey(config.trainer.seed), 3)

    if config.data_seed is not None:
        data_key = jrandom.PRNGKey(config.data_seed)

    train_dataset = mk_dataset(config, tokenizer)
    train_dataset = ShuffleDataset(train_dataset, data_key, buffer_size=1000 * 1000)

    optimizer = config.optimizer.build(config.trainer.num_train_steps)

    with Trainer(config.trainer, optimizer) as trainer:
        # how we shard parameters across devices
        parameter_axis_mapping = config.trainer.parameter_axis_mapping

        # load the underlying hf model
        logger.info(f"Loading pretrained model from {converter.reference_checkpoint}")
        model: LmHeadModel = converter.load_pretrained(  # type: ignore
            model_config, axis_mapping=parameter_axis_mapping, dtype=trainer.mp.compute_dtype
        )

        # Major difference from Alpaca: we loraize the model.

        @hax.named_jit(axis_resources=parameter_axis_mapping, donate_args=(True))
        def loraize_hf_model(model):
            return loraize(model, config.lora, key=lora_key)

        model = loraize_hf_model(model)

        lora_param_filter = lora_trainable_params_filter(model)

        state = trainer.initial_state(training_key, model=model, is_trainable=lora_param_filter)

        # log some info about the model
        all_param_count = parameter_count(state.model)
        just_lora_params = parameter_count(eqx.filter(state.model, lora_param_filter))

        levanter.tracker.log_summary(
            {
                "parameter_count": all_param_count,
                "trainable_parameter_count": just_lora_params,
                "fraction_trainable": just_lora_params * 1.0 / all_param_count,
            }
        )

        logger.info(f"Total parameter count: {all_param_count}")
        logger.info(f"Trainable parameter count: {just_lora_params}")
        logger.info(f"Fraction of parameters that are trainable: {just_lora_params * 1.0 / all_param_count%.3}")

        # Levanter has two kinds of data loaders: sharded and replicated. Replicated is simpler and allows for
        # single pass training. Sharded only loads a subset of the data on each device, and is more efficient for large
        # datasets. We use replicated here since the dataset is small.
        loader = trainer.replicated_loader(train_dataset, trainer.TrainBatch)
        loader = non_caching_cycle(loader)

        if int(state.step) != 0:
            logger.info(f"Resuming training from step {state.step}")
            for i in range(state.step):
                next(loader)  # type: ignore

        # Save HF PEFT checkpoints periodically (and at the end of training), which is just the lora weights
        if config.hf_save_path is not None:
            full_save_path = os.path.join(config.hf_save_path, trainer.run_id)
            trainer.add_hook(
                save_peft_checkpoint_callback(
                    full_save_path, config.lora, config.model_name_or_path, tokenizer, config.hf_upload
                ),
                every=config.hf_save_steps,
            )

        # Save merged HF checkpoints if requested
        if config.merged_hf_save_path is not None:
            full_save_path = os.path.join(config.merged_hf_save_path, trainer.run_id)
            trainer.add_hook(
                save_merged_hf_checkpoint_callback(full_save_path, converter, config.merged_hf_upload),
                every=config.hf_save_steps,
            )

        trainer.train(state, loader)


if __name__ == "__main__":
    levanter.config.main(train)()
