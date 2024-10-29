import logging
import os
from dataclasses import dataclass
from typing import Optional, Union

import jax.random as jrandom
import transformers

import haliax as hax

import levanter
from levanter.compat.hf_checkpoints import HFCheckpointConverter, save_hf_checkpoint_callback
from levanter.data import PermutationDataset
from levanter.data.text import EpochDataset, LMSupervisedDatasetConfig, mk_supervised_dataset
from levanter.models.lm_model import LmHeadModel, compute_next_token_loss
from levanter.optim import OptimizerConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.py_utils import non_caching_cycle


logger = logging.getLogger(__name__)

# Define default special tokens
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class TrainArgs:
    optimizer: OptimizerConfig
    trainer: TrainerConfig

    max_tune_length: int = 2048  # maximum length of the input to the model during tuning

    # Supervision config
    supervised_data: LMSupervisedDatasetConfig = LMSupervisedDatasetConfig()
    input_field: str = "instruction"  # field name for input in dataset
    output_field: str = "output"  # field name for output in dataset
    data_cache_dir: str = "cache/"  # Path to cache the tokenized data

    model_name_or_path: str = "meta-llama/Llama-2-7b-hf"
    trust_remote_code: bool = False  # Trust remote code when loading from HuggingFace checkpoints.
    model_cache_dir: Optional[str] = None  # Path to cache the model. must be local.

    hf_save_path: Optional[str] = "sft_hf_ckpts"  # Path to save the HuggingFace checkpoint
    hf_upload: Union[bool, str] = False  # Name of the HuggingFace repo to upload to (if any)
    hf_save_steps: int = 1000  # How often to save the HuggingFace checkpoint

    epochs: int = 0  # Number of epochs to train for


def train(config: TrainArgs):
    levanter.initialize(config)

    converter = HFCheckpointConverter.from_hf(config.model_name_or_path, trust_remote_code=config.trust_remote_code)
    model_config = converter.default_config

    if config.max_tune_length > model_config.Pos.size:
        logger.warning(
            f"max_tune_length ({config.max_tune_length}) is greater than the model's maximum length"
            f" ({model_config.Pos.size}). "
        )

    training_key, data_key = jrandom.split(jrandom.PRNGKey(config.trainer.seed), 2)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        cache_dir=config.model_cache_dir,
        model_max_length=config.max_tune_length,
        padding_side="right",
    )
    num_new_tokens = add_special_tokens(tokenizer)
    logger.info(f"Added {num_new_tokens} new tokens")

    # modify converter to use our tokenizer
    converter = converter.replaced(tokenizer=tokenizer)

    # Configure supervised dataset
    supervised_config = config.supervised_data

    # Create supervised dataset using generic machinery
    logger.info("Creating supervised dataset")
    train_dataset = mk_supervised_dataset(supervised_config, tokenizer)
    logger.info("Supervised dataset created")
    train_dataset = PermutationDataset(train_dataset, data_key)

    # Then wrap for epochs
    if config.epochs > 0:
        logger.info(f"Wrapping dataset for {config.epochs} epochs")
        train_dataset = EpochDataset(train_dataset, max_epochs=config.epochs)

    logger.info("Creating optimizer")
    optimizer = config.optimizer.build(config.trainer.num_train_steps)

    with Trainer(config.trainer, optimizer, loss_fn=compute_next_token_loss) as trainer:
        parameter_axis_mapping = trainer.parameter_axis_mapping

        logger.info(f"Loading pretrained model from {converter.reference_checkpoint}")
        model: LmHeadModel = converter.load_pretrained(
            model_config.model_type, axis_mapping=parameter_axis_mapping, dtype=trainer.mp.param_dtype
        )

        model = hax.named_jit(lambda m: m.resize_vocab(len(tokenizer)))(model)

        loader = trainer.data_loader(train_dataset, trainer.TrainBatch)
        loader = non_caching_cycle(loader)

        state = trainer.initial_state(training_key, model=model)

        if int(state.step) != 0:
            logger.info(f"Resuming training from step {state.step}")
            for i in range(state.step):
                next(loader)

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
