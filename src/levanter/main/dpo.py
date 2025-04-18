import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union

import jax.random as jrandom
import transformers

from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.trainer import Trainer, TrainerConfig


logger = logging.getLogger(__name__)

# Define default special tokens
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

from levanter.data.text import DpoSourceConfig

@dataclass
class DPOConfig:
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LmConfig = field(default_factory=LlamaConfig)
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)

    beta: float = 0.1  # temperature parameter for DPO loss
    reference_free: bool = False  # whether to use reference-free DPO
    max_prompt_length: int = 512
    max_response_length: int = 1536
    max_seq_len: int = 2048

    # config related to model loading and saving
    initialize_from_hf: Union[bool, str] = False
    hf_save_path: Optional[str] = None
    hf_upload: Optional[str] = None
    hf_save_steps: int = 0

    model_name_or_path: str = "meta-llama/Llama-2-7b-hf"
    tokenizer: str = "meta-llama/Llama-2-7b-hf"

    # DPO dataset config
    dpo_data: DpoSourceConfig = field(default_factory=DpoSourceConfig)


def compute_dpo_loss(model: LmHeadModel, batch, beta=0.1, reference_free=False):
    """Compute DPO loss"""
    
    # TODO: implement
    raise NotImplementedError


def train(config: DPOConfig):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.tokenizer,
        model_max_length=config.max_seq_len,
        padding_side="right",
        trust_remote_code=True,
    )
    logger.info(f"Loaded tokenizer {tokenizer}")

    if config.initialize_from_hf:
        if config.trainer.initialize_from is not None:
            raise ValueError("Cannot use both --initialize_from_hf and --initialize_from")

        assert isinstance(config.model, HFCompatConfig)
        converter = HFCheckpointConverter.from_hf(config.model_name_or_path, trust_remote_code=True)
        
        if hasattr(tokenizer, "vocab") and tokenizer.vocab != converter.tokenizer.vocab:
            logger.warning("The tokenizers appear to be different. You may want to check this.")

    # initialize model and optimizer
    model = config.model.init(key=jrandom.PRNGKey(config.trainer.seed))
    
    # add special tokens if needed
    add_special_tokens(tokenizer)

    # create DPO dataset and DataLoader
    from levanter.data.loader import DataLoader

    dpo_dataset = config.dpo_data.mk_dataset(
        tokenizer=tokenizer,
        split="train",
        batch_size=config.trainer.batch_size if hasattr(config.trainer, 'batch_size') else 8,
    )
    train_loader = DataLoader(dpo_dataset, batch_size=config.trainer.batch_size if hasattr(config.trainer, 'batch_size') else 8)

    # create trainer with DPO loss
    trainer = Trainer(
        model=model,
        optimizer=config.optimizer,
        loss_fn=lambda m, batch: compute_dpo_loss(m, batch, config.beta, config.reference_free),
        config=config.trainer,
    )

    if config.hf_save_path:
        trainer.add_callback(
            save_hf_checkpoint_callback(
                save_dir=config.hf_save_path,
                tokenizer=tokenizer,
                save_steps=config.hf_save_steps,
                upload_repo=config.hf_upload,
            )
        )

    trainer.train(train_loader)



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
