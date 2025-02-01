import dataclasses
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union, Iterator

import jax.random as jrandom
import jax.numpy as jnp
import transformers

import haliax as hax
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning
from optax.tree_utils import tree_zeros_like
import time

import levanter
from levanter import callbacks
from levanter.compat.hf_checkpoints import HFCheckpointConverter, HFCompatConfig, save_hf_checkpoint_callback
from levanter.data import PermutationDataset, batched
from levanter.data.text import (
    ChatUrlDataSourceConfig,
    EpochDataset,
    SupervisedSourceConfig,
    mk_chat_sft_dataset,
    mk_supervised_dataset,
)
from levanter.models.attention import AttentionMask
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig, LmHeadModel, LmExample, compute_next_token_loss
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.data.loader import stack_tree
from levanter.data.packing import PromptCompletion, pack_prompt_completions
from levanter.utils.background_iterable import BackgroundIterator
from levanter.utils.hf_utils import HfTokenizer
from levanter.utils.jax_utils import broadcast_shard, use_cpu_device
from levanter.data.dataset import AsyncDataset

import asyncio

logger = logging.getLogger(__name__)

# Define default special tokens
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


class DatasetType(str, Enum):
    """Type of dataset to use"""

    HUGGINGFACE = "huggingface"  # Use HF dataset
    CHAT_JSONL = "chat_jsonl"  # Use JSONL files with chat format


@dataclass
class SFTConfig:
    # inherit most of the config from TrainLmConfig
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LmConfig = field(default_factory=LlamaConfig)
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)
    supervised_data: Optional[SupervisedSourceConfig | dict[str, SupervisedSourceConfig]] = None

    # config related to continued pretraining
    initialize_from_hf: Union[bool, str] = False
    hf_save_path: Optional[str] = None
    hf_upload: Optional[str] = None
    hf_save_steps: int = 0

    max_seq_len: int = 2048
    model_name_or_path: str = "meta-llama/Llama-2-7b-hf"
    tokenizer: str = "meta-llama/Llama-2-7b-hf"

    # Add dataset type and chat-specific fields
    dataset_type: DatasetType = DatasetType.CHAT_JSONL
    chat_train_urls: Optional[List[str]] = None
    messages_field: str = "messages"
    input_role: str = "user"
    output_role: str = "assistant"

    data_seed: Optional[int] = None  # if provided, will override the data seed from the trainer

    # if provided, will initialize from this checkpoint, used for llama style data mixture
    epoch: int = 0


def train(config: SFTConfig):
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
        if isinstance(config.initialize_from_hf, str):
            converter = converter.replaced(reference_checkpoint=config.initialize_from_hf, tokenizer=tokenizer)
        else:
            converter = converter.replaced(tokenizer=tokenizer)

        model_config = converter.default_config
        model_config = dataclasses.replace(converter.default_config, seq_len=config.max_seq_len)
    elif config.trainer.initialize_from is None:
        raise ValueError("Must specify either --initialize_from_hf or --initialize_from")
    else:
        if config.hf_save_steps:
            converter = HFCheckpointConverter.from_hf(config.model_name_or_path, trust_remote_code=True)
            converter = converter.replaced(tokenizer=tokenizer)
        else:
            converter = None
        model_config = config.model

    levanter.initialize(config)

    num_new_tokens = add_special_tokens(tokenizer)
    logger.info(f"Added {num_new_tokens} new tokens")
    # randomness in jax is tightly controlled by "keys" which are the states of the random number generators
    # this makes deterministic training pretty easy
    seed = config.trainer.seed
    data_key, _, model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 4)

    if config.data_seed is not None:
        logger.info(f"Overriding data seed with {config.data_seed}")
        data_key = jrandom.PRNGKey(config.data_seed)

    # Create supervised dataset using generic machinery
    logger.info("Creating supervised dataset")
    if config.dataset_type == DatasetType.CHAT_JSONL:
        assert config.chat_train_urls is not None
        assert config.supervised_data is not None

        # Get the cache_dir safely
        cache_dir = (
            config.supervised_data.cache_dir
            if not isinstance(config.supervised_data, dict)
            else next(iter(config.supervised_data.values())).cache_dir
        )

        chat_config = ChatUrlDataSourceConfig(
            cache_dir=cache_dir,
            train_urls=config.chat_train_urls,  # No validation in this config
            messages_field=config.messages_field,
            input_role=config.input_role,
            output_role=config.output_role,
        )
        train_dataset = mk_chat_sft_dataset(chat_config, tokenizer, model_config.Pos)
    else:
        assert config.supervised_data is not None
        if isinstance(config.supervised_data, dict):
            # TODO: figure out what actually makes sense here
            # for marin we will just use the url code path
            config_to_use = next(iter(config.supervised_data.values()))
        else:
            config_to_use = config.supervised_data
        train_dataset = mk_supervised_dataset(config_to_use, "train", tokenizer, model_config.Pos)
    logger.info("Supervised dataset created")
    train_dataset = PermutationDataset(train_dataset, data_key)

    # Then wrap for epochs
    if config.epoch > 0:
        logger.info(f"Wrapping dataset for {config.epoch} epochs")
        train_dataset = EpochDataset(train_dataset, max_epochs=config.epoch)

    logger.info("Creating optimizer")
    optimizer = config.optimizer.build(config.trainer.num_train_steps)

    # Using the trainer as a context manager does 3 things:
    # 1. Sets the device mesh
    # 2. Sets the axis mapping (for fsdp)
    # 3. Sets the global metrics tracker
    with Trainer(config.trainer, optimizer, loss_fn=compute_next_token_loss) as trainer:  # type: ignore
        parameter_axis_mapping = trainer.parameter_axis_mapping

        # We have two axis_mappings: one for storing the model and optimizer states, and one for compute
        # This allows Zero-3-style parameter sharding, where we shard the parameters and optimizer state across the mesh
        parameter_axis_mapping = trainer.parameter_axis_mapping

        # some axes we need
        Pos = config.model.Pos
        # to do partitioning, our dimensions have to be divisible by the size of the physical axes they're mapped to
        # For most things, we just insist you specify the config right, but tokenizers often have strange numbers of
        # tokens: gpt-2 has 50257, for example. So we round up.
        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), parameter_axis_mapping)
        if config.initialize_from_hf:
            logger.info(f"Loading pretrained model from {converter.reference_checkpoint}")
            model: LmHeadModel = converter.load_pretrained(
                model_config.model_type, axis_mapping=parameter_axis_mapping, dtype=trainer.mp.param_dtype
            )  # type: ignore
            model = hax.named_jit(lambda m: m.resize_vocab(len(tokenizer)))(model)
            state = trainer.initial_state(training_key, model=model)
        else:
            if vocab_size != Vocab.size:
                logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")
            state = trainer.initial_state(training_key, model_init=lambda: config.model.build(Vocab, key=model_key))

        flops_per_token = config.model.flops_per_token(vocab_size)
        flops_per_example = 3 * flops_per_token * Pos.size if flops_per_token is not None else None
        trainer.add_hook(
            callbacks.log_performance_stats(Pos.size, trainer.config.train_batch_size, flops_per_example), every=1
        )
        # Get current step from trainer state
        current_step = int(state.step)

        
       

        # change iterate tokenized requests to take a dict rather than a list
        # of where the first element is prompt ands econd is response

        # then pass into tierate tokenizer requests, go to pack requests
        # and then you have the correct loader, just pass to trainer.train()

        # TODO figure out if there's a better heuristic for max segements to pack per example?
        logger.info("Creating prompt completion iterator")
        prompt_completion_iterator = create_prompt_completion_iterator(train_dataset, Pos)

        if current_step > 0:
            logger.info(f"Resuming training from step {current_step}")
            # Calculate how many examples to skip based on batch size
            examples_to_skip = current_step * trainer.config.train_batch_size
            
            # Skip through the iterator until we reach the right position
            for _ in range(examples_to_skip):
                try:
                    next(prompt_completion_iterator)
                except StopIteration:
                    logger.warning("Ran out of examples while seeking - restarting from beginning")
                    # Recreate iterator and continue skipping
                    prompt_completion_iterator = create_prompt_completion_iterator(train_dataset, Pos)
        else:
            logger.info("Starting SFT from scratch")

        logger.info("Packing prompt completions")
        packed_iterator = _pack_requests(prompt_completion_iterator, tokenizer, Pos, max_pack_size=4)
        logger.info("Stacking batches to train batch")
        packed_iterator = stack_batches(example_iterator=packed_iterator, Pos=Pos, TrainBatch=trainer.TrainBatch)
        # TODO  what's a good number for max_capacity?
        logger.info("Creating data loader")
        packed_loader = BackgroundIterator(packed_iterator, max_capacity=1024)

        # to be moved 
        #loader = trainer.data_loader(train_dataset, trainer.TrainBatch)

        if config.hf_save_path is not None:
            # bit gross to reach this far into the config, but it's fine
            if config.trainer.checkpointer.append_run_id_to_base_path:
                full_save_path = os.path.join(config.hf_save_path, trainer.run_id)
            else:
                full_save_path = config.hf_save_path

            trainer.add_hook(
                save_hf_checkpoint_callback(full_save_path, converter, upload_to_hf=config.hf_upload or False),
                every=config.hf_save_steps,
            )

        trainer.train(state, packed_loader)

def create_prompt_completion_iterator(cached_dataset: AsyncDataset, Pos: hax.Axis) -> Iterator[PromptCompletion]:
    """
    Creates an iterator that yields PromptCompletion objects from a cached dataset.
    
    Args:
        cached_dataset: The AsyncDataset containing preprocessed examples
        Pos: The position axis defining maximum sequence length
    
    Returns:
        An iterator yielding PromptCompletion objects
    """
    # AsyncDataset already has a current_len method that returns current length or None
    length = asyncio.run(cached_dataset.async_len())
    
    if length is None:
        raise ValueError("Dataset length cannot be None")
        
    # TODO play around with batch size
    for batch_indicies in batched(range(length), 4096):
        # put timer here
        # start_time = time.perf_counter()
        examples = asyncio.run(cached_dataset.get_batch(batch_indicies))
        # end_time = time.perf_counter()
        # elapsed_time = end_time - start_time
        # print(f"Elapsed time for get batches: {elapsed_time:.6f} seconds", flush=True)

        for i in range(len(examples)):
            example = examples[i]
            sources_len = example["sources_len"].item()
            if sources_len > Pos.size - 1:
                continue

            ids = example["input_ids"].tolist()
            if len(ids) > Pos.size:
                ids = ids[:Pos.size]

            if len(ids) <= sources_len:
                continue

            try:
                yield PromptCompletion(
                    ids=ids,
                    prompt_length=sources_len,
                    segment_id=batch_indicies[i]
                )
            except ValueError:
                continue

def _pack_requests(
    prompt_completion_iterator: Iterator[PromptCompletion], tokenizer: HfTokenizer, Pos: hax.Axis, max_pack_size: int
) -> Iterator[LmExample]:
    # TODO: use a better packing algorithm?
    yield from pack_prompt_completions(
        Pos,
        prompt_completion_iterator,
        max_segments_per_example=max_pack_size,
        pad_token=tokenizer.pad_token_id,
        max_buffered_examples=16,
    )

"""
Helper function to create a dummy instance with the same shape as the batch.
When we reach the end of the dataset but we want a full batch,
will give a batch of zeros with -1 segment mask so it doesn't affect loss
"""
def _make_dummy_instance(batch, Pos):
        dummy_instance: LmExample = tree_zeros_like(batch[0])
        dummy_segment_mask = hax.full(Pos, -1, dtype=jnp.int32)
        dummy_attn = AttentionMask.causal().with_segment_ids(dummy_segment_mask)
        dummy_instance = dataclasses.replace(dummy_instance, attn_mask=dummy_attn)
        return dummy_instance

def stack_batches(example_iterator, Pos, TrainBatch):
        """
        Stack examples from an iterator into a batch.

        Args:
            TrainBatch: The batch axis.
            Pos: The position axis.
            example_iterator: An iterator of examples.

        Returns:
            A batch of examples.
        """
        # add timer here as well and profile
        with use_cpu_device():
            batch_count = 0
            for batch in batched(example_iterator, TrainBatch.size):
                batch_count += 1
                #start_time_loop = time.perf_counter()
                if len(batch) < TrainBatch.size:
                    dummy_instance = _make_dummy_instance(batch, Pos)
                    batch.extend([dummy_instance] * (TrainBatch.size - len(batch)))
                # # Start timing before calling stack_tree
                # start_time = time.perf_counter()
                result = stack_tree(TrainBatch, batch)  # Capture the result
                # stack_time = time.perf_counter() - start_time  # Calculate elapsed time

                # print(f"Stack tree execution time: {stack_time:.6f} seconds", flush=True)
                yield result  # Yield the computed result
                # end_time_loop = time.perf_counter()
                # loop_time = end_time_loop - start_time_loop
                # print(f"Loop takes {loop_time}")
                # print(f"Iterator time is {loop_time - stack_time}")

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
