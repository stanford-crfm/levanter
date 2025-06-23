import functools
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Union

import jax.numpy as jnp
import jax.random as jrandom
import transformers

import haliax as hax
from haliax import Axis

import levanter
from levanter.compat.hf_checkpoints import HFCompatConfig, save_hf_checkpoint_callback
from levanter.data.dpo_processor import DpoProcessor
from levanter.data.text import DpoSourceConfig, legacy_mk_dpo_dataset
from levanter.models.attention import AttentionMask
from levanter.models.dpo_example import DpoExample
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.trainer import Trainer, TrainerConfig


logger = logging.getLogger(__name__)

# define default special tokens
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class DPOConfig:
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LmConfig = field(default_factory=LlamaConfig)
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)

    beta: float = 0.1  # temperature parameter for DPO loss
    reference_free: bool = True  # whether to use reference-free DPO
    max_prompt_length: int = 512
    max_response_length: int = 1536
    max_seq_len: int = 2048

    # config related to model loading and saving
    initialize_from_hf: Union[bool, str] = False
    use_hf_model_config: bool = False
    hf_save_path: Optional[str] = None
    hf_upload: Optional[str] = None
    hf_save_steps: int = 10000
    hf_save_dtype: Optional[str] = None

    model_name_or_path: str = "meta-llama/Llama-2-7b-hf"
    tokenizer_name_or_path: str = "meta-llama/Llama-2-7b-hf"

    # DPO dataset config
    dpo_data: DpoSourceConfig = field(default_factory=DpoSourceConfig)


def compute_dpo_loss(model: LmHeadModel, ex: DpoExample, beta=0.1, reference_free=True, *, key=None):
    # Unpack the DPO example fields
    p = ex.prompt_ids
    c = ex.chosen_ids
    r = ex.rejected_ids
    prompt_len = ex.prompt_len
    response_len = ex.response_len

    # Rename prompt and response axes to model's position axis (e.g., 'position')
    pos_name = model.Pos.name
    p_pos = p.rename({p.axes[-1].name: pos_name})
    c_pos = c.rename({c.axes[-1].name: pos_name})
    r_pos = r.rename({r.axes[-1].name: pos_name})

    # Build concatenated sequences for chosen and rejected
    seq_chosen = hax.concatenate(pos_name, [p_pos, c_pos])
    seq_rejected = hax.concatenate(pos_name, [p_pos, r_pos])

    # Build causal attention mask
    mask = AttentionMask.causal()
    # Compute logits for both sequences with causal mask
    logits_ch = model(seq_chosen, mask, key=key)
    logits_rj = model(seq_rejected, mask, key=key)

    # Determine where responses start (end of prompt)
    start = prompt_len - 1
    length_c = min(response_len, c_pos.shape[-1])
    length_r = min(response_len, r_pos.shape[-1])

    # Slice out response logits
    resp_ch = hax.slice(logits_ch, pos_name, start=start, length=length_c)
    resp_rj = hax.slice(logits_rj, pos_name, start=start, length=length_r)

    # Log-softmax over vocab
    lp_ch = hax.nn.log_softmax(resp_ch, axis="vocab")
    lp_rj = hax.nn.log_softmax(resp_rj, axis="vocab")

    # Sum log probabilities for the actual tokens
    c_tokens = hax.slice(c_pos, pos_name, start=0, length=length_c)
    r_tokens = hax.slice(r_pos, pos_name, start=0, length=length_r)
    logp_ch = hax.sum(hax.take(lp_ch, "vocab", c_tokens), axis=pos_name)
    logp_rj = hax.sum(hax.take(lp_rj, "vocab", r_tokens), axis=pos_name)

    # Reference model log probabilities if needed
    if not reference_free and hasattr(model, "reference_model"):
        # Use same causal mask for reference model
        logits_ref_ch = model.reference_model(seq_chosen, mask, key=key)
        logits_ref_rj = model.reference_model(seq_rejected, mask, key=key)
        ref_ch = hax.slice(logits_ref_ch, pos_name, start=start, length=length_c)
        ref_rj = hax.slice(logits_ref_rj, pos_name, start=start, length=length_r)
        lp_ref_ch = hax.nn.log_softmax(ref_ch, axis="vocab")
        lp_ref_rj = hax.nn.log_softmax(ref_rj, axis="vocab")
        logp_ref_ch = hax.sum(hax.take(lp_ref_ch, "vocab", c_tokens), axis=pos_name)
        logp_ref_rj = hax.sum(hax.take(lp_ref_rj, "vocab", r_tokens), axis=pos_name)
    else:
        logp_ref_ch = 0
        logp_ref_rj = 0

    # Compute the DPO loss
    diff = (logp_ch - logp_rj) - (logp_ref_ch - logp_ref_rj)
    loss = -hax.nn.log_sigmoid(beta * diff)

    # Return mean loss over the batch axis
    return hax.mean(loss, axis="batch").array


def main(config: DPOConfig):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.tokenizer_name_or_path,
        model_max_length=config.max_seq_len,
        padding_side="right",
        trust_remote_code=True,
    )
    logger.info(f"Loaded tokenizer {tokenizer}")

    # Add special tokens if needed
    add_special_tokens(tokenizer)

    # Handle HF checkpoint initialization similar to train_lm.py
    if config.initialize_from_hf:
        if config.trainer.initialize_from is not None:
            raise ValueError("Cannot specify both initialize_from_hf and initialize_from")

        assert isinstance(config.model, HFCompatConfig)
        converter = config.model.hf_checkpoint_converter()
        if hasattr(tokenizer, "vocab") and tokenizer.vocab != converter.tokenizer.vocab:
            logger.warning("The tokenizers appear to be different. You may want to check this.")

        if isinstance(config.initialize_from_hf, str):
            converter = converter.replaced(reference_checkpoint=config.initialize_from_hf, tokenizer=tokenizer)
        else:
            converter = converter.replaced(tokenizer=tokenizer)

        if config.use_hf_model_config:
            # TODO: log diff of old and new config
            # NB: gross mutability
            config.model = converter.config_from_hf_config(converter.default_hf_config)
    elif isinstance(config.model, HFCompatConfig):
        converter = config.model.hf_checkpoint_converter()
        converter = converter.replaced(tokenizer=tokenizer)
    else:
        converter = None

    levanter.initialize(config)
    optimizer = config.optimizer.build(config.trainer.num_train_steps)

    # Create DPO loss function
    loss_function = functools.partial(compute_dpo_loss, beta=config.beta, reference_free=config.reference_free)

    # Using the trainer as a context manager does 3 things:
    # 1. Sets the device mesh
    # 2. Sets the axis mapping (for fsdp)
    # 3. Sets the global metrics tracker
    with Trainer(config.trainer, optimizer, loss_function) as trainer:
        # randomness in jax is tightly controlled by "keys" which are the states of the random number generators
        # this makes deterministic training pretty easy
        seed = config.trainer.seed
        data_key, loader_key, model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 4)

        # We have two axis_mappings: one for storing the model and optimizer states, and one for compute
        # This allows Zero-3-style parameter sharding, where we shard the parameters and optimizer state across the mesh
        compute_axis_mapping = trainer.compute_axis_mapping
        parameter_axis_mapping = trainer.parameter_axis_mapping

        # Get vocab size and create Vocab axis
        vocab_size = len(tokenizer)
        from haliax import Axis
        from haliax.partitioning import round_axis_for_partitioning

        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), parameter_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        # Create DPO dataset using legacy loader and wrap into first-class DpoExample in batches
        # Raw dataset of dicts (legacy)
        raw_ds = legacy_mk_dpo_dataset(
            urls=config.dpo_data.get_urls("train"),
            tokenizer=tokenizer,
            prompt_field=config.dpo_data.prompt_field,
            chosen_field=config.dpo_data.chosen_field,
            rejected_field=config.dpo_data.rejected_field,
            max_prompt_length=config.max_prompt_length,
            max_response_length=config.max_response_length,
            max_seq_len=config.max_seq_len,
            debug=False,
        )
        # Wrap raw examples into DpoExample with NamedArray axes
        Prompt = hax.Axis("prompt", config.max_prompt_length)
        Response = hax.Axis("response", config.max_response_length)
        processor = DpoProcessor(tokenizer, Prompt, Response)
        # Wrap raw examples into DpoExample; batch sizing happens in Trainer.data_loader
        train_dataset = raw_ds.map_batches(processor)

        state = trainer.initial_state(training_key, model_init=lambda: config.model.build(Vocab, key=model_key))

        if int(state.step) == 0 and config.initialize_from_hf:
            # initialize from an hf pretrained model
            logger.info(
                "No training checkpoint found. Initializing model from HF checkpoint"
                f" '{converter.reference_checkpoint}'"
            )
            # this is a bit gross, but we want to free up the memory from the model we just built
            import dataclasses
            import gc

            state = dataclasses.replace(state, model=None)
            gc.collect()

            from haliax.partitioning import named_jit

            model = converter.load_pretrained(
                config.model.model_type,
                config=config.model if not config.use_hf_model_config else None,
                axis_mapping=parameter_axis_mapping,
                dtype=trainer.mp.compute_dtype,
            )
            model = named_jit(trainer.mp.cast_to_param, parameter_axis_mapping)(model)
            state = dataclasses.replace(state, model=model)
        else:
            logger.info("No checkpoint found. Starting from scratch.")

        from levanter.utils.jax_utils import parameter_count

        levanter.tracker.log_summary({"parameter_count": parameter_count(state.model)})

        # Add HF checkpoint saving if configured
        if config.hf_save_path is not None:
            # bit gross to reach this far into the config, but it's fine
            if config.trainer.checkpointer.append_run_id_to_base_path:
                full_save_path = os.path.join(config.hf_save_path, trainer.run_id)
            else:
                full_save_path = config.hf_save_path

            save_dtype: Optional[jnp.dtype] = None
            if config.hf_save_dtype is not None:
                try:
                    save_dtype = jnp.dtype(config.hf_save_dtype)
                except TypeError:
                    logger.warning(f"Invalid hf_save_dtype: {config.hf_save_dtype}. Defaulting to None.")

            trainer.add_hook(
                save_hf_checkpoint_callback(
                    full_save_path, converter, upload_to_hf=config.hf_upload or False, save_dtype=save_dtype
                ),
                every=config.hf_save_steps,
            )

        # Create data loader from DPO dataset
        train_loader = trainer.data_loader(train_dataset)

        if state.step > 0:
            logger.info(f"Resuming training from step {state.step}")
            train_loader = train_loader.iter_from_step(state.step)
        else:
            train_loader = train_loader.iter_from_step(0)

        ## OK, actually run training!
        trainer.train(state, train_loader)

    # This isn't necessary except when Levanter is run in a subprocess (as happens w/ ray)
    trainer.tracker.finish()


def add_special_tokens(tokenizer, use_unk_instead_of_adding=False):
    """Add special tokens to tokenizer if they don't exist."""
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
    levanter.config.main(main)()
