"""
Direct Preference Optimization (DPO) training with optional KL divergence penalty.

This module implements DPO training for language models with the following features:

1. Standard DPO loss using preference data (chosen vs rejected responses)
2. Optional KL divergence penalty to prevent model deviation from reference
3. Support for both separate and concatenated forward passes for efficiency
4. Reference-free and reference-based DPO modes
5. Integration with HuggingFace checkpoints and model saving

The KL penalty feature adds a regularization term that penalizes the KL divergence
between the current model's output distribution and the reference model's distribution.
This helps prevent the model from deviating too far from the reference during training,
which can improve stability and prevent overfitting to the preference data.

Usage:
    python -m levanter.main.dpo --config config/dpo_with_kl_penalty.yaml

Key parameters:
    - beta: DPO temperature parameter (default: 0.1)
    - kl_penalty_weight: Weight for KL divergence penalty (default: 0.0, disabled)
    - reference_free: Whether to use reference-free DPO (default: True)
    - use_concatenated_forward: Use efficient concatenated forward passes (default: True)
"""

import functools
import logging
import os
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Union

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import transformers

import haliax as hax
from haliax import Axis, NamedArray

import levanter
from levanter.compat.hf_checkpoints import HFCompatConfig, save_hf_checkpoint_callback
from levanter.data._preprocessor import BatchProcessor
from levanter.data.text import DpoSourceConfig, legacy_mk_dpo_dataset
from levanter.models.attention import AttentionMask
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.hf_utils import num_cpus_used_by_tokenizer


logger = logging.getLogger(__name__)

# define default special tokens
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


class DpoExample(eqx.Module):
    prompt_ids: NamedArray  # axes=(Pos,)
    chosen_ids: NamedArray  # axes=(Response,)
    rejected_ids: NamedArray  # axes=(Response,)
    prompt_len: int = 0  # number of prompt tokens before padding
    response_len: int = 0  # number of response tokens before padding

    @staticmethod
    def from_dict(
        raw: dict,
        tokenizer,
        Pos: Axis,
        Response: Axis,
    ) -> "DpoExample":
        """
        Build a DpoExample from raw token id lists in raw dict.
        Pads/truncates to Pos.size and Response.size, wraps in NamedArray without batch axis.
        """
        pad_id = getattr(tokenizer, "pad_token_id", 0)

        def pad(seq, target_len):
            # Unwrap NamedArray to raw array if needed
            if hasattr(seq, "array"):
                raw = seq.array
            else:
                raw = seq
            # Convert to Python list
            if isinstance(raw, np.ndarray):
                lst = raw.tolist()
            else:
                lst = list(raw)
            # Truncate or pad to target_len
            out = lst[:target_len]
            if len(out) < target_len:
                out += [pad_id] * (target_len - len(out))
            return out

        raw_prompt = raw["prompt_ids"]
        raw_chosen = raw["chosen_ids"]
        raw_rejected = raw["rejected_ids"]

        def seq_len(seq):
            logger.info(f"seq: {seq}")
            logger.info(f"type of seq: {type(seq)}")
            if hasattr(seq, "shape") and len(seq.shape) > 0:
                return int(seq.shape[-1])
            elif hasattr(seq, "__len__"):
                return len(seq)
            else:
                # Fallback for other types
                return len(list(seq))

        prompt_len = int(raw.get("prompt_len", min(seq_len(raw_prompt), Pos.size)))
        chosen_len = min(seq_len(raw_chosen), Response.size)
        rejected_len = min(seq_len(raw_rejected), Response.size)

        response_len = int(raw.get("response_len", min(chosen_len, rejected_len)))

        prompt = np.array(pad(raw_prompt, Pos.size), dtype=np.int32)
        chosen = np.array(pad(raw_chosen, Response.size), dtype=np.int32)
        rejected = np.array(pad(raw_rejected, Response.size), dtype=np.int32)

        # wrap in NamedArray without batch axis
        prompt_na = hax.named(prompt, (Pos,))
        chosen_na = hax.named(chosen, (Response,))
        rejected_na = hax.named(rejected, (Response,))

        return DpoExample(
            prompt_ids=prompt_na,
            chosen_ids=chosen_na,
            rejected_ids=rejected_na,
            prompt_len=prompt_len,
            response_len=response_len,
        )


@dataclass
class DPOConfig:
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LmConfig = field(default_factory=LlamaConfig)
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)

    beta: float = 0.1  # temperature parameter for DPO loss
    reference_free: bool = True  # whether to use reference-free DPO
    kl_penalty_weight: float = 0.0  # weight for KL divergence penalty from reference model (0.0 disables KL penalty)
    max_prompt_length: int = 512
    max_response_length: int = 1536
    max_seq_len: int = 2048
    
    # Efficiency optimizations
    use_concatenated_forward: bool = True  # use concatenated forward passes for better FSDP efficiency
    precompute_ref_log_probs: bool = False  # precompute reference model log probabilities

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


class DpoProcessor(BatchProcessor[Mapping, DpoExample]):
    """
    Batch processor that converts raw DPO dicts to DpoExample instances with NamedArray fields.
    """

    def __init__(
        self,
        tokenizer,
        Prompt: Axis,
        Response: Axis,
    ):
        self.tokenizer = tokenizer
        self.Prompt = Prompt
        self.Response = Response

    def __call__(self, batch: Sequence[Mapping]) -> list[DpoExample]:
        return [DpoExample.from_dict(raw, self.tokenizer, self.Prompt, self.Response) for raw in batch]

    @property
    def output_exemplar(self) -> DpoExample:
        # exemplar with zeros for one example (no batch axis)
        pad_id = getattr(self.tokenizer, "pad_token_id", 0)
        import haliax as hax

        prompt = np.full((self.Prompt.size,), pad_id, dtype=np.int32)
        chosen = np.full((self.Response.size,), pad_id, dtype=np.int32)
        rejected = np.full((self.Response.size,), pad_id, dtype=np.int32)
        return DpoExample(
            prompt_ids=hax.named(prompt, (self.Prompt,)),
            chosen_ids=hax.named(chosen, (self.Response,)),
            rejected_ids=hax.named(rejected, (self.Response,)),
            prompt_len=0,
            response_len=0,
        )

    @property
    def num_cpus(self) -> int:
        return num_cpus_used_by_tokenizer(self.tokenizer)

    @property
    def num_gpus(self) -> int:
        return 0

    @property
    def metadata(self) -> dict:
        return {}
    

def compute_dpo_loss_separate(model: LmHeadModel, ex: DpoExample, beta=0.1, reference_free=True, *, key=None):
    """
    Compute DPO loss using separate forward passes (original implementation).
    
    This approach does two separate forward passes - one for chosen and one for rejected.
    Less efficient but simpler to understand and debug.
    """
    # Unpack the DPO example fields
    p = ex.prompt_ids
    c = ex.chosen_ids
    r = ex.rejected_ids
    prompt_len = ex.prompt_len
    response_len = ex.response_len

    # Rename prompt and response axes to model's position axis
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
    # Note: This approach uses two separate forward passes, which is less efficient
    # for FSDP but simpler to understand and debug
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


def compute_dpo_loss_concatenated(model: LmHeadModel, ex: DpoExample, beta=0.1, reference_free=True, *, key=None):
    """
    Compute DPO loss using concatenated forward passes for better efficiency.
    
    This approach concatenates the chosen and rejected sequences into a single batch,
    performs one forward pass, then splits the results. This is more efficient for
    FSDP and reduces memory overhead.
    """
    # Unpack the DPO example fields
    p = ex.prompt_ids
    c = ex.chosen_ids
    r = ex.rejected_ids
    prompt_len = ex.prompt_len
    response_len = ex.response_len

    # Rename prompt and response axes to model's position axis
    pos_name = model.Pos.name
    p_pos = p.rename({p.axes[-1].name: pos_name})
    c_pos = c.rename({c.axes[-1].name: pos_name})
    r_pos = r.rename({r.axes[-1].name: pos_name})

    # Build individual sequences
    seq_chosen = hax.concatenate(pos_name, [p_pos, c_pos])
    seq_rejected = hax.concatenate(pos_name, [p_pos, r_pos])

    # Concatenate chosen and rejected sequences along batch dimension
    # This creates a single batch with 2*original_batch_size sequences
    # This is more efficient for FSDP as it reduces communication overhead
    concatenated_seq = hax.concatenate("batch", [seq_chosen, seq_rejected])
    
    # Build causal attention mask for concatenated sequence
    # The mask automatically handles the increased batch size
    mask = AttentionMask.causal()
    
    # Single forward pass for both sequences
    # This reduces memory allocation/deallocation cycles and improves FSDP efficiency
    logits = model(concatenated_seq, mask, key=key)
    
    # Split the logits back into chosen and rejected
    batch_size = seq_chosen.shape[0]
    logits_ch = hax.slice(logits, "batch", start=0, length=batch_size)
    logits_rj = hax.slice(logits, "batch", start=batch_size, length=batch_size)

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
        # Use same concatenated approach for reference model
        logits_ref = model.reference_model(concatenated_seq, mask, key=key)
        logits_ref_ch = hax.slice(logits_ref, "batch", start=0, length=batch_size)
        logits_ref_rj = hax.slice(logits_ref, "batch", start=batch_size, length=batch_size)
        
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


def compute_dpo_loss(model: LmHeadModel, ex: DpoExample, beta=0.1, reference_free=True, use_concatenated_forward=True, *, key=None):
    """
    Compute DPO loss with configurable forward pass strategy.
    
    Args:
        model: The language model
        ex: DPO example containing prompt, chosen, and rejected sequences
        beta: DPO temperature parameter
        reference_free: Whether to use reference-free DPO
        use_concatenated_forward: Whether to use concatenated forward passes (more efficient for FSDP)
        key: JAX random key
    
    Returns:
        DPO loss value
    """
    if use_concatenated_forward:
        return compute_dpo_loss_concatenated(model, ex, beta, reference_free, key=key)
    else:
        return compute_dpo_loss_separate(model, ex, beta, reference_free, key=key)


def test_dpo_implementations_equivalence(model: LmHeadModel, ex: DpoExample, beta=0.1, reference_free=True, *, key=None):
    """
    Test that both DPO implementations produce the same results.
    
    This is useful for debugging and ensuring the concatenated implementation
    is mathematically equivalent to the separate implementation.
    """
    # Test separate forward passes
    loss_separate = compute_dpo_loss_separate(model, ex, beta, reference_free, key=key)
    
    # Test concatenated forward passes
    loss_concatenated = compute_dpo_loss_concatenated(model, ex, beta, reference_free, key=key)
    
    # Check if they're close (within numerical precision)
    diff = abs(loss_separate - loss_concatenated)
    if diff > 1e-6:
        logger.warning(f"DPO implementations differ by {diff:.2e}")
        logger.warning(f"Separate: {loss_separate:.6f}, Concatenated: {loss_concatenated:.6f}")
    else:
        logger.info("DPO implementations produce equivalent results ✓")
    
    return loss_separate, loss_concatenated


def compute_dpo_loss_with_kl_penalty_separate(model: LmHeadModel, ex: DpoExample, beta=0.1, reference_free=True, kl_penalty_weight=0.0, *, key=None):
    """
    Compute DPO loss with KL divergence penalty using separate forward passes.
    
    This adds a KL divergence penalty between the current model and reference model
    to prevent the model from deviating too far from the reference during training.
    
    Args:
        model: The language model
        ex: DPO example containing prompt, chosen, and rejected sequences
        beta: DPO temperature parameter
        reference_free: Whether to use reference-free DPO
        kl_penalty_weight: Weight for KL divergence penalty (0.0 disables KL penalty)
        key: JAX random key
    
    Returns:
        DPO loss with KL penalty
    """
    # Unpack the DPO example fields
    p = ex.prompt_ids
    c = ex.chosen_ids
    r = ex.rejected_ids
    prompt_len = ex.prompt_len
    response_len = ex.response_len

    # Rename prompt and response axes to model's position axis
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
    dpo_loss = -hax.nn.log_sigmoid(beta * diff)

    # Compute KL divergence penalty if enabled and reference model exists
    kl_penalty = 0.0
    if kl_penalty_weight > 0.0 and hasattr(model, "reference_model"):
        # Compute KL divergence between current model and reference model
        # KL(p_current || p_reference) = sum(p_current * (log(p_current) - log(p_reference)))
        
        # Get probabilities from current model
        p_current_ch = hax.nn.softmax(resp_ch, axis="vocab")
        p_current_rj = hax.nn.softmax(resp_rj, axis="vocab")
        
        # Get probabilities from reference model
        logits_ref_ch = model.reference_model(seq_chosen, mask, key=key)
        logits_ref_rj = model.reference_model(seq_rejected, mask, key=key)
        ref_ch = hax.slice(logits_ref_ch, pos_name, start=start, length=length_c)
        ref_rj = hax.slice(logits_ref_rj, pos_name, start=start, length=length_r)
        p_ref_ch = hax.nn.softmax(ref_ch, axis="vocab")
        p_ref_rj = hax.nn.softmax(ref_rj, axis="vocab")
        
        # Compute KL divergence for chosen responses
        kl_ch = hax.sum(p_current_ch * (hax.nn.log_softmax(resp_ch, axis="vocab") - hax.nn.log_softmax(ref_ch, axis="vocab")), axis="vocab")
        kl_rj = hax.sum(p_current_rj * (hax.nn.log_softmax(resp_rj, axis="vocab") - hax.nn.log_softmax(ref_rj, axis="vocab")), axis="vocab")
        
        # Average KL over sequence length and batch
        kl_ch_mean = hax.mean(kl_ch, axis=pos_name)
        kl_rj_mean = hax.mean(kl_rj, axis=pos_name)
        kl_penalty = hax.mean(kl_ch_mean + kl_rj_mean, axis="batch")

    # Combine DPO loss with KL penalty
    total_loss = dpo_loss + kl_penalty_weight * kl_penalty

    # Return mean loss over the batch axis
    return hax.mean(total_loss, axis="batch").array


def compute_dpo_loss_with_kl_penalty_concatenated(model: LmHeadModel, ex: DpoExample, beta=0.1, reference_free=True, kl_penalty_weight=0.0, *, key=None):
    """
    Compute DPO loss with KL divergence penalty using concatenated forward passes for better efficiency.
    
    This adds a KL divergence penalty between the current model and reference model
    to prevent the model from deviating too far from the reference during training.
    
    Args:
        model: The language model
        ex: DPO example containing prompt, chosen, and rejected sequences
        beta: DPO temperature parameter
        reference_free: Whether to use reference-free DPO
        kl_penalty_weight: Weight for KL divergence penalty (0.0 disables KL penalty)
        key: JAX random key
    
    Returns:
        DPO loss with KL penalty
    """
    # Unpack the DPO example fields
    p = ex.prompt_ids
    c = ex.chosen_ids
    r = ex.rejected_ids
    prompt_len = ex.prompt_len
    response_len = ex.response_len

    # Rename prompt and response axes to model's position axis
    pos_name = model.Pos.name
    p_pos = p.rename({p.axes[-1].name: pos_name})
    c_pos = c.rename({c.axes[-1].name: pos_name})
    r_pos = r.rename({r.axes[-1].name: pos_name})

    # Build individual sequences
    seq_chosen = hax.concatenate(pos_name, [p_pos, c_pos])
    seq_rejected = hax.concatenate(pos_name, [p_pos, r_pos])

    # Concatenate chosen and rejected sequences along batch dimension
    concatenated_seq = hax.concatenate("batch", [seq_chosen, seq_rejected])
    
    # Build causal attention mask for concatenated sequence
    mask = AttentionMask.causal()
    
    # Single forward pass for both sequences
    logits = model(concatenated_seq, mask, key=key)
    
    # Split the logits back into chosen and rejected
    batch_size = seq_chosen.shape[0]
    logits_ch = hax.slice(logits, "batch", start=0, length=batch_size)
    logits_rj = hax.slice(logits, "batch", start=batch_size, length=batch_size)

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
        # Use same concatenated approach for reference model
        logits_ref = model.reference_model(concatenated_seq, mask, key=key)
        logits_ref_ch = hax.slice(logits_ref, "batch", start=0, length=batch_size)
        logits_ref_rj = hax.slice(logits_ref, "batch", start=batch_size, length=batch_size)
        
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
    dpo_loss = -hax.nn.log_sigmoid(beta * diff)

    # Compute KL divergence penalty if enabled and reference model exists
    kl_penalty = 0.0
    if kl_penalty_weight > 0.0 and hasattr(model, "reference_model"):
        # Compute KL divergence between current model and reference model
        # Get probabilities from current model
        p_current_ch = hax.nn.softmax(resp_ch, axis="vocab")
        p_current_rj = hax.nn.softmax(resp_rj, axis="vocab")
        
        # Get probabilities from reference model (reuse logits from above)
        p_ref_ch = hax.nn.softmax(ref_ch, axis="vocab")
        p_ref_rj = hax.nn.softmax(ref_rj, axis="vocab")
        
        # Compute KL divergence for chosen and rejected responses
        kl_ch = hax.sum(p_current_ch * (hax.nn.log_softmax(resp_ch, axis="vocab") - hax.nn.log_softmax(ref_ch, axis="vocab")), axis="vocab")
        kl_rj = hax.sum(p_current_rj * (hax.nn.log_softmax(resp_rj, axis="vocab") - hax.nn.log_softmax(ref_rj, axis="vocab")), axis="vocab")
        
        # Average KL over sequence length and batch
        kl_ch_mean = hax.mean(kl_ch, axis=pos_name)
        kl_rj_mean = hax.mean(kl_rj, axis=pos_name)
        kl_penalty = hax.mean(kl_ch_mean + kl_rj_mean, axis="batch")

    # Combine DPO loss with KL penalty
    total_loss = dpo_loss + kl_penalty_weight * kl_penalty

    # Return mean loss over the batch axis
    return hax.mean(total_loss, axis="batch").array


def compute_dpo_loss_with_kl_penalty(model: LmHeadModel, ex: DpoExample, beta=0.1, reference_free=True, kl_penalty_weight=0.0, use_concatenated_forward=True, *, key=None):
    """
    Compute DPO loss with KL divergence penalty with configurable forward pass strategy.
    
    Args:
        model: The language model
        ex: DPO example containing prompt, chosen, and rejected sequences
        beta: DPO temperature parameter
        reference_free: Whether to use reference-free DPO
        kl_penalty_weight: Weight for KL divergence penalty (0.0 disables KL penalty)
        use_concatenated_forward: Whether to use concatenated forward passes (more efficient for FSDP)
        key: JAX random key
    
    Returns:
        DPO loss with KL penalty
    """
    if use_concatenated_forward:
        return compute_dpo_loss_with_kl_penalty_concatenated(model, ex, beta, reference_free, kl_penalty_weight, key=key)
    else:
        return compute_dpo_loss_with_kl_penalty_separate(model, ex, beta, reference_free, kl_penalty_weight, key=key)


def test_dpo_kl_penalty_implementations_equivalence(model: LmHeadModel, ex: DpoExample, beta=0.1, reference_free=True, kl_penalty_weight=0.1, *, key=None):
    """
    Test that both DPO KL penalty implementations produce the same results.
    
    This is useful for debugging and ensuring the concatenated implementation
    is mathematically equivalent to the separate implementation.
    """
    # Test separate forward passes
    loss_separate = compute_dpo_loss_with_kl_penalty_separate(model, ex, beta, reference_free, kl_penalty_weight, key=key)
    
    # Test concatenated forward passes
    loss_concatenated = compute_dpo_loss_with_kl_penalty_concatenated(model, ex, beta, reference_free, kl_penalty_weight, key=key)
    
    # Check if they're close (within numerical precision)
    diff = abs(loss_separate - loss_concatenated)
    if diff > 1e-6:
        logger.warning(f"DPO KL penalty implementations differ by {diff:.2e}")
        logger.warning(f"Separate: {loss_separate:.6f}, Concatenated: {loss_concatenated:.6f}")
    else:
        logger.info("DPO KL penalty implementations produce equivalent results ✓")
    
    return loss_separate, loss_concatenated


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

    # handle HF checkpoint initialization (same logic as used in train_lm.py)
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

    # Log the forward pass strategy being used
    if config.use_concatenated_forward:
        logger.info("Using concatenated forward passes for DPO (more efficient for FSDP)")
    else:
        logger.info("Using separate forward passes for DPO (simpler, less efficient)")

    # Log whether KL penalty is being used
    if config.kl_penalty_weight > 0.0:
        logger.info(f"Using DPO with KL penalty (weight: {config.kl_penalty_weight})")
        # Define the DPO loss function with KL penalty
        loss_function = functools.partial(
            compute_dpo_loss_with_kl_penalty, 
            beta=config.beta, 
            reference_free=config.reference_free,
            kl_penalty_weight=config.kl_penalty_weight,
            use_concatenated_forward=config.use_concatenated_forward
        )
    else:
        logger.info("Using standard DPO loss (no KL penalty)")
        # Define the standard DPO loss function
        loss_function = functools.partial(
            compute_dpo_loss, 
            beta=config.beta, 
            reference_free=config.reference_free,
            use_concatenated_forward=config.use_concatenated_forward
        )

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

        # actually run training
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
