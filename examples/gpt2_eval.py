import logging
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import jmp
import pyrallis
from transformers import GPT2Tokenizer

import haliax as hax
from haliax import Axis
from haliax.partitioning import named_pjit, round_axis_for_partitioning
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import load_hf_gpt2_checkpoint
from levanter.config import TrainerConfig
from levanter.data.sharded import GlobalBatchDataset
from levanter.data.text import CachedLMDatasetConfig, TokenSeqDataset
from levanter.modeling_utils import cross_entropy_loss
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel


logger = logging.getLogger(__name__)


@dataclass
class EvalGpt2Config:
    checkpoint_path: Optional[str] = None
    hf_checkpoint: Optional[str] = None
    hf_revision: Optional[str] = None
    trainer: TrainerConfig = TrainerConfig()
    data: CachedLMDatasetConfig = CachedLMDatasetConfig()
    model: Gpt2Config = Gpt2Config()

    compare_torch: bool = False


@pyrallis.wrap()
def main(config: EvalGpt2Config):
    config.trainer.initialize(config)
    tokenizer: GPT2Tokenizer = config.data.the_tokenizer

    compute_axis_mapping = config.trainer.compute_axis_mapping
    parameter_axis_mapping = config.trainer.parameter_axis_mapping

    EvalBatch = Axis("batch", config.trainer.eval_batch_size)

    eval_dataset = GlobalBatchDataset(
        TokenSeqDataset(config.data.build_or_load_document_cache("validation"), config.model.seq_len),
        config.trainer.device_mesh,
        EvalBatch,
        compute_axis_mapping,
    )

    # some axes we use outside the model proper
    SeqLen = config.model.SeqLen
    KeySeqLen = config.model.KeySeqLen

    with config.trainer.device_mesh:
        key = jax.random.PRNGKey(0)

        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), compute_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        mp: jmp.Policy = config.trainer.mp

        # don't want to compute the mask w.r.t. the final token
        loss_mask = 1 - hax.nn.one_hot(-1, SeqLen, dtype=jnp.float32)  # one everywhere except the last token

        def compute_loss(model: Gpt2LMHeadModel, input_ids, attn_mask):
            with hax.axis_mapping(compute_axis_mapping):
                model = mp.cast_to_compute(model)

                pred_y = model(input_ids, attn_mask, inference=True, key=None)
                pred_y = mp.cast_to_output(pred_y)

                # need to roll the target tokens back by one so that each token is predicting the next token
                target_y = hax.roll(input_ids, -1, SeqLen)
                target_y = hax.nn.one_hot(target_y, Vocab, dtype=pred_y.dtype)

                loss = cross_entropy_loss(pred_y, Vocab, target_y)
                loss = hax.mean(loss, where=loss_mask)

                return loss.scalar()

        @named_pjit(axis_resources=compute_axis_mapping)
        def eval_loss(model, input_ids):
            input_ids = hax.named(input_ids, (EvalBatch, SeqLen))
            # just use causal mask for evaluation
            mask = hax.nn.attention.causal_mask(SeqLen, KeySeqLen)
            return compute_loss(model, input_ids, mask)

        def evaluate(model):
            with hax.axis_mapping(compute_axis_mapping):
                # standard evaluation loop
                loss = 0.0
                n = 0

                for batch in eval_dataset:
                    loss += eval_loss(model, batch).item()
                    n += 1

                if n > 0:
                    loss /= n

            logger.info(f"validation loss: {loss:.3f}")
            return loss

        # initialize the model
        if config.checkpoint_path is not None:

            @named_pjit(axis_resources=parameter_axis_mapping)
            def init_model():
                model = Gpt2LMHeadModel(Vocab, config.model, key=key)
                model = config.trainer.mp.cast_to_param(model)
                return model

            model = init_model()

            model, _, _ = load_checkpoint(model, None, config.checkpoint_path)
            loss = evaluate(model)

            del model
            print("Loss from Levanter model: ", loss)

        if config.hf_checkpoint is not None:
            # load the huggingface model
            with jax.default_device(jax.devices("cpu")[0]):
                hf_model = load_hf_gpt2_checkpoint(config.hf_checkpoint, revision=config.hf_revision)
            jax.lib.xla_bridge.get_backend().defragment()
            hf_model = named_pjit(lambda m: m, donate_argnums=(0,))(hf_model)
            jax.lib.xla_bridge.get_backend().defragment()
            loss = evaluate(hf_model)

            print("Loss from HF model: ", loss)

            del hf_model

            if config.compare_torch:
                import torch
                from transformers import GPT2LMHeadModel

                torch_model = GPT2LMHeadModel.from_pretrained(config.hf_checkpoint)

                def torch_compute_loss(model, input_ids):
                    input_ids = torch.from_numpy(input_ids)
                    loss = model(input_ids, labels=input_ids, return_dict=True).loss

                    return loss.item()

                total_loss = 0.0
                n = 0
                for batch in eval_dataset:
                    total_loss += torch_compute_loss(torch_model, batch)
                    n += 1

                total_loss /= n

                print("Loss from Torch model: ", total_loss)


if __name__ == "__main__":
    main()
