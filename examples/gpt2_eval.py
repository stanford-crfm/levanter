import logging
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import jmp
import numpy
import tqdm
from equinox import filter_vmap
from transformers import GPT2Tokenizer

import haliax as hax
import levanter
from haliax import Axis
from haliax.nn import cross_entropy_loss
from haliax.partitioning import named_pjit, round_axis_for_partitioning
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import load_hf_gpt2_checkpoint
from levanter.config import TrainerConfig
from levanter.data.sharded import LocalBatchDataset
from levanter.data.text import CachedLMDatasetConfig, TokenSeqDataset
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
    eval_on_train: bool = False


@levanter.config.main()
def main(config: EvalGpt2Config):
    config.trainer.initialize(config)
    tokenizer: GPT2Tokenizer = config.data.the_tokenizer

    EvalBatch = Axis("batch", config.trainer.eval_batch_size)

    if config.eval_on_train:
        raw_dataset = TokenSeqDataset(config.data.build_or_load_document_cache("train"), config.model.seq_len)
    else:
        raw_dataset = TokenSeqDataset(config.data.build_or_load_document_cache("validation"), config.model.seq_len)

    eval_dataset = LocalBatchDataset(raw_dataset, config.trainer.device_mesh, EvalBatch)

    # some axes we use outside the model proper
    SeqLen = config.model.SeqLen

    compute_axis_mapping = config.trainer.compute_axis_mapping
    parameter_axis_mapping = config.trainer.parameter_axis_mapping

    with config.trainer.device_mesh, hax.axis_mapping(parameter_axis_mapping):
        key = jax.random.PRNGKey(0)

        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), compute_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        mp: jmp.Policy = config.trainer.mp

        # don't want to compute the mask w.r.t. the final token
        loss_mask = 1 - hax.nn.one_hot(-1, SeqLen, dtype=jnp.float32)  # one everywhere except the last token

        def compute_loss(model: Gpt2LMHeadModel, input_ids):
            with hax.axis_mapping(compute_axis_mapping):
                model = mp.cast_to_compute(model)
                attn_mask = hax.nn.attention.causal_mask(config.model.SeqLen, config.model.KeySeqLen)
                pred_y = model(input_ids, inference=True, key=None, attn_mask=attn_mask)
                pred_y = mp.cast_to_output(pred_y)

                # need to roll the target tokens back by one so that each token is predicting the next token
                target_y = hax.roll(input_ids, -1, SeqLen)
                target_y = hax.nn.one_hot(target_y, Vocab, dtype=pred_y.dtype)

                loss = cross_entropy_loss(pred_y, Vocab, target_y)
                loss = hax.mean(loss, where=loss_mask)

            return loss.scalar()

        def mean_loss(model: Gpt2LMHeadModel, input_ids):
            # None here means the first argument (the model) is not vectorized but instead broadcasted
            input_ids = hax.named(input_ids, (EvalBatch, SeqLen))
            return hax.mean(hax.vmap(compute_loss, EvalBatch)(model, input_ids))
            compute_loss_vmap = filter_vmap(compute_loss, args=(None,))
            input_ids = hax.named(input_ids, (EvalBatch, SeqLen))
            return jnp.mean(compute_loss_vmap(model, input_ids))

        compute_loss_pjit = named_pjit(
            mean_loss,
            in_axis_resources=parameter_axis_mapping,
            out_axis_resources=compute_axis_mapping,
            axis_resources=compute_axis_mapping,
        )

        total = config.trainer.max_eval_batches

        def evaluate(model):

            # standard evaluation loop
            loss = 0.0
            n = 0

            with hax.axis_mapping(compute_axis_mapping):
                for batch in tqdm.tqdm(eval_dataset, total=total, desc="Evaluating"):
                    loss += compute_loss_pjit(model, batch).item()
                    n += 1
                    if total is not None and n >= total:
                        break

            return loss / n

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
            # hf_model = named_pjit(lambda m: m, donate_argnums=(0,))(hf_model)
            loss = evaluate(hf_model)

            print("Loss from HF model: ", loss)

            if config.compare_torch:
                import torch
                from transformers import GPT2LMHeadModel as TorchGPT2LMHeadModel

                torch_model: TorchGPT2LMHeadModel = TorchGPT2LMHeadModel.from_pretrained(config.hf_checkpoint)
                torch_model.eval()
                torch_model.to("cpu")

                loss = 0.0
                n = 0
                for batch in tqdm.tqdm(eval_dataset, total=total, desc="Evaluating (torch)"):
                    torch_ids = torch.from_numpy(numpy.array(batch)).to(torch.int64)
                    with torch.no_grad():
                        loss += torch_model(input_ids=torch_ids, labels=torch_ids)[0].item()
                    n += 1
                    if total is not None and n >= total:
                        break

                print("Loss from Torch model: ", loss / n)


if __name__ == "__main__":
    main()
