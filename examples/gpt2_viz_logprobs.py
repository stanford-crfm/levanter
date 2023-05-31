import logging
from dataclasses import dataclass

import jax
import jmp
from transformers import GPT2Tokenizer

import haliax as hax
import levanter
from haliax import Axis
from haliax.partitioning import named_jit, round_axis_for_partitioning
from levanter import callbacks
from levanter.checkpoint import load_checkpoint
from levanter.config import TrainerConfig
from levanter.data.sharded import LocalBatchDataset
from levanter.data.text import LMDatasetConfig, TokenSeqDataset
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel
from levanter.models.loss import next_token_loss
from levanter.trainer_hooks import StepInfo


logger = logging.getLogger(__name__)


@dataclass
class EvalGpt2Config:
    checkpoint_path: str
    output_dir: str = "logprob_viz"
    trainer: TrainerConfig = TrainerConfig()
    data: LMDatasetConfig = LMDatasetConfig()
    model: Gpt2Config = Gpt2Config()

    num_docs: int = 256


@levanter.config.main()
def main(config: EvalGpt2Config):
    config.trainer.initialize(config)
    tokenizer: GPT2Tokenizer = config.data.the_tokenizer

    EvalBatch = Axis("batch", config.trainer.eval_batch_size)

    # some axes we use outside the model proper
    Pos = config.model.Pos
    KeyPos = config.model.KeyPos

    eval_dataset = LocalBatchDataset(
        TokenSeqDataset(config.data.build_or_load_cache("validation"), Pos),
        config.trainer.device_mesh,
        EvalBatch,
    )

    # some axes we use outside the model proper
    Pos = config.model.Pos
    KeyPos = config.model.KeyPos

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

        @named_jit(axis_resources=parameter_axis_mapping)
        def compute_log_probs(model, input_ids):
            attn_mask = hax.nn.attention.causal_mask(Pos, KeyPos)
            attn_mask = hax.auto_sharded(attn_mask)

            with hax.axis_mapping(compute_axis_mapping):
                model = mp.cast_to_compute(model)

                pred_y = model(input_ids, attn_mask, inference=True, key=None)
                pred_y = mp.cast_to_output(pred_y)

                return next_token_loss(Pos, Vocab, pred_y, input_ids).scalar()

        # initialize the model
        @named_jit(axis_resources=parameter_axis_mapping)
        def init_model():
            model = Gpt2LMHeadModel.init(Vocab, config.model, key=key)
            model = config.trainer.mp.cast_to_param(model)
            return model

        model = init_model()

        ckpt = load_checkpoint(model, None, config.checkpoint_path)

        assert ckpt is not None
        model, _, _ = ckpt

        cb = callbacks.compute_and_visualize_log_probs(
            eval_dataset, tokenizer, compute_log_probs, config.output_dir, max_docs=config.num_docs
        )
        cb(StepInfo(model=model, step=0, opt_state=None, loss=0.0, step_duration=0.0, next_key=0.0))

        del model


if __name__ == "__main__":
    main()
