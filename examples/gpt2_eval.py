import itertools
from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import optax
import pyrallis
from jax.experimental.pjit import pjit
from jax.interpreters.pxla import PartitionSpec
from transformers import GPT2Tokenizer

from haliax import Axis
from levanter import callbacks
from levanter.axis_names import ResourceAxis, infer_resource_partitions
from levanter.checkpoint import load_checkpoint
from levanter.compat.torch_checkpoints import load_hf_gpt2_checkpoint
from levanter.config import TrainerConfig
from levanter.data import CachedLMDatasetConfig
from levanter.data.sharded import ShardedIndexedDataset
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel
from levanter.trainer_hooks import StepInfo


@dataclass
class EvalGpt2Config:
    checkpoint_path: str
    hf_checkpoint: str = "gpt2-medium"
    hf_revision: Optional[str] = None
    trainer: TrainerConfig = TrainerConfig()
    data: CachedLMDatasetConfig = CachedLMDatasetConfig()
    model: Gpt2Config = Gpt2Config()

    dtype: jnp.dtype = jnp.bfloat16


@pyrallis.wrap()
def main(config: EvalGpt2Config):
    tokenizer: GPT2Tokenizer = config.data.the_tokenizer

    # first load our checkpoint
    key = jax.random.PRNGKey(0)
    vocab = Axis("vocab", len(tokenizer))

    with config.trainer.device_mesh:
        eval_dataset = ShardedIndexedDataset(
            config.data.build_or_load_document_cache("validation"),
            config.trainer.eval_mesh_info,
            config.model.seq_len,
            microbatched=False,
        )

        resource_partitions = {
            "hidden": ResourceAxis.MODEL,
            # "mlp": ResourceAxis.MODEL,
            "batch": ResourceAxis.DATA,
        }

        # initialize the model
        model = Gpt2LMHeadModel(vocab, config.model, key=key)
        model_resources = infer_resource_partitions(model, resource_partitions)
        model = jax.tree_map(lambda array: array.astype(config.dtype), model)

        model, _, _ = load_checkpoint(model, None, config.checkpoint_path)

        def eval_dataloader():
            # TODO: only do one pass
            for batch in itertools.islice(eval_dataset, 50):
                yield (batch,)

        def compute_loss(model: Gpt2LMHeadModel, input_ids, key):
            pred_y = model(input_ids, key)
            token_loss = jnp.mean(
                optax.softmax_cross_entropy(
                    pred_y[:-1],
                    jax.nn.one_hot(input_ids[1:], num_classes=tokenizer.vocab_size),
                )
            )

            return token_loss

        compute_loss_vmap = jax.vmap(compute_loss, in_axes=[None, 0, 0], spmd_axis_name=ResourceAxis.DATA)

        def mean_loss(model: Gpt2LMHeadModel, input_ids, key):
            return jnp.mean(compute_loss_vmap(model, input_ids, key))

        compute_loss_pjit = pjit(
            partial(mean_loss, key=None),
            in_axis_resources=(model_resources, PartitionSpec(ResourceAxis.DATA, None)),
            out_axis_resources=None,
        )

        evaluate = callbacks.compute_validation_loss(compute_loss_pjit, eval_dataloader)

        loss = evaluate(StepInfo(0, model, None, 0.0, None, 0.0))

        del model
        print("Loss from Levanter model: ", loss)

        # load the huggingface model
        hf_model = load_hf_gpt2_checkpoint(config.hf_checkpoint, revision=config.hf_revision)
        hf_model = jax.tree_map(lambda array: array.astype(config.dtype), hf_model)

        evaluate = callbacks.compute_validation_loss(compute_loss_pjit, eval_dataloader)

        loss = evaluate(StepInfo(0, hf_model, None, 0.0, None, 0.0))

        print("Loss from HF model: ", loss)


if __name__ == "__main__":
    main()
