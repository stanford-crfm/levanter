import itertools
from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import jmp
import optax
import pyrallis
from jax.experimental.pjit import pjit
from jax.interpreters.pxla import PartitionSpec
from transformers import GPT2Tokenizer

from haliax import Axis
from levanter import callbacks
from levanter.axis_names import ResourceAxis, infer_resource_partitions, named_pjit
from levanter.checkpoint import load_checkpoint
from levanter.compat.torch_checkpoints import load_hf_gpt2_checkpoint
from levanter.config import TrainerConfig
from levanter.data import CachedLMDatasetConfig
from levanter.data.sharded import ShardedIndexedDataset
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel
from levanter.trainer_hooks import StepInfo


@dataclass
class EvalGpt2Config:
    checkpoint_path: Optional[str] = None
    hf_checkpoint: Optional[str] = None
    hf_revision: Optional[str] = None
    trainer: TrainerConfig = TrainerConfig()
    data: CachedLMDatasetConfig = CachedLMDatasetConfig()
    model: Gpt2Config = Gpt2Config()


@pyrallis.wrap()
def main(config: EvalGpt2Config):
    tokenizer: GPT2Tokenizer = config.data.the_tokenizer

    # first load our checkpoint
    key = jax.random.PRNGKey(0)
    Vocab = Axis("vocab", len(tokenizer))

    with config.trainer.device_mesh:
        eval_dataset = ShardedIndexedDataset(
            config.data.build_or_load_document_cache("validation"),
            config.trainer.eval_mesh_info,
            config.model.seq_len,
            microbatched=False,
        )

        resource_partitions = {
            "batch": ResourceAxis.DATA,
            # ZERO-3
            # "embed": ResourceAxis.DATA,
            # "vocab": ResourceAxis.MODEL,
            "mlp": ResourceAxis.MODEL,
            # "qkv": ResourceAxis.MODEL,
            # "heads": ResourceAxis.MODEL,
            # "total_head_dim": ResourceAxis.MODEL,
        }

        def compute_loss(model: Gpt2LMHeadModel, input_ids, key):
            pred_y = model(input_ids, inference=True, key=key)
            token_loss = jnp.mean(
                optax.softmax_cross_entropy(pred_y[:-1], jax.nn.one_hot(input_ids[1:], num_classes=Vocab.size))
            )

            return token_loss

        compute_loss_vmap = jax.vmap(compute_loss, in_axes=[None, 0, 0], spmd_axis_name=ResourceAxis.DATA)

        def mean_loss(model: Gpt2LMHeadModel, input_ids, key):
            return jnp.mean(compute_loss_vmap(model, input_ids, key))

        def eval_dataloader():
            # TODO: only do one pass
            for batch in itertools.islice(eval_dataset, 100):
                yield (batch,)

        # initialize the model
        if config.checkpoint_path is not None:
            model = Gpt2LMHeadModel(Vocab, config.model, key=key, mp=config.trainer.mp)
            model_resources = infer_resource_partitions(model, resource_partitions)
            model = config.trainer.mp.cast_to_param(model)

            model, _, _ = load_checkpoint(model, None, config.checkpoint_path)

            compute_loss_pjit = pjit(
                partial(mean_loss, key=None),
                in_axis_resources=(model_resources, PartitionSpec(ResourceAxis.DATA, None)),
                out_axis_resources=None,
            )

            evaluate = callbacks.compute_validation_loss(compute_loss_pjit, eval_dataloader)

            loss = evaluate(StepInfo(0, model, None, 0.0, None, 0.0))

            del model
            print("Loss from Levanter model: ", loss)

        if config.hf_checkpoint is not None:
            # load the huggingface model
            with jax.default_device(jax.devices("cpu")[0]):
                hf_model = load_hf_gpt2_checkpoint(config.hf_checkpoint, revision=config.hf_revision, mp=jmp.get_policy("bfloat16"))
            jax.lib.xla_bridge.get_backend().defragment()
            hf_model = named_pjit(lambda m: m, axis_resources=resource_partitions, donate_argnums=(0,))(hf_model)
            jax.lib.xla_bridge.get_backend().defragment()
            model_resources = infer_resource_partitions(hf_model, resource_partitions)

            compute_loss_pjit = pjit(
                partial(mean_loss, key=None),
                in_axis_resources=(model_resources, PartitionSpec(ResourceAxis.DATA, None)),
                out_axis_resources=None,
            )

            evaluate = callbacks.compute_validation_loss(compute_loss_pjit, eval_dataloader)

            loss = evaluate(StepInfo(0, hf_model, None, 0.0, None, 0.0))

            print("Loss from HF model: ", loss)


if __name__ == "__main__":
    main()
