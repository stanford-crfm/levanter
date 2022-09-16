import itertools
from dataclasses import dataclass

import jax
import jax.nn as jnn
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
import pyrallis
from jax.experimental.pjit import pjit
from jax.interpreters.pxla import PartitionSpec
from tqdm import tqdm
from transformers import GPT2Tokenizer

from haliax import Axis
from haliax.partitioning import ResourceAxis, axis_mapping, infer_resource_partitions
from levanter.checkpoint import load_checkpoint
from levanter.config import TrainerConfig
from levanter.data import CachedLMDatasetConfig
from levanter.data.sharded import ShardedIndexedDataset
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel


@dataclass
class EvalGpt2Config:
    checkpoint_path: str
    trainer: TrainerConfig = TrainerConfig()
    data: CachedLMDatasetConfig = CachedLMDatasetConfig()
    model: Gpt2Config = Gpt2Config()


@pyrallis.wrap()
def main(config: EvalGpt2Config):
    tokenizer: GPT2Tokenizer = config.data.the_tokenizer

    # first load our checkpoint
    key = jax.random.PRNGKey(0)
    Vocab = Axis("vocab", len(tokenizer))

    resource_partitions = {
        "embed": ResourceAxis.MODEL,
        # "mlp": ResourceAxis.MODEL,
        "batch": ResourceAxis.DATA,
    }

    with config.trainer.device_mesh, axis_mapping(resource_partitions):
        eval_dataset = ShardedIndexedDataset(
            config.data.build_or_load_document_cache("validation"),
            config.trainer.eval_mesh_info,
            config.model.seq_len,
            microbatched=False,
        )

        # initialize the model
        model = Gpt2LMHeadModel(Vocab, config.model, key=key)
        model_resources = infer_resource_partitions(model)
        model = config.trainer.mp.cast_to_param(model)

        model, _, _ = load_checkpoint(model, None, config.checkpoint_path)

        def eval_dataloader():
            # TODO: only do one pass
            for batch in itertools.islice(eval_dataset, 50):
                yield (batch,)

        def compute_logz(model: Gpt2LMHeadModel, input_ids):
            # [seq_len, vocab]
            pred_y = model(input_ids, inference=True, key=key)
            pred_y = jnn.logsumexp(pred_y, axis=-1)  # seq_len

            return pred_y

        # [batch, seq_len]
        compute_logz_vmap = jax.vmap(compute_logz, in_axes=[None, 0], spmd_axis_name=ResourceAxis.DATA)

        compute_logz_pjit = pjit(
            compute_logz_vmap,
            in_axis_resources=(model_resources, PartitionSpec(ResourceAxis.DATA, None)),
            out_axis_resources=None,
        )

        pbar = tqdm(eval_dataloader(), desc="eval", position=1, leave=False)
        all_logz = []
        for batch in pbar:
            logz = compute_logz_pjit(model, *batch)
            all_logz.append(logz)

        all_logz = jnp.concatenate(all_logz, axis=0)
        all_logz = onp.array(all_logz)
        # print a histogram of the logz
        plt.hist(all_logz, bins=50)
        plt.savefig("logz.png")


if __name__ == "__main__":
    main()
