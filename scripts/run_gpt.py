# import functools
import logging
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from tqdm import tqdm

import haliax as hax
from haliax import Axis
from haliax.partitioning import named_jit, round_axis_for_partitioning

import levanter
from levanter.data.text import LMDatasetConfig
from levanter.models.gpt2 import Gpt2Config
from levanter.models.lm_model import LmConfig
from levanter.trainer import TrainerConfig


# from jax_automin.interpreter import automin_function


logger = logging.getLogger(__name__)


@dataclass
class EvalLmConfig:
    trainer: TrainerConfig = TrainerConfig()
    data: LMDatasetConfig = LMDatasetConfig()
    model: LmConfig = Gpt2Config()


def main(config: EvalLmConfig):
    config.trainer.initialize(config)
    tokenizer = config.data.the_tokenizer

    Batch = Axis("batch", config.trainer.eval_batch_size)
    Pos = config.model.Pos
    Embed = config.model.Embed

    compute_axis_mapping = config.trainer.compute_axis_mapping
    parameter_axis_mapping = config.trainer.parameter_axis_mapping

    with config.trainer.device_mesh, hax.axis_mapping(parameter_axis_mapping):
        key = jax.random.PRNGKey(0)

        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), compute_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        def model(tokens, key, inference):
            embed = jnp.take(jnp.ones((Vocab.size, Embed.size)), tokens.array, axis=0)
            # dumb fake gpt2 attn
            for i in range(0, config.model.num_layers):  # type: ignore
                attn = jnp.einsum("...ld,...kd->...lk", embed, embed)

                if not inference and config.model.attn_pdrop > 0.0:  # type: ignore
                    key, subkey = jax.random.split(key)
                    dout = jax.random.bernoulli(subkey, config.model.attn_pdrop, shape=attn.shape)  # type: ignore
                    attn = jnp.where(dout, jnp.zeros_like(attn), attn)  # type: ignore

                attn = jax.nn.softmax(attn, axis=-1)
                embed = jnp.einsum("...ld,...lk->...kd", attn, embed)

            out = jnp.einsum("...ld,...kd->...lk", embed, jnp.ones((Vocab.size, Embed.size)))

            return out

        def compute_loss(example, key, inference):
            with hax.axis_mapping(compute_axis_mapping):
                pred_y = model(example, key=key, inference=inference)
                return jnp.sum(pred_y)

        def compute_loss_vmap(examples, key, inference):
            key = jax.random.split(key, Batch.size)
            per_ex_loss = hax.vmap(compute_loss, "batch")(examples, key, inference=inference)
            return hax.mean(per_ex_loss)

        compute_loss_pjit = named_jit(compute_loss_vmap, axis_resources=parameter_axis_mapping)

        total_loss = 0.0
        total_time = 0.0
        n = 0

        batch = jnp.ones((Batch.size, Pos.size), dtype=jnp.int32)
        batch = hax.named(batch, (Batch, Pos))
        pbar = tqdm(desc="eval")

        for n in range(100):
            this_key, key = jax.random.split(key)
            time_in = time.time()
            loss = compute_loss_pjit(batch, this_key, inference=False)

            total_loss += loss.item()
            time_out = time.time()
            if n > 0:
                total_time += time_out - time_in
            n += 1

            if n > 1:
                pbar.set_postfix(loss=total_loss / n, time=total_time / (n - 1))

        #
        #
        # pbar = tqdm(eval_loader, desc="eval", position=1, leave=False)
        # for batch in pbar:
        #     this_key, key = jax.random.split(key)
        #     time_in = time.time()
        #     if config.loss_only:
        #         loss = compute_loss_pjit(batch, this_key, inference=False)
        #     else:
        #         loss, grad_norm = compute_loss_and_grad_norm_pjit(batch, this_key, inference=False)
        #         total_grad_norm += grad_norm
        #
        #     total_loss += loss.item()
        #     time_out = time.time()
        #     if n > 0:
        #         total_time += time_out - time_in
        #     n += 1
        #
        #     if n > 1:
        #         pbar.set_postfix(loss=total_loss / n, time=total_time / (n - 1))
        #
        #     if total and n >= total:
        #         break

        logger.info(f"eval loss: {total_loss / n:.3f}")
        logger.info(f"eval time: {total_time / (n-1):.3f}")
        print(f"eval loss: {total_loss / n:.3f}")
        print(f"eval time: {total_time / (n-1):.3f}")


if __name__ == "__main__":
    levanter.config.main(main)()
