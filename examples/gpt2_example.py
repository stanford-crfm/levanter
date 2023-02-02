import logging
from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
import jmp
import pyrallis
from jax.interpreters.pxla import PartitionSpec
from transformers import GPT2Tokenizer

import haliax as hax
import haliax.random
import wandb
from haliax import Axis
from haliax.partitioning import ResourceAxis, named_pjit, round_axis_for_partitioning
from levanter import callbacks
from levanter.config import TrainerConfig
from levanter.data.sharded import GlobalBatchDataset
from levanter.data.text import LMDatasetConfig, TokenSeqDataset
from levanter.grad_accum import accumulate_gradients_sharded
from levanter.jax_utils import global_key_array, parameter_count
from levanter.logging import capture_time, log_time_to_wandb
from levanter.modeling_utils import cross_entropy_loss_and_log_normalizers
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel
from levanter.trainer_hooks import StepInfo, TrainerHooks
from py_utils import non_caching_cycle


logger = logging.getLogger(__name__)


# cf https://github.com/google-research/language/blob/aa58066bec83d30de6c8f9123f0af7b81db3aeba/language/mentionmemory/training/trainer.py


@dataclass
class TrainGpt2Config:
    data: LMDatasetConfig = LMDatasetConfig()
    trainer: TrainerConfig = TrainerConfig()
    model: Gpt2Config = Gpt2Config()

    log_z_regularization: float = 0.0
    fcm_prob: float = 0.0  # forgetful context masking prob. recommended 0.15


@pyrallis.wrap()
def main(config: TrainGpt2Config):
    config.trainer.initialize(config)

    tokenizer: GPT2Tokenizer = config.data.the_tokenizer

    # some axes we need
    Batch = Axis("batch", config.trainer.train_batch_size)
    EvalBatch = Axis("batch", config.trainer.eval_batch_size)
    SeqLen = config.model.SeqLen
    KeySeqLen = config.model.KeySeqLen

    # We have two axis_mappings: one for storing the model and optimizer states, and one for compute
    # This allows Zero-3-style parameter sharding, where we shard the parameters and optimizer state across the mesh
    compute_axis_mapping = config.trainer.compute_axis_mapping
    parameter_axis_mapping = config.trainer.parameter_axis_mapping

    dataset = GlobalBatchDataset(
        TokenSeqDataset(config.data.build_or_load_document_cache("train"), config.model.seq_len),
        config.trainer.device_mesh,
        Batch,
        compute_axis_mapping,
    )

    eval_dataset = GlobalBatchDataset(
        TokenSeqDataset(config.data.build_or_load_document_cache("validation"), config.model.seq_len),
        config.trainer.device_mesh,
        EvalBatch,
        compute_axis_mapping,
    )

    with config.trainer.device_mesh as mesh:
        # randomness in jax is tightly controlled by "keys" which are the states of the random number generators
        # this makes deterministic training pretty easy
        seed = config.trainer.seed
        data_key, loader_key, model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 4)

        # to do partitioning, our dimensions have to be divisible by the size of the physical axes they're mapped to
        # For most things, we just insist you specify the config right, but tokenizers often have strange numbers of
        # tokens: gpt-2 has 50257, for example. So we round up.
        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), compute_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        # Mixed Precision: We use the "jmp" library to handle mixed precision training. It basically has three dtypes:
        # 1) compute (typically bfloat16)
        # 2) parameter (typically float32)
        # 3) output (sometimes float32)
        # I like to think of these as "semantic" dtypes: compute is the dtype we do most of our math in, parameter is
        # the dtype we store our parameters in, and output is the dtype we use for loss calculations.
        mp: jmp.Policy = config.trainer.mp

        # initialize the model
        # This function
        # 1) initializes model weights
        # 2) ensures all model weights are the right dtype
        # 3) ensures the model is partitioned across the mesh according to the parameter_axis_mapping
        @named_pjit(axis_resources=parameter_axis_mapping)
        def init_model():
            model = Gpt2LMHeadModel(Vocab, config.model, key=model_key)
            return mp.cast_to_param(model)

        model = init_model()

        wandb.summary["parameter_count"] = parameter_count(model)

        # initialize the optimizer
        # This is basically the same as the model.
        optimizer = config.trainer.optimizer()
        opt_state = named_pjit(optimizer.init, axis_resources=parameter_axis_mapping)(model)

        # masks for attention and loss
        def attention_mask(inference, fcm_key):
            causal_mask = hax.nn.attention.causal_mask(SeqLen, KeySeqLen)

            # forgetful causal masking
            if not inference and config.fcm_prob > 0:
                fcm_mask = hax.nn.attention.forgetful_causal_mask(KeySeqLen, config.fcm_prob, key=fcm_key)
                causal_mask = causal_mask & fcm_mask
            return causal_mask

        # don't want to compute the loss w.r.t. the final token
        loss_mask = 1 - hax.nn.one_hot(-1, SeqLen, dtype=jnp.float32)  # one everywhere except the last token

        # loss function: this computes the loss with respect to a single example
        def compute_loss(model: Gpt2LMHeadModel, input_ids, attn_mask, key, inference):
            with hax.axis_mapping(compute_axis_mapping):
                model = mp.cast_to_compute(model)

                pred_y = model(input_ids, attn_mask, key=key, inference=inference)
                pred_y = mp.cast_to_output(pred_y)

                # need to roll the target tokens back by one so that each token is predicting the next token
                target_y = haliax.roll(input_ids, -1, SeqLen)
                target_y = haliax.nn.one_hot(target_y, Vocab, dtype=pred_y.dtype)

                loss, log_normalizers = cross_entropy_loss_and_log_normalizers(pred_y, Vocab, target_y)
                loss = hax.mean(loss, where=loss_mask)

                if not inference and config.log_z_regularization > 0:
                    logz_mse = hax.mean((log_normalizers**2))
                    loss += config.log_z_regularization * logz_mse

                return loss.scalar()

        def train_batch_loss(model, input_ids, attn_mask, key):
            return hax.mean(hax.vmap(compute_loss, Batch)(model, input_ids, attn_mask, key, inference=False))

        # training loop
        # donate args to conserve memory
        @named_pjit(axis_resources=parameter_axis_mapping, donate_args=True)
        def train_step(model, opt_state, input_ids, keys):

            attn_mask = hax.vmap(attention_mask, Batch)(False, keys)
            attn_mask = hax.auto_sharded(attn_mask)

            loss, grads = accumulate_gradients_sharded(
                eqx.filter_value_and_grad(train_batch_loss),
                Batch,
                model,
                input_ids,
                attn_mask,
                keys,
                per_device_parallelism=config.trainer.per_device_parallelism,
                parameter_axis_mapping=parameter_axis_mapping,
            )

            # distribute gradients across the mesh and apply them
            updates, opt_state = optimizer.update(grads, opt_state, params=model)
            model = eqx.apply_updates(model, updates)

            return loss, model, opt_state

        # evaluation loss and loop

        @named_pjit(axis_resources=compute_axis_mapping)
        def eval_loss(model, input_ids):
            input_ids = hax.named(input_ids, (EvalBatch, SeqLen))
            # just use causal mask for evaluation
            mask = hax.nn.attention.causal_mask(SeqLen, KeySeqLen)
            return compute_loss(model, input_ids, mask, None, True)

        # Set up evaluation
        def evaluate_step(info: StepInfo):
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
            if wandb.run is not None:
                wandb.log({"eval/loss": loss}, step=info.step)

            return loss

        # boilerplate hooks and such
        engine = TrainerHooks()
        engine.add_hook(callbacks.pbar_logger(total=config.trainer.num_train_steps), every=1)
        engine.add_hook(callbacks.log_to_wandb, every=1)
        engine.add_hook(
            callbacks.log_performance_stats(config.model.seq_len, config.trainer.train_batch_size), every=1
        )
        # engine.add_hook(evaluate_step, every=config.trainer.steps_per_eval)
        engine.add_hook(callbacks.wandb_xla_logger(config.trainer.wandb), every=config.trainer.steps_per_eval)
        checkpointer = config.trainer.checkpointer.create(config.trainer.run_name)
        engine.add_hook(checkpointer.on_step, every=1)  # checkpointer manages its own frequency

        # data loader
        iter_data = non_caching_cycle(dataset)

        # load the last checkpoint and resume if we want
        resume_step = None
        if config.trainer.load_last_checkpoint:
            checkpoint = checkpointer.load_checkpoint(
                model,
                (opt_state, training_key),
                config.trainer.load_checkpoint_path,
            )

            if checkpoint is not None:
                model, (opt_state, training_key), resume_step = checkpoint
                assert training_key.shape == jrandom.PRNGKey(0).shape
            elif config.trainer.load_checkpoint_path:
                raise ValueError("No checkpoint found")
            else:
                logger.info("No checkpoint found. Starting from scratch")

        if resume_step is not None:
            # step is after the batch, so we need to seek to step
            # TODO: iter_data.seek(resume_step +1)
            import tqdm

            for _ in tqdm.tqdm(range(resume_step + 1), desc="seeking data for resume"):
                next(iter_data)
            resume_step = resume_step + 1
        else:
            resume_step = 0

        # finally, run the training loop
        for step in range(resume_step, config.trainer.num_train_steps):
            with capture_time() as step_time:
                with log_time_to_wandb("throughput/loading_time", step=step):
                    input_ids = next(iter_data)
                    input_ids = hax.named(input_ids, (Batch, SeqLen))
                    my_key, training_key = jrandom.split(training_key, 2)
                    example_keys = global_key_array(
                        my_key, config.trainer.train_batch_size, mesh, PartitionSpec(ResourceAxis.DATA)
                    )

                step_loss, model, opt_state = train_step(model, opt_state, input_ids, example_keys)
                step_loss = step_loss.item()

            with log_time_to_wandb("throughput/hook_time", step=step):
                engine.run_hooks(StepInfo(step, model, opt_state, step_loss, training_key, step_duration=step_time()))

        last_step = StepInfo(
            config.trainer.num_train_steps,
            model,
            opt_state,
            step_loss,
            training_key,
            step_duration=step_time(),
        )

        evaluate_step(last_step)
        checkpointer.on_step(last_step, force=True)


if __name__ == "__main__":
    main()
