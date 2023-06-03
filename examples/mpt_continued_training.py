import logging
import os
from dataclasses import dataclass
from typing import Optional

import equinox as eqx
import jax.random as jrandom
import jmp
from jax.interpreters.pxla import PartitionSpec
from transformers import AutoConfig, GPT2Tokenizer

import haliax as hax
import haliax.random
import levanter
import wandb
from haliax import Axis
from haliax.partitioning import ResourceAxis, named_jit
from levanter import callbacks
from levanter.config import TrainerConfig
from levanter.data.sharded import GlobalBatchDataset, LocalBatchDataset
from levanter.data.text import LMDatasetConfig, TokenSeqDataset
from levanter.grad_accum import accumulate_gradients_sharded
from levanter.logging import capture_time, log_time_to_wandb
from levanter.models.gpt2 import Gpt2LMHeadModel
from levanter.models.loss import next_token_loss
from levanter.models.mpt import MptLmHeadModel
from levanter.trainer_hooks import StepInfo, TrainerHooks
from levanter.utils.jax_utils import global_key_array, parameter_count
from levanter.utils.py_utils import non_caching_cycle


logger = logging.getLogger(__name__)


# cf https://github.com/google-research/language/blob/aa58066bec83d30de6c8f9123f0af7b81db3aeba/language/mentionmemory/training/trainer.py


@dataclass
class TrainMptConfig:
    data: LMDatasetConfig = LMDatasetConfig()
    trainer: TrainerConfig = TrainerConfig()
    initialize_from: str = "mosaicml/mpt-7b"

    seq_len: Optional[int] = None

    fcm_prob: float = 0.0  # forgetful context masking prob. recommended 0.15

    hf_save_path: Optional[str] = None
    hf_upload: Optional[str] = None
    hf_save_steps: int = 10000


@levanter.config.main()
def main(config: TrainMptConfig):
    config.trainer.initialize(config)

    tokenizer: GPT2Tokenizer = config.data.the_tokenizer

    # We have two axis_mappings: one for storing the model and optimizer states, and one for compute
    # This allows Zero-3-style parameter sharding, where we shard the parameters and optimizer state across the mesh
    compute_axis_mapping = config.trainer.compute_axis_mapping
    parameter_axis_mapping = config.trainer.parameter_axis_mapping

    with config.trainer.device_mesh as mesh:
        # randomness in jax is tightly controlled by "keys" which are the states of the random number generators
        # this makes deterministic training pretty easy
        seed = config.trainer.seed
        data_key, loader_key, model_key, training_key, resize_key = jrandom.split(jrandom.PRNGKey(seed), 5)

        mpt_config = AutoConfig.from_pretrained(config.initialize_from, trust_remote_code=True)

        vocab_size = len(tokenizer)

        if config.seq_len is not None:
            mpt_config.update({"max_seq_len": config.seq_len})

        model = MptLmHeadModel.from_hf_pretrained(config.initialize_from, axis_mapping=parameter_axis_mapping)

        if vocab_size != model.Vocab.size:
            logger.warn(f"Resizing model vocab from {model.Vocab.size} to {vocab_size}")

            @named_jit(axis_resources=parameter_axis_mapping)
            def resize_axis(model):
                return haliax.tree_util.resize_axis(model, model.Vocab.resize(vocab_size), resize_key)

            model = resize_axis(model)

        model_config = model.config

        # some axes we need
        Batch = Axis("batch", config.trainer.train_batch_size)
        EvalBatch = Axis("batch", config.trainer.eval_batch_size)
        SeqLen = model_config.SeqLen
        KeySeqLen = model_config.KeySeqLen

        dataset = GlobalBatchDataset(
            TokenSeqDataset(config.data.build_or_load_cache("train"), model_config.SeqLen),
            config.trainer.device_mesh,
            Batch,
            compute_axis_mapping,
        )

        eval_dataset = LocalBatchDataset(
            TokenSeqDataset(config.data.build_or_load_cache("validation"), model_config.SeqLen),
            config.trainer.device_mesh,
            EvalBatch,
            compute_axis_mapping,
        )

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
        wandb.summary["parameter_count"] = parameter_count(model)

        # initialize the optimizer
        # This is basically the same as the model.
        optimizer = config.trainer.optimizer()
        opt_state = named_jit(optimizer.init, axis_resources=parameter_axis_mapping)(model)

        # masks for attention and loss
        def attention_mask(inference, fcm_key):
            causal_mask = hax.nn.attention.causal_mask(SeqLen, KeySeqLen)

            # forgetful causal masking
            if not inference and config.fcm_prob > 0:
                fcm_mask = hax.nn.attention.forgetful_causal_mask(KeySeqLen, config.fcm_prob, key=fcm_key)
                causal_mask = causal_mask & fcm_mask
            return causal_mask

        # loss function: this computes the loss with respect to a single example
        def compute_loss(model: Gpt2LMHeadModel, input_ids, attn_mask):
            with hax.axis_mapping(compute_axis_mapping):
                model = mp.cast_to_compute(model)

                pred_y = model(input_ids, attn_mask)
                pred_y = mp.cast_to_output(pred_y)

                return next_token_loss(SeqLen, model.Vocab, pred_y, input_ids)

        def train_batch_loss(model, input_ids, attn_mask, key):
            per_ex_loss = hax.vmap(compute_loss, "batch")(model, input_ids, attn_mask)
            return hax.mean(per_ex_loss, "batch").scalar()

        # training loop
        @named_jit(axis_resources=parameter_axis_mapping, donate_args=True)
        def train_step(model, opt_state, input_ids, keys):
            attn_mask = hax.vmap(attention_mask, Batch)(False, keys)
            attn_mask = hax.auto_sharded(attn_mask)

            grad_loss = eqx.filter_value_and_grad(train_batch_loss)

            loss, grads = accumulate_gradients_sharded(
                grad_loss,
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

        @named_jit(axis_resources=parameter_axis_mapping)
        def eval_loss(model, input_ids):
            mask = hax.nn.attention.causal_mask(SeqLen, KeySeqLen)
            return hax.mean(compute_loss(model, input_ids, mask))

        # Set up evaluation
        def evaluate_step(info: StepInfo):
            with hax.axis_mapping(compute_axis_mapping):
                # standard evaluation loop
                loss = 0.0
                n = 0

                for batch in eval_dataset:
                    this_loss = eval_loss(model, batch)
                    loss += this_loss.item()
                    n += 1
                    if config.trainer.max_eval_batches is not None and n >= config.trainer.max_eval_batches:
                        break

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
        engine.add_hook(callbacks.log_performance_stats(SeqLen.size, config.trainer.train_batch_size), every=1)
        engine.add_hook(evaluate_step, every=config.trainer.steps_per_eval)
        engine.add_hook(callbacks.wandb_xla_logger(config.trainer.wandb), every=config.trainer.steps_per_eval)
        # engine.add_hook(callbacks.log_memory_usage(), every=1)
        checkpointer = config.trainer.checkpointer.create(config.trainer.run_name)
        engine.add_hook(checkpointer.on_step, every=1)  # checkpointer manages its own frequency
        if config.hf_save_path is not None:
            full_save_path = os.path.join(config.hf_save_path, config.trainer.run_name)
            from levanter.compat.hf_checkpoints import save_hf_gpt2_checkpoint_callback

            engine.add_hook(save_hf_gpt2_checkpoint_callback(full_save_path), every=config.hf_save_steps)

        # visualize log probs
        @named_jit(axis_resources=parameter_axis_mapping)
        def compute_log_probs(model, input_ids):
            attn_mask = hax.vmap(attention_mask, EvalBatch)(True, None)
            attn_mask = hax.auto_sharded(attn_mask)

            with hax.axis_mapping(compute_axis_mapping):
                model = mp.cast_to_compute(model)

                pred_y = model(input_ids, attn_mask, inference=True, key=None)
                pred_y = mp.cast_to_output(pred_y)
                loss = next_token_loss(SeqLen, Vocab, pred_y, input_ids, reduction=None)
                logprobs = -loss
                # roll forward to get the loss for each predicted token
                logprobs = haliax.roll(logprobs, 1, SeqLen)
                return logprobs.rearrange(("batch", SeqLen)).array

        engine.add_hook(
            callbacks.compute_and_visualize_log_probs(
                eval_dataset, tokenizer, compute_log_probs, f"{config.trainer.run_dir}/log_probs"
            ),
            every=config.trainer.steps_per_eval,
        )

        # data loader
        iter_data = non_caching_cycle(dataset)

        # load the last checkpoint and resume if we want
        model, (opt_state, training_key), resume_step = config.trainer.maybe_load_checkpoint(
            model, (opt_state, training_key)
        )

        if resume_step is not None:
            # step is after the batch, so we need to seek to step
            # TODO: implement iter_data.seek(resume_step +1)
            import tqdm

            for _ in tqdm.tqdm(range(resume_step + 1), desc="seeking data for resume"):
                next(iter_data)
            initial_step = resume_step + 1
        else:
            initial_step = 0

        # assign these here in case num_train_steps == 0
        step_loss = 0.0
        step_time = lambda: 0.0  # noqa: E731

        # finally, run the training loop
        for step in range(initial_step, config.trainer.num_train_steps):
            if step < 10 or step % 50 == 0:
                print(f"step {step}/{config.trainer.num_train_steps}")
            with capture_time() as step_time:
                with log_time_to_wandb("throughput/loading_time", step=step):
                    input_ids = next(iter_data)
                    my_key, training_key = jrandom.split(training_key, 2)
                    example_keys = global_key_array(
                        my_key, config.trainer.train_batch_size, mesh, PartitionSpec(ResourceAxis.DATA)
                    )

                jax_step_loss, model, opt_state = train_step(model, opt_state, input_ids, example_keys)
                step_loss = jax_step_loss.item()
                if step < 10 or step % 50 == 0:
                    print(f"step loss: {step_loss:.3f}")

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

        engine.run_hooks(last_step, force=True)
        checkpointer.on_step(last_step, force=True)


if __name__ == "__main__":
    main()
