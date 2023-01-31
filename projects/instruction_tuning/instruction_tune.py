import logging
from dataclasses import dataclass
from typing import List, Optional

import datasets
import equinox as eqx
import jax
import jax.random as jrandom
import numpy.random
import pyrallis
from instruction_tuning.encdec_dataset import InstructionDatasetConfig, InstructionTuningDataset
from jax.experimental.pjit import pjit
from jax.interpreters.pxla import PartitionSpec
from transformers import GPT2Tokenizer

import haliax as hax
import wandb
from haliax import Axis
from haliax.partitioning import ResourceAxis, named_pjit
from levanter import callbacks
from levanter.config import TrainerConfig
from levanter.data.sharded import GlobalBatchDataset, build_batch
from levanter.data.text import CachedLMDatasetConfig, TokenSeqDataset
from levanter.data.ul2r import DecoderOnlyExample, DenoisingTaskConfig, Ul2rDataset, convert_to_decoder_only
from levanter.grad_accum import accumulate_gradients_sharded
from levanter.jax_utils import global_key_array
from levanter.logging import capture_time, log_time_to_wandb
from levanter.modeling_utils import cross_entropy_loss
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel
from levanter.trainer_hooks import StepInfo, TrainerHooks
from py_utils import non_caching_cycle


@dataclass
class InstructionTuneConfig:
    hf_model: str

    trainer: TrainerConfig
    data: CachedLMDatasetConfig

    instruction_datasets: List[dict]  # should be InstructionDatasetConfig but pyrallis doesn't like it
    instruction_weight: float = 0.5

    ul2r_phase_fraction: float = 0.5  # fraction of training steps to spend in UL2R phase

    hf_revision: Optional[str] = None

    def build_instruction_dataset(self):
        instruction_configs = [InstructionDatasetConfig(**d) for d in self.instruction_datasets]
        instruction_hf_dsets = [c.canonicalize_hf() for c in instruction_configs]
        interleaved = datasets.interleave_datasets(instruction_hf_dsets, stopping_strategy="all_exhausted")
        # interleaved = interleaved.shuffle(self.trainer.seed + 43, buffer_size=10000)

        return InstructionTuningDataset(interleaved, self.data.the_tokenizer)


@pyrallis.wrap()
def main(config: InstructionTuneConfig):
    config.trainer.initialize(config)
    tokenizer: GPT2Tokenizer = config.data.the_tokenizer

    if tokenizer.pad_token_id is None:
        logging.warning(f"Adding pad token to tokenizer: {tokenizer.eos_token_id}")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # We follow two steps, more or less following the recipe from together's gpt-jt https://huggingface.co/togethercomputer/GPT-JT-6B-v1
    # 1. do continued pretraining on the model following ul2r for ul2r_phase_fraction of the training steps
    # 2. do fine-tuning on the instruction data for the remaining steps
    compute_axis_mapping = config.trainer.compute_axis_mapping
    parameter_axis_mapping = config.trainer.parameter_axis_mapping

    mp = config.trainer.mp

    ul2r_key, itune_key, training_key = jax.random.split(jax.random.PRNGKey(config.trainer.seed), 3)

    # load a model
    # with jax.default_device(jax.devices("cpu")[0]):
    #     model = load_hf_gpt2_checkpoint(config.hf_model, revision=config.hf_revision)

    tiny_model_config = Gpt2Config(num_layers=4, num_heads=4, hidden_dim=32)

    # 1. ul2r phase
    base_dataset = TokenSeqDataset(config.data.build_or_load_document_cache("train"), tiny_model_config.SeqLen.size)
    task_configs = DenoisingTaskConfig.ul2r_configs()
    base_dataset = Ul2rDataset(  # type: ignore
        base_dataset, tiny_model_config.SeqLen, tiny_model_config.KeySeqLen, ul2r_key, tokenizer, task_configs
    )

    # NB we can't make this until we have added all our tokens to the tokenizer
    Vocab = Axis("vocab", len(tokenizer))

    with config.trainer.device_mesh as mesh:
        # model = named_pjit(lambda m: m, axis_resources=parameter_axis_mapping)(model)
        @named_pjit(axis_resources=parameter_axis_mapping)
        def init_model():
            model = Gpt2LMHeadModel.init(Vocab, tiny_model_config, key=ul2r_key)
            return mp.cast_to_param(model)

        model = init_model()

        Batch = Axis("batch", config.trainer.train_batch_size)
        # TODO: evaluation
        SeqLen = model.SeqLen
        KeySeqLen = model.config.KeySeqLen

        optimizer = config.trainer.optimizer()
        opt_state = optimizer.init(model)

        def compute_loss(model: Gpt2LMHeadModel, ex: DecoderOnlyExample, inference, key):
            with hax.axis_mapping(compute_axis_mapping):
                model = mp.cast_to_compute(model)

                pred_y = model(ex.tokens, ex.attn_mask, key=key, inference=inference)
                pred_y = mp.cast_to_output(pred_y)

                loss = cross_entropy_loss(pred_y, Vocab, hax.nn.one_hot(ex.targets, Vocab))
                loss = hax.mean(loss, where=ex.loss_mask)

                return loss.scalar()

        def train_batch_loss(model, example, key):
            return hax.mean(hax.vmap(compute_loss, Batch)(model, example, key=key, inference=False))

        # training loop
        # donate args to conserve memory
        @named_pjit(axis_resources=parameter_axis_mapping, donate_args=True)
        def train_step(model, opt_state, batch: DecoderOnlyExample, keys):
            loss, grads = accumulate_gradients_sharded(
                eqx.filter_value_and_grad(train_batch_loss),
                Batch,
                model,
                batch,
                keys,
                per_device_parallelism=config.trainer.per_device_parallelism,
                parameter_axis_mapping=parameter_axis_mapping,
            )

            # distribute gradients across the mesh and apply them
            updates, opt_state = optimizer.update(grads, opt_state, params=model)
            model = eqx.apply_updates(model, updates)

            return loss, model, opt_state

        # hooks
        engine = TrainerHooks()
        engine.add_hook(callbacks.pbar_logger(total=config.trainer.num_train_steps), every=1)
        engine.add_hook(callbacks.log_to_wandb, every=1)
        engine.add_hook(callbacks.log_performance_stats(SeqLen.size, Batch.size), every=1)
        engine.add_hook(callbacks.wandb_xla_logger(config.trainer.wandb), every=config.trainer.steps_per_eval)
        checkpointer = config.trainer.checkpointer.create(config.trainer.run_name)
        engine.add_hook(checkpointer.on_step, every=1)  # checkpointer manages its own frequency

        base_dataset = GlobalBatchDataset(base_dataset, config.trainer.device_mesh, Batch)  # type: ignore

        iter_data = non_caching_cycle(base_dataset)

        # Interlude: load the dataset for phase two as a sanity check before doing learning
        itune_dataset = config.build_instruction_dataset()

        ul2r_steps = int(config.trainer.num_train_steps * config.ul2r_phase_fraction)
        logging.info(f"Starting UL2R phase for {ul2r_steps} steps")

        for step in range(ul2r_steps):
            with capture_time() as step_time:
                with log_time_to_wandb("throughput/loading_time", step=step):
                    batch = next(iter_data)
                    my_key, training_key = jrandom.split(training_key, 2)
                    example_keys = global_key_array(
                        my_key, config.trainer.train_batch_size, mesh, PartitionSpec(ResourceAxis.DATA)
                    )

                step_loss, model, opt_state = train_step(model, opt_state, batch, example_keys)
                step_loss = step_loss.item()
                wandb.log({"phase": 1}, step=step)

            with log_time_to_wandb("throughput/hook_time", step=step):
                engine.run_hooks(StepInfo(step, model, opt_state, step_loss, training_key, step_duration=step_time()))

        if ul2r_steps > 0:
            logging.info("UL2R phase finished. Saving checkpoint")
            checkpointer.save_checkpoint(
                StepInfo(step, model, opt_state, step_loss, training_key, step_duration=step_time()), "ul2r_finished"
            )

        # 2. fine-tuning phase
        logging.info("Starting instruction-tuning phase")

        # build our sampling scheme
        def sampling_iterator():
            prng = numpy.random.default_rng(numpy.array(itune_key))

            iter_core = iter_data
            iter_itune = iter(itune_dataset)

            while True:
                if prng.uniform() < config.instruction_weight:
                    example = next(iter_itune)
                    yield convert_to_decoder_only(example, tokenizer.pad_token_id, SeqLen, KeySeqLen)
                else:
                    yield next(iter_core)

        def batch_sampler():
            # Todo this is not ideal
            shard_batch = pjit(
                lambda x: x,
                in_axis_resources=None,
                out_axis_resources=PartitionSpec(ResourceAxis.DATA),
                donate_argnums=0,
            )

            while True:
                batch = []
                for i in range(config.trainer.train_batch_size):
                    batch.append(next(sampling_iterator()))
                batch = shard_batch(build_batch(Batch, batch, unchecked=False))
                yield batch

        iter_data_2 = batch_sampler()

        for x in range(10):
            batch = next(iter_data_2)
            print(batch)

        return

        for step in range(ul2r_steps, config.trainer.num_train_steps):
            with capture_time() as step_time:
                with log_time_to_wandb("throughput/loading_time", step=step):
                    batch = next(iter_data_2)
                    my_key, training_key = jrandom.split(training_key, 2)
                    example_keys = global_key_array(
                        my_key, config.trainer.train_batch_size, mesh, PartitionSpec(ResourceAxis.DATA)
                    )

                step_loss, model, opt_state = train_step(model, opt_state, batch, example_keys)
                step_loss = step_loss.item()
                wandb.log({"phase": 1}, step=step)

            with log_time_to_wandb("throughput/hook_time", step=step):
                engine.run_hooks(StepInfo(step, model, opt_state, step_loss, training_key, step_duration=step_time()))

        logging.info("Training finished. Saving checkpoint")
        checkpointer.save_checkpoint(
            StepInfo(step, model, opt_state, step_loss, training_key, step_duration=step_time()), "instruction_tuned"
        )


if __name__ == "__main__":
    main()
