import itertools
import logging
from collections import Counter
from dataclasses import dataclass
from functools import partial

import equinox as eqx
import jax
import jmp
from equinox import filter_vmap
from jax.experimental.pjit import pjit

from haliax import Axis
from haliax.partitioning import axis_mapping, infer_resource_partitions, named_pjit, round_axis_for_partitioning
from levanter import callbacks
from levanter.callbacks import log_performance_stats, log_to_wandb, pbar_logger, wandb_xla_logger
from levanter.data import CachedLMDatasetConfig
from levanter.data.sharded import ShardedIndexedDataset
from levanter.logging import capture_time, log_time_to_wandb
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel


print(Counter([type(dev) for dev in jax.devices()]))
import jax.numpy as jnp
import jax.profiler
import jax.random as jrandom
import pyrallis
from transformers import GPT2Tokenizer

import wandb
from levanter.checkpoint import load_checkpoint
from levanter.config import TrainerConfig
from levanter.jax_utils import global_key_array, parameter_count
from levanter.modeling_utils import accumulate_gradients_sharded, cross_entropy_loss_and_log_normalizers
from levanter.trainer_hooks import StepInfo, TrainerHooks


logger = logging.getLogger(__name__)


# cf https://github.com/google-research/language/blob/aa58066bec83d30de6c8f9123f0af7b81db3aeba/language/mentionmemory/training/trainer.py


@dataclass
class TrainGpt2Config:
    data: CachedLMDatasetConfig = CachedLMDatasetConfig()
    trainer: TrainerConfig = TrainerConfig()
    model: Gpt2Config = Gpt2Config()

    log_z_regularization: float = 0.0


@pyrallis.wrap()
def main(config: TrainGpt2Config):
    config.trainer.initialize(config)

    tokenizer: GPT2Tokenizer = config.data.the_tokenizer
    dataset = ShardedIndexedDataset(
        config.data.build_or_load_document_cache("train"),
        config.trainer.train_mesh_info,
        config.model.seq_len,
    )

    eval_dataset = ShardedIndexedDataset(
        config.data.build_or_load_document_cache("validation"),
        config.trainer.eval_mesh_info,
        config.model.seq_len,
    )

    with config.trainer.device_mesh as mesh, axis_mapping(config.trainer.axis_resources):
        # randomness in jax is tightly controlled by "keys" which are the states of the random number generators
        # this makes deterministic training pretty easy
        seed = config.trainer.seed
        data_key, loader_key, model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 4)

        # to do partitioning, our dimensions have to be divisible by the size of the physical axes they're mapped to
        # For most things, we just insist you specify the config right, but tokenizers often have strange numbers of
        # tokens: gpt-2 has 50257, for example. So we round up.
        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size))
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        # Mixed Precision: We use the "jmp" library to handle mixed precision training. It basically has three dtypes:
        # 1) compute (typically bfloat16)
        # 2) parameter (typically float32)
        # 3) output (sometimes float32)
        # I like to think of these as "semantic" dtypes: compute is the dtype we do most of our math in, parameter is
        # the dtype we store our parameters in, and output is the dtype we use for loss calculations.
        mp: jmp.Policy = config.trainer.mp

        # initialize the model and optimizer
        # this doesn't actually perform any compute because Jax is lazy
        model = Gpt2LMHeadModel(Vocab, config.model, key=model_key)
        wandb.summary["parameter_count"] = parameter_count(model)

        optim = config.trainer.optimizer()

        # now shard the model and optimizer across the mesh, using "parameter axis resources".
        # This allows Zero-3-style parameter sharding, where we shard the parameters and optimizer state across the mesh
        # and then use a "reducer" to aggregate the gradients across the mesh.
        with axis_mapping(config.trainer.parameter_axis_resources, merge=True):

            def init_state():
                # TODO: think about how we want to do this if we want to load a checkpoint
                _model = mp.cast_to_param(model)
                opt_state = optim.init(_model)
                return model, opt_state

            # doing this in a pjit means that the model and optimizer states are already sharded
            model, opt_state = named_pjit(init_state)()

            # model and opt_state are what Jax calls "pytrees", which is just a fancy word for "python object that
            # has 'leaf' attributes". Typically leaves are ndarrays or scalars, but they can be be anything.

            # we need to tell Jax how to split the pytree across the mesh, so we use the infer_resource_partitions
            # function, which returns a Pytree with the same structure as the input, but all of the leaves are replaced
            # with PartitionSpecs, which tell Jax how to split the leaf across the mesh. We'll use these with pjit
            opt_state_resources = infer_resource_partitions(opt_state)
            model_resources = infer_resource_partitions(model)

        with axis_mapping(config.trainer.axis_resources):
            compute_model_resources = infer_resource_partitions(model)
            data_resources = dataset.partition_spec

        # when it's time to do actual compute, we want to convert the model to the compute dtype and shard it
        # for inference
        prepare_model_for_compute = pjit(
            lambda m: mp.cast_to_compute(m),
            in_axis_resources=(model_resources,),
            out_axis_resources=compute_model_resources,
        )

        # loss function: this computes the loss with respect to a single example
        def compute_loss(model: Gpt2LMHeadModel, input_ids, key, inference):
            pred_y = model(input_ids, inference=inference, key=key)
            pred_y = mp.cast_to_output(pred_y)

            # TODO: would prefer to do this in haliax name land, but it's not clear how to do that
            # could add a where mask which is pretty normal
            pred_y = pred_y[:-1]
            labels = jax.nn.one_hot(input_ids[1:], Vocab.size)

            loss, log_normalizers = cross_entropy_loss_and_log_normalizers(pred_y, labels)

            if not inference and config.log_z_regularization > 0:
                logz_mse = jnp.mean((log_normalizers**2))
                loss += config.log_z_regularization * logz_mse

            return loss

        # mean_loss: this computes the mean loss over a batch of examples
        def mean_loss(model: Gpt2LMHeadModel, input_ids, key, inference):
            # None here means the first argument (the model) is not vectorized but instead broadcasted
            compute_loss_vmap = filter_vmap(compute_loss, args=(None,))
            return jnp.mean(compute_loss_vmap(model, input_ids, key, inference))

        # get the gradient using a wrapper around jax.value_and_grad
        compute_loss_and_grad = eqx.filter_value_and_grad(partial(mean_loss, inference=False))

        # pjit tells jax how to split the arguments across the mesh and how the returned values should be sharded
        # we use the resources we inferred above
        compute_loss_pjit = pjit(
            partial(mean_loss, inference=True, key=None),
            in_axis_resources=(compute_model_resources, data_resources),
            out_axis_resources=None,
        )

        # Set up evaluation: dataloader, loop
        def eval_dataloader():
            # TODO: only do one pass
            yield from itertools.islice(eval_dataset, 50)

        def evaluate_step(info: StepInfo):
            model_inf = prepare_model_for_compute(info.model)

            # standard evaluation loop
            loss = 0.0
            n = 0

            for batch in eval_dataloader():
                loss += compute_loss_pjit(model_inf, batch)
                n += 1

            if n > 0:
                loss /= n

            logger.info(f"validation loss: {loss:.3f}")
            if wandb.run is not None:
                wandb.log({"eval/loss": loss}, step=info.step)

            return loss

        # boilerplate hooks and such
        engine = TrainerHooks()
        engine.add_hook(pbar_logger(total=config.trainer.num_train_steps), every=1)
        engine.add_hook(log_to_wandb, every=1)
        engine.add_hook(log_performance_stats(config.model.seq_len, config.trainer.train_batch_size), every=1)
        engine.add_hook(evaluate_step, every=config.trainer.steps_per_eval)
        save = callbacks.save_model(config.trainer.checkpoint_path)
        engine.add_hook(save, every=config.trainer.steps_per_save)
        engine.add_hook(wandb_xla_logger(config.trainer.wandb), every=config.trainer.steps_per_eval)

        # data loader
        iter_data = iter(dataset)

        # load the last checkpoint and resume if we want
        resume_step = None
        if config.trainer.load_last_checkpoint:
            with jax.default_device(jax.devices("cpu")[0]):
                checkpoint = load_checkpoint(
                    model,
                    (opt_state, training_key),
                    config.trainer.load_checkpoint_path or config.trainer.checkpoint_path,
                )
            if checkpoint is not None:
                model, (opt_state, training_key), resume_step = checkpoint
                # TODO: switch to GlobalDeviceArray loading/saving or something so we don't have to do this
                #model = pjit(lambda m: m, in_axis_resources=(None,), out_axis_resources=model_resources)(model)
                #opt_state = pjit(lambda m: m, in_axis_resources=(None,), out_axis_resources=opt_state_resources)(
                #    opt_state
                #)
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

        # training loop
        mesh_info = config.trainer.train_mesh_info

        def train_step(model, opt_state, input_ids, keys):
            model_inf = prepare_model_for_compute(model)

            parameter_axis_mapping = dict(config.trainer.axis_resources)
            parameter_axis_mapping.update(config.trainer.parameter_axis_resources)

            loss, grads = accumulate_gradients_sharded(
                compute_loss_and_grad,
                model_inf,
                input_ids,
                keys,
                data_axis_size=mesh_info.data_axis_size,
                per_device_parallelism=mesh_info.per_device_parallelism,
                compute_axis_mapping=config.trainer.axis_resources,
                parameter_axis_mapping=parameter_axis_mapping,
            )

            with jax.named_scope("optimizer"), axis_mapping(config.trainer.parameter_axis_resources, merge=True):
                # distribute gradients across the mesh and apply them
                updates, opt_state = optim.update(grads, opt_state, params=model)
                model = eqx.apply_updates(model, updates)

            return loss, model, opt_state

        train_step = pjit(
            train_step,
            in_axis_resources=(
                model_resources,
                opt_state_resources,
                data_resources,  # input_ids
                data_resources,  # keys are shared same as data
            ),
            out_axis_resources=(None, model_resources, opt_state_resources),
            donate_argnums=(0, 1),
        )

        for step in range(resume_step, config.trainer.num_train_steps):
            with capture_time() as step_time:
                with log_time_to_wandb("throughput/loading_time", step=step):
                    input_ids = next(iter_data)
                    my_key, training_key = jrandom.split(training_key, 2)
                    micro_keys = global_key_array(my_key, input_ids.shape[:-1], mesh, dataset.partition_spec[:-1])

                step_loss, model, opt_state = train_step(model, opt_state, input_ids, micro_keys)
                step_loss = jnp.mean(step_loss).item()

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
        save(last_step)


if __name__ == "__main__":
    main()
