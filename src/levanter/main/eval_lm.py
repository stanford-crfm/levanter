import logging
from dataclasses import dataclass
from typing import Optional

import equinox as eqx
import jax
import jmp
import numpy
import tqdm
import wandb

import haliax as hax
from haliax import Axis
from haliax.partitioning import fsdp, round_axis_for_partitioning, named_jit

import levanter
from levanter import callbacks
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef
from levanter.data import ReplicatedBatchLoader
from levanter.data.text import CausalLmDataset, LMDatasetConfig
from levanter.models.gpt2 import Gpt2Config
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel
from levanter.models.llama import LlamaLMHeadModel
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device
from levanter.utils.tree_utils import inference_mode
import jax.tree_util as tree_util
from levanter.utils.jax_utils import parameter_count, flops_estimate, is_inexact_arrayish, multihost_broadcast_sync


logger = logging.getLogger(__name__)


@dataclass
class EvalLmConfig:
    checkpoint_path: Optional[str] = None
    hf_checkpoint: Optional[RepoRef] = None
    trainer: TrainerConfig = TrainerConfig()
    data: LMDatasetConfig = LMDatasetConfig()
    model: LmConfig = Gpt2Config()
    second_hf_checkpoint: Optional[RepoRef] = None

    compare_torch: bool = False
    eval_on_train: bool = False
    alpha: float = 0.5


def main(config: EvalLmConfig):
    # HACK: This is a workaround for the fact that the config is not properly deserialized
    config.second_hf_checkpoint.model_name_or_path = config.second_hf_checkpoint.model_name_or_path[
        "model_name_or_path"
    ]
    config.trainer.initialize(config)
    tokenizer = config.data.the_tokenizer

    Batch = Axis("batch", config.trainer.eval_batch_size)
    Pos = config.model.Pos
    KeyPos = config.model.KeyPos

    if config.eval_on_train:
        raw_dataset = CausalLmDataset(config.data.train_set(Pos.size), Pos, KeyPos)
    else:
        raw_dataset = CausalLmDataset(config.data.validation_set(Pos.size), Pos, KeyPos)  # type: ignore

    eval_loader = ReplicatedBatchLoader(raw_dataset, config.trainer.device_mesh, Batch)
    compute_axis_mapping = config.trainer.compute_axis_mapping
    parameter_axis_mapping = config.trainer.parameter_axis_mapping

    with config.trainer.device_mesh, hax.axis_mapping(parameter_axis_mapping):
        key = jax.random.PRNGKey(0)

        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), compute_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        mp: jmp.Policy = config.trainer.mp


        @fsdp(parameter_axis_mapping, compute_axis_mapping)
        def l2_norm_diff(model1: LmHeadModel, model2: LmHeadModel):
            def squared_diff_mean(x, y):
                diff = x - y
                squared_diff = diff ** 2
                mean_squared_diff = hax.mean(squared_diff)
                return mean_squared_diff

            tree_diff = tree_util.tree_map(squared_diff_mean, model1, model2)
            total_loss = tree_util.tree_reduce(lambda x, y: x + y, tree_diff, initializer=0.0)
            num_params = len(tree_util.tree_leaves(model1))

            return total_loss / num_params


        @fsdp(parameter_axis_mapping, compute_axis_mapping)
        def compute_logits_diff(logit_fn, model1: LmHeadModel, model2: LmHeadModel, batch: LmExample):
            logits1 = logit_fn(model1, batch)
            logits2 = logit_fn(model2, batch)

            diff = logits1 - logits2
            squared_diff = diff ** 2
            mean_squared_diff = hax.mean(squared_diff)
            loss = mean_squared_diff

            return loss
        
        @fsdp(parameter_axis_mapping, compute_axis_mapping)
        def compute_loss(model: LmHeadModel, example: LmExample):
            model = inference_mode(model, True)
            model = mp.cast_to_compute(model)
            return model.compute_loss(example, key=None)

        @fsdp(parameter_axis_mapping, compute_axis_mapping)
        def compute_logit(model: LmHeadModel, example: LmExample):
            model = inference_mode(model, True)
            model = mp.cast_to_compute(model)
            return model.compute_logits(example, key=None)
        
        @fsdp(parameter_axis_mapping, compute_axis_mapping)
        def compute_jsd_loss(model1: LmHeadModel, model2: LmHeadModel, example: LmExample):
            model1 = inference_mode(model1, True)
            model1 = mp.cast_to_compute(model1)
            model2 = inference_mode(model2, True)
            model2 = mp.cast_to_compute(model2)
            
            logits = model1.compute_logits(example, key=None)
            logits2 = model2.compute_logits(example, key=None)
            
            # Compute log probabilities using log_softmax for numerical stability
            log_p1 = hax.nn.log_softmax(logits, axis=model1.Vocab)
            log_p2 = hax.nn.log_softmax(logits2, axis=model2.Vocab)
            
            # Obtain probabilities from log probabilities using exp
            p1 = hax.exp(log_p1)
            p2 = hax.exp(log_p2)
            
            # Mean probability distribution
            m = (p1 + p2) / 2
            
            # Compute KL divergences using the formula KL(p||q) = sum(p * log(p / q))
            kl1 = hax.dot(p1, log_p1 - hax.log(m), axis=model1.Vocab)
            kl2 = hax.dot(p2, log_p2 - hax.log(m), axis=model2.Vocab)
            
            # Sum KL divergences and normalize to get Jensen-Shannon Divergence
            jsd = 0.5 * (kl1 + kl2)
            
            # Compute the mean JSD across the batch and sequence dimensions
            mean_jsd = hax.mean(jsd, axis=(jsd.axes[0], jsd.axes[1]))
            
            return mean_jsd

        total = config.trainer.max_eval_batches

        # initialize the model
        if config.checkpoint_path is not None:
            # initialize the model
            with use_cpu_device():
                model = eqx.filter_eval_shape(config.model.build, Vocab, key=key)
                # TODO: don't load the entire checkpoint into CPU memory when we only need our share of the model
                ckpt = load_checkpoint(model, None, config.checkpoint_path)

            assert ckpt is not None
            model, _, _ = ckpt

            model = hax.shard_with_axis_mapping(model, parameter_axis_mapping)

            # TODO: switch to throwing instead of returning None
            loss = callbacks.eval_loss_loop(compute_loss, model, eval_loader, max_batches=total)

            del model
            print("Loss from Levanter model: ", loss)

        if config.hf_checkpoint is not None:
            # load the huggingface model
            model_config = config.model
            # if not hasattr(model_config, "hf_checkpoint_converter"):
            #    raise ValueError("Model config does not have an HF checkpoint converter. Can't load HF checkpoint.")
            converter: HFCheckpointConverter = model_config.default_hf_checkpoint_converter
            converter = converter.replaced(reference_checkpoint=config.hf_checkpoint, tokenizer=tokenizer)

            logger.info(f"Loading first model from {converter.reference_checkpoint}")
            logger.info(f"Loading first model from {config.model}")
            logger.info(f"model config {model_config}")
            model_1 = converter.load_pretrained(model_config, config.hf_checkpoint)

            converter = converter.replaced(reference_checkpoint=config.second_hf_checkpoint, tokenizer=tokenizer)
            logger.info(f"Loading second model from {converter.reference_checkpoint}")
            logger.info(f"Loading second model from {config.model}")
            model_2 = converter.load_pretrained(model_config)

            jsd = callbacks.jsd_loss_loop(compute_jsd_loss, model_1, model_2, eval_loader, max_batches=total)
            l2_norm_num = callbacks.l2_norm_diff(l2_norm_diff, model_1, model_2)
            logits_diff = callbacks.logits_diff_loop(compute_logit, compute_logits_diff, model_1, model_2, eval_loader, max_batches=total)
            # Generate alphas from 0 to 1 with a step of 0.05
            
            wandb.log({"eval/logits_diff": logits_diff})
            wandb.log({"eval/l2_norm_diff": l2_norm_num})
            wandb.log({"eval/jsd": jsd})
            alphas = [round(alpha * 0.05, 2) for alpha in range(21)]

            for alpha in alphas:
                print(f"alpha: {alpha}")

                def add_floats(path, x, y):
                    print(path)
                    if is_inexact_arrayish(x) and is_inexact_arrayish(y):
                        # Linearly interpolate between the two models
                        minus_alpha = 1.0 - alpha
                        return x * alpha + y * minus_alpha
                    else:
                        return x

                # Use the rounded alpha for merging models
                merged_model = named_jit(
                    lambda m1, m2: jax.tree_util.tree_map_with_path(add_floats, m1, m2), donate_args=False
                )(model_1, model_2)

                # Use the rounded alpha for merging models
                # Evaluate the loss for the merged model
                loss = callbacks.eval_loss_loop(compute_loss, merged_model, eval_loader, max_batches=total)

                # Log the rounded alpha and loss to W&B
                wandb.log({"eval/loss": loss, "alpha": alpha})

                print(f"Loss from merged model (alpha={alpha}): ", loss)
                del merged_model

            if config.compare_torch:
                import torch
                from transformers import GPT2LMHeadModel as TorchGPT2LMHeadModel

                torch_model: TorchGPT2LMHeadModel = TorchGPT2LMHeadModel.from_pretrained(
                    config.hf_checkpoint.model_name_or_path, revision=config.hf_checkpoint.revision
                )
                torch_model.eval()
                torch_model.to("cpu")

                loss = 0.0
                n = 0
                for batch in tqdm.tqdm(eval_loader, total=total, desc="Evaluating (torch)"):
                    torch_ids = torch.from_numpy(numpy.array(batch)).to(torch.int64)
                    with torch.no_grad():
                        loss += torch_model(input_ids=torch_ids, labels=torch_ids)[0].item()
                    n += 1
                    if total is not None and n >= total:
                        break

                print("Loss from Torch model: ", loss / n)


if __name__ == "__main__":
    levanter.config.main(main)()
