import functools
import logging
from dataclasses import dataclass
from typing import Optional

import jax
import jmp
import numpy
import tqdm

import haliax as hax
from haliax import Axis
from haliax.jax_utils import filter_eval_shape
from haliax.nn import cross_entropy_loss
from haliax.partitioning import named_jit, round_axis_for_partitioning

import levanter
from levanter import callbacks
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef
from levanter.data import ReplicatedBatchLoader
from levanter.data.text import CausalLmDataset, LMDatasetConfig, LmExample
from levanter.models.gpt2 import Gpt2Config
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.trainer import TrainerConfig


logger = logging.getLogger(__name__)


@dataclass
class EvalLmConfig:
    checkpoint_path: Optional[str] = None
    hf_checkpoint: Optional[RepoRef] = None
    trainer: TrainerConfig = TrainerConfig()
    data: LMDatasetConfig = LMDatasetConfig()
    model: LmConfig = Gpt2Config()

    compare_torch: bool = False
    eval_on_train: bool = False


def main(config: EvalLmConfig):
    config.trainer.initialize(config)
    tokenizer = config.data.the_tokenizer

    Batch = Axis("batch", config.trainer.eval_batch_size)
    Pos = config.model.Pos
    KeyPos = config.model.Pos

    if config.eval_on_train:
        raw_dataset = CausalLmDataset(config.data.token_seq_dataset("train", Pos.size), Pos, KeyPos)
    else:
        raw_dataset = CausalLmDataset(config.data.token_seq_dataset("validation", Pos.size), Pos, KeyPos)

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

        @named_jit(axis_resources=parameter_axis_mapping)
        def compute_loss(model: LmHeadModel, example: LmExample, key, inference):
            with hax.axis_mapping(compute_axis_mapping):
                model = mp.cast_to_compute(model)

                pred_y = model(example.tokens, example.attn_mask, key=key, inference=inference)
                pred_y = mp.cast_to_output(pred_y)

                target_y = hax.nn.one_hot(example.targets, Vocab, dtype=pred_y.dtype)

                per_ex_loss = cross_entropy_loss(pred_y, Vocab, target_y, where=example.loss_mask, reduction_axis=Pos)
                return hax.mean(per_ex_loss).scalar()

        compute_loss = functools.partial(compute_loss, inference=True, key=None)

        total = config.trainer.max_eval_batches

        # initialize the model
        if config.checkpoint_path is not None:
            # initialize the model
            with jax.default_device(jax.devices("cpu")[0]):
                model = filter_eval_shape(config.model.build, Vocab, key=key)
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
            if not hasattr(model_config, "hf_checkpoint_converter"):
                raise ValueError("Model config does not have an HF checkpoint converter. Can't load HF checkpoint.")
            converter: HFCheckpointConverter = model_config.hf_checkpoint_converter
            converter = converter.replaced(reference_checkpoint=config.hf_checkpoint, tokenizer=tokenizer)
            model_from_hf_checkpoint = converter.load_pretrained(model_config.model_type, config.hf_checkpoint)
            loss = callbacks.eval_loss_loop(compute_loss, model_from_hf_checkpoint, eval_loader, max_batches=total)

            print("Loss from HF model: ", loss)

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
