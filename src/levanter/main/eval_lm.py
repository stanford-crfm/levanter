import logging
from dataclasses import dataclass, field
from typing import Optional

import equinox as eqx
import jax
import jmp
import numpy
import tqdm

import haliax as hax
from haliax import Axis
from haliax.partitioning import fsdp, round_axis_for_partitioning

import levanter
from levanter import callbacks
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef
from levanter.data import DataLoader
from levanter.data.text import CausalLmDataset, LMDatasetConfig
from levanter.models.gpt2 import Gpt2Config
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel, compute_next_token_loss
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device
from levanter.utils.tree_utils import inference_mode


logger = logging.getLogger(__name__)


@dataclass
class EvalLmConfig:
    checkpoint_path: Optional[str] = None
    hf_checkpoint: Optional[RepoRef] = None
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    data: LMDatasetConfig = field(default_factory=LMDatasetConfig)
    model: LmConfig = field(default_factory=Gpt2Config)

    compare_torch: bool = False
    eval_on_train: bool = False


def main(config: EvalLmConfig):
    levanter.initialize(config)
    tokenizer = config.data.the_tokenizer

    Batch = Axis("batch", config.trainer.eval_batch_size)
    Pos = config.model.Pos
    KeyPos = config.model.KeyPos

    if config.eval_on_train:
        raw_dataset = CausalLmDataset(config.data.train_set(Pos.size, key=jax.random.PRNGKey(0)), Pos, KeyPos)
    else:
        validation_set = config.data.validation_set(Pos.size)
        if validation_set is None:
            raise ValueError("Can't eval on validation_set b/c there isn't one!")

        raw_dataset = CausalLmDataset(validation_set, Pos, KeyPos)  # type: ignore

    eval_loader = DataLoader(
        Batch, raw_dataset, None, config.trainer.device_mesh, config.trainer.parameter_axis_mapping
    )
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
        def compute_loss(model: LmHeadModel, example: LmExample):
            model = inference_mode(model, True)
            model = mp.cast_to_compute(model)
            return compute_next_token_loss(model, example, key=None)

        total = config.trainer.max_eval_batches

        # initialize the model
        if config.checkpoint_path is not None:
            # initialize the model
            with use_cpu_device():
                model = eqx.filter_eval_shape(config.model.build, Vocab, key=key)
                # TODO: don't load the entire checkpoint into CPU memory when we only need our share of the model
                model = load_checkpoint(model, config.checkpoint_path, subpath="model")

            model = hax.shard_with_axis_mapping(model, parameter_axis_mapping)

            loss = callbacks.eval_loss_loop(compute_loss, model, eval_loader, max_batches=total)

            del model
            print("Loss from Levanter model: ", loss)

        if config.hf_checkpoint is not None:
            # load the huggingface model
            model_config = config.model
            if not hasattr(model_config, "hf_checkpoint_converter"):
                raise ValueError("Model config does not have an HF checkpoint converter. Can't load HF checkpoint.")
            converter: HFCheckpointConverter = model_config.hf_checkpoint_converter()
            converter = converter.replaced(reference_checkpoint=config.hf_checkpoint, tokenizer=tokenizer)
            model_from_hf_checkpoint = converter.load_pretrained(
                model_config.model_type, ref=config.hf_checkpoint, dtype=mp.compute_dtype
            )
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
