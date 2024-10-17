import logging
from dataclasses import dataclass, field
from typing import Union

import equinox as eqx
import jax.random as jrandom

from haliax import Axis
from haliax.partitioning import named_jit, round_axis_for_partitioning

import levanter
from levanter.compat.hf_checkpoints import HFCompatConfig
from levanter.data.text import CausalLmDataset, LMMixtureDatasetConfig
from levanter.doremi import DoReMiConfig, estimate_mixture_weights
from levanter.models.gpt2 import Gpt2Config
from levanter.models.lm_model import LmConfig, LmHeadModel, compute_next_token_loss
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.trainer import TrainerConfig
from levanter.utils.tree_utils import inference_mode


logger = logging.getLogger(__name__)


@dataclass
class TrainLmConfig:
    ref_model_path: str
    ref_model_from_hf: bool = False

    data: LMMixtureDatasetConfig = field(default_factory=LMMixtureDatasetConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LmConfig = field(default_factory=Gpt2Config)
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)
    doremi: DoReMiConfig = field(default_factory=DoReMiConfig)

    # config related to continued pretraining
    initialize_from_hf: Union[bool, str] = False
    """if provided, this will override the model config in the config. if true, use the default hf checkpoint for this model class"""
    use_hf_model_config: bool = False  # if true, replace the model config with the hf config from the checkpoint

    # TODO: atm we don't support loading from a checkpoint that has a different tokenizer. this is a bit annoying
    # TODO: atm you have to at least specify a levanter model config with the same type as the hf checkpoint


def main(config: TrainLmConfig):
    levanter.initialize(config)

    tokenizer = config.data.the_tokenizer

    # this is some unpleasant code to allow us to initialize from a hf checkpoint. If this is your first read through,
    # I recommend skipping it for now
    if config.initialize_from_hf:
        if config.trainer.initialize_from is not None:
            raise ValueError("Cannot specify both initialize_from_hf and initialize_from")

        assert isinstance(config.model, HFCompatConfig)
        converter = config.model.hf_checkpoint_converter()
        if hasattr(tokenizer, "vocab") and tokenizer.vocab != converter.tokenizer.vocab:
            logger.warning("The tokenizers appear to be different. You may want to check this.")

        if isinstance(config.initialize_from_hf, str):
            converter = converter.replaced(reference_checkpoint=config.initialize_from_hf, tokenizer=tokenizer)
        else:
            converter = converter.replaced(tokenizer=tokenizer)

        if config.use_hf_model_config:
            # TODO: log diff of old and new config
            # NB: gross mutability
            config.model = converter.config_from_hf_config(converter.default_hf_config)
    elif isinstance(config.model, HFCompatConfig):
        converter = config.model.hf_checkpoint_converter()
        converter = converter.replaced(tokenizer=tokenizer)
    else:
        converter = None

    optimizer = config.optimizer.build(config.trainer.num_train_steps)

    parameter_axis_mapping = config.trainer.parameter_axis_mapping

    loss_function = compute_next_token_loss

    with config.trainer.device_mesh:
        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), parameter_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        # initialize the ref model
        if config.ref_model_from_hf:
            assert converter is not None
            ref_model = converter.load_pretrained(config.model.model_type, dtype=config.trainer.mp.compute_dtype)
        else:
            ref_model_shape = eqx.filter_eval_shape(config.model.build, Vocab, key=jrandom.PRNGKey(0))
            ref_model = levanter.checkpoint.load_checkpoint(
                ref_model_shape, config.ref_model_path, axis_mapping=parameter_axis_mapping, subpath="model"
            )

        ref_model = inference_mode(ref_model, True)
        assert isinstance(ref_model, LmHeadModel)

        training_key, model_key = jrandom.split(jrandom.PRNGKey(config.trainer.seed), 2)

        @named_jit(axis_resources=parameter_axis_mapping)
        def init_proxy_model():
            return config.model.build(Vocab, key=model_key)

        proxy_model = init_proxy_model()

        train_datasets = config.data.training_sets(ref_model.Pos.size)
        valid_datasets = config.data.validation_sets(ref_model.Pos.size)

        train_datasets = {
            k: CausalLmDataset(v, config.model.Pos, config.model.KeyPos, ignore_index=config.data.ignore_token_id)
            for k, v in train_datasets.items()
        }
        valid_datasets = {
            k: CausalLmDataset(v, config.model.Pos, config.model.KeyPos, ignore_index=config.data.ignore_token_id)
            for k, v in valid_datasets.items()
        }

        mixture_weights = estimate_mixture_weights(
            loss_function,
            proxy_model,
            ref=ref_model,
            data_sources=train_datasets,
            trainer_config=config.trainer,
            optimizer=optimizer,
            domain_weight_step_size=config.doremi.domain_weight_step_size,
            sampling_weights=config.doremi.sampling_weights,
            validation_sets=valid_datasets,
            key=training_key,
        )

        print(mixture_weights)

        # dump to a yaml file
        weights_path = "mixture_weights.yaml"
        with open(weights_path, "w") as f:
            import yaml

            yaml.dump(mixture_weights, f)

        # log as an artifact
        levanter.tracker.current_tracker().log_artifact(weights_path, name="mixture_weights.yaml")


if __name__ == "__main__":
    levanter.config.main(main)()
