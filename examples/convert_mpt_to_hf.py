import logging
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import jax
from transformers import GPT2Tokenizer

import haliax as hax
import haliax.tree_util as htu
import levanter
from haliax import Axis
from haliax.jax_utils import filter_eval_shape
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef
from levanter.models.mpt import MptConfig, MptLmHeadModel
from levanter.tensorstore_serialization import tree_deserialize_leaves_tensorstore
from levanter.utils.hf_utils import load_tokenizer


logger = logging.getLogger(__name__)


@dataclass
class ConvertMptConfig:
    checkpoint_path: str
    output_dir: str
    upload_to_hf: Optional[RepoRef] = None

    model: Optional[MptConfig] = None

    save_tokenizer: bool = True  # if True, save the tokenizer to the output directory

    tokenizer: str = "gpt2"

    override_vocab_size: Optional[int] = None  # if specified, override the vocab size in the config
    override_sequence_length: Optional[int] = None  # if specified, override the sequence length in the config

    @cached_property
    def the_tokenizer(self):
        return load_tokenizer(self.tokenizer)


@levanter.config.main()
def main(config: ConvertMptConfig):
    levanter.logging.init_logger("convert.log", logging.INFO)
    tokenizer: GPT2Tokenizer = config.the_tokenizer

    vocab_size = config.override_vocab_size or len(tokenizer)
    Vocab = Axis("vocab", vocab_size)

    key = jax.random.PRNGKey(0)

    with jax.default_device(jax.devices("cpu")[0]):
        # we want to call this in case we're on a TPU node
        jax.distributed.initialize()
        converter = HFCheckpointConverter(MptConfig, "mosaicml/mpt-7b", trust_remote_code=True, tokenizer=tokenizer)
        if config.model is None:
            model_config = converter.config_from_hf_config(converter.default_hf_config)
        else:
            model_config = config.model

        if config.override_sequence_length is not None:
            model_config.max_seq_length = config.override_sequence_length

        model = filter_eval_shape(MptLmHeadModel.init, Vocab, model_config, key=key)

        with hax.enable_shape_checks(False):
            model = tree_deserialize_leaves_tensorstore(f"{config.checkpoint_path}/model", model)

        print(f"Resizing model to vocab size from {model.Vocab.size} to {Vocab.size}...")
        model = htu.resize_axis(model, Vocab.resize(vocab_size), key=key)

        converter.save_pretrained(
            model, config.output_dir, upload_to_hf=config.upload_to_hf or False, save_tokenizer=config.save_tokenizer
        )


if __name__ == "__main__":
    main()
