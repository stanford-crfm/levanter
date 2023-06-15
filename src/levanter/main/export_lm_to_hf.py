import logging
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import jax

import haliax as hax
import haliax.tree_util as htu
import levanter
from haliax import Axis
from haliax.jax_utils import filter_eval_shape
from levanter.compat.hf_checkpoints import RepoRef, load_tokenizer
from levanter.models.gpt2 import Gpt2Config
from levanter.models.lm_model import LmConfig
from levanter.tensorstore_serialization import tree_deserialize_leaves_tensorstore


logger = logging.getLogger(__name__)


@dataclass
class ConvertLmConfig:
    checkpoint_path: str
    output_dir: str
    upload_to_hf: Optional[RepoRef] = None  # if specified, attempt to upload this checkpoint to the hf hub

    model: LmConfig = Gpt2Config()

    save_tokenizer: bool = True  # if True, save the tokenizer to the output directory

    tokenizer: str = "gpt2"

    override_vocab_size: Optional[int] = None  # if specified, override the vocab size in the config

    config_overrides: Optional[dict] = None  # if specified, override the config with these values

    @cached_property
    def the_tokenizer(self):
        return load_tokenizer(self.tokenizer)


@levanter.config.main()
def main(config: ConvertLmConfig):
    logger.setLevel(logging.INFO)
    tokenizer = config.the_tokenizer

    vocab_size = config.override_vocab_size or len(tokenizer)
    Vocab = Axis("vocab", vocab_size)

    key = jax.random.PRNGKey(0)

    with jax.default_device(jax.devices("cpu")[0]):
        model = filter_eval_shape(config.model.build(Vocab, key=key), Vocab, config.model, key=key)

        with hax.enable_shape_checks(False):
            model = tree_deserialize_leaves_tensorstore(f"{config.checkpoint_path}/model", model)

        model = htu.resize_axis(model, Vocab.resize(vocab_size), key=key)

        converter = model.config.default_hf_checkpoint_converter.replaced(tokenizer=tokenizer)

        converter.save_pretrained(model, config.output_dir, upload_to_hf=config.upload_to_hf or False)


if __name__ == "__main__":
    main()
