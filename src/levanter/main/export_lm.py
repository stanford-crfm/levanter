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
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef, load_tokenizer
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel
from levanter.tensorstore_serialization import tree_deserialize_leaves_tensorstore


logger = logging.getLogger(__name__)


@dataclass
class ConvertGpt2Config:
    checkpoint_path: str
    output_dir: str
    upload_to_hf: Optional[RepoRef] = None

    model: Gpt2Config = Gpt2Config()

    save_tokenizer: bool = True  # if True, save the tokenizer to the output directory

    tokenizer: str = "gpt2"

    override_vocab_size: Optional[int] = None  # if specified, override the vocab size in the config

    @cached_property
    def the_tokenizer(self):
        return load_tokenizer(self.tokenizer)


@levanter.config.main()
def main(config: ConvertGpt2Config):
    logger.setLevel(logging.INFO)
    tokenizer = config.the_tokenizer

    vocab_size = config.override_vocab_size or len(tokenizer)
    Vocab = Axis("vocab", vocab_size)

    key = jax.random.PRNGKey(0)

    with jax.default_device(jax.devices("cpu")[0]):
        model = filter_eval_shape(Gpt2LMHeadModel.init, Vocab, config.model, key=key)

        with hax.enable_shape_checks(False):
            model = tree_deserialize_leaves_tensorstore(f"{config.checkpoint_path}/model", model)

        model = htu.resize_axis(model, Vocab.resize(vocab_size), key=key)

        if not isinstance(model, Gpt2LMHeadModel):
            raise TypeError("Can't export a non-GPT2 model to HF checkpoint format with this script just yet!")

        converter = HFCheckpointConverter(Gpt2Config, "gpt2")

        converter.save_pretrained(model, config.output_dir, upload_to_hf=config.upload_to_hf or False)


if __name__ == "__main__":
    main()
