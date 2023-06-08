import logging
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import jax
from jax.random import PRNGKey
from transformers import GPT2Tokenizer

import haliax as hax
import haliax.tree_util as htu
import levanter
from haliax import Axis
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef
from levanter.models.backpack import BackpackConfig, BackpackLMHeadModel
from levanter.tensorstore_serialization import tree_deserialize_leaves_tensorstore
from levanter.utils.hf_utils import load_tokenizer


logger = logging.getLogger(__name__)


@dataclass
class ConvertConfig:
    checkpoint_path: str
    output_dir: str
    hf_checkpoint: Optional[RepoRef] = None  # if specified, attempt to upload this checkpoint to the hf hub

    model: BackpackConfig = BackpackConfig()

    save_tokenizer: bool = True  # if True, save the tokenizer to the output directory

    tokenizer: str = "gpt2"

    override_vocab_size: Optional[int] = None  # if specified, override the vocab size in the config

    config_overrides: Optional[dict] = None  # if specified, override the config with these values

    @cached_property
    def the_tokenizer(self):
        return load_tokenizer(self.tokenizer)


@levanter.config.main()
def main(config: ConvertConfig):
    logger.setLevel(logging.INFO)
    tokenizer: GPT2Tokenizer = config.the_tokenizer

    key = jax.random.PRNGKey(0)

    vocab_size = config.override_vocab_size or len(tokenizer)
    Vocab = Axis("vocab", vocab_size)

    converter = HFCheckpointConverter(
        BackpackConfig,
        "stanford-crfm/levanter-backpacks-test",
        trust_remote_code=True,
    )

    if config.config_overrides:
        converter = converter.with_config_overrides(config.config_overrides)

    with jax.default_device(jax.devices("cpu")[0]):
        model = BackpackLMHeadModel(Vocab, config.model, key=key)

        with hax.enable_shape_checks(False):
            model = tree_deserialize_leaves_tensorstore(f"{config.checkpoint_path}/model", model)

        model = htu.resize_axis(model, Vocab.resize(vocab_size), key=PRNGKey(0))

        converter.save_model(
            model,
            config.output_dir,
            save_tokenizer=True,
            upload_to_hf=config.hf_checkpoint or False,
            commit_message="convert to hf checkpoint",
        )


if __name__ == "__main__":
    main()
