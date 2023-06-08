import logging
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import jax
import jax.numpy as jnp
from transformers import GPT2Tokenizer

import haliax as hax
import levanter
from haliax import Axis, NamedArray
from haliax.util import is_named_array
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

        def patch_vocab(array):
            if is_named_array(array):
                return patch_vocab_size(array.array, array)
            else:
                return array

        model = jax.tree_util.tree_map(patch_vocab, model, is_leaf=is_named_array)

        converter.save_model(
            model,
            config.output_dir,
            save_tokenizer=True,
            upload_to_hf=config.hf_checkpoint or False,
            commit_message="convert to hf checkpoint",
        )


def patch_vocab_size(inner: jnp.ndarray, like: NamedArray):
    # for partitioning reasons we frequently round the vocab size, but we need to patch it back
    # to the original size for the HF checkpoint to work
    if any(ax.name == "vocab" for ax in like.axes):
        index_of_vocab = next(i for i, ax in enumerate(like.axes) if ax.name == "vocab")
        desired_vocab_size = like.axes[index_of_vocab].size
        vocab_size = inner.shape[index_of_vocab]
        if vocab_size != desired_vocab_size:
            logger.info(f"Patching vocab size from {vocab_size} back to {desired_vocab_size} for HF checkpoint")
            inner = jnp.take(inner, jnp.arange(desired_vocab_size), axis=index_of_vocab)
    return inner


if __name__ == "__main__":
    main()
