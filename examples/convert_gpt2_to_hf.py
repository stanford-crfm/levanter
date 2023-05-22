import logging
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import fsspec
import jax
import jax.numpy as jnp
from equinox import default_deserialise_filter_spec
from jaxtyping import PyTree
from transformers import GPT2Tokenizer

import haliax as hax
import levanter
from haliax import Axis, NamedArray
from haliax.util import is_named_array
from levanter.checkpoint import _assert_same
from levanter.compat.hf_checkpoints import save_hf_gpt2_checkpoint
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel
from levanter.tensorstore_serialization import tree_deserialize_leaves_tensorstore
from levanter.utils.hf_utils import load_tokenizer


logger = logging.getLogger(__name__)


@dataclass
class ConvertGpt2Config:
    checkpoint_path: str
    output_dir: str
    hf_checkpoint: Optional[str] = None  # if specified, attempt to upload this checkpoint to the hf hub
    hf_revision: Optional[str] = None  # if specified, use this branch name when uploading a checkpoint

    old_style_model: bool = False  # if True, use the old-style model serialization format (equinox-native

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
    tokenizer: GPT2Tokenizer = config.the_tokenizer

    key = jax.random.PRNGKey(0)

    vocab_size = config.override_vocab_size or len(tokenizer)
    Vocab = Axis("vocab", vocab_size)

    with jax.default_device(jax.devices("cpu")[0]):
        # we want to call this in case we're on a TPU node
        jax.process_index()

        model = Gpt2LMHeadModel.init(Vocab, config.model, key=key)

        if config.old_style_model:
            model = deserialize_checkpoint_and_patch_vocab_dim(f"{config.checkpoint_path}/model.eqx", model)
        else:
            with hax.enable_shape_checks(False):
                model = tree_deserialize_leaves_tensorstore(f"{config.checkpoint_path}/model", model)

            def patch_vocab(array):
                if is_named_array(array):
                    return patch_vocab_size(array.array, array)
                else:
                    return array

            model = jax.tree_util.tree_map(patch_vocab, model, is_leaf=is_named_array)

        save_hf_gpt2_checkpoint(model, config.output_dir, hf_repo=config.hf_checkpoint, hf_revision=config.hf_revision)


def deserialize_checkpoint_and_patch_vocab_dim(
    path: str,
    like: PyTree,
) -> PyTree:

    path = str(path)

    fs, _, (path_to_open,) = fsspec.get_fs_token_paths(path)

    with fs.open(path_to_open, "rb") as f:

        def deserialize_spec(x):
            if isinstance(x, NamedArray):
                inner = default_deserialise_filter_spec(f, x.array)

                inner = patch_vocab_size(inner, x)

                return NamedArray(inner, x.axes)
            else:
                inner = default_deserialise_filter_spec(f, x)
                return inner

        out = jax.tree_util.tree_map(deserialize_spec, like, is_leaf=is_named_array)
    jax.tree_util.tree_map(_assert_same, out, like)
    return out


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
