import logging
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import fsspec
import jax
import jax.numpy as jnp
import pyrallis
from equinox import default_deserialise_filter_spec
from jaxtyping import PyTree
from transformers import AutoTokenizer, GPT2Tokenizer

from haliax import Axis, NamedArray
from haliax.util import is_named_array
from levanter.checkpoint import _assert_same
from levanter.compat.torch_checkpoints import save_hf_gpt2_checkpoint
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel


logger = logging.getLogger(__name__)


@dataclass
class ConvertGpt2Config:
    checkpoint_path: Optional[str] = None
    hf_checkpoint: Optional[str] = "hf_out"
    model: Gpt2Config = Gpt2Config()

    tokenizer: str = "gpt2"

    @cached_property
    def the_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.tokenizer)


@pyrallis.wrap()
def main(config: ConvertGpt2Config):
    tokenizer: GPT2Tokenizer = config.the_tokenizer

    # first load our checkpoint
    key = jax.random.PRNGKey(0)

    vocab_size = len(tokenizer)
    Vocab = Axis("vocab", vocab_size)

    if config.checkpoint_path is not None:
        model = Gpt2LMHeadModel(Vocab, config.model, key=key)

        with jax.default_device(jax.devices("cpu")[0]):
            model = deserialize_checkpoint_and_patch_vocab_dim(f"{config.checkpoint_path}/model.eqx", model)
            save_hf_gpt2_checkpoint(config.hf_checkpoint, model)


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

                # for partitioning reasons we frequently round the vocab size, but we need to patch it back
                # to the original size for the HF checkpoint to work
                if any(ax.name == "vocab" for ax in x.axes):
                    index_of_vocab = next(i for i, ax in enumerate(x.axes) if ax.name == "vocab")
                    desired_vocab_size = x.axes[index_of_vocab].size
                    vocab_size = inner.shape[index_of_vocab]
                    if vocab_size != desired_vocab_size:
                        logger.info(
                            f"Patching vocab size from {vocab_size} back to {desired_vocab_size} for HF checkpoint"
                        )
                        inner = jnp.take(inner, jnp.arange(desired_vocab_size), axis=index_of_vocab)

                return NamedArray(inner, x.axes)
            else:
                inner = default_deserialise_filter_spec(f, x)
                return inner

        out = jax.tree_util.tree_map(deserialize_spec, like, is_leaf=is_named_array)
    jax.tree_util.tree_map(_assert_same, out, like)
    return out


if __name__ == "__main__":
    main()
