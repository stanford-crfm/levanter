import tempfile

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import numpy as onp
import pytest
from jax.random import PRNGKey
from transformers import AutoModelForCausalLM
from transformers import GPT2Config as HfGpt2Config
from transformers import GPT2LMHeadModel as HfGpt2LMHeadModel

from levanter.compat.torch_checkpoints import (
    load_hf_gpt2_checkpoint,
    load_hf_model_checkpoint,
    save_hf_gpt2_checkpoint,
)


def has_torch():
    try:
        import torch  # noqa F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not has_torch(), reason="torch not installed")
def test_hf_gpt2_roundtrip():
    import torch

    config, data = load_hf_model_checkpoint("gpt2")
    config = HfGpt2Config.from_dict(config)
    torch_model: HfGpt2LMHeadModel = AutoModelForCausalLM.from_pretrained("gpt2")
    torch_model.eval()

    model = load_hf_gpt2_checkpoint("gpt2")

    def rand_input(key: PRNGKey, num: int, seq_len: int) -> jnp.ndarray:
        return jrandom.randint(
            key,
            (
                num,
                seq_len,
            ),
            0,
            config.vocab_size,
        )

    input = rand_input(PRNGKey(0), 1, config.n_positions)

    torch_out = torch_model(torch.from_numpy(onp.array(input)).to(torch.int32))
    torch_out = torch_out.logits[0].detach().cpu().numpy()

    jax_out = model(input[0], key=None)

    assert torch_out.shape == jax_out.shape, f"{torch_out.shape} != {jax_out.shape}"
    assert np.isclose(torch_out, onp.array(jax_out)).all(), f"{torch_out} != {jax_out}"

    with tempfile.TemporaryDirectory() as tmpdir:
        save_hf_gpt2_checkpoint(tmpdir, model)

        torch_model2: HfGpt2LMHeadModel = AutoModelForCausalLM.from_pretrained(tmpdir, config=config)
        torch_model2.eval()

        torch_out2 = torch_model2(torch.from_numpy(onp.array(input)).to(torch.int32))
        torch_out2 = torch_out2.logits[0].detach().cpu().numpy()
        assert np.isclose(torch_out2, onp.array(jax_out)).all(), f"{torch_out2} != {jax_out}"
