import tempfile

import equinox
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as onp
import optax
import pytest
from jax.random import PRNGKey
from transformers import AutoModelForCausalLM
from transformers import GPT2Config as HfGpt2Config
from transformers import GPT2LMHeadModel as HfGpt2LMHeadModel

from levanter.config import TrainerConfig
from levanter.models.gpt2 import Gpt2LMHeadModel


def has_torch():
    try:
        import torch  # noqa F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not has_torch(), reason="torch not installed")
def test_hf_gpt2_roundtrip():
    _roundtrip_compare_gpt2_checkpoint("gpt2", None)


@pytest.mark.skipif(not has_torch(), reason="torch not installed")
def test_mistral_gpt2_roundtrip():
    _roundtrip_compare_gpt2_checkpoint("stanford-crfm/expanse-gpt2-small-x777", "checkpoint-60000")


def _rand_input(key: PRNGKey, seq_len: int, vocab_size) -> jnp.ndarray:
    return jrandom.randint(key, (seq_len,), 0, vocab_size)


def _roundtrip_compare_gpt2_checkpoint(model_id, revision):
    import torch

    from levanter.compat.hf_checkpoints import (
        load_hf_gpt2_checkpoint,
        load_hf_model_checkpoint,
        save_hf_gpt2_checkpoint,
    )

    config, data = load_hf_model_checkpoint(model_id, revision=revision)
    config = HfGpt2Config.from_dict(config)
    torch_model: HfGpt2LMHeadModel = AutoModelForCausalLM.from_pretrained(model_id, config=config, revision=revision)
    torch_model.eval()

    model = load_hf_gpt2_checkpoint(model_id, revision=revision)

    input = _rand_input(PRNGKey(0), config.n_positions, config.vocab_size)

    # we compare softmaxes because the numerics are wonky and we usually just care about the softmax
    torch_out = torch_model(torch.from_numpy(onp.array(input)).to(torch.int32).unsqueeze(0))
    torch_out = torch_out.logits[0].detach().cpu().numpy()
    torch_out = jax.nn.softmax(torch_out, axis=-1)

    def compute(input):
        return jax.nn.softmax(model(input, inference=True, key=None), axis=-1)

    compute = jax.jit(compute)
    jax_out = compute(input)
    assert torch_out.shape == jax_out.shape, f"{torch_out.shape} != {jax_out.shape}"
    assert onp.isclose(torch_out, onp.array(jax_out), rtol=1e-2, atol=1e-2).all(), f"{torch_out} != {jax_out}"

    with tempfile.TemporaryDirectory() as tmpdir:
        save_hf_gpt2_checkpoint(tmpdir, model)

        torch_model2: HfGpt2LMHeadModel = AutoModelForCausalLM.from_pretrained(tmpdir, config=config)
        torch_model2.eval()

        torch_out2 = torch_model2(torch.from_numpy(onp.array(input)).to(torch.int32).unsqueeze(0))
        torch_out2 = torch_out2.logits[0].detach().cpu().numpy()
        torch_out2 = jax.nn.softmax(torch_out2, axis=-1)
        assert onp.isclose(torch_out2, onp.array(jax_out), rtol=1e-2, atol=1e-2).all(), f"{torch_out2} != {jax_out}"


# Gradient tests


@pytest.mark.skipif(not has_torch(), reason="torch not installed")
def test_hf_gradient():
    _compare_gpt2_checkpoint_gradients("gpt2", None)


def _compare_gpt2_checkpoint_gradients(model_id, revision):
    import torch

    from levanter.compat.hf_checkpoints import load_hf_gpt2_checkpoint, load_hf_model_checkpoint

    config, data = load_hf_model_checkpoint(model_id, revision=revision)
    config = HfGpt2Config.from_dict(config)
    torch_model: HfGpt2LMHeadModel = AutoModelForCausalLM.from_pretrained(model_id, config=config, revision=revision)
    torch_model.eval()

    model = load_hf_gpt2_checkpoint(model_id, revision=revision)

    input = _rand_input(PRNGKey(0), config.n_positions, config.vocab_size)

    def torch_loss(model, input_ids) -> torch.Tensor:
        return model(input_ids, labels=input_ids)[0]

    torch_out = torch_loss(torch_model, torch.from_numpy(onp.array(input)).to(torch.int64).unsqueeze(0))

    def compute_loss(model, input_ids):
        pred_y = model(input_ids, key=None, inference=True)
        token_loss = jnp.mean(
            optax.softmax_cross_entropy(
                pred_y[:-1],
                jax.nn.one_hot(input_ids[1:], num_classes=model.vocab_size),
            )
        )

        return token_loss

    jax_compute_grad = jax.value_and_grad(compute_loss)
    jax_loss, jax_grad = jax_compute_grad(model, input)

    # gradients are kind of a pain to get at in torch, but we do it anyway
    torch_out.backward()
    torch_dict = torch_model.transformer.state_dict(keep_vars=True)
    torch_dict = {k: v.grad for k, v in torch_dict.items()}

    jax_grad: Gpt2LMHeadModel

    jax_grad_dict = jax_grad.to_torch_dict()

    for jax_key, jax_g in jax_grad_dict.items():
        if jax_key not in torch_dict:
            assert jax_key == "token_out_embeddings"
            continue

        torch_g = torch_dict[jax_key]
        assert onp.isclose(jax_g, torch_g.detach().cpu().numpy(), rtol=1e-2, atol=1e-2).all(), f"{jax_g} != {torch_g}"

    # now we also want to check that the optimizers do similar things
    trainer_config = TrainerConfig(weight_decay=0.0, learning_rate=1e-3, warmup_ratio=0.0)

    if trainer_config.max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(torch_model.parameters(), trainer_config.max_grad_norm)
    torch_optimizer = torch.optim.AdamW(
        torch_model.parameters(),
        lr=trainer_config.learning_rate,
        weight_decay=trainer_config.weight_decay,
        betas=(trainer_config.beta1, trainer_config.beta2),
        eps=trainer_config.epsilon,
    )

    torch_optimizer.step()

    jax_optimizer = trainer_config.optimizer()
    state = jax_optimizer.init(model)
    updates, state = jax_optimizer.update(updates=jax_grad, state=state, params=model)
    new_model = equinox.apply_updates(model, updates)

    new_model_dict = new_model.to_torch_dict()
    torch_dict = torch_model.transformer.state_dict(keep_vars=True)

    # now compare new params
    for key, jax_p in new_model_dict.items():
        if key not in torch_dict:
            assert key == "token_out_embeddings"
            continue
        torch_p = torch_dict[key]
        assert onp.isclose(
            jax_p, torch_p.detach().cpu().numpy(), rtol=1e-3, atol=2e-3
        ).all(), f"{key}: {onp.linalg.norm(jax_p - torch_p.detach().cpu().numpy(), ord=onp.inf)}"
