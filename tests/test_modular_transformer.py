import equinox as eqx
import jax.random as jrandom

import haliax as hax

from levanter.layers.attention import AttentionMask
from levanter.models.modular_transformer import ModularConfig, ModularLMHeadModel
from levanter.utils.activation import ActivationFunctionEnum


class ZeroAttention(eqx.Module):
    """Simple attention module that returns zeros for testing."""

    @staticmethod
    def init(config, *, key):
        return ZeroAttention()

    def __call__(self, x, mask, *, key=None, pos_ids=None):
        return hax.zeros_like(x)


class MlpOnlyLayer(eqx.Module):
    """Layer that applies only an MLP and layer norm."""

    config: ModularConfig = eqx.field(static=True)
    mlp: eqx.Module
    ln: eqx.Module

    @staticmethod
    def init(config: ModularConfig, *, key):
        mlp = config.mlp_cls.init(
            config.Embed, config.Mlp, config.activation_function, key=key, use_bias=config.use_bias
        )
        ln = config.mk_LayerNorm(config.Embed)
        return MlpOnlyLayer(config, mlp, ln)

    def __call__(self, x, mask, *, key=None, pos_ids=None):
        y = self.ln(x)
        y = self.mlp(y, key=key)
        return x + y


def _random_input(cfg: ModularConfig):
    Batch = hax.Axis("batch", 2)
    Vocab = hax.Axis("vocab", 32)
    inputs = hax.random.randint(jrandom.PRNGKey(0), (Batch, cfg.Pos), 0, Vocab.size)
    mask = AttentionMask.causal()
    return Batch, Vocab, inputs, mask


def test_can_swap_activation():
    cfg = ModularConfig(num_layers=1, activation_function=ActivationFunctionEnum.relu)
    Batch, Vocab, inputs, mask = _random_input(cfg)
    model = ModularLMHeadModel.init(Vocab, cfg, key=jrandom.PRNGKey(1))
    logits = model(inputs, mask)
    assert logits.array.shape == (Batch.size, cfg.Pos.size, Vocab.size)


def test_can_swap_attention():
    cfg = ModularConfig(num_layers=1, attention_cls=ZeroAttention)
    Batch, Vocab, inputs, mask = _random_input(cfg)
    model = ModularLMHeadModel.init(Vocab, cfg, key=jrandom.PRNGKey(2))
    logits = model(inputs, mask)
    assert logits.array.shape == (Batch.size, cfg.Pos.size, Vocab.size)


def test_can_swap_layer_cls():
    cfg = ModularConfig(num_layers=1, layer_cls=MlpOnlyLayer, attention_cls=ZeroAttention)
    Batch, Vocab, inputs, mask = _random_input(cfg)
    model = ModularLMHeadModel.init(Vocab, cfg, key=jrandom.PRNGKey(3))
    logits = model(inputs, mask)
    assert logits.array.shape == (Batch.size, cfg.Pos.size, Vocab.size)
