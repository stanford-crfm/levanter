import equinox as eqx
import jax

import haliax as hax
import haliax.nn as hnn

from levanter.lora import LoraConfig, LoraLinear, lora_state_dict, loraize
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel


In = hax.Axis("In", 10)
Mid = hax.Axis("Mid", 20)
Out = hax.Axis("Out", 5)


def test_loraize_simple():

    k0 = jax.random.PRNGKey(0)
    k1 = jax.random.PRNGKey(1)

    class Module(eqx.Module):
        first: hnn.Linear
        second: hnn.Linear

        def __call__(self, x):
            return self.second(self.first(x))

    module = Module(first=hnn.Linear.init(In, Mid, key=k0), second=hnn.Linear.init(Mid, Out, key=k1))

    loraized = loraize(module, LoraConfig(r=8, target_modules=["first"]), key=k0)
    assert isinstance(loraized.first, LoraLinear)
    assert isinstance(loraized.second, hnn.Linear)

    loraized = loraize(module, LoraConfig(r=8, target_modules=["second"]), key=k0)
    assert isinstance(loraized.first, hnn.Linear)
    assert isinstance(loraized.second, LoraLinear)

    input = hax.random.normal(k0, (In,))
    assert not hax.all(hax.isclose(module(input), loraized(input)))


def test_lora_scan_layers():
    class Module(eqx.Module):
        first: hnn.Linear
        second: hnn.Linear

        def __call__(self, x):
            return self.second(self.first(x))

        @staticmethod
        def init(*, key):
            k1, k2 = jax.random.split(key)
            first = hnn.Linear.init(In, Mid, key=k1)
            second = hnn.Linear.init(Mid, In, key=k2)
            return Module(first, second)

    Layers = hax.Axis("Layers", 3)

    k0 = jax.random.PRNGKey(0)
    module: hnn.Stacked[Module] = hnn.Stacked.init(Layers, Module)(key=jax.random.split(k0, 3))

    loraized = loraize(module, LoraConfig(r=8, target_modules=["first"]), key=k0)
    assert isinstance(loraized, hnn.Stacked)
    assert isinstance(loraized.stacked.first, LoraLinear)
    assert isinstance(loraized.stacked.second, hnn.Linear)

    assert loraized.stacked.first.lora_A.weight.axes == (Layers, hax.Axis("LORA_R", 8), In)
    assert loraized.stacked.first.lora_B.weight.axes == (Layers, Mid, hax.Axis("LORA_R", 8))

    assert loraized.stacked.second.weight.axes == (Layers, Mid, In)
    input = hax.random.normal(k0, (In,))
    assert not hax.all(hax.isclose(module.fold(input), loraized.fold(input)))


def test_lora_peft_integration():
    import peft
    from transformers import AutoModelForCausalLM

    base_hf_model = AutoModelForCausalLM.from_pretrained("stanford-crfm/expanse-gpt2-small-x777")
    peft_config = peft.tuners.LoraConfig(
        base_model_name_or_path="stanford-crfm/expanse-gpt2-small-x777",
        peft_type="lora",
    )
    model = peft.get_peft_model(base_hf_model, peft_config)

    from peft.utils.save_and_load import get_peft_model_state_dict

    hf_dict = get_peft_model_state_dict(model)

    converter = Gpt2Config.default_hf_checkpoint_converter
    lev_model = converter.load_pretrained(Gpt2LMHeadModel, "stanford-crfm/expanse-gpt2-small-x777")

    lora_lev_model = loraize(lev_model, LoraConfig(r=8, target_modules=["c_attn"]), key=jax.random.PRNGKey(0))
    # for some dumb reason, the hf state dict starts with this prefix
    lev_dict = lora_state_dict(lora_lev_model, "base_model.model.transformer")

    assert lev_dict.keys() == hf_dict.keys()

    for k, v in lev_dict.items():
        assert v.shape == hf_dict[k].shape
