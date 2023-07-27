import equinox as eqx
import jax

import haliax as hax
import haliax.nn as hnn

from levanter.lora import LoraConfig, LoraLinear, loraize


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

    module: hnn.Stacked[Module] = hnn.Stacked.init(Layers, Module)(key=jax.random.split(jax.random.PRNGKey(0), 3))

    loraized = loraize(module, LoraConfig(r=8, target_modules=["first"]), key=jax.random.PRNGKey(0))
    assert isinstance(loraized, hnn.Stacked)
    assert isinstance(loraized.stacked.first, LoraLinear)
    assert isinstance(loraized.stacked.second, hnn.Linear)

    assert loraized.stacked.first.lora_A.weight.axes == (Layers, In, hax.Axis("LORA_R", 8))
    assert loraized.stacked.first.lora_B.weight.axes == (Layers, hax.Axis("LORA_R", 8), Mid)

    assert loraized.stacked.second.weight.axes == (Layers, Mid, In)
