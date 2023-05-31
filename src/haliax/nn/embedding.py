import equinox as eqx

import haliax as hax

from ..core import Axis, NamedArray
from ..jax_utils import named_call


class Embedding(eqx.Module):
    weight: NamedArray

    # axes
    Vocab: Axis = eqx.static_field()
    Embed: Axis = eqx.static_field()

    def __init__(
        self,
        Vocab: Axis,
        Embed: Axis,
        initializer_range: float = 0.02,
        *,
        key,
    ):
        super().__init__()

        self.Vocab = Vocab
        self.Embed = Embed

        self.weight = hax.random.generate_sharded(hax.random.normal)(key, (Vocab, Embed)) * initializer_range

    @named_call
    def __call__(self, input_ids, inference, *, key):
        return self.embed(input_ids)

    def embed(self, input_ids):
        input_embeds = self.weight.take(self.Vocab, input_ids)
        return input_embeds

    def unembed(self, input_embeds):
        return input_embeds.dot(self.Embed, self.weight)
