import equinox as eqx
import jax
import jax.random as jrandom

import haliax as hax

from levanter.models.gpt2 import Gpt2Config
from levanter.optim import AdamConfig


def test_weight_decay_masking():
    def tree_at_mask(params):
        # let's mask all leaves as False
        params = jax.tree_util.tree_map(lambda _: False, params)

        def apply_weight_decay(tree):
            # there is no weight decay performed in LayerNorms and bias
            nodes = []

            # apply on embedding
            nodes.append(tree.embeddings.token_embeddings.weight.array)
            nodes.append(tree.embeddings.position_embeddings.weight.array)

            # apply on attention
            nodes.append(tree.transformer.blocks.stacked.attn.c_attn.weight.array)
            nodes.append(tree.transformer.blocks.stacked.attn.c_proj.weight.array)

            # apply on MLP
            nodes.append(tree.transformer.blocks.stacked.mlp.c_fc.weight.array)
            nodes.append(tree.transformer.blocks.stacked.mlp.c_proj.weight.array)

            return nodes

        # apply weight decay when necessary
        params = eqx.tree_at(
            where=apply_weight_decay,
            pytree=params,
            replace_fn=lambda _: True,
        )

        return params

    gpt_config = Gpt2Config()
    Vocab = hax.Axis("vocab", 100)
    model = gpt_config.build(Vocab, key=jrandom.PRNGKey(0))
    string_list_config = AdamConfig(
        weight_decay_modules=[
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
            "token_embeddings.weight",
            "position_embeddings.weight",
        ]
    )
    regex_config = AdamConfig(
        weight_decay_modules=r".*attn.*weight|.*mlp.*weight|.*token_embeddings|.*position_embeddings",
    )
    # masking using `equinox.tree_at`
    true_mask = tree_at_mask(model)
    # masking using list of module path
    list_string_mask = string_list_config.build_weight_decay_mask()(model)

    regex_mask = regex_config.build_weight_decay_mask()(model)

    assert eqx.tree_equal(list_string_mask, true_mask)
    assert eqx.tree_equal(regex_mask, true_mask)
