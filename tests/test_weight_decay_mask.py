import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

import haliax as hax

from levanter.models.gpt2 import Gpt2Config
from levanter.optim import AdamConfig
from levanter.optim.config import TagPattern
from levanter.utils.jax_utils import leaf_key_paths


def test_weight_decay_masking():
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
    true_mask = _tree_at_mask(model)
    # masking using list of module path
    list_string_mask = string_list_config.build_weight_decay_mask()(model)

    regex_mask = regex_config.build_weight_decay_mask()(model)

    assert eqx.tree_equal(list_string_mask, true_mask)
    assert eqx.tree_equal(regex_mask, true_mask)


def test_weight_decay_masking_with_class_names():
    gpt_config = Gpt2Config()
    Vocab = hax.Axis("vocab", 100)
    model = gpt_config.build(Vocab, key=jrandom.PRNGKey(0))
    string_list_config = AdamConfig(
        weight_decay_modules=[
            "Linear.weight",
            "Embedding.weight",
        ]
    )

    # masking using `equinox.tree_at`
    true_mask = _tree_at_mask(model)
    # masking using list of module path
    list_string_mask = string_list_config.build_weight_decay_mask()(model)

    assert eqx.tree_equal(list_string_mask, true_mask)


def test_weight_decay_masking_with_param_tags():
    gpt_config = Gpt2Config()
    Vocab = hax.Axis("vocab", 100)
    model = gpt_config.build(Vocab, key=jrandom.PRNGKey(0))

    tag_config = AdamConfig(
        param_tags=[
            TagPattern(
                pattern=[
                    "attn.c_attn.weight",
                    "attn.c_proj.weight",
                    "mlp.c_fc.weight",
                    "mlp.c_proj.weight",
                    "token_embeddings.weight",
                    "position_embeddings.weight",
                ],
                tag="decay",
            )
        ],
        weight_decay_modules=["decay"],
    )

    true_mask = _tree_at_mask(model)
    tag_mask = tag_config.build_weight_decay_mask()(model)

    assert eqx.tree_equal(tag_mask, true_mask)


def test_weight_decay_tag_priority_over_pattern():
    gpt_config = Gpt2Config()
    Vocab = hax.Axis("vocab", 100)
    model = gpt_config.build(Vocab, key=jrandom.PRNGKey(0))

    config = AdamConfig(
        param_tags=[TagPattern(pattern="mlp.c_fc.weight", tag="hidden")],
        weight_decay_modules=["mlp.c_fc.weight"],
    )

    mask = config.build_weight_decay_mask()(model)

    paths = leaf_key_paths(model)
    flat_mask, _ = jax.tree_util.tree_flatten(mask)
    flat_paths, _ = jax.tree_util.tree_flatten(paths)
    for m, p in zip(flat_mask, flat_paths):
        if "mlp.c_fc.weight" in p:
            assert not m


def test_default_weight_decay_masking():
    gpt_config = Gpt2Config()
    Vocab = hax.Axis("vocab", 100)
    model = gpt_config.build(Vocab, key=jrandom.PRNGKey(0))
    default_config = AdamConfig(default_weight_decay_mask=True)
    no_default_config = AdamConfig(default_weight_decay_mask=False)
    equivalent_default_config = AdamConfig(weight_decay_modules=None, default_weight_decay_mask=None)

    no_default_mask = no_default_config.build_weight_decay_mask()
    assert no_default_mask is None

    # masking using `equinox.tree_at`
    tree_at_mask = _tree_at_mask(model, decay_embeddings=False)
    # masking using list of module path
    default_mask = default_config.build_weight_decay_mask()(model)
    equivalent_default_tree = equivalent_default_config.build_weight_decay_mask()(model)

    assert eqx.tree_equal(default_mask, equivalent_default_tree)
    assert eqx.tree_equal(default_mask, tree_at_mask)


def test_learning_rate_tags_scale_updates():
    params = {"w": jnp.array(1.0), "b": jnp.array(1.0)}
    grads = {"w": jnp.array(1.0), "b": jnp.array(1.0)}

    cfg = AdamConfig(
        learning_rate={"default": 1.0, "bias": 0.5},
        weight_decay=0.0,
        max_grad_norm=None,
        param_tags=[TagPattern(pattern="b", tag="bias")],
    )

    opt = cfg.build(10)
    state = opt.init(params)
    updates, _ = opt.update(grads, state, params)

    ratio = (updates["b"] / updates["w"]).item()
    assert np.isclose(ratio, 0.5)


def test_learning_rate_tags_multiple_and_default():
    params = {"w": jnp.array(1.0), "b": jnp.array(1.0), "c": jnp.array(1.0)}
    grads = {"w": jnp.array(1.0), "b": jnp.array(1.0), "c": jnp.array(1.0)}

    cfg = AdamConfig(
        learning_rate={"default": 1.0, "bias": 0.5, "special": 0.25},
        weight_decay=0.0,
        max_grad_norm=None,
        param_tags=[
            TagPattern(pattern="b", tag="bias"),
            TagPattern(pattern="c", tag="special"),
        ],
    )

    opt = cfg.build(10)
    state = opt.init(params)
    updates, _ = opt.update(grads, state, params)

    assert np.isclose((updates["b"] / updates["w"]).item(), 0.5)
    assert np.isclose((updates["c"] / updates["w"]).item(), 0.25)


def test_weight_decay_pattern_used_when_no_tag():
    gpt_config = Gpt2Config()
    Vocab = hax.Axis("vocab", 100)
    model = gpt_config.build(Vocab, key=jrandom.PRNGKey(0))

    cfg = AdamConfig(
        param_tags=[TagPattern(pattern="bias", tag="bias")],
        weight_decay_modules=["mlp.c_proj.weight"],
    )

    mask = cfg.build_weight_decay_mask()(model)

    paths = leaf_key_paths(model)
    flat_mask, _ = jax.tree_util.tree_flatten(mask)
    flat_paths, _ = jax.tree_util.tree_flatten(paths)

    found = False
    for m, p in zip(flat_mask, flat_paths):
        if "mlp.c_proj.weight" in p:
            found = True
            assert m

    assert found


def test_param_tags_first_match_wins():
    gpt_config = Gpt2Config()
    Vocab = hax.Axis("vocab", 100)
    model = gpt_config.build(Vocab, key=jrandom.PRNGKey(0))

    cfg = AdamConfig(
        param_tags=[
            TagPattern(pattern="mlp", tag="first"),
            TagPattern(pattern="mlp.c_fc.weight", tag="second"),
        ],
        weight_decay_modules=["first"],
    )

    mask = cfg.build_weight_decay_mask()(model)

    paths = leaf_key_paths(model)
    flat_mask, _ = jax.tree_util.tree_flatten(mask)
    flat_paths, _ = jax.tree_util.tree_flatten(paths)

    for m, p in zip(flat_mask, flat_paths):
        if "mlp.c_fc.weight" in p:
            assert m


def _tree_at_mask(params, decay_embeddings=True):
    # let's mask all leaves as False
    params = hax.tree_util.tree_map(lambda _: False, params)

    def apply_weight_decay(tree):
        # there is no weight decay performed in LayerNorms and bias
        nodes = []

        # apply on embedding
        if decay_embeddings:
            nodes.append(tree.embeddings.token_embeddings.weight)
            nodes.append(tree.embeddings.position_embeddings.weight)

        # apply on attention
        nodes.append(tree.transformer.blocks.stacked.attn.c_attn.weight)
        nodes.append(tree.transformer.blocks.stacked.attn.c_proj.weight)

        # apply on MLP
        nodes.append(tree.transformer.blocks.stacked.mlp.c_fc.weight)
        nodes.append(tree.transformer.blocks.stacked.mlp.c_proj.weight)

        return nodes

    # apply weight decay when necessary
    params = eqx.tree_at(
        where=apply_weight_decay,
        pytree=params,
        replace_fn=lambda _: True,
    )

    return params
