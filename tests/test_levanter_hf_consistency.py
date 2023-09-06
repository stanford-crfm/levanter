import numpy as onp
import transformers
from jax.random import PRNGKey
from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel

import haliax as hax
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning

from levanter.checkpoint import load_checkpoint
from levanter.models.backpack import BackpackLMHeadModel
from levanter.models.gpt2 import Gpt2LMHeadModel
from levanter.trainer import TrainerConfig
from levanter.utils.tree_utils import inference_mode
from test_utils import skip_if_checkpoint_not_accessible, skip_if_hf_model_not_accessible, skip_if_no_torch


HF_BACKPACK = "stanford-crfm/levanter-backpacks-test"
LEVANTER_BACKPACK_CHECKPOINT = "gs://levanter-checkpoints/backpacks/backpack_170M_100k_steps_run_0424-new/step-100000/"
HF_GPT2 = "stanford-crfm/levanter-gpt2-small-for-test"
LEVANTER_GPT2_CHECKPOINT = "gs://levanter-checkpoints/dev/clean-snowball-990-new/step-10000"


@skip_if_checkpoint_not_accessible(LEVANTER_BACKPACK_CHECKPOINT)
@skip_if_hf_model_not_accessible(HF_BACKPACK)
@skip_if_no_torch
def test_hf_backpack_consistency():
    hf_model_config = transformers.AutoConfig.from_pretrained(HF_BACKPACK, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(HF_BACKPACK, config=hf_model_config, trust_remote_code=True)
    hf_model.cuda().eval()

    from levanter.models.backpack import BackpackConfig

    model_config: BackpackConfig = BackpackConfig.from_hf_config(hf_model_config)
    trainer_config = TrainerConfig()

    vocab_size = hf_model_config.vocab_size
    Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), trainer_config.compute_axis_mapping)
    model_key = PRNGKey(0)
    model_levanter = BackpackLMHeadModel.init(Vocab, model_config, key=model_key)
    model_levanter, (_, _), _ = load_checkpoint(
        model_levanter,
        (None, None),
        checkpoint_path=LEVANTER_BACKPACK_CHECKPOINT,
        discover_latest=True,
    )
    mp = trainer_config.mp
    model_levanter = mp.cast_to_param(model_levanter)
    _compare_models_output(model_1=model_levanter, model_2=hf_model)


@skip_if_checkpoint_not_accessible(LEVANTER_GPT2_CHECKPOINT)
@skip_if_hf_model_not_accessible(HF_GPT2)
@skip_if_no_torch
def test_hf_gpt2_consistency():
    hf_model_config = GPT2Config.from_pretrained(HF_GPT2)
    hf_model = GPT2LMHeadModel.from_pretrained(HF_GPT2)
    hf_model.cuda().eval()

    from levanter.models.gpt2 import Gpt2Config

    model_config: GPT2Config = Gpt2Config.from_hf_config(hf_model_config)
    trainer_config = TrainerConfig()

    vocab_size = hf_model_config.vocab_size
    Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), trainer_config.compute_axis_mapping)
    model_key = PRNGKey(0)
    model_levanter = Gpt2LMHeadModel.init(Vocab, model_config, key=model_key)
    model_levanter, (_, _), _ = load_checkpoint(
        model_levanter,
        (None, None),
        checkpoint_path=LEVANTER_GPT2_CHECKPOINT,
        discover_latest=True,
    )
    mp = trainer_config.mp
    model_levanter = mp.cast_to_param(model_levanter)
    _compare_models_output(model_1=model_levanter, model_2=hf_model)


def _compare_models_output(model_1, model_2):
    import torch

    model_1 = inference_mode(model_1, True)
    model_2 = inference_mode(model_2, True)

    input = hax.random.randint(PRNGKey(0), model_1.Pos, 0, model_1.Vocab.size)
    out_1, out_2 = None, None
    if model_1 is not None:
        # model_1 output
        attn_mask = hax.nn.attention.causal_mask(model_1.Pos, model_1.config.KeyPos)

        def compute(input):
            return hax.nn.softmax(
                model_1(input, key=None, attn_mask=attn_mask),
                axis=model_1.Vocab,
            )

        out_1 = compute(input).array
        assert out_1.shape == (
            model_1.Pos.size,
            model_1.Vocab.size,
        ), f"{out_1.shape} != {(model_1.Pos, model_1.Vocab.size)}"

    if model_2 is not None:  # for debugging
        # model_2 output
        with torch.cuda.amp.autocast():
            out_2 = model_2(torch.from_numpy(onp.array(input.array)).to(torch.int32).unsqueeze(0).cuda())
        out_2 = torch.nn.functional.softmax(out_2.logits, dim=-1)
        out_2 = out_2.detach().cpu().numpy()[0]

    if out_1 is not None and out_2 is not None:
        assert out_2.shape == out_1.shape, f"{out_2.shape} != {out_1.shape}"
        assert onp.isclose(out_2, onp.array(out_1), rtol=1e-2, atol=1e-2).all(), f"{out_2} != {out_1}"
