from typing import cast

import jax.numpy as jnp
from datasets import load_dataset
from jax.random import PRNGKey
from transformers import WhisperConfig as HfWhisperConfig
from transformers import WhisperModel as HfWhisperModel
from transformers import WhisperProcessor

import haliax as hax
from haliax import Axis

from levanter.compat.hf_checkpoints import RepoRef
from levanter.models.whisper import WhisperConfig, WhisperModel
from test_utils import skip_if_no_torch


def test_basic_forward_whisper():
    c = HfWhisperConfig.from_pretrained("openai/whisper-tiny")
    conf = WhisperConfig.from_hf_config(c)
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperModel.init(conf.Vocab, conf, key=PRNGKey(42))
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    audio_sample = ds[3]
    speech_data = audio_sample["audio"]["array"]
    inputs = processor.feature_extractor(speech_data, sampling_rate=16_000, return_tensors="np")

    na = hax.NamedArray(
        inputs["input_features"],
        axes=(Axis(name="batch", size=1), conf.Mels, Axis(name="position", size=3000)),
    )
    inp = hax.NamedArray(
        jnp.array([processor.get_decoder_prompt_ids()])[:, :, 1],
        axes=(
            Axis("batch", size=1),
            Axis("position", size=1),
        ),
    )
    model(na, inp)


@skip_if_no_torch
def test_hf_roundtrip():
    import torch

    model_id = "openai/whisper-tiny"

    converter = WhisperConfig.default_hf_checkpoint_converter
    c = HfWhisperConfig.from_pretrained(model_id)
    config = WhisperConfig.from_hf_config(c)
    processor = WhisperProcessor.from_pretrained(model_id)
    torch_model: HfWhisperModel = HfWhisperModel.from_pretrained(model_id)
    torch_model.eval()

    model: WhisperModel = cast(WhisperModel, converter.load_pretrained(config, RepoRef(model_id)))
    model = inference_mode(model, True)

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    inputs = processor.feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
    input_features = inputs.input_features
    decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
    # we compare softmaxes because the numerics are wonky and we usually just care about the softmax
    torch_out = torch_model(input_features, decoder_input_ids=decoder_input_ids)
    torch_out = torch_out.logits[0].detach().cpu().numpy()
    torch_out = jax.nn.softmax(torch_out, axis=-1)

    na = hax.NamedArray(
        input_features.cpu().numpy(),
        axes=(Axis(name="batch", size=1), conf.Mels, Axis(name="position", size=3000)),
    )
    inp = hax.NamedArray(
        decoder_input_ids.cpu().numpy(),
        axes=(
            Axis("batch", size=1),
            Axis("position", size=1),
        ),
    )

    def compute(input):
        return hax.nn.softmax(
            model(na, inp),
            axis=model.Vocab,
        )

    compute = jax.jit(compute)
    jax_out = compute(input).array
    assert torch_out.shape == jax_out.shape, f"{torch_out.shape} != {jax_out.shape}"
    assert onp.isclose(torch_out, onp.array(jax_out), rtol=1e-2, atol=1e-2).all(), f"{torch_out} != {jax_out}"

    with tempfile.TemporaryDirectory() as tmpdir:
        converter.save_pretrained(model, tmpdir)

        torch_model2: HfGpt2LMHeadModel = AutoModelForCausalLM.from_pretrained(tmpdir)
        torch_model2.eval()

        torch_out2 = torch_model2(torch.from_numpy(onp.array(input.array)).to(torch.int32).unsqueeze(0))
        torch_out2 = torch_out2.logits[0].detach().cpu().numpy()
        torch_out2 = jax.nn.softmax(torch_out2, axis=-1)
        assert onp.isclose(torch_out2, onp.array(jax_out), rtol=1e-2, atol=1e-2).all(), f"{torch_out2} != {jax_out}"
