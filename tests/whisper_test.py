import jax.numpy as jnp
from datasets import load_dataset
from jax.random import PRNGKey
from transformers import WhisperConfig as HfWhisperConfig
from transformers import WhisperProcessor

import haliax as hax
from haliax import Axis

from levanter.models.whisper import WhisperConfig, WhisperModel


def test_basic_forward_whisper():
    c = HfWhisperConfig.from_pretrained("openai/whisper-small")
    conf = WhisperConfig.from_hf_config(c)
    model = WhisperModel.init(conf, key=PRNGKey(42))
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
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
