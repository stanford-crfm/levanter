from datasets import load_dataset
from jax.random import PRNGKey
from transformers import LlamaConfig as HfLlamaConfig
from transformers import PretrainedConfig as HfConfig  # noqa: E402
from transformers import WhisperConfig as HfWhisperConfig
from transformers import WhisperProcessor

import haliax as hax
from haliax import Axis

from levanter.models.llama import LlamaLMHeadModel
from levanter.models.via import ViaConfig, ViaModel
from levanter.utils.tree_utils import inference_mode
from test_utils import skip_if_no_soundlibs


@skip_if_no_soundlibs
def test_basic_forward_via():
    # Model Setup
    hf_enc_config = HfWhisperConfig.from_pretrained("openai/whisper-tiny")
    hf_dec_config = HfLlamaConfig.from_pretrained("WillHeld/debug_llama")
    merged_config = {
        "encoder": hf_enc_config.to_dict(),
        "decoder": hf_dec_config.to_dict(),
        "time_dialation": 50,
    }
    c = HfConfig.from_dict(merged_config)
    conf = ViaConfig.from_hf_config(c)
    Vocab = hax.Axis("vocab", 1000)
    model = ViaModel.init(Vocab, conf, key=PRNGKey(42), dec_cls=LlamaLMHeadModel)
    model = inference_mode(model, True)

    # Data
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    ds = load_dataset("WillHeld/test_librispeech_parquet", split="validation")
    audio_sample = ds[3]
    speech_data = audio_sample["audio"]["array"]
    inputs = processor.feature_extractor(speech_data, sampling_rate=16_000, return_tensors="np")

    Batch = Axis(name="batch", size=1)
    na = hax.NamedArray(
        inputs["input_features"],
        axes=(Batch, conf.enc_config.Mels, Axis(name="position", size=3000)),
    )
    inp = hax.arange(Axis("position", size=50)).broadcast_axis(Batch)
    model(na, inp)
