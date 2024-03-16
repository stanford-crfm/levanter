from datasets import load_dataset
from transformers import AutoProcessor

from levanter.data.audio import AudioDatasetSourceConfig, AudioIODatasetConfig, BatchAudioProcessor
from test_utils import skip_if_hf_model_not_accessible, skip_if_no_soundlibs


@skip_if_no_soundlibs
@skip_if_hf_model_not_accessible("openai/whisper-tiny")
def test_whisper_batch_processor():
    processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
    ds = load_dataset("WillHeld/test_librispeech_parquet", split="validation").select_columns(["audio", "text"])
    batch_processor = BatchAudioProcessor(processor)
    inputs = [(audio["array"], audio["sampling_rate"], text) for audio, text in zip(ds[:16]["audio"], ds[:16]["text"])]
    batch_processor(inputs)


@skip_if_no_soundlibs
@skip_if_hf_model_not_accessible("openai/whisper-tiny")
def test_hf_audio_loading():
    # Use the Real Librispeech Valudation. Testing one doesn't support streaming.
    ac = AudioDatasetSourceConfig(id="WillHeld/test_librispeech_parquet", text_key="text")
    audio_iterator = ac.doc_iterator("validation")
    for i in range(10):
        audio, sample, text = next(audio_iterator)


@skip_if_no_soundlibs
@skip_if_hf_model_not_accessible("openai/whisper-tiny")
def test_hf_audio_loading_source():
    # Use the Real Librispeech Valudation. Testing one doesn't support streaming.
    ac = AudioDatasetSourceConfig(id="WillHeld/test_librispeech_parquet", text_key="text")
    audio_iterator = iter(ac.get_shard_source("validation"))
    for i in range(10):
        audio, sample, text = next(audio_iterator)


@skip_if_no_soundlibs
@skip_if_hf_model_not_accessible("openai/whisper-tiny")
def test_hf_audio_ray_pipeline():
    # Use the Real Librispeech Valudation. Testing one doesn't support streaming.
    ac = AudioIODatasetConfig(id="WillHeld/test_librispeech_parquet", text_key="text")
    audio_iterator = iter(ac.validation_set(batch_size=10))
    for i in range(10):
        t = next(audio_iterator)
        assert t["input_features"].shape == (80, 3000), t["input_features"].shape
        assert t["input_ids"].shape == (1024,), t["input_ids"].shape
        assert t["attention_mask"].shape == (1024,), t["attention_mask"].shape
