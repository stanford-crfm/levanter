from datasets import load_dataset
from transformers import AutoProcessor

from levanter.data.audio import AudioDatasetSourceConfig, AudioIODatasetConfig, BatchAudioProcessor
from test_utils import skip_if_hf_model_not_accessible, skip_if_no_soundlibs


@skip_if_no_soundlibs
@skip_if_hf_model_not_accessible("openai/whisper-tiny")
def test_whisper_batch_processor():
    processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").select_columns(
        ["audio", "text"]
    )
    batch_processor = BatchAudioProcessor(processor)
    inputs = [(audio["array"], audio["sampling_rate"], text) for audio, text in zip(ds[:16]["audio"], ds[:16]["text"])]
    batch_processor(inputs)


def test_hf_audio_loading():
    # Use the Real Librispeech Valudation. Testing one doesn't support streaming.
    ac = AudioDatasetSourceConfig(id="librispeech_asr", name="clean", text_key="text")
    audio_iterator = ac.doc_iterator("validation")
    for i in range(10):
        audio, sample, text = next(audio_iterator)


def test_hf_audio_loading_source():
    # Use the Real Librispeech Valudation. Testing one doesn't support streaming.
    ac = AudioDatasetSourceConfig(id="librispeech_asr", name="clean", text_key="text")
    audio_iterator = iter(ac.get_shard_source("validation"))
    for i in range(10):
        audio, sample, text = next(audio_iterator)


def test_hf_audio_batching():
    # Use the Real Librispeech Valudation. Testing one doesn't support streaming.
    ac = AudioIODatasetConfig(id="librispeech_asr", name="clean", text_key="text", audio_key="audio")
    audio_iterator = ac.validation_set(batch_size=2)
    for i in range(5):
        audio, text = next(audio_iterator)
