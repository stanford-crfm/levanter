from datasets import load_dataset
from transformers import AutoProcessor

from levanter.data.audio import BatchAudioProcessor
from test_utils import skip_if_hf_model_not_accessible


@skip_if_hf_model_not_accessible("openai/whisper-tiny")
def test_whisper_batch_processor():
    processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").select_columns(
        ["audio", "text"]
    )
    batch_processor = BatchAudioProcessor(processor)
    inputs = [(audio, text) for audio, text in zip(ds[:16]["audio"], ds[:16]["text"])]
    batch_processor(inputs)
