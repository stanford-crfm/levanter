import tempfile

from levanter.data.sharded_datasource import AudioTextUrlDataSource, _sniff_format_for_dataset
from test_utils import skip_if_no_soundlibs


def test_sniff_format_for_json():
    # this tests where some people use ".json" to mean a jsonlines file
    # and others use it to mean a json file

    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        f.write(b'[{"text": "hello world"}, {"text": "hello world!"]')
        f.flush()
        assert _sniff_format_for_dataset(f.name) == ".json"

    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        f.write(b'{"text": "hello world"}\n{"text": "hello world!"}\n')
        f.flush()
        assert _sniff_format_for_dataset(f.name) == ".jsonl"

    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        f.write(b'{\n"ids": [1, 2, 3]\n}\n')
        f.flush()
        assert _sniff_format_for_dataset(f.name) == ".json"


@skip_if_no_soundlibs
def test_resolve_audio_pointer():
    AudioTextUrlDataSource.resolve_audio_pointer("https://ccrma.stanford.edu/~jos/mp3/trumpet.mp3", 16_000)
