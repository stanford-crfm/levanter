import tempfile

from levanter.data.sharded_dataset import _sniff_format_for_dataset


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
