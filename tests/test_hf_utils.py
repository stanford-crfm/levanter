import os

from fsspec import AbstractFileSystem

from levanter.compat.hf_checkpoints import load_tokenizer


def test_load_tokenizer_in_memory_fs():
    # sort of like a gs:// path insasmuch as it uses fsspec machinery
    import fsspec

    fs: AbstractFileSystem = fsspec.filesystem("memory")
    directory_of_this_test = os.path.dirname(os.path.abspath(__file__))
    fs.put(f"{directory_of_this_test}/gpt2_tokenizer_config.json", "memory://foo/tokenizer.json")

    with fsspec.open("memory://foo/config.json", "w") as f:
        f.write(
            """{
         "model_type": "gpt2",
         "vocab_size": 5027
         }"""
        )
    tokenizer = load_tokenizer("memory://foo/")
    assert len(tokenizer) == 5027
