import os

from fsspec import AbstractFileSystem

from levanter.compat.hf_checkpoints import load_tokenizer
from levanter.utils.hf_utils import byte_length_of_token
from test_utils import skip_if_hf_model_not_accessible


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


@skip_if_hf_model_not_accessible("meta-llama/Llama-2-7b-hf")
def test_byte_length_of_token():
    tok = load_tokenizer("meta-llama/Llama-2-7b-hf")
    ids = tok("this is hello a test", add_special_tokens=False)["input_ids"]
    assert byte_length_of_token(tok, ids[2]) == len(" hello".encode("utf-8"))
    assert byte_length_of_token(tok, 25) == 1
    # llama prepends a space to the token. ideally it wouldn't b/c it technically throws off our bpb calculations
    # but it's a small difference
    assert byte_length_of_token(tok, ids[0]) == len(" this".encode("utf-8"))

    bos = tok.bos_token_id
    assert byte_length_of_token(tok, bos) == 0


@skip_if_hf_model_not_accessible("gpt2")
def test_byte_length_of_token_gpt2():
    tok = load_tokenizer("gpt2")
    ids = tok("this is hello a test", add_special_tokens=False)["input_ids"]
    assert byte_length_of_token(tok, ids[2]) == len(" hello".encode("utf-8"))

    eos = tok.eos_token_id
    assert byte_length_of_token(tok, eos) == 0
