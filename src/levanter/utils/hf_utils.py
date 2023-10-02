import os

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from levanter.utils.py_utils import logical_cpu_core_count


HFTokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast


def num_cpus_used_by_tokenizer(tokenizer) -> int:
    if getattr(tokenizer, "is_fast", False):
        print("\n\nIS FAST SET TO TRUE\n\n")
        if os.getenv("TOKENIZERS_PARALLELISM", "true").lower() in _HF_TOKENIZER_OFF_VALUES:
            print("\n\nTOKENIZER PARALLELISM IS FALSE")
            return 1
        else:
            print("\n\nTOKENIZER PARALLELISM IS TRUE")
            print(f"\n\nLOGICAL CPU CORE COUNT IS {logical_cpu_core_count()}")
            # This is a bit hacky, but HF's fast tokenizers are parallelized under the hood.
            # we reserve a couple of cores just so Ray has somewhere to run the coordinator.
            return max(1, logical_cpu_core_count() - 2)
    else:
        print("\n\nIS FAST SET TO FALSE\n\n")
        return 1


_HF_TOKENIZER_OFF_VALUES = {"off", "false", "f", "no", "n", "0"}
