import os
from dataclasses import dataclass
from typing import List, Optional

import levanter
from levanter.data.text import LMDatasetConfig


@dataclass
class TrainTokenizerConfig(LMDatasetConfig):
    new_vocab_size: Optional[int] = None
    enforce_bos_eos: bool = True

    special_tokens: Optional[List[str]] = None

    output_dir: str = "output"
    upload_to_hub: bool = False
    hub_name: Optional[str] = None


@levanter.config.main()
def main(config: TrainTokenizerConfig):
    old_tokenizer = config.the_tokenizer
    vocab_size = config.new_vocab_size or old_tokenizer.vocab_size

    if config.enforce_bos_eos:
        # By default HF BPE tokenizers don't append EOS tokens to the end of the text
        from tokenizers.processors import RobertaProcessing

        old_tokenizer._tokenizer.post_processor = RobertaProcessing(
            sep=(old_tokenizer.bos_token, old_tokenizer.bos_token_id),
            cls=(old_tokenizer.eos_token, old_tokenizer.eos_token_id),
        )

    import tqdm

    data = tqdm.tqdm(config.doc_iterator("train"), "reading docs", unit="docs")

    new_tokenizer = old_tokenizer.train_new_from_iterator(
        data, vocab_size=vocab_size, new_special_tokens=config.special_tokens
    )

    new_tokenizer.save_pretrained(config.output_dir)

    if config.upload_to_hub:
        dest = config.hub_name or os.path.basename(config.output_dir)
        print(f"Uploading to HuggingFace Hub at {dest}...")
        new_tokenizer.push_to_hub(dest)


if __name__ == "__main__":
    main()
