from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Union

import jax.numpy as jnp
import pyrallis.utils
from datasets import load_dataset
from jaxtyping import PyTree
from transformers import AutoTokenizer

from haliax import Axis
from levanter.data import Dataset
from levanter.data.ul2r import DecoderOnlyExample, Ul2Example, convert_to_decoder_only
from levanter.shapes import NamedShapeSpec, ShapeSpec


@dataclass
class InstructionDatasetConfig:
    id: str

    input_fields: List[str] = pyrallis.field(default_factory=lambda: ["inputs"])
    output_field: str = "targets"

    def canonicalize_hf(self):
        """Returns a dataset that has (at least) two fields: inputs and outputs"""
        dataset = load_dataset(self.id, split="train", streaming=True)

        def map_fn(example):
            example["inputs"] = "\n".join(example[field] for field in self.input_fields)
            example["targets"] = example[self.output_field]

            return example

        return dataset.map(map_fn)

    def build(self, tokenizer) -> "InstructionTuningDataset":
        dataset = load_dataset(self.id, split="train", streaming=True)
        return InstructionTuningDataset(
            dataset, tokenizer, input_fields=self.input_fields, output_field=self.output_field
        )


class InstructionTuningDataset(Dataset[Ul2Example]):
    def __init__(
        self,
        base_dataset: Iterable[Dict[str, str]],
        tokenizer,
        task_token: Optional[str] = None,
        input_fields: Union[str, List[str]] = "inputs",
        output_field: str = "targets",
    ):
        self.base_dataset = base_dataset
        self.tokenizer = tokenizer

        if task_token is not None:
            self.tokenizer.add_special_tokens({"additional_special_tokens": [task_token]})
        self.task_token_id = tokenizer.convert_tokens_to_ids(task_token)

        if isinstance(input_fields, str):
            input_fields = [input_fields]
        self.input_fields = input_fields
        self.output_field = output_field

    def __iter__(self) -> Iterator[DecoderOnlyExample]:
        for example in self.base_dataset:
            yield self._process_example(example)

    def _process_example(self, example) -> Ul2Example:
        input_text = "\n".join([example[field] for field in self.input_fields])
        input_ids = self.tokenizer.encode(input_text)
        output_text = example[self.output_field]
        output_ids = self.tokenizer.encode(output_text)

        return Ul2Example(inputs=input_ids, outputs=output_ids, task_token=self.task_token_id)

    @property
    def item_shape(self) -> PyTree[Union[ShapeSpec, NamedShapeSpec]]:
        return Ul2Example(
            inputs=ShapeSpec((self.max_length,), dtype=jnp.int32),  # type: ignore
            outputs=ShapeSpec((self.max_length,), dtype=jnp.int32),  # type: ignore
            task_token=self.task_token_id,
        )  # type: ignore

    def __len__(self) -> int:
        return len(self.base_dataset)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # dataset_name = 'Muennighoff/P3'
    dataset_name = "Muennighoff/natural-instructions"
    dataset = InstructionTuningDataset(
        load_dataset(dataset_name, split="train", streaming=True), tokenizer, input_fields=["definition", "inputs"]
    )

    def render(example):
        print(
            f"<<Task>>:{tokenizer.convert_ids_to_tokens(example.task_token or 0)}\n\n"
            f"<<Input>>: {tokenizer.decode(example.inputs)}\n\n"
            f"<<Output>>: {tokenizer.decode(example.outputs)}\n\n"
        )

    import time

    time_in = time.time()
    for i, example in zip(range(1024), dataset):
        # render(example)
        convert_to_decoder_only(example, 0, Axis("SeqLen", 1024), Axis("KSeqLen", 1024))
        # pass

    print(f"Time: {time.time() - time_in}")
