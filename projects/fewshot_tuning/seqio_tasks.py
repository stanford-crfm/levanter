import functools
import gzip
from typing import Iterable, Mapping, Optional, Union, Tuple
import json

import fsspec
import numpy as np
import seqio
import tensorflow as tf
from fewshot_tuning.hf_vocab import HfVocabulary
from seqio import preprocessors
from transformers import AutoTokenizer

from levanter import logging


# from https://github.com/q-hwang/LM-random-tokens/blob/129ade385beea8a85158566d9dcb5067e1e01712/meliad/transformer/text_dataset.py#L607
#
# Copyright 2022 Google.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class GCJsonDataSource(seqio.DataSource):
    """A `DataSource` that reads a file to provide the input dataset."""

    def __init__(
        self,
        split_to_filepattern: Mapping[str, Union[str, Iterable[str]]],
        num_input_examples: Optional[Mapping[str, int]] = None,
        caching_permitted: bool = True,
        file_shuffle_buffer_size: Optional[int] = None,
    ):
        def gc_lmd_generator(files):
            for file in files:
                # with tf.io.gfile.GFile(file, "rb+") as f:
                with fsspec.open(file, 'rt', compression='infer') as f:
                    for line in f:
                        item = json.loads(line)
                        result = dict()
                        result["targets"] = item["text"]
                        # result['meta'] = item['meta']
                        yield result

        def read_file_fn(file_dataset):

            # return tf.data.Dataset.from_tensor_slices([file_dataset])
            out_sig = {
                "targets": tf.TensorSpec(shape=(), dtype=tf.string),
            }

            def generator():
                return gc_lmd_generator(list(s.decode("utf-8") for s in file_dataset.as_numpy_iterator()))

            return tf.data.Dataset.from_generator(generator, output_signature=out_sig)

        self._split_to_filepattern = split_to_filepattern
        self._reader = read_file_fn
        self._file_shuffle_buffer_size = file_shuffle_buffer_size
        super().__init__(
            splits=split_to_filepattern.keys(),
            num_input_examples=num_input_examples,
            caching_permitted=caching_permitted,
        )

    @property
    def supports_arbitrary_sharding(self) -> bool:
        return False

    def get_dataset(
        self,
        split: str,
        shuffle: bool = True,
        seed: Optional[int] = None,
        shard_info: Optional[seqio.ShardInfo] = None,
    ) -> tf.data.Dataset:
        files = self.list_shards(split)

        if not files:
            raise ValueError(f"No file is found for the file pattern: {self._split_to_filepattern[split]}.")
        files_ds = tf.data.Dataset.from_tensor_slices(np.array(files, dtype=str))

        if shard_info:
            if len(files) < shard_info.num_shards:
                raise ValueError(
                    f"Dataset has too few files to shard. {len(files)} files vs "
                    f"{shard_info.num_shards} shards requested."
                )
            files_ds = files_ds.shard(shard_info.num_shards, shard_info.index)

        if shuffle:
            if self._file_shuffle_buffer_size:
                logging.warning(
                    "`file_shuffle_buffer_size` is explicitly set to %d; this may lead "
                    "to an imperfect file shuffle. Leave `file_shuffle_buffer_size` "
                    "unset for a perfect shuffle.",
                    self._file_shuffle_buffer_size,
                )
            file_shuffle_buffer_size = self._file_shuffle_buffer_size or len(files)
            files_ds = files_ds.shuffle(buffer_size=file_shuffle_buffer_size, seed=seed)

        return self._reader(files_ds)

    def list_shards(self, split: str):
        return tf.io.gfile.glob(self._split_to_filepattern[split])


# LARGER_VOCAB = seqio.SentencePieceVocabulary("gs://n2formal-public-data-netherland/random_token_ckpts_tony/spm/m4_meena_vocab_0304.bos1eos2.spm.64k.model")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
vocab = HfVocabulary(tokenizer)


seqio.TaskRegistry.add(
    "the_pile_gcs",
    GCJsonDataSource(
        split_to_filepattern={
            "train": "gs://levanter-data/pile/train/*.jsonl.zst",
            "validation": "gs://levanter-data/pile/val.jsonl.zst",
            "test": "gs://levanter-data/pile/test.jsonl.zst",
        }
    ),
    preprocessors=[
        #functools.partial(preprocessors.rekey, key_map={"inputs": None, "targets": "text"}),
        seqio.preprocessors.tokenize,
        # t5.data.preprocessors.concatenate_and_split_to_fixed_length,
    ],
    output_features={
        "targets": seqio.Feature(vocab, add_eos=False, dtype=tf.int32),
    },
)

seqio.TaskRegistry.add(
    "the_filtered_pile",
    GCJsonDataSource(split_to_filepattern={"train": "gs://levanter-data/filtered_pile/*.json",}),
    preprocessors=[
        seqio.preprocessors.tokenize,
        # t5.data.preprocessors.concatenate_and_split_to_fixed_length,
    ],
    output_features={
        "targets": seqio.Feature(vocab, add_eos=False, dtype=tf.int32),
    },
)

seqio.MixtureRegistry.add(
    "the_pile_gcs_and_filtered",
    [("the_pile_gcs", 1), ("the_filtered_pile", 1)],
)


#seqio.get_mixture_or_task("the_pile_gcs_and_filtered").get_dataset(
#    split="train",
#    sequence_length={"targets": 1024},
#    shuffle=True,
#    seed=0,
#    use_cached=False,
#)


def do_nothing():
    pass
