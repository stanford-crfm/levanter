import functools
import gzip
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Union, Tuple
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


def bytes_to_tokens(s: str):
    """Convert a byte string to an array of integers."""
    return np.fromiter((char for char in s), count=len(s), dtype=np.int32)


def pad_chunk(s: Optional[np.ndarray], sequence_length: int):
    """Pad an array s out to the given sequence_length."""
    if s is None:
        return np.zeros(sequence_length, dtype=np.int32)
    assert np.ndim(s) == 1
    chunk_len = len(s)
    assert chunk_len <= sequence_length
    if chunk_len == sequence_length:
        return s
    else:
        return np.pad(s, (0, sequence_length - chunk_len), mode="constant", constant_values=0)


def split_article(
    tokens: np.ndarray, sequence_length: int, split: str, include_loss_mask: bool
) -> (Iterable[Tuple[np.ndarray, np.ndarray]]):
    """Split an array into segments of length sequence_length."""
    assert np.ndim(tokens) == 1
    if include_loss_mask:
        loss_mask = loss_mask_from_tokens(tokens, split)

    for k in range(0, len(tokens), sequence_length):
        segment = pad_chunk(tokens[k : k + sequence_length], sequence_length)
        if include_loss_mask:
            segment_loss_mask = pad_chunk(loss_mask[k : k + sequence_length], sequence_length).astype(bool)
        else:
            segment_loss_mask = np.array(True, dtype=bool)  # dummy mask
        yield (segment, segment_loss_mask)


def nonzero_tokens(tokens: np.ndarray, loss_mask: Optional[np.ndarray]) -> list[int]:
    """Removes tokens that are not predicted by the model."""
    # TODO(delesley): Fix the model so that it predicts the first token.
    # The language model doesn't predict the first token.
    toks = [int(tokens[i]) for i in range(1, len(tokens)) if (tokens[i] != 0 and (loss_mask is None or loss_mask[i]))]
    return


def split_and_batch(
    ds: tf.data.Dataset,
    split: str,
    extract_fn: Callable[[Any], Any],
    sequence_length: int,
    batch_size: int,
    auto_rewind: bool = False,
    vocab: Optional[seqio.Vocabulary] = None,
    include_loss_mask: bool = False,
    verbose: bool = False,
) -> tf.data.Dataset:
    """Converts articles to tokens and chops and batches them.
    See batched_interleave_generator for more details.
    Args:
      ds:                A dataset of articles.
      split:             Which dataset split is to be computed, e.g. 'train'.
      extract_fn:        Return a sequence of tokens from article.
      sequence_length:   The number of tokens in each sequence.
      batch_size:        The number of examples in each batch.
      auto_rewind:       If True, will automatically rewind at end of epoch.
      vocab:             Vocabulary, used to count characters.
      include_loss_mask: Return a loss mask for each batch.
      verbose:           Write article info to log as they are read.
    Returns:
      A dataset which yields examples of shape {
          "targets": int32[batch_size, sequence_length],
          "start_of_sequence": bool[batch_size],
          "epoch": int32[batch_size],
          "loss_mask": bool[batch_size, sequence_length],
          "num_chars": A count of the number of detokenized characters.
          "nonzero_tokens": A count of the number of nonzero predicted tokens.
      }
    """

    # Tokenize article, compute loss mask, split into multiple chunks.
    # The entire article must fit into memory.
    def wrap_split_article(article):
        tokens = extract_fn(article)
        if isinstance(tokens, str) or isinstance(tokens, bytes):
            tokens = bytes_to_tokens(tokens)
        elif isinstance(tokens, np.ndarray):
            tokens = tokens.astype(np.int32)
        else:
            raise TypeError("Unusupported sequence type: %s" % str(type(tokens)))
        return split_article(tokens, sequence_length, split=split, include_loss_mask=include_loss_mask)

    # Handle None values.
    def wrap_pad_chunk(s):
        return pad_chunk(s, sequence_length)

    def wrap_batched_interleave_generator():
        return _batched_interleave_generator(
            ds,
            flat_map_func=wrap_split_article,
            post_map_func=wrap_pad_chunk,
            batch_size=batch_size,
            vocab=vocab,
            include_loss_mask=include_loss_mask,
            auto_rewind=auto_rewind,
        )

    out_sig = {
        "targets": tf.TensorSpec(shape=(batch_size, sequence_length), dtype=tf.int32),
        "start_of_sequence": tf.TensorSpec(shape=(batch_size,), dtype=tf.bool),
        "epoch": tf.TensorSpec(shape=(batch_size,), dtype=tf.int32),
        "num_chars": tf.TensorSpec(shape=(batch_size,), dtype=tf.int32),
        "nonzero_tokens": tf.TensorSpec(shape=(batch_size,), dtype=tf.int32),
    }
    if include_loss_mask:
        out_sig["loss_mask"] = tf.TensorSpec(shape=(batch_size, sequence_length), dtype=tf.bool)

    cds = tf.data.Dataset.from_generator(wrap_batched_interleave_generator, output_signature=out_sig)
    return cds


def _batched_interleave_generator(
    ds: tf.data.Dataset,
    flat_map_func: Callable[[str], Iterable[Tuple[np.ndarray, np.ndarray]]],
    post_map_func,
    batch_size: int,
    vocab: Optional[seqio.Vocabulary] = None,
    include_loss_mask: bool = False,
    auto_rewind: bool = False,
) -> Iterable[Dict[str, np.ndarray]]:
    """Generator which combines the interleave and batch dataset operations.
    Given a set of articles from ds, flat_map_func is mapped over the articles
    to break each article up into an iterable of chunks and their loss masks.
    The generator will return the examples from each article in sequential order,
    for transformer-XL style models that process long articles over multiple
    training steps.
    Articles are combined into batches of size batch_size, where each example in
    the batch is pulled from a different article. When one article ends, the
    generator will start pulling examples from the next article.  The overall
    result is similar to tf.Data.Dataset.interleave, except that interleave does
    not always maintain the same order of articles.  If this generator starts
    pulling from article "foo" as the 3rd item in the batch, then consecutive
    examples from "foo" will remain as the 3rd item until the article ends.  This
    guarantee is necessary to pass state from one training step to the next.
    If auto_rewind, then the generator will automatically grab a new iterator
    from ds at the end of the epoch, and increment the epoch counter. Otherwise,
    it will yield empty datasets until all articles in the batch have been
    completed.
    Args:
      ds:            A dataset of articles.
      flat_map_func: A function which returns an iterator over chunks of tokens
        and the loss masks associated with those tokens.
      post_map_func: A function which post-processes each item to fixed size.
      batch_size:    The number of articles in a batch.
      vocab:         The vocabulary to detokenize strings and count characters.
      include_loss_mask: If true, will return a loss mask with the tokens.
      auto_rewind:   Automatically rewind ds at end of epoch.
    Yields:
      Batches of consecutive examples from articles.
      Each example has type: {
        "targets": int32[batch_size, sequence_length],
        "start_of_sequence": bool[batch_size],
        "epoch": int32[batch_size],
        "loss_mask": bool[batch_size, sequence_length],
      }
    """

    ds_iter = ds.as_numpy_iterator()

    document_start = [True] * batch_size  # At start of each article.
    readers = [None] * batch_size  # Iterator for each article
    still_reading = [True] * batch_size  # End of current article?
    item_epochs = [0] * batch_size  # Epoch of the given item.
    epoch = 0

    # Main generator loop
    while any(still_reading):
        targets = [None] * batch_size
        loss_mask = [None] * batch_size
        for i in range(0, batch_size):
            targets_i = None
            loss_mask_i = None
            while targets_i is None and still_reading[i]:
                if readers[i] is not None:
                    try:
                        # Grab the next item from the article.
                        targets_i, loss_mask_i = next(readers[i])
                    except StopIteration:
                        # Article has ended; continue the while loop to grab a new one.
                        readers[i] = None
                else:
                    # Grab the next article from ds if the current one has ended.
                    dsi = None
                    try:
                        dsi = iter(flat_map_func(next(ds_iter)))
                    except StopIteration:
                        logging.info("End of epoch %d.", epoch)
                        if auto_rewind:
                            epoch = epoch + 1
                            logging.info("Starting epoch %d.", epoch)
                            ds_iter = ds.as_numpy_iterator()
                            dsi = iter(flat_map_func(next(ds_iter)))
                        else:
                            still_reading[i] = False  # No more articles on i
                    if dsi is not None:
                        # Start reading the new article.
                        # Continue while loop to grab the first chunk.
                        readers[i] = dsi
                        document_start[i] = True
                        item_epochs[i] = epoch

            # post_map_func must handle None values, and return stackable np.arrays.
            targets[i] = post_map_func(targets_i)  # handles None
            if include_loss_mask:
                loss_mask[i] = post_map_func(loss_mask_i).astype(bool)  # handles None

        # If we've reached the end of all articles, stop immediately.
        if not any(still_reading):
            break

        doc_start_orig = document_start.copy()  # Return doc_start_orig.
        for i in range(0, batch_size):
            # Now that we've read an item, set /start/ to false for each reader.
            document_start[i] = False

        # Decode the tokenized segement back to characters, to count the number
        # of characters for the bits-per-character computation.
        num_chars = [0] * batch_size
        nz_toks = [0] * batch_size
        for i in range(0, batch_size):
            lmask = loss_mask[i] if include_loss_mask else None
            toks = nonzero_tokens(targets[i], lmask)
            if vocab is not None:
                bchars = decode_tokens_1d(toks, vocab, max_length=len(targets[i]), raw_string=True)
                num_chars[i] = len(bchars)
            else:
                num_chars[i] = len(toks)
            nz_toks[i] = len(toks)

        item = {
            "targets": np.stack(targets),
            "start_of_sequence": np.array(doc_start_orig),
            "epoch": np.array(item_epochs),
            "num_chars": np.stack(num_chars),
            "nonzero_tokens": np.stack(nz_toks),
        }
        if include_loss_mask:
            item["loss_mask"] = np.stack(loss_mask)
        yield item


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
