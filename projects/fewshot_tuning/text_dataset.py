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

"""Load text datasets for long-range transformer models."""

from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import jax
import numpy as np
import seqio
import tensorflow as tf

from levanter import logging


_DEFAULT_DATA_DIRECTORY = None


def get_loss_mask_tokens(
    split: str,
    loss_mask_start_tokens: Sequence[int] = (),
    loss_mask_end_tokens: Sequence[int] = (),
    splits: Sequence[str] = ("all",),
) -> Tuple[Sequence[int], Sequence[int]]:
    """Returns two token sequences to indicate start and end of the loss.

    Please configure loss_mask_start_tokens, loss_mask_end_tokens, and
    split_filter via gin. Example gin config to only apply loss between tokens 2
    and 1 for the test set (and everywhere for any other data split):

    ```
    text_dataset.get_loss_mask_tokens:
      loss_mask_start_tokens=(2,)
      loss_mask_end_tokens=(1,)
      restrict_to_splits=("test",)
    ```

    Args:
      split: The mode ("test", "train", ...)
      loss_mask_start_tokens: token sequence to starts the loss
      loss_mask_end_tokens: token sequence to stop the loss
      splits: Only compute the loss mask for splits in this list.
        By default it is 'all', which is a reserved split string that applies to
        all splits.
    """
    if "all" in splits or split in splits:
        return loss_mask_start_tokens, loss_mask_end_tokens
    return (), ()


def load_text_dataset(
    name: str,
    split: str,
    sequence_length: int,
    batch_size: int,
    sequential: bool = True,
    shard_dataset: bool = True,
    verbose: bool = False,
) -> Tuple[tf.data.Dataset, seqio.Vocabulary]:
    """Load a text dataset of long articles or books, and split_and_batch them.

    The input dataset must produce complete books or articles, where each article
    is a dictionary containing a "tokens" field.
    See split_and_batch for more information on the output dataset.

    Args:
      name:  The name of the seqio task which produces the dataset.
      split: The name of the split to use, e.g. "train" or "test".
      sequence_length: Split text into sequences of this length.
      batch_size: Draw from batch_size articles in each batch.
      sequential: If True, return the chunks of each article in sequence.
      shard_dataset: If True, split data set into shards.
      verbose: Log (an excerpt) of every text example loaded from disk. If False,
        will only print 1 excerpt every 60 seconds.

    Returns:
      (dataset, vocabulary)
      where vocabulary is the seqio.Vocabulary which is used to encode "targets".
    """

    logging.info("Loading text data set %s, split=%s, shape=(%d, %d)", name, split, batch_size, sequence_length)

    # Bypass the seqio "feature converter", and get the task directly.
    task = seqio.get_mixture_or_task(name)
    vocab = task.output_features["targets"].vocabulary

    # Create the task input pipeline.
    if shard_dataset:
        logging.info("Shards: %d of %d", jax.process_index(), jax.process_count())
        shard_info = seqio.ShardInfo(index=jax.process_index(), num_shards=jax.process_count())
    else:
        shard_info = None

    if sequential:
        task_seqlen = None  # We do our own splitting.
        shuffle_buffer_size = 1000  # Number of full-length books.
    else:
        task_seqlen = {"targets": sequence_length}  # Ask the task to do splitting.
        shuffle_buffer_size = 10_000  # Number of chunks.

    ds = task.get_dataset(
        sequence_length=task_seqlen,
        split=split,
        use_cached=False,
        shuffle=True,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=42,
        shard_info=shard_info,
        num_epochs=1,
    )

    if sequence_length == 0:
        return (ds, vocab)  # Don't chop into subsequences.

    def extract_fn(article):
        return article["targets"]

    include_loss_mask = bool(get_loss_mask_tokens(split)[0])
    ds = split_and_batch(
        ds,
        split=split,
        extract_fn=extract_fn,
        sequence_length=sequence_length,
        batch_size=batch_size,
        auto_rewind=True,
        vocab=vocab,
        include_loss_mask=include_loss_mask,
        verbose=verbose,
    )
    return (ds, vocab)


def pretty_print_article(
    article, vocab_map: Mapping[str, Optional[seqio.Vocabulary]], max_length: int = 60, max_batch_size=3
) -> str:
    """Convert the contents of a long article to a short string."""
    if not hasattr(article, "items"):
        return pretty_print_value(article, max_length, max_batch_size)  # Not a dictionary.
    dstr = "{"
    for (k, v) in article.items():
        if vocab_map and k in vocab_map:
            vstr = decode_tokens(v, vocab_map[k], max_length, max_batch_size)
        else:
            vstr = pretty_print_value(v, max_length, max_batch_size)
        dstr += "\n  " + k + ": " + vstr
    return dstr + "\n}"


def pretty_print_value(value, max_length: int, max_batch_size: int) -> str:
    """Convert a possibly large value to a short string."""
    if isinstance(value, bytes):
        if len(value) <= max_length:
            return str(value)
        else:
            return f"bytes[{len(value)}] " + str(value[:max_length]) + "..."
    elif isinstance(value, str):
        if len(value) <= max_length:
            return value
        else:
            return f"str[{len(value)}] " + value[:max_length] + "..."
    elif isinstance(value, np.ndarray):
        vstr = f"ndarray({value.shape}, {value.dtype.str})"
        if value.size <= (max_length / 4):
            vstr += " = " + str(value)
        elif np.ndim(value) == 2:
            vstr += " = " + str(value[:max_batch_size, :max_length])
        return vstr
    elif np.ndim(value) == 0:
        return str(value)  # Scalar data.
    else:
        return str(type(value))


def decode_tokens(tokens: Any, vocab: seqio.Vocabulary, max_length: int, max_batch_size: int = 3) -> str:
    """Convert tokens to a human-readable string."""
    if isinstance(tokens, np.ndarray):
        tstr = f"ndarray({tokens.shape}, {tokens.dtype.str}) = "
    else:
        tstr = f"{str(type(tokens))} = "

    if np.ndim(tokens) == 1:
        tstr += decode_tokens_1d(tokens, vocab, max_length)  # type: ignore
    elif np.ndim(tokens) == 2:
        chunks = [decode_tokens_1d(s, vocab, max_length) for s in tokens[:max_batch_size]]
        jtstr = ",\n    ".join(chunks)  # type: ignore
        tstr += f"[\n    {jtstr}\n  ]"
    else:
        tstr = pretty_print_value(tokens, max_length, max_batch_size)
    return tstr


def decode_tokens_1d(tokens: Any, vocab: Any, max_length: int, raw_string: bool = False) -> Union[str, bytes]:
    """Convert a 1D array of tokens to a human-readable string.

    Args:
      tokens:     1-dimensional array of integers.
      vocab:      The vocabulary to detokenize the array.
      max_length: The maximum number of tokens to detokenize.
      raw_string: If True, return the string as bytes.
                  If false, pretty print it (e.g. with "\n").

    Returns:
      The detokenized string.
    """

    assert np.ndim(tokens) == 1
    # The type of tokens is np.ndarray((sequence_length,), "int32")
    # We have to convert this to an actual list of python integers, NOT numpy
    # integers, or decode will blow up, and fail to marshall the data to C++.
    dtoks = [int(i) for i in tokens[:max_length]]
    tstr = vocab.decode(dtoks)

    # Convert the decoded string to a byte string.
    # PassThroughVocabulary returns a list, not a string.
    try:
        if isinstance(tstr, str):
            tstr = bytes(tstr.encode("utf-8"))
        else:
            tstr = bytes(tstr)
    except:  # noqa: E722
        if len(tokens) > max_length:
            return str(dtoks) + "..."
        else:
            return str(dtoks)

    # If raw_string, return immediately.
    if raw_string:
        return tstr

    # Otherwise format it for pretty-printing.
    # Converting bytes to str will convert, e.g., newlines as "\n".
    tstr = str(tstr)
    if len(tokens) > max_length:
        tstr += "..."
    return tstr


def bytes_to_tokens(s: str):
    """Convert a byte string to an array of integers."""
    return np.fromiter((char for char in s), count=len(s), dtype=np.int32)


def pad_chunk(s: Optional[np.ndarray], sequence_length: int, pad_token: int):
    """Pad an array s out to the given sequence_length."""
    if s is None:
        return np.zeros(sequence_length, dtype=np.int32)
    assert np.ndim(s) == 1
    chunk_len = len(s)
    assert chunk_len <= sequence_length
    if chunk_len == sequence_length:
        return s
    else:
        return np.pad(s, (0, sequence_length - chunk_len), mode="constant", constant_values=pad_token)


def split_article(
    tokens: np.ndarray, sequence_length: int, split: str, include_loss_mask: bool, pad_token: int
) -> (Iterable[Tuple[np.ndarray, np.ndarray]]):
    """Split an array into segments of length sequence_length."""
    assert np.ndim(tokens) == 1, tokens.shape
    if include_loss_mask:
        loss_mask = loss_mask_from_tokens(tokens, split)  # noqa: F841

    for k in range(0, len(tokens), sequence_length):
        segment = pad_chunk(tokens[k : k + sequence_length], sequence_length, pad_token)
        if include_loss_mask:
            # segment_loss_mask = pad_chunk(loss_mask[k : k + sequence_length], sequence_length, pad_token).astype(bool)
            raise NotImplementedError("loss_mask is not implemented yet")
        else:
            segment_loss_mask = np.array(True, dtype=bool)  # dummy mask
        yield (segment, segment_loss_mask)


def nonzero_tokens(tokens: np.ndarray, loss_mask: Optional[np.ndarray]) -> list[int]:
    """Removes tokens that are not predicted by the model."""
    # TODO(delesley): Fix the model so that it predicts the first token.
    # The language model doesn't predict the first token.
    toks = [int(tokens[i]) for i in range(1, len(tokens)) if (tokens[i] != 0 and (loss_mask is None or loss_mask[i]))]
    return toks


def _find_subsequence_idxs(sequence: np.ndarray, subsequence: Sequence[int]):
    """Returns the indices where `subsequence` occurs in `sequence`."""
    subsequence = np.asarray(subsequence, dtype=np.int32)
    # use np.where as an efficient way to iterate over the whole array; but we can
    # only test for a single token, unfortunately.
    potential_matches = np.where(sequence == subsequence[0])[0]
    match_indices = []
    for start_index in potential_matches:
        if np.array_equal(sequence[start_index : start_index + len(subsequence)], subsequence):
            match_indices.append(start_index)
    return match_indices


def loss_mask_from_tokens(tokens: np.ndarray, split: str) -> np.ndarray:
    """Compute a mask for language modelling loss using start and end tokens."""
    assert np.ndim(tokens) == 1
    tokens = tokens.astype(np.int32)

    # Position offset of loss mask and target positions. Typically -1, which
    # indicates that targets are shifted 1 position left compared to inputs.
    offset = -1

    start_tokens, end_tokens = get_loss_mask_tokens(split=split)
    if not start_tokens:
        # default to not masking out any loss
        return np.ones_like(tokens, dtype=bool)

    start = 0
    end = len(tokens)  # include end_tokens
    start_indices = _find_subsequence_idxs(tokens, start_tokens)
    if start_indices:
        if end_tokens:
            end_indices = _find_subsequence_idxs(tokens, end_tokens)
        else:
            end_indices = []
        if len(start_indices) > 1 or len(end_indices) > 1:
            logging.error("Multiple start or end tokens for loss mask: %s, %s", start_indices, end_indices)
        start = start_indices[0]
        if end_indices and end_indices[0] >= start:
            end = end_indices[0]

        # We include the start_tokens and the end_tokens, which represents that the
        # model must predict the location, the content, and the end of the
        # subsequence.
        start += offset
        start = max(0, start)  # to prevent offset creating negative indices
        end += len(end_tokens) + offset

    # Create the actual mask. Roughly equivalent to
    # mask = np.array([i >= start && i <= end for i in range(len(tokens))])
    mask = np.concatenate(
        [
            np.zeros((start,), dtype=bool),
            np.ones((end - start,), dtype=bool),
            np.zeros((len(tokens) - end,), dtype=bool),
        ]
    )
    return mask


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
                        targets_i, loss_mask_i = next(readers[i])  # type: ignore
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
                        readers[i] = dsi  # type: ignore
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
                bchars = decode_tokens_1d(toks, vocab, max_length=len(targets[i]), raw_string=True)  # type: ignore
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

    if vocab:
        pad_token = vocab.pad_id
    else:
        pad_token = 0

    # Tokenize article, compute loss mask, split into multiple chunks.
    # The entire article must fit into memory.
    def wrap_split_article(article):
        #if verbose:
        #    logging.info("Reading article: %s", pretty_print_article(article, {}))
        #else:
        #    logging.log_every_n_seconds(logging.INFO, "Reading article: %s", 60, pretty_print_article(article, {}))
        tokens = extract_fn(article)
        if isinstance(tokens, str) or isinstance(tokens, bytes):
            tokens = bytes_to_tokens(tokens)
        elif isinstance(tokens, np.ndarray):
            tokens = tokens.astype(np.int32)
        else:
            raise TypeError("Unusupported sequence type: %s" % str(type(tokens)))
        return split_article(tokens, sequence_length, split=split, include_loss_mask=include_loss_mask, pad_token=pad_token)


    # Handle None values.
    def wrap_pad_chunk(s):
        return pad_chunk(s, sequence_length, pad_token)

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
