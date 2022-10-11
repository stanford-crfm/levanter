## Current Status as of 2022-10-10

The current data loader (in levanter/data/text.py and levanter/data/sharded.py) works as follows:

### TokenizedDocumentCache
* We build a TokenizedDocumentCache, which creates a (user-specified) number of shards (default 128 for training). Documents are tokenized (via an HF tokenizer) and written to the cache in batches of 1000 (by default), with each batch being written to the *smallest* shard.
* The underlying format is a Parquet file, for text data this means a sequence of input_ids stored in a batched columnar layout and compressed
* When you iterate through the TokenizedDocumentCache, it reads the shards in a round-robin fashion, and yields batches of documents, as they were written.
* It can optionally "flatten" a batch of documents into a single doc (which are delimited by eos), which is what we do with IndexedDataset.


### IndexedDataset
* At load time, a TokenizedDocumentCache is typically wrapped in an IndexedDataset, which just wraps the
TokenizedDocumentCache and sets a max_seq_len. This is the fundamental data structure that is used by the data loader.
* The IndexedDataset iterates through the TokenizedDocumentCache one batch at a time. The docs are (implicitly)
concatenated together. If a concatenated doc is longer than max_seq_len, then it is split into chunks of max_seq_len. Any left over at the end of a batch is currently thrown out, matching Mistral's behavior.

### ShardedIndexedDataset

* Recall that model computation is done by creating a 2-D grid of TPUs, with the first axis being "data" and the other being "model". All devices on the same row process the same slice of a batch. Typically a row does not span multiple nodes, but usually a node will have multiple rows.
* We can conceptually group the rows into "row groups" such that either a row group is just 1 row, or it spans all rows that are on the same node.
* The job of the ShardedIndexedDataset is to shard the IndexedDataset into a number of shards and loads the data so that each row gets its own data. Each row group of the 2-d mesh is assigned a shard (i.e. a set of cache files) that it loads from exclusively.
* For each batch, a node reads however many examples it needs to fill its row group. We then create a GlobalDeviceArray which orchestrates the shards together.

### Misc notes / problems
* There's no randomness anywhere.
* If documents are very long, this means we're reading from the same doc repeatedly for a batch, which is not ideal.
* We get bitwise determinism so long as the grid configuration doesn't change.
* Because we write to the smallest shard, with a large enough dataset, we should have roughly the same number of tokens in each shard, but it won't be exact.
* Because of the above (and because I didn't know how to do it) we don't have a way for one process to signal that it's done. So we just loop the dataset forever. This isn't ideal for evaluation, if nothing else.
* We haven't implemented seeking in the DataLoader, so resumes are expensive. This is not super hard in principle, but it's not implemented.
* Mentioning again that we drop the last batch of a shard if it's not full. This is not ideal. We should pad it and return masks.

## Resumable, Streaming Dataset with good randomness

Goal: a streaming dataset that is:
1. disk-seek efficient (meaning we don't jump to a random position in a random shard for every sample)
2. reasonably random, including robust to long documents.
3. resumable (meaning it's relatively cheap to resume if a run crashes)
4. doesn't eat too much disk

It's easy to get (1) with streaming, and (2) by jumping to random offsets for every sample. Shuffle buffers
get you (1) and (2) together, but only if documents aren't too long. (3) comes easily
if you do streaming OR random jumping constantly, but is a bit painful with a shuffle buffer.
You can get (1), (2) and (3) if you are willing to lay out the entire shuffled dataset on disk for every epoch. But that's not ideal.

We take a middle path: we stream into a shuffle buffer, but we jump to random offsets after every K samples. Moreover,
we serialize the shuffle buffer, the positions of the datasets,
and the random seed to disk when we checkpoint, so that we can resume easily.

### Tasks

#### IndexedDataset
* [] supports seek that jumps to a random offset in a random shard in the TokenizedDocumentCache
* [] can report current position for serialization
* [] can resume from a position

#### JumpingDataset
* [] has a policy for jumping around in an IndexedDataset
* [] has a random key and a position within the policy
* [] can report key and current position for serialization

#### ShuffleBufferDataset
* [] has a shuffle buffer of size ≤N
* [] has a random key
* [] can report key and shuffle buffer for serialization

#### Misc
* [] dataset hierarchy that exposes the interfaces we need (tree_leaves probably for serialization?)
* [] serialize the dataset on all nodes. This logic might need to be a bit different than for models, since the models all use GDAs and only write plain old arrays once.
* [] make sure we can resume from a checkpoint with bitwise determinism as before
