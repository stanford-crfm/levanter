# Data Loader Design
## Design as of 2023-04-18

### Goals

We want to support the following:
1) Deterministic batches, even for a changing number of readers (or writers). That is, for any cluster size
during training, we want the same batches to be generated in the same order.
2) Sharded reading and writing. We want to be able to read and write from multiple shards in parallel.
3) Simultaneous reading and writing of shards. We want to be able to start training while we are still building the cache.
4) Fast resumption without losing too much progress. This applies to both *writing* and *reading* the cache. That is, when we
   resume a training run, we want to finish producing the cache and also jump to the right place in the cache for reads.
5) (eventually) shuffling/random access
6) We want to be able to build the cache offline too.
7) We want to support batches that are composed of fragments of documents. In particular, we take a moving window of tokens
   from documents. This implies that the mapping from "documents" to "batches" is not 1:1, or easy to compute.

We want to support the following use cases:
1) We have a larger training dataset, and we want to draw samples from it more or less independently on a large number of machines.
   We don't really care about "epochs"/"passes", but we do want to be able to handle resumes and be deterministic. Ideally, each
   machine only reads from the chunks that it needs to read from.
2) We have a smaller validation dataset, and we want to do a single pass over it. We don't care about resuming, and it's ok if
we have to read the whole dataset on each machine.
3) (Eventually) Like (1) but we want to jump around the dataset. We still care about resuming and determinism, but don't care about epochs.

We focus on (1) and (2) for now.


## Some terminology

* **Shard**: A shard is a list of *raw* documents that not been tokenized/preprocessed.
* **Chunk**: A chunk is a list of *processed* documents that have been tokenized/preprocessed.
* **Reader**: A reader is a process that reads from the cache. Typically there is one reader per machine.
* **Writer**: A writer is a process that writes to the cache. Typically there is one writer per machine.
* **Global ordering**: The global ordering is the ordering of chunks in the cache. This is the order in which
  documents are read by readers. The global ordering is defined with respect to an "idealized" number of readers R*. (See below.)
* **Processor** or **Tokenizer**: A function that takes a raw document and returns a processed document.
* **Example** is a single datum that is fed into the model. Examples are typically composed of fragments of documents.
  For example, we might take a moving window of tokens from the concatenation of a list of preprocessed documents.


We say there are K input shards, W writers, R readers. We assume K >= W (though typically K is not too large), and W ≈ R.
We produce N chunks. We also define an idealized number of readers R*, which defines the global ordering over the data.
Typically R* should be the maximum number of readers we expect to actually use.


## Cache structure
We define a shard cache as a list of "chunks", where each chunk is a parquet file (plus metadata) with an equal
number of documents (except for the last chunks for each shard.)
Each chunk is a list of processed documents. Chunks are ordered round robin from the input shards, so that the c'th global chunk is the
c%K'th chunk of the c/K'th shard, so long as all shards have at least c/K chunks. (After that, we remove shards that
have been exhausted and continue round robin.)
We keep the following metadata:
* For each shard, we keep a list of chunks written so far and whether or not we are done processing that shard.
* For each chunk, we keep the number of documents, token counts/length of various fields, and the number of bytes.
  (This metadata can be used for seeking.)
* For the cache overall, we keep the global ordering of chunks, the number of chunks, and the number of documents.

### Chunk format

A Chunk is an Apache Parquet file with schema dependent on the task. For example, for language modeling, we might have
just a sequence of input_ids per document. We use Apache Parquet because it's compact and doesn't require us to know
much about the datatypes we're using.

Chunks also have metadata stored in a separate json file. This metadata includes the total number of documents in the
chunk, as well as token counts/lengths of various fields. This metadata is used for seeking.

## Cache construction

We use Ray to handle the construction of the cache. There are 4 types of processes/actors that we create using Ray:

* A ChunkCacheBroker actor, whose job is to dispense chunks to readers while the cache is being built. It is also
  responsible for keeping track of the global ordering of chunks.
* A ChunkCacheBuilder actor, which is responsible for building the cache. It forks off processes for processing
   input shards. It acts as a callback for these processes, and accepts chunk metadata from them.
* Shard writer processes, one per input shard. The function _produce_cache_for_shard is the entry point for these processes.
  This function is responsible for reading from the input shard and forking off processes to process chunks of documents.
* Chunk processor processes, which are responsible for processing documents and creating chunks. _produce_chunk is the
  entry point for these processes.

Readers are managed by the model training processes, which read by sending requests to the broker via the Ray. They
are not themselves Ray actors/processes.

## Reproducible Sharded Reading for Training

We want to be able to read from the cache in a way that is deterministic and reproducible, even if the number of readers
changes. We also want readers to only read from the chunks that they need to read from.
We pretend the list of data is infinite by cycling. We do track epochs when reading this way.

NB Our goal is a deterministic ordering over examples, and not merely chunks or even documents.

Given a list of chunks and the idealized number of readers R*, we define the global ordering over chunks as follows:
First define R* iterators over chunks, with `chunk_iterators[r]` being defined as `loop(all_chunks)[r::R*]`.

Next, define a function `mk_examples(chunk_iterator)` that takes a list of iterators over chunks and returns
a list of examples. Define `chunk_examples[r] = mk_examples(chunk_iterators[r])`.
This function depends on our sequence length, etc. Then the ordering over examples is:

`chunk_examples[0][0], chunk_examples[1][0], ..., chunk_examples[R*-1][0], ..., chunk_examples[0][1], chunk_examples[1][1], ..., chunk_examples[R*-1][1], ...`
that is, `example[i] == chunk_examples[i % R*][i // R*]`

If we have $R*$ readers, then each `reader_iterator[r][j] == chunk_examples[r][j] == example[j * R* + r]`.
Moreover, if either R or R* is a multiple of the other, then we still get a nice property where
each reader reads from a strided slice of the chunk_iterators:

(Boring math)
* If we have R readers, then `reader_iterator[r][j] == example[j * R + r] == chunk_examples[(j * R + r) % R*][(j * R + r) // R*]`
* If we have `R == n * R*`, then `reader_iterator[r][j] == example[j * R + r] == chunk_examples[(j * R + r) % R*][(j * R + r) // R*]
 == chunk_examples[r % R*][(j * n * R* + r) // R*] == chunk_examples[r % R*][j * n + r // R*],` so each reader reads from
a strided slice (specifically `islice(..., r//R*, None, n)`)
* If we have `R* == n * R`, then `reader_iterator[r][j] == example[j * R + r] == chunk_examples[(j * R + r) % R*][(j * R + r) // R*]
== chunk_examples[R * (j % n) + r][(j * R + r) // R*]` and so each reader reads from n different chunk_exampless.
so we round-robin over a slice of the chunk_examples.

For other cases (R and R* don't divide each other), there's no simple relationship between the reader and chunk iterators
and you end up reading from everywhere, but that's ok.

# Single-Pass Reading for Evaluation
When we want to do a single pass over the data, we don't cycle and we don't shuffle. We just read the data in order. Boring
and simple.


## Resuming

We need to think about resuming in two cases: resuming writes and resuming reads.

## Resuming Writes

Resuming writes is relatively easy, since we can just keep track of the number of chunks written for each shard and the
number of documents written for each chunk. Then you just skip to the appropriate document and start writing.

### Resuming Reads

We want to understand how to seek to the b'th batch.

There are two cases of resuming we need to think about:

1) The "easy" case where 1 example == 1 (preprocessed) document.
2) The "hard" case where the mapping from examples to documents is not 1:1, but there is some easily computable relationship.

In the first case, each reader `r` reads `documents[r::R]`. The `b`th batch
is `documents[b * batch_size:(b+1) * batch_size]`. Assuming `batch_size % R == 0`, then for the b'th batch, reader r
needs to read `documents[b * batch_size + r: (b+1) * batch_size + r: R] == docs(chunk_iterator[r])[b * batch_size // R:(b+1) * batch_size // R]`.
If we know how many documents are in each chunk, then we can seek to the right place in the chunk.

The second case is broadly similar. In particular, we consider the case where we take moving windows of concatenated documents.
If our metadata includes token counts, then we can skip chunks until we pass `batch_size * tokens_per_example // R` tokens.


## Shuffling

### A brief digression
Why do we want to shuffle during training? Shuffling reduces variance in the gradients. If we have batches
where every example is from the same document/domain, then the gradients for those batches will be correlated.

That said, in our setting where we use moving windows from documents, if we round-robin from chunks (which are produced
from different documents), and R* is roughly equal to the batch size, then we will read from a different chunk for every
example in a batch, which reduces correlation within a batch.

However, we still have (undesirable) correlation between batches: if we
read from chunks consecutively and our documents are long, then many examples in the next batch will be from the
same document as an example in the previous batch. Ideally this wouldn't happen. I'm not convinced that it matters
that much.

Proper shuffling is incompatible with streaming at a fundamental level. Our choices are something like:

* Randomly shuffle before preprocessing. Makes life a bit less pleasant for people with a new dataset. Can't be changed after preprocessing. Doesn't solve the problem of correlated batches.
* Reservoir sampling. Makes resumes hard, but is easy to implement.
* "Epochal" reservoir sampling, where we periodically "flush" the reservoir. Resumes are easier because you can start from the latest "epoch"
* No shuffling in the first pass, but shuffle in subsequent passes.
* Shuffle within a range of chunks that grows as the run progresses.

My hunch is that we can skip this for now, and revisit if we find that it's a problem.


## Current Status as of 2022-10-10

The current data loader (in levanter/data/text.py and levanter/data/sharded.py) works as follows:

### TokenizedDocumentCache
* We build a TokenizedDocumentCache, which creates a (user-specified) number of shards (default 128 for training). Documents are tokenized (via an HF tokenizer) and written to the cache in batches of 1000 (by default), with each batch being written to the *smallest* shard.
* The underlying format is a Parquet file, for text data this means a sequence of input_ids stored in a batched columnar layout and compressed
* When you iterate through the TokenizedDocumentCache, it reads the shards in a round-robin fashion, and yields batches of documents, as they were written.
* It can optionally "flatten" a batch of documents into a single doc (which are delimited by eos), which is what we do with TokenSeqDataset.


### TokenSeqDataset
* At load time, a TokenizedDocumentCache is typically wrapped in an TokenSeqDataset, which just wraps the
TokenizedDocumentCache and sets a max_seq_len. This is the fundamental data structure that is used by the data loader.
* The TokenSeqDataset iterates through the TokenizedDocumentCache one batch at a time. The docs are (implicitly)
concatenated together. If a concatenated doc is longer than max_seq_len, then it is split into chunks of max_seq_len. Any left over at the end of a batch is currently thrown out, matching Mistral's behavior.

### ShardedTokenSeqDataset

* Recall that model computation is done by creating a 2-D grid of TPUs, with the first axis being "data" and the other being "model". All devices on the same row process the same slice of a batch. Typically a row does not span multiple nodes, but usually a node will have multiple rows.
* We can conceptually group the rows into "row groups" such that either a row group is just 1 row, or it spans all rows that are on the same node.
* The job of the ShardedTokenSeqDataset is to shard the TokenSeqDataset into a number of shards and loads the data so that each row gets its own data. Each row group of the 2-d mesh is assigned a shard (i.e. a set of cache files) that it loads from exclusively.
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
5. fully replicable with same configuration (meaning that if you run the same code on the same data, you get the same results)
6. (stretch) fully replicable with different configurations (meaning that if you run the same code on the same data, you get the same results, even if you change the number of nodes)

It's easy to get (1) with streaming, and (2) by jumping to random offsets for every sample. Shuffle buffers get you (1)
and (2) together, but only if documents aren't too long. (3) comes easily if you do streaming OR random jumping
constantly, but is a bit painful with a shuffle buffer. You can get (1), (2) and (3) if you are willing to lay out the
entire shuffled dataset on disk for every epoch. But that's not ideal.


For (1)-(4), we take a middle path: we stream into a shuffle buffer, but we jump to random offsets after every K samples. Moreover,
we serialize the shuffle buffer, the positions of the datasets, and the random seed to disk when we checkpoint, so that we can resume easily.

(5) is easy if you serialize the relevant state, or can just restart your iterators deterministically.
(6) is hard to do in a sharded way. It's easy to "scale down" by emulating a larger number of nodes with a smaller
number of nodes, but it's hard to "scale up". To do this, we can think of each row as having its own stream of data,
perhaps sliced out of a larger stream? TODO for version 3


### Tasks

#### TokenSeqDataset
* [] supports seek that jumps to a random offset in a random shard in the TokenizedDocumentCache
* [] can report current position for serialization
* [] can resume from a position

#### JumpingDataset
* [] has a policy for jumping around in an TokenSeqDataset
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
