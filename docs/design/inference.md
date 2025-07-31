# Draft Design Document for Inference

## References

* [JAX LLM Example](https://github.com/jax-ml/jax-llm-examples/blob/b282713880943cebe7183815918fb7dd60922b14/llama3/llama3_jax/model.py) <-- pretty basic kv cache, but easy to follow
* [Fancier JAX LLM Example](https://github.com/jax-ml/jax-llm-examples/pull/22/files)

### MaxText

* [Page Manager](https://github.com/AI-Hypercomputer/maxtext/blob/eac885edb371e6141a2bb784f9060f816ce17b23/MaxText/inference/page_manager.py)
* [Paged Attention](https://github.com/AI-Hypercomputer/maxtext/blob/main/MaxText/inference/paged_attention.py#L298)
* [Jetstream](https://github.com/AI-Hypercomputer/JetStream) <-- session management and such
* [MaxEngine](https://github.com/AI-Hypercomputer/maxtext/blob/main/MaxText/maxengine.py) <-- maxtext impl of jetstream protocol

### EasyDel

* [VSurge](https://github.com/erfanzar/EasyDeL/tree/main/easydel/inference/vsurge)

### JAX Repo

- [Ragged Paged Attention](https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py)

### Nano-VLLM

torch, not much interesting here given jetstream. focused on batch inference I think?

* https://github.com/GeeeekExplorer/nano-vllm/

### SGLang

* [SGLang](https://github.com/sgl-project/sglang)


## Tasks

- [x] Attention: do prefill for a batch of inputs
- [x] Attention: do decode for a batch of inputs
- [x] Attention: unit test that decode gives same output as prefill
- [x] simple sampler
- [x] initialize cache for a model (add a method to the lmheadmodel class?)
- [x] make a main to test the above
- [x] use paged attention for simple kvcache decoding (should be doable)
- [x] add support for ragged paged attention
- [x] stop sequences

## Offline Milestone
- [ ] make an offline batch inference that does continuous decoding
- [x] Update JIT scheduler to recognize sequences properly using `hax.where` with size/fill
- [x] Remove tokens from the queue based on sequence ID using a JIT-safe mask
- [x] Automatically slide tokens forward when removing entries
- [ ] Integrate stop sequence into decode state
- [ ] Implement max tokens per sequence limit
- [ ] Add option to record logprobs
- [ ] add swapping in new sequences into jit scheduler
- [ ] per-seq PRNG states inside the JIT scheduler
- [ ] Rename jit scheduler to something more appropriate (DecodeState?)
- [ ] Finish offline batch inference support

## LM Eval Harness milestone

- [ ] Support LM Eval Harness generation tasks
- [ ] Stop sequences
- [ ] max num tokens
- [ ] temperature

##  Prefix Cache Integration
- [ ] Use radix tree draft to actually drive scheduling
- [ ] Enable automatic prefix caching
- [ ] When returning from JIT scheduler, insert allocated pages into the radix cache. careful to handle the case where there were the sampled tokens already happen to exist.
- [ ] Radix cache must return freed pages when a sequence is released
- [ ] Manage list of free pages outside the radix cache
- [ ] Double check radix LRU evict

## Serving Milestone
- [ ] Basic server for serving with OpenAI-compatible API
  - [ ] Implement stop token handling
  - [ ] Support `max_new_tokens` and `temperature` params
  - [ ] Support `logprobs` and `echo` params

## Optimization Ideas
- [ ] microoptimize jit dispatch
- [ ] Investigate why tensor parallelism doesn't work for logits / vocab mat mul
- [ ] Avoid computing full logit matrix during prefill unless `logprobs` are requested
- [ ] figure out why the profiler isn't giving me anything useful

## Sample LM Integration
- [ ] expose a free list of pages in `PageTable`
- [ ] allocate pages from the free list inside the generation loop
- [x] store partial sequences in `DecodeState` instead of `JitScheduler`
- [x] check `DecodeState.is_finished` during generation
- [ ] remove `PageTable` from the core decoding loop
- [ ] integrate any remaining pieces needed for `sample_lm`
