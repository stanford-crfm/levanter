# Draft Design Document for Inference

## References

* [JAX LLM Example](https://github.com/jax-ml/jax-llm-examples/blob/b282713880943cebe7183815918fb7dd60922b14/llama3/llama3_jax/model.py) <-- pretty basic kv cache, but easy to follow

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


## Tasks

- [x] Attention: do prefill for a batch of inputs
- [x] Attention: do decode for a batch of inputs
- [x] Attention: unit test that decode gives same output as prefill
- [x] simple sampler
- [x] initialize cache for a model (add a method to the lmheadmodel class?)
- [x] make a main to test the above
- [x] use paged attention for simple kvcache decoding (should be doable)
- [x] add support for ragged paged attention
- [ ] make an offline batch inference that does continuous decoding
- [ ] make an online scheduler that handles new requests and does continuous decoding
- [ ] implement OpenAI compatible API for inference
- [ ] automatic prefix caching
- [ ] microoptimize jit dispatch
- [ ] figure out why the profiler isn't giving me anything useful
