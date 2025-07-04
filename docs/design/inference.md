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

- [ ] Attention: do prefill for a batch of inputs
- [ ] Attention: do decode for a batch of inputs
- [ ] Attention: unit test that decode gives same output as prefill
- [ ] simple sampler
- [ ] initialize cache for a model (add a method to the lmheadmodel class?)
- [ ] make a main to test the above
- [ ] use paged attention for simple kvcache decoding (should be doable)
- [ ] add support for ragged paged attention
