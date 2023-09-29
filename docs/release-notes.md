# Levanter Release Notes

## 1.1

### Major Features

* Added Llama, including Llama 1 and Llama 2.
* Added LoRA support:
    * We added a new main entry point called `levanter.main.lora_lm` that can read in a Hugging Face checkpoint and loraize it, then train on a corpuse. Great for lightweight finetuning.
    * Checkpoints are fully compatible with Hugging Face's [PEFT](https://github.com/huggingface/peft/)!
* Extensive new documentation including:
  * [A new tutorial on training your own LM, including continued pretraining](./Training-On-Your-Data.md)
  * [Replicating Alpaca with Llama 1 and Llama 2](./tutorials/Replicating-Alpaca.md)
  * [Creating an Alpaca with LoRA](./tutorials/Alpaca-LoRA.md)
  * [An in-depth guide to configuration](./Configuration-Guide.md)
  * [A detailed guide to porting models](./Port-Models.md)
* Pure-JAX implementation of Flash-Attention 2 added to GPT-2 implementation.
  * (Note: doesn't really improve speed in JAX except for very long sequences.)

### Misc Improvements

* Added background data loading, reducing overhead for training small models substantially.
* Various performance improvments to loading checkpoints
* Bumped various dependencies.
* Improved TPU spin-up scripts.

## 1.0

See [the announcement post for 1.0](./Levanter-1.0-Release.md).
