# Add New Models to Levanter

This guide outlines the process of adding new models into Levanter. While we emphasize adding models that are implemented in Hugging Face, these steps are applicable to models from other sources.

We'll start with a detailed walkthrough on implementing your model and testing it. Subsequently, we'll describe how to configure and run a training job with your model. To conclude, we'll share insights and recommendations to enhance your model's training efficiency.

## Write Your Model
Writing a new model in Levanter is very similar to Hugging Face. You start by adding a new file in [models/](https://github.com/stanford-crfm/levanter/tree/main/src/levanter/models). You first create a config class to register the key hyperparameters and axes of your model. Then, you write the model class that includes key layers and components of your model.

### Write Config
We start by writing your model config class. This class will register all the hyperparameters and axes of your models. We want to define them as the first step because they will be used immidately in the next step that implements your model.

Note that you don't need to write all of configurations at once. You can start with the key hyperparameters and axes, and add more as you implement the model.

**Hyperparameters** should be declared with their type and a default value, illustrated below:

```python
seq_len: int = 2048
hidden_dim: int = 4096
intermediate_dim: int = 11008
```

**Model axes** are used for parallelization. An Axis is registered with its name and size. The size of an Axis is normally associated with a hyperparameter. For example:

```python
Pos = property(lambda self: Axis(name="position", size=self.seq_len))
Embed = property(lambda self: Axis(name="embed", size=self.hidden_dim))
Mlp = property(lambda self: Axis(name="mlp", size=self.intermediate_dim))
```

For real examples and deeper understanding, check out `Gpt2Config` in [gpt2.py](https://github.com/stanford-crfm/levanter/blob/main/src/levanter/models/gpt2.py) and `LlamaConfig` in [llama.py](https://github.com/stanford-crfm/levanter/blob/main/src/levanter/models/llama.py).

To convert your config class to and from Hugging Face config class, you can write class functions `to_hf_config()` and `from_hf_config()` that maps the input parameters to the corresponding Hugging Face parameters. For example, in Llama, we have the following:

```python
from transformers import PretrainedConfig as HfPretrainedConfig

# ...

@classmethod
def from_hf_config(cls, hf_config: HfPretrainedConfig):
    return LlamaConfig(
        seq_len=hf_config.max_position_embeddings,
        hidden_dim=hf_config.hidden_size,
        intermediate_dim=hf_config.intermediate_size,
        num_layers=hf_config.num_hidden_layers,
        num_heads=hf_config.num_attention_heads,
        activation_function=hf_config.hidden_act,
        initializer_range=hf_config.initializer_range,
        layer_norm_epsilon=hf_config.rms_norm_eps,
        rope_scaling=hf_config.rope_scaling,
    )
```

Lastly, you should register your head model's class name as a class property. This step can be deferred until the head class is constructed.
This class property would make it easier to call your model with the config class. For example:

```python
@property
def model_type(cls) -> Type["LlamaLMHeadModel"]:
    return LlamaLMHeadModel
```

### Write Model
After you have defined your config class, you can start writing your model.
You can follow the breakdown of the model in your Hugging Face implmentation. This would make it easier to validate the implementation through unit tests.

For example, in GPT2, we have the following breakdown:
- `Gpt2Mlp`
- `Gpt2Attention`
- `Gpt2Block`: a block of Gpt2Mlp and Gpt2Attention
- `Gpt2Transformer`: a stack of Gpt2Block
- `Gpt2Embeddings`: token and position embeddings
- `Gpt2LMHead`: a complete GPT2 model with embedding, transformer, and LM head.

We follow the same breakdown in the implementation of Llama in Levanter.

#### Note on the Implementation Format
- Each class will have its key layers and components defined as attributes and be initialized with a static method `init()`.
- Each class will inherit from `StateDictSerializationMixin` from `torch_serialization` and Equinox's `Module` class.

### Mapping weights from/to Hugging Face
To load weights from Hugging Face, you will need to write a class function `from_hf_state_dict()` in each of your model class. It takes in a Hugging Face state_dict and returns a Levanter state_dict.

For implementation, there are a few helper classes from `torch_serialization` that you can use:
- To add specific prefix to the keys of Hugging Face state_dict, you can use the helper function `apply_prefix()`. The prefix comes from the name of attributes defined at the beginning of your model class.
- To unflatten the linear layers of Hugging Face, you can use the helper function `unflatten_linear_params()`.
- To unstack the transformer blocks of Hugging Face, you can use the helper function `unstack_transformer_blocks()`.

For example, below is the implementation of `from_hf_state_dict()` in LlamaAttention:

```python
def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> "LlamaAttention":
    # unflatten the linear layers of HF state_dict to match the shape of LlamaAttention
    d = {}
    d.update(unflatten_linear_layers(apply_prefix(prefix, "q_proj"), state_dict, self.q_proj, True))
    d.update(unflatten_linear_layers(apply_prefix(prefix, "k_proj"), state_dict, self.k_proj, True))
    d.update(unflatten_linear_layers(apply_prefix(prefix, "v_proj"), state_dict, self.v_proj, True))
    d.update(unflatten_linear_layers(apply_prefix(prefix, "o_proj"), state_dict, self.o_proj, True))

    return super().from_state_dict(d, prefix)
```

Similarly, to save weights to Hugging Face, you will need to write a class function `to_hf_state_dict()` in each of your model class.

The correctness of your implementation can be validated through serialization tests, which will be discussed in the next section.

## Write Tests
There are two types of tests that you should write for your model: unit tests and serialization tests.

### Unit Tests
Unit tests are very useful for testing the correctness of your model implementation.
It is recommended to have at least one test for each of your modules, so that you can make sure that each module is working as expected and capture any surprises early on, before you test them end-to-end.

In a unit test, you test your module in the following aspects:
1. The forward pass is successful.
2. The output shape is correct.
3. The output value is correct.

The point 3 is the most important one, but it is also the most difficult one to implement. If you have a reference implementation in another framework, like HuggingFace, you can use it to test your implementation on the output consistency.
Here is a piece of example code to test a same module of Levanter against HuggingFace:

```python
x = random.normal(key, (1, seq_len))
x_torch = torch.from_numpy(np.array(x))  # convert to torch tensor

levanter_rope = LlamaRotaryEmbedding(HeadSize=HeadSize, Pos=Pos)
levanter_output = levanter_rope(seq_len=seq_len)
hf_rope = HFLlamaRotaryEmbedding(dim=hidden_dim, max_position_embeddings=seq_len, device="cpu")
hf_output = hf_rope(x_torch, seq_len=seq_len)

# compare the output values
for jax_out, torch_out in zip(levanter_output, hf_output):
    torch_out = torch_out.numpy()
    assert np.isclose(torch_out, np.array(jax_out.array), rtol=1e-4, atol=1e-4).all(), f"{torch_out} != {jax_out}"
```

For input variables that are common among tests, you can create a helper function to generate them.
For example, below is the helper function for unit tests in Llama:

```python
def _get_random_inputs(config: LlamaConfig):
    Embed = config.Embed
    Pos = config.Pos
    Batch = hax.Axis("batch", 2)
    x = hax.random.normal(random.PRNGKey(0), (Batch, Pos, Embed))
    mask = hax.nn.attention.causal_mask(config.Pos, config.KeyPos)
    return x, mask
```

If your module contains model weights, such consistency tests would require you to load weight from your module in Levanter to the reference module in HuggingFace. We will discuss how to do that in the next section.

### Serialization Tests

In order to validate the implementation on Levanter matches with the reference implementation in HuggingFace, we would like to pass the same input to both implementations and compare the output.
However, if the model are initialized with different weights, such comparison would not be meaningful. Therefore, we will need to serialize the weights from HuggingFace and load them into Levanter, or vice versa. We call such test "serialization test".

For modules like Attention, Mlp, and Embeddings, you can read the weight from Levanter in memory and load into corresponding modules in HuggingFace. For example, in Llama, we did the following:

```python
# initialize the module in Levanter
attention = LlamaAttention.init(config=config, key=random.PRNGKey(0))

# read the weights from Levanter
state = attention.to_state_dict()
state = {k: torch.from_numpy(np.array(v)) for k, v in state.items()}

# load the weights into HuggingFace
hf_attention = HFLlamaAttention(config.to_hf_config(32000))
hf_attention.load_state_dict(state, strict=True)

# ...

# compare the output values
out = attention(x, mask)
hf_out = hf_attention(x_torch, mask_torch)

assert np.isclose(
    hf_out[0].detach().cpu().numpy(), np.array(outarray),
    rtol=1e-4, atol=1e-4
).all()
```

For the end-to-end model, you can save the model weight to disk as a checkpoint and load it into the reference model. For example, in Llama, we can do the following:

```python
converter = LlamaConfig.default_hf_checkpoint_converter

# initialize the model in HF...
hf_model = transformer.AutoModelForCausalLM(...)

# save the model weight to disk
with tempfile.TemporaryDirectory() as tmpdir:
    ck_path = f"{tmpdir}/hf_model"
    hf_model.save_pretrained(ck_path)

    model = converter.load_pretrained(
        LlamaLMHeadModel,
        ck_path,
        resize_vocab_to_match_tokenizer=False
    )

# compare the output values between Levanter and HF
# ...
```

The serialization tests are very useful for testing the correctness of your implementation and make sure you can load your pretrained HuggingFace model into Levanter.

## Training
After you have implemented your model and validated it through tests, you can start training your model with Levanter.

### Write Training Configuration
To launch a training job, you will need to write a training configuration file in yaml. It includes the dataset, model, and trainer specifications for the training job. You can find many examples in [configs/](https://github.com/stanford-crfm/levanter/tree/main/config).

Under the model section, you will need to specify the model name as `type` and modify the hyperparameters that you would like to change. For parameters that are not specified, the default values will be used.

For example, the following configuration uses Llama with default hyperparameters:

```yaml
model:
    type: llama
```

To initialize with the model configuration and checkpoint from Hugging Face, you can specify the `initialize_from_hf` and `use_hf_model_config` parameters:

```yaml
model:
    type: llama
initialize_from_hf: "meta-llama/Llama-2-7b-hf"
use_hf_model_config: true
```

Sometimes it is helpful to start with a small model for debugging. For example, the configuration below specifies a very small Llama model with only 4M parameters. It can be run on a single chip with ease and is useful for debugging.

```yaml
model:
  type: llama
  hidden_dim: 32
  num_heads: 4
  num_layers: 2
```

For more details on the training configuration, please refer to [Configuration Guide](./Configuration-Guide.md).

### Launch Training Job
Once you have your training configuration ready and your training environment set up, you can launch a training job with the following command:

```bash
# HUGGING_FACE_HUB_TOKEN is only needed if you want to load private checkpoint from Hugging Face
WANDB_API_KEY=$WANDB_API_KEY \
HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN \
python levanter/src/levanter/main/train_lm.py --config_path $CONFIG_PATH
```

Check out [Training on Your Own Data](./Training-On-Your-Data.md) for more detailed guide on how to spin off a training cluster and launch a training job.

### Profile Your Model
If you are interested in profiling the training throughput of your model, good news is that it comes for free with automatic job monitoring in Levanter, powered through Weights & Biases.

Once you run a training job, on the corresponding job page on Weights & Biases, you will be able to find a section named "Throughput". It reports metrics like `examples_per_second` and `tokens_per_second` across the training time.

## Tips for Optimization
1. Avoid upcasting to float32. Levanter uses float16 by default, which is more memory efficient and faster for training. You should avoid upcasting to float32 unless it is necessary.
2. For attention, rearrange the heads and position axes to make the computation more efficient. For example, in Llama, we did the following:

```python
q = self.q_proj(x, key=key_q).rearrange((..., "heads", "position", "head_size"))
k = self.k_proj(x, key=key_k).rearrange((..., "heads", "position", "head_size"))
v = self.v_proj(x, key=key_v).rearrange((..., "heads", "position", "head_size"))
```
