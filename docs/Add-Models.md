# Add New Models to Levanter

## Write Your Model
### Write Config
We start by writing your model config class. 
This class will register all the hyperparameters and axes of your models. We want to define them as the first step because they will be used in the implementation of the model. 
Note that you don't need to write all of them at once. You can start with the key hyperparameters and axes, and add more as you implement the model.

Hyperparameters are defined with the type and the default value. For example:

```python
seq_len: int = 2048
hidden_dim: int = 4096
intermediate_dim: int = 11008
```

Model axies are used for parallelization. An Axis is registered with its name and size. The size of an Axis is normally associated with a hyperparameter. For example:

```python
Pos = property(lambda self: Axis(name="position", size=self.seq_len))
Embed = property(lambda self: Axis(name="embed", size=self.hidden_dim))
Mlp = property(lambda self: Axis(name="mlp", size=self.intermediate_dim))
```

You can find examples like `Gpt2Config` in [gpt2.py](TODO) and `LlamaConfig` in [llama.py](TODO).

- Converting to HF
- Property

### Write Model
- Break down into modules
    - Mlp
    - Attention
    - Decoder Blocks
    - Transformer
    - Embeddings
    - LM Head
- Write in Jax and Haliax

## Write Tests
### Unit Tests
Unit tests are very useful for testing the correctness of your model implementation. 
It is recommended to have at least one test for each of your modules, so that you can make sure that each module is working as expected and capture any surprises early on, before you test them end-to-end. 

In a unit test, you should test your module in the following aspects:
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

## Training
### Write Training Configuration

### Launch Training Job

### Profile Your Model

## Tips for Optimization
