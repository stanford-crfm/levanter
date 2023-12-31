# Fine-Tuning for Function Calling with Levanter

Function Calling describes the task of generating a function call and its arguments given a user query. 
It is a very common need for fine-tuning a Large Language Model, so that it can generate structured output that can be consumed by a backend service. 
It can be extended to other structured output generation tasks, such as SQL generation, JSON generation, etc.

In this post, we will show how to fine-tune a Llama2 model for function calling using Levanter. 

## Example Task
The example task that we will use is based on the [GEM ViGGO](https://huggingface.co/datasets/GEM/viggo) dataset. 
It is an English-to-Function-Call dataset in the video game domain. In each example, the input is conversational query in plain English, and the output is in structural representation with pre-defined rules on function names and arguments. 

Below are some examples from the dataset:

```
Query: `Dirt: Showdown from 2012 is a sport racing game for the PlayStation, Xbox, PC rated E 10+ (for Everyone 10 and Older). It's not available on Steam, Linux, or Mac.`
Expected Response: `inform(name[Dirt: Showdown], release_year[2012], esrb[E 10+ (for Everyone 10 and Older)], genres[driving/racing, sport], platforms[PlayStation, Xbox, PC], available_on_steam[no], has_linux_release[no], has_mac_release[no])`

Query: `Were there even any terrible games in 2014?`
Expected Response: `request(release_year[2014], specifier[terrible])`

Query: `Adventure games that combine platforming and puzzles can be frustrating to play, but the side view perspective is perfect for them. That's why I enjoyed playing Little Nightmares.`
Expected Response: `give_opinion(name[Little Nightmares], rating[good], genres[adventure, platformer, puzzle], player_perspective[side view])`
```

The dataset is relatively small, with 5,103 training examples and 714 examples for evaluation.

## Fine-tuning with Levanter

### Step 1: Prepare the Dataset

First, we need to prepare the dataset. 
In this example, the dataset is already in a clean, tabular format, so there is very little preprocessing needed.
We primarily want to rename the columns and convert the dataset into the JSONL format that is compatible with Levanter.
See [Training on Your Data](docs/Training-on-Your-Data.md) for more details.


Below is an example code snippet for preparing the dataset. 
Note that the `prompt` is a set of instructions for the model to better understand the task. 
In this case, the prompt enumerates the possible function names and arguments, and the model can use them to generate the output. 
It is nice-to-have but not required for fine-tuning.

```python
import json
import datasets


PROMPT = """
You are a helpful chatbot to assist users to convert natural language sentences into function call.

# omit the rest of the prompt for brevity
... 
"""

# load the dataset
train_dataset = datasets.load_dataset("GEM/viggo", split="train")

# rename the columns
train_dataset = train_dataset.map(
    lambda example: {
        "instruction_field": PROMPT,
        "input": example["target"],
        "output": example["meaning_representation"],
    }
)

# save the dataset in JSONL format
with open("train.jsonl", "w") as f:
    for example in train_dataset:
        json.dump(example, f)
        f.write("\n")
```

### Step 2: Fine-tune the Model

Next, we can fine-tune the model using Levanter. 
In this example, we conducted both full-weight fine-tuning, as used in Alpaca, and more efficient LoRA fine-tuning with Levanter. 
Both methods have been described thoroughly in the doc [Fine-Tuning](docs/Fine-Tuning.md) and [LoRA](docs/LoRA.md). Here I will provide an overview of the high-level differences between the two methods:
- Full-weight fine-tuning: it fine-tunes the entire model weights to better follow the instruction and examples in the training dataset. It is able to leverage the entire  model capacity, but it is expensive and prone to overfitting. 
- LoRA fine-tuning: it adapts the model to the task by adding a small number of parameters (0.1% to 1%) to the model, and train only those parameters. The new parameters are sufficient to capture the task-specific patterns and enable the model to generate the desired output. After training, we merge the new parameters into the original model to be used for inference. It is much more efficient than full-weight fine-tuning, and it is less prone to overfitting.

Levanter provides good support for both methods. Therefore, we can easily try both methods and compare the results.

#### Full-weight Fine-tuning

Below is our configuration for full-weight fine-tuning:

```yaml
data: "gs://levanter-data/fine-tuning/gem-viggo/GEM_viggo_train.jsonl"
data_cache_dir: "gs://levanter-data/tokenized/GEM_viggo_llama/"
trainer:
  wandb:
    project: "levanter"
    tags: ["viggo", "llama", "full-weight"]
  mp: p=f32,c=bfloat16
  train_batch_size: 128
  num_train_steps: 80  # 5103 examples / 128 batch size * 2 epochs
  tensor_parallel_axes: ["mlp", "heads"]
optimizer:
  learning_rate: 2E-5
```

#### LoRA Fine-tuning

Below is our configuration for LoRA fine-tuning. Note that it is very similar to the full-weight fine-tuning configuration, except for a few differences:
- We increased the number of steps by 1 more epoch
- We increased the learning rate
- We added the `lora` section to specify the LoRA parameters.

```yaml
data: "gs://levanter-data/fine-tuning/gem-viggo/GEM_viggo_train.jsonl"
data_cache_dir: "gs://levanter-data/tokenized/GEM_viggo_lora/"
trainer:
  wandb:
    project: "levanter"
    tags: ["viggo", "llama", "lora"]

  mp: p=f32,c=bfloat16
  train_batch_size: 128
  num_train_steps: 120  # 5103 examples / 128 batch size * 3 epochs
  tensor_parallel_axes: ["mlp", "heads"]
optimizer:
  # values in qlora paper
  learning_rate: 3e-4
lora:
  r: 8  # rank of LoRA transform
  alpha: 8.0  # scaling factor for LoRA transform
  dropout: 0.0  # dropout probability for LoRA layers
```

## Results

| Metric | Llama2-7B Chat | Full-weight Fine-tuning | LoRA Fine-tuning |
| ------------------------ | ---- | ----- | ----- |
| Function Name Accuracy   | 0.00 | 0.577 | 0.517 |
| Attribute Set Accuracy   | 0.00 | 0.822 | 0.845 |
| Attribute Value Accuracy | 0.00 | 0.942 | 0.881 |
| Overall Accuracy         | 0.00 | 0.780 | 0.748 |
