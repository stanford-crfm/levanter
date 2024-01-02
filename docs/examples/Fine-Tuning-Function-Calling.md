# Fine-Tuning for Function Calling with Levanter

Function Calling involves generating a function call and its corresponding arguments in response to a user query. 
This is a common need for fine-tuning a Large Language Model to produce structured output suitable for backend service consumption. 
This task can be extended to other structured output tasks like SQL and JSON generation. 

In this post, we'll guide you through the process of fine-tuning a Llama2 model for function calling with the help of Levanter.

## Example Task
Our example task is based on the [GEM ViGGO](https://huggingface.co/datasets/GEM/viggo) dataset. 
It translates conversational English queries into function calls within the video game domain. 
Each example features a plain English input and a structured output that adheres to predefined rules for function names and arguments

Below are some examples from the dataset:

```
Query: `Dirt: Showdown from 2012 is a sport racing game for the PlayStation, Xbox, PC rated E 10+ (for Everyone 10 and Older). It's not available on Steam, Linux, or Mac.`
Expected Response: `inform(name[Dirt: Showdown], release_year[2012], esrb[E 10+ (for Everyone 10 and Older)], genres[driving/racing, sport], platforms[PlayStation, Xbox, PC], available_on_steam[no], has_linux_release[no], has_mac_release[no])`

Query: `Were there even any terrible games in 2014?`
Expected Response: `request(release_year[2014], specifier[terrible])`

Query: `Adventure games that combine platforming and puzzles can be frustrating to play, but the side view perspective is perfect for them. That's why I enjoyed playing Little Nightmares.`
Expected Response: `give_opinion(name[Little Nightmares], rating[good], genres[adventure, platformer, puzzle], player_perspective[side view])`
```

If we test with the Llama2-7B chat model on some examples, we can see its limitation: it struggles to generate accurate function calls and outputs incorrect arguments. 
This highlights the need for fine-tuning—to enhance the model's understanding on this task and producing the intended output.

```
Query: SpellForce 3 is a pretty bad game. The developer Grimlore Games is clearly a bunch of no-talent hacks, and 2017 was a terrible year for games anyway.
Target: give_opinion(name[SpellForce 3], release_year[2017], developer[Grimlore Games], rating[poor])
Generation from Llama2-7B Chat: \nOutput: inform(name[SpellForce 3], release_year[2017], esrb[NA], genres[fantasy,\naction, adventure], platforms[PC], available_on_steam[no], has_linux_release[no],\nhas_mac_release[no])

Query: Have any of the games you've played for the PC had a third person perspective, such as Little Big Adventure?
Target:: suggest(name[Little Big Adventure], player_perspective[third person], platforms[PC])
Generation from Llama2-7B Chat: \nOutput: inform(name[Little Big Adventure], release_year[1994],\nesrb[E (for Everyone)], genres[adventure, platformer],\nplayer_perspective[third person], has_mac_release[no])
```

The dataset has 5,103 training examples and 714 examples for evaluation. 
Although smaller than the Alpaca dataset, it provides enough data to effectively adapt the model to the task.

## Fine-tuning with Levanter

### Step 1: Prepare the Dataset

Let's begin by preparing the dataset. 
Since our dataset is already in a clean, tabular format, minimal preprocessing effort is required. 
Our main tasks are to rename the columns and convert the data into the JSONL format, which is compatible with Levanter. 
For detailed instructions, refer to the [Training on Your Data](docs/Training-on-Your-Data.md) section in our documentation."

Below is a code snippet for dataset preparation. 
The `prompt` provides the model with instructions to enhance its understanding of the task at hand. 
In our example, the prompt details the potential function names and arguments, aiding the model in generating the correct output. 
While helpful, including a prompt is optional for fine-tuning.

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

Now, let's proceed to fine-tune the model with Levanter. 
For this example, we've explored both comprehensive full-weight fine-tuning, similar to the approach used in Alpaca, and the more resource-efficient LoRA fine-tuning. Detailed descriptions of both methods are available in the documentation: [Fine-Tuning](docs/Fine-Tuning.md) and [LoRA](docs/LoRA.md). 
Here's a brief comparison of the two:

- Full-weight fine-tuning: it fine-tunes the entire model weights to better follow the instruction and examples in the training dataset. It is able to leverage the entire  model capacity, but it is expensive and prone to overfitting. 
- LoRA fine-tuning: it adapts the model to the task by adding a small number of parameters (0.1% to 1%) to the model, and train only those parameters. The new parameters are sufficient to capture the task-specific patterns and enable the model to generate the desired output. After training, we merge the new parameters into the original model to be used for inference. It is much more efficient than full-weight fine-tuning, and it is less prone to overfitting.

Levanter provides good support for both methods. Therefore, we can easily try both methods and compare their results.

#### Full-weight Fine-tuning

We start with full-weight fine-tuning. Below is our configuration. Noteably:
- The base model is `meta-llama/Llama-2-7b-hf`. It is set as the default value, so we don't need to specify it explicitly.
- On batch size: We set the batch size to 128, which is the maximum batch size that can fit into a single TPUv3-8. 
- On learnign rate: We compared the results with 3 epochs vs 2 epochs, and found that 2 epochs is sufficient to achieve the best results, while 3 epochs leads to overfitting.

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

Given the small dataset and high efficiency of Levanter, the entire training job completed quickly in only 21 min on a single TPUv3-8. 

Below is an example of the output from the original chat model and the fine-tuned model. We can see that the fine-tuned model is able to generate the correct function call and precisely capture the right arguments, while the original model is not able to do so.

```
Query: SpellForce 3 is a pretty bad game. The developer Grimlore Games is clearly a bunch of no-talent hacks, and 2017 was a terrible year for games anyway.
Target: give_opinion(name[SpellForce 3], release_year[2017], developer[Grimlore Games], rating[poor])
Generation from Llama2-7B Chat: \nOutput: inform(name[SpellForce 3], release_year[2017], esrb[NA], genres[fantasy,\naction, adventure], platforms[PC], available_on_steam[no], has_linux_release[no],\nhas_mac_release[no])
Generation from fine-tuned Llama2-7B: give_opinion(name[SpellForce 3], release_year[2017], developer[Grimlore Games], rating[poor])
```

#### LoRA Fine-tuning

Below is our configuration for LoRA fine-tuning. Note that it is very similar to the full-weight fine-tuning configuration, except for a few differences:
- We increased the number of steps by 1 more epoch. LoRA fine-tuning is more efficient and less prone to overfitting, so we can train for more steps.
- We increased the learning rate to 3e-4, but we did not do thorough hyperparameter tuning. We expect that a better learning rate can lead to better results.
- We found weight decay at 0.1 to lead to better results than 0, so we set it at 0.1.
- We added the `lora` section to specify the LoRA parameters. All of the parameters are set to the default values. 

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
  learning_rate: 3e-4
  weight_decay: 0.1
lora:
  r: 8  # rank of LoRA transform
  alpha: 8.0  # scaling factor for LoRA transform
  dropout: 0.0  # dropout probability for LoRA layers
```

After training, Levanter will automatically merge the new parameters into the original model and save the new model to the specified output directory.

## Results

| Metric | Llama2-7B Chat | Full-weight Fine-tuning | LoRA Fine-tuning |
| ------------------------ | ---- | ----- | ----- |
| Function Name Accuracy   | 0.00 | 0.577 | 0.517 |
| Attribute Set Accuracy   | 0.00 | 0.822 | 0.845 |
| Attribute Value Accuracy | 0.00 | 0.942 | 0.881 |
| Overall Accuracy         | 0.00 | 0.780 | 0.748 |
