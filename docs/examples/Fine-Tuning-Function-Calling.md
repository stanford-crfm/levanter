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

