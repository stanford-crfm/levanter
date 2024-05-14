from transformers import AutoTokenizer, LlamaForCausalLM


PATH = "llama_1b_hf"  # "meta-llama/Llama-2-7b-hf"
model = LlamaForCausalLM.from_pretrained(PATH)
tokenizer = AutoTokenizer.from_pretrained(PATH)

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids, max_length=30)
output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output)
