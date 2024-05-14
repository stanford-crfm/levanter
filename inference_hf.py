from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM


path = "allenai/OLMo-1B" # "stanford-crfm/llama-1b-dolma"  "llama_1b_hf" "allenai/OLMo-1B"
if "OLMo" in path:
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
else:
    model = LlamaForCausalLM.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids, max_length=100)
output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output)
