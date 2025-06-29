from transformers.models.auto.modeling_auto import AutoModel

from levanter.compat.hf_checkpoints import load_tokenizer
from levanter.generator import Generator, GreedySamplerArgs, Prompt
from levanter.models.llama import LlamaConfig


def main():
    hf_model_name = "meta-llama/Llama-3.2-1B"
    tokenizer = load_tokenizer(hf_model_name)

    hf_config = AutoModel.from_pretrained(pretrained_model_name_or_path=hf_model_name)

    levanter_config = LlamaConfig.from_hf_config(hf_config.config)

    print("Initializing HuggingFace checkpoint converter...")

    # Port the HF checkpoint's config to levanter-friendly config
    converter = levanter_config.hf_checkpoint_converter()
    converter = converter.replaced(reference_checkpoint=hf_model_name, tokenizer=tokenizer)

    llama_model = converter.load_pretrained(converter.default_config.model_type, config=converter.default_config)

    myGen = Generator(hf_config, tokenizer, llama_model)

    myPrompt = Prompt(
        system_instruction="Translate the following phrase into French.",
        input_prompt="I love you.",
    )

    samplingArgs = GreedySamplerArgs(
        prompt=myPrompt,
    )

    output = myGen.generate(samplingArgs, "greedy_sampling")
    print(output, "\nTime Taken:", output.total_time_taken)
    print("Tok/s: ", output.num_tokens / output.total_time_taken)

    output = myGen.scan_generate(samplingArgs, "greedy_sampling")
    print(output, "\nTime Taken (w/ `scan`):", output.total_time_taken)
    print("Tok/s: ", output.num_tokens / output.total_time_taken)


if __name__ == "__main__":
    main()
