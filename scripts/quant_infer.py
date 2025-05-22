import torch
from transformers.models.auto.modeling_auto import AutoModel

from levanter.compat.hf_checkpoints import load_tokenizer
from levanter.generator import Generator, GreedySamplerArgs, Prompt
from levanter.models.llama import LlamaConfig


LOAD_DTYPE = torch.float32


def main():
    hf_model_name = "meta-llama/Llama-3.2-1B"

    tokenizer = load_tokenizer(hf_model_name)

    hf_config = AutoModel.from_pretrained(pretrained_model_name_or_path=hf_model_name, torch_dtype=LOAD_DTYPE).config

    levanter_config = LlamaConfig.from_hf_config(hf_config)

    print("Initializing HuggingFace checkpoint converter...")
    converter = levanter_config.hf_checkpoint_converter()
    converter = converter.replaced(reference_checkpoint=hf_model_name, tokenizer=tokenizer)
    chkp_config = converter.default_config

    llama_model = converter.load_pretrained(chkp_config.model_type, config=chkp_config)
    myGen = Generator(hf_config, tokenizer, llama_model)

    myPrompt = Prompt(system_instruction="Translate the following phrase into French.", input_prompt="I love you.")

    samplingArgs = GreedySamplerArgs(
        prompt=myPrompt,
    )
    output = myGen.generate(samplingArgs, "greedy_samping")

    myPrompt = Prompt(system_instruction="Translate the following phrase into German.", input_prompt="I love you.")

    samplingArgs = GreedySamplerArgs(
        prompt=myPrompt,
    )

    output = myGen.generate(samplingArgs, "greedy_samping")

    print(f"\nFull decoded sentence:\n{'=' * 10}\n{output.decoded_output}\n{'=' * 10}")


if __name__ == "__main__":
    main()
