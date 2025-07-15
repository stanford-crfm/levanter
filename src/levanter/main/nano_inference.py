import argparse
from pathlib import Path

from levanter.inference import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description="Run simple LLM inference")
    parser.add_argument("model", help="Path to HF checkpoint")
    parser.add_argument("prompts", help="Text file with one prompt per line")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=256)
    args = parser.parse_args()

    with open(Path(args.prompts), "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    llm = LLM(args.model)
    sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    outputs = llm.generate(prompts, sp)

    for prompt, out in zip(prompts, outputs):
        print(f"Prompt: {prompt}\nCompletion: {out['text']}\n")


if __name__ == "__main__":
    main()
