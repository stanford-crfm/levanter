# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
Interactive REPL for Levanter model inference.

Allows loading/unloading models and submitting chat/text completion requests.

Usage:
    uv run src/levanter/main/inference_repl.py
    uv run src/levanter/main/inference_repl.py --checkpoint_path /path/to/checkpoint
    uv run src/levanter/main/inference_repl.py --hf_checkpoint meta-llama/Llama-3.2-1B
"""

import asyncio
import logging
import shlex
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import levanter
from levanter.compat.hf_checkpoints import RepoRef
from levanter.inference.engine import InferenceEngineConfig
from levanter.inference.openai import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
    InferenceServer,
    InferenceServerConfig,
    _create_chat_completion,
    _create_completion,
)
from levanter.models.llama import LlamaConfig

logger = logging.getLogger(__name__)
console = Console()

# Global state - simple and direct
server: Optional[InferenceServer] = None
server_config: InferenceServerConfig = InferenceServerConfig()
model_name: Optional[str] = None
max_tokens: int = 64  # Default generation max tokens


def parse_options(args):
    """Parse command line options into dict."""
    options = {}
    i = 0
    while i < len(args):
        if args[i].startswith("--"):
            key = args[i][2:]
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                options[key] = args[i + 1]
                i += 2
            else:
                options[key] = True
                i += 1
        else:
            i += 1
    return options


def _reload_server():
    """Reload server with current server_config."""
    global server

    if server:
        server.shutdown()

    server = InferenceServer.create(server_config)


def load_model(model_path: str, **options):
    """Load a model and set it as current."""
    global server, server_config, model_name, max_tokens

    # Reset max_tokens
    max_tokens = int(options.get("max_tokens", 64))

    # Unload existing model first
    if server:
        unload_model()

    # Create config
    base_config = InferenceServerConfig(
        model=options.get("model_type", LlamaConfig()),
        temperature=float(options.get("temperature", 0.7)),
        seed=int(options.get("seed", 42)),
        service=InferenceEngineConfig(max_seqs=2, page_size=8, max_pages_per_seq=16, max_queued_tokens=8),
    )

    is_hf_model = not (
        "://" in model_path
        or model_path.startswith("/")
        or model_path.startswith("./")
        or model_path.startswith("../")
    )

    if is_hf_model:
        server_config = InferenceServerConfig(
            hf_checkpoint=RepoRef.from_string(model_path),
            tokenizer=model_path,
            model=base_config.model,
            temperature=base_config.temperature,
            seed=base_config.seed,
            trainer=base_config.trainer,
            service=base_config.service,
        )
        model_name = model_path
    else:
        tokenizer = options["tokenizer"]  # Will raise KeyError if missing
        server_config = InferenceServerConfig(
            checkpoint_path=model_path,
            tokenizer=tokenizer,
            model=base_config.model,
            temperature=base_config.temperature,
            seed=base_config.seed,
            trainer=base_config.trainer,
            service=base_config.service,
        )
        model_name = Path(model_path).name

    console.print(f"[blue]Loading {model_name}...[/blue]")
    server = InferenceServer.create(server_config)
    console.print(f"[green]✓ Loaded {model_name}[/green]")


def unload_model():
    """Unload the current model."""
    global server, server_config, model_name

    if server:
        console.print(f"[blue]Unloading {model_name}...[/blue]")
        server.shutdown()
        server = None
        server_config = None
        model_name = None
        console.print("[green]✓ Model unloaded[/green]")
    else:
        console.print("[yellow]No model loaded[/yellow]")


def chat(text: Optional[str] = None):
    """Submit a chat completion request."""
    if not server:
        console.print("[red]No model loaded. Use /load first[/red]")
        return

    if text:
        # Single message mode
        messages = [ChatMessage(role="user", content=text)]
        run_chat_completion(messages)
    else:
        # Multi-turn conversation mode
        console.print("[cyan]Entering conversation mode. Type '/exit' to leave[/cyan]")
        messages = []

        while True:
            user_input = input("You: ").strip()
            if user_input == "/exit":
                break

            messages.append(ChatMessage(role="user", content=user_input))
            response = run_chat_completion(messages, print_response=False)

            if response:
                assistant_msg = response.choices[0].message.content
                messages.append(ChatMessage(role="assistant", content=assistant_msg))
                console.print(f"[green]Assistant: {assistant_msg}[/green]")


def run_chat_completion(messages, print_response=True):
    """Run async chat completion."""
    request = ChatCompletionRequest(
        model=model_name or "model",
        messages=messages,
        stop=[server.inference_context.tokenizer.eos_token],
        max_tokens=max_tokens,
        temperature=server_config.temperature,
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    response = loop.run_until_complete(_create_chat_completion(server.inference_context, request))
    loop.close()

    if print_response:
        print_completion_response(response)
    return response


def complete(prompt_text: str):
    """Complete a text prompt."""
    if not server:
        console.print("[red]No model loaded. Use /load first[/red]")
        return

    request = CompletionRequest(
        model=model_name or "model",
        prompt=prompt_text,
        max_tokens=max_tokens,
        temperature=server_config.temperature,
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    response = loop.run_until_complete(_create_completion(server.inference_context, request))
    loop.close()

    print_completion_response(response)


def reload_model(checkpoint_path: str):
    """Reload model from a new checkpoint."""
    global server_config

    if not server:
        console.print("[red]No model loaded to reload[/red]")
        return

    # Update checkpoint path in config
    server_config = InferenceServerConfig(
        checkpoint_path=checkpoint_path,
        hf_checkpoint=None,  # Clear HF checkpoint if reloading from path
        tokenizer=server_config.tokenizer,
        model=server_config.model,
        temperature=server_config.temperature,
        seed=server_config.seed,
        trainer=server_config.trainer,
        service=server_config.service,
    )

    console.print(f"[blue]Reloading from {checkpoint_path}...[/blue]")

    def weight_loader(current_model):
        import equinox as eqx
        import haliax as hax
        import jax.random as jrandom
        from haliax import Axis
        from haliax.partitioning import round_axis_for_partitioning

        from levanter.checkpoint import load_checkpoint
        from levanter.utils.jax_utils import use_cpu_device

        with (
            server_config.trainer.device_mesh,
            hax.axis_mapping(server_config.trainer.compute_axis_mapping),
        ):
            with use_cpu_device():
                key = jrandom.PRNGKey(server_config.seed)
                vocab_size = len(server.inference_context.tokenizer)
                Vocab = round_axis_for_partitioning(
                    Axis("vocab", vocab_size), server_config.trainer.compute_axis_mapping
                )
                model = eqx.filter_eval_shape(server_config.model.build, Vocab, key=key)
                model = load_checkpoint(model, checkpoint_path, subpath="model")
                model = server_config.trainer.mp.cast_to_compute(model)
            return model

    server.reload(weight_loader)
    console.print("[green]✓ Model reloaded successfully[/green]")


def show_info():
    """Show current model information."""
    if not server:
        console.print("[yellow]No model loaded[/yellow]")
        return

    table = Table(title="Model Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Model", model_name or "Unknown")
    table.add_row("Tokenizer", str(server_config.tokenizer))
    table.add_row("Vocab Size", str(len(server.inference_context.tokenizer)))
    table.add_row("Max Sequences", str(server_config.service.max_seqs))
    table.add_row("Temperature", str(server_config.temperature))
    table.add_row("Seed", str(server_config.seed))
    table.add_row("Max Tokens", str(max_tokens))

    console.print(table)


def update_config(key: str, value: str):
    """Update configuration and reload server if needed."""
    global server_config, max_tokens, server

    if not server:
        console.print("[red]No model loaded[/red]")
        return

    # Define configurable parameters with types and help
    PARAMS = {
        "temperature": (float, "Sampling temperature (0.0-2.0)", lambda x: 0.0 <= x <= 2.0),
        "seed": (int, "Random seed for generation", lambda x: x >= 0),
        "max_tokens": (int, "Maximum tokens to generate (1-4096)", lambda x: 1 <= x <= 4096),
    }

    if key == "help":
        table = Table(title="Configurable Parameters")
        table.add_column("Parameter", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("Description", style="white")
        for param, (ptype, desc, _) in PARAMS.items():
            table.add_row(param, ptype.__name__, desc)
        console.print(table)
        return

    if key not in PARAMS:
        console.print(f"[yellow]Unknown parameter: {key}. Use '/config help' for available options[/yellow]")
        return

    ptype, desc, validator = PARAMS[key]
    try:
        parsed_value = ptype(value)
        if not validator(parsed_value):
            console.print(f"[red]Invalid value for {key}: {desc}[/red]")
            return
    except ValueError:
        console.print(f"[red]Invalid {ptype.__name__} value: {value}[/red]")
        return

    # Special handling for max_tokens (not in server config)
    if key == "max_tokens":
        max_tokens = parsed_value
        console.print(f"[green]Max tokens set to {parsed_value}[/green]")
        return

    # For server config params, update config and reload server
    console.print(f"[blue]Updating {key} to {parsed_value} and reloading server...[/blue]")

    # Create new config with updated value
    server_config = InferenceServerConfig(
        checkpoint_path=server_config.checkpoint_path,
        hf_checkpoint=server_config.hf_checkpoint,
        tokenizer=server_config.tokenizer,
        model=server_config.model,
        temperature=parsed_value if key == "temperature" else server_config.temperature,
        seed=parsed_value if key == "seed" else server_config.seed,
        trainer=server_config.trainer,
        service=server_config.service,
    )

    # Reload server with new config
    _reload_server()
    console.print(f"[green]✓ {key.capitalize()} updated to {parsed_value}[/green]")


def print_completion_response(response):
    """Pretty print completion response."""
    for i, choice in enumerate(response.choices):
        if hasattr(choice, "message"):  # Chat completion
            content = choice.message.content
        else:  # Text completion
            content = choice.text

        panel = Panel(content, title=f"Response {i + 1}/{len(response.choices)}", border_style="green")
        console.print(panel)

    # Show usage stats
    if response.usage:
        console.print(
            f"[dim]Tokens - Prompt: {response.usage.prompt_tokens}, "
            f"Completion: {response.usage.completion_tokens}, "
            f"Total: {response.usage.total_tokens}[/dim]"
        )


def show_help():
    """Show help text."""
    help_text = """
[bold cyan]Commands:[/bold cyan]
  /load <path|hf:model>     Load model (e.g., /load meta-llama/Llama-3.2-1B or ./checkpoints/llama)
                            Options: --tokenizer <name> --max_tokens <n>
  /unload                   Unload current model
  /chat [text]              Chat with model (interactive if no text)
  /complete <text>          Complete text prompt
  /reload <checkpoint>      Reload from new checkpoint
  /config <key>=<value>     Update config (use '/config help' for options)
  /info                     Show model information
  /help                     Show this help
  /quit                     Exit

[bold yellow]Direct Input:[/bold yellow]
  Type text to chat with the model
    """
    console.print(Panel(help_text, border_style="blue"))


def _dispatch():
    user_input = input("🤖 > ").strip()

    if not user_input:
        return

    # Handle slash commands
    if user_input.startswith("/"):
        parts = shlex.split(user_input[1:])
        if not parts:
            return

        cmd = parts[0]
        args = parts[1:]

        if cmd in ["quit", "exit"]:
            raise EOFError()

        elif cmd == "load":
            if not args:
                console.print("[red]Usage: /load <path|hf:model> [--tokenizer <name>] [--max_tokens <n>][/red]")
                return
            model_path = args[0]
            options = parse_options(args[1:])
            load_model(model_path, **options)
        elif cmd == "unload":
            unload_model()
        elif cmd == "chat":
            chat(" ".join(args) if args else None)
        elif cmd == "complete":
            if args:
                complete(" ".join(args))
            else:
                console.print("[red]Usage: /complete <prompt>[/red]")
        elif cmd == "reload":
            if args:
                reload_model(args[0])
            else:
                console.print("[red]Usage: /reload <checkpoint_path>[/red]")
        elif cmd == "config":
            if not args:
                console.print("[red]Usage: /config <key>=<value> or /config help[/red]")
            elif args[0] == "help":
                update_config("help", "")
            elif "=" in args[0]:
                key, value = args[0].split("=", 1)
                update_config(key.strip(), value.strip())
            else:
                console.print("[red]Usage: /config <key>=<value> or /config help[/red]")
        elif cmd == "info":
            show_info()
        elif cmd == "help":
            show_help()
        else:
            console.print(f"[red]Unknown command: /{cmd}[/red]")
    else:
        # Default to chat
        chat(user_input)


def main(initial_config: InferenceServerConfig):
    """Main entry point for the REPL."""
    console.print(
        Panel.fit(
            "[bold blue]Levanter Inference REPL[/bold blue]\nType [bold]/help[/bold] for commands",
            border_style="blue",
        )
    )

    # Auto-load if config provided
    if initial_config.checkpoint_path:
        load_model(initial_config.checkpoint_path, tokenizer=initial_config.tokenizer)
    elif initial_config.hf_checkpoint:
        load_model(f"hf:{initial_config.hf_checkpoint.model_name_or_path}")

    while True:
        try:
            _dispatch()
        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    # Cleanup on exit
    if server:
        server.shutdown()
    console.print("[blue]Goodbye![/blue]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    levanter.config.main(main)()
