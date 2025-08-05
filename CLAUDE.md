# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

### Testing
```bash
# Default test command (CI equivalent with proper XLA flags)
XLA_FLAGS=--xla_force_host_platform_device_count=8 PYTHONPATH=tests:src:. uv run pytest tests

# Run tests excluding slow, entry, and ray tests
XLA_FLAGS=--xla_force_host_platform_device_count=8 PYTHONPATH=tests:src:. uv run pytest tests -m "not entry and not slow and not ray"

# Run specific test categories
uv run pytest tests -m "slow"        # Only slow tests
uv run pytest tests -m "entry"       # Only entry point tests  
uv run pytest tests -m "ray"         # Only Ray-dependent tests

# Set up test environment
export PYTHONPATH=/path/to/levanter/src:path/to/levanter/tests:$PYTHONPATH
wandb offline  # Disable wandb for local testing
```

### Code Quality
```bash
# Run pre-commit hooks on all files (use uv run)
uv run pre-commit run --all-files

# Install pre-commit hooks (required for development)
pre-commit install

# Additional code quality tools
uv run ruff check    # Linting and formatting
uv run mypy         # Type checking
```

### Training Commands
```bash
# Train GPT2-nano model (quick test)
python -m levanter.main.train_lm --config_path config/gpt2_nano.yaml

# Train with custom dataset
python -m levanter.main.train_lm --config_path config/gpt2_small.yaml --data.id openwebtext

# Evaluation with language model evaluation harness
python -m levanter.main.eval_lm --config_path config/harness/eval_llama3.yaml

# Supervised fine-tuning (SFT)
python -m levanter.main.sft --config_path config/sft_llama3_mixture.yaml

# Export trained model to Hugging Face format
python -m levanter.main.export_lm_to_hf --config_path path/to/trained/model/config.yaml
```

### Installation and Setup
```bash
# Development installation (use uv for dependency management)
pip install -e .

# Install with test dependencies
pip install -e .[test]

# If developing both Haliax and Levanter together
git clone https://github.com/stanford-crfm/haliax.git
cd haliax && pip install -e . && cd ../levanter

# Command-line config overrides supported
python -m levanter.main.train_lm --config_path config/gpt2_small.yaml --trainer.num_train_steps 10000
```

## Project Architecture

Levanter is a framework for training large language models built on JAX, Equinox, and Haliax (named tensor library).

### Core Architecture Components

**Configuration System**: Uses Draccus for hierarchical configuration management with YAML files in `config/`. All training parameters are configurable through these files.

**Model Architecture**: Located in `src/levanter/models/`, supports:
- GPT-2 variants
- LLaMA (1/2/3)
- Mistral/Mixtral
- Gemma
- Backpack models
- Qwen
- Whisper (for ASR)

**Training System**: 
- `trainer.py`: Core distributed training logic with FSDP and tensor parallelism
- `trainer_state.py`: Training state management and checkpointing
- Supports TPU and GPU training with hardware-agnostic design

**Data Pipeline**: 
- `data/`: Streaming data processing with caching
- Supports mixture datasets, custom tokenizers
- Online preprocessing with distributed caching via TensorStore
- HuggingFace dataset integration

**Optimization**: 
- `optim/`: Custom optimizers including Sophia, AdamW variants
- Support for gradient accumulation and mixed precision training

**Distributed Training**:
- Built on JAX's distributed training primitives
- Supports FSDP (Fully Sharded Data Parallel) and tensor parallelism
- Automatic device mesh configuration

### Key Directories Structure

- `src/levanter/main/`: Entry points for training, evaluation, and inference
- `src/levanter/models/`: Model implementations using Haliax named tensors
- `src/levanter/data/`: Data loading, preprocessing, and caching
- `src/levanter/optim/`: Custom optimizers and training utilities  
- `src/levanter/callbacks/`: Training callbacks for logging, checkpointing
- `src/levanter/tracker/`: Integration with WandB, TensorBoard for experiment tracking
- `config/`: YAML configuration files for different model sizes and datasets
- `docs/`: Comprehensive documentation for setup and usage
- `infra/`: Infrastructure scripts for cloud deployment (TPU/GPU clusters)

### Integration Points

**Hugging Face Ecosystem**: Full compatibility for importing/exporting models, tokenizers, and datasets via SafeTensors format.

**Checkpointing**: Distributed checkpointing via TensorStore with support for preemption recovery and cross-host resumption.

**Logging**: Rich metrics collection with support for WandB, TensorBoard, and custom trackers.

### Named Tensor System (Haliax)

Levanter uses Haliax for named tensor operations, which provides:
- Axis-aware tensor operations (e.g., `Axis("batch", 32)`)
- Automatic partitioning for distributed training
- Type-safe tensor manipulations
- Composable model definitions

This makes the codebase more readable and less error-prone compared to traditional positional tensor libraries.

## Development Guidelines

### Code Style (from haliax/AGENTS.md)
- **Python ≥3.10** target version
- **Use `uv run`** for all commands (pytest, pre-commit, etc.)
- **Ruff** for formatting/linting via pre-commit
- **MyPy** for type checking (config in pyproject.toml)
- **Google-style docstrings** for all public functions and classes
- **Use `assert_allclose`** with appropriate tolerances (1e-4 for complex, 1e-5 for simple operations)
- **Never relax floating point test tolerances** without team discussion

### Testing Requirements
- Use `@skip_if_no_torch` decorator for PyTorch-dependent tests
- Default test command includes XLA flags for proper device simulation
- Mark slow tests with `@pytest.mark.slow`, entry tests with `@pytest.mark.entry`

### Design Preferences
- **Generic code**: Use TypeVars and dataclasses where possible
- **Haliax conventions**: Use `NamedArray` and explicit `Axis` objects over positional dimensions
- **Reproducibility**: Avoid nondeterminism unless explicitly required
- **Prefer Stacked operations** with fold/scan over custom loops for better compile times
- **Configuration as dataclasses**: Use Draccus for typed, declarative configs

### Haliax Library Conventions
- APIs should accept axes or axis names rather than hard-coding positional dimensions
- Write utilities that work with arbitrary axis names when possible
- Use `haliax.nn` or Equinox modules for neural network layers
- Type annotations: Use `ht.f32[NamedArray, "batch"]` for shaped arrays

## Hardware Support

### TPU (Recommended)
- **Bitwise deterministic training** - reproducible results even with preemption/resumption
- Excellent JAX performance and ecosystem support
- See `docs/Getting-Started-TPU-VM.md` for setup instructions

### GPU
- CUDA support available but still in development
- See `docs/Getting-Started-GPU.md` for current status and setup

## Key Features

### Performance & Scalability
- **Sophia optimizer**: Up to 2x faster than Adam for many workloads
- **Distributed training**: FSDP and tensor parallelism for large models
- **Performance rivals** commercial frameworks (MosaicML Composer, Google MaxText)
- **Gradient checkpointing**: Memory optimization for large models

### Data & Checkpointing
- **Online data preprocessing** with caching for faster resume times
- **Distributed checkpointing** via Google TensorStore with cross-host resumption
- **Multiple data mixture support** without retokenization
- **Automatic decompression** of .gz, .zstd files via fsspec

### Integration & Compatibility
- **Hugging Face ecosystem**: Full compatibility for models, tokenizers, datasets via SafeTensors
- **Multiple architectures**: GPT-2, LLaMA (1/2/3), Mistral/Mixtral, Gemma, Qwen, Backpack, Whisper
- **Flexible data sources**: URLs, GCS, local files, Hugging Face datasets

## Configuration System

Uses Draccus for YAML-to-dataclass configuration with command-line overrides:

**Key Config Types:**
- `TrainerConfig`: Training parameters, checkpointing, logging
- `LmConfig`: Model architecture configs (Gpt2Config, LlamaConfig, etc.)  
- `LMDatasetConfig`: Dataset specification and preprocessing
- `OptimizerConfig`: Adam, Sophia, and other optimizer configurations

## Logging and Tracking

- **WandB integration** (primary tracking backend)
- **TensorBoard support** available
- **Rich metrics**: Comprehensive loss and performance tracking
- **Inside-jit logging**: Ability to log from within JAX compiled functions

## Documentation Resources

- **Main docs**: [levanter.readthedocs.io](https://levanter.readthedocs.io/)
- **Haliax docs**: [haliax.readthedocs.io](https://haliax.readthedocs.io/)
- **Key guides**: `docs/Getting-Started-Training.md`, `docs/Installation.md`
- **Configuration reference**: `docs/reference/Configuration.md`
- **Community**: #levanter channel on [JAX LLM Discord](https://discord.gg/CKazXcbbBm)