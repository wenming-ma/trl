# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TRL (Transformer Reinforcement Learning) is a library for post-training foundation models using techniques like Supervised Fine-Tuning (SFT), Proximal Policy Optimization (PPO), Direct Preference Optimization (DPO), and Group Relative Policy Optimization (GRPO). Built on top of Hugging Face Transformers, TRL integrates with Accelerate, PEFT, and DeepSpeed for scalable training.

## Development Setup

```bash
# Install development dependencies
pip install -e .[dev]

# Alternatively, install specific optional dependencies:
pip install -e .[test]      # Testing only
pip install -e .[peft]      # PEFT integration
pip install -e .[deepspeed] # DeepSpeed support
pip install -e .[vllm]      # vLLM integration
pip install -e .[vlm]       # Vision-language models
```

## Common Commands

### Testing
```bash
# Run all tests (excludes slow and low_priority tests)
make test

# Run a specific test file
pytest tests/test_sft_trainer.py

# Run tests in parallel with verbose output
pytest -n auto -s -v tests/test_dpo_trainer.py

# Run slow tests only
make slow_tests

# Run experimental tests
make test_experimental

# Run tests for a specific feature
pytest -k "test_sft"
```

### Code Quality
```bash
# Run all pre-commit hooks, linting, and formatting
make precommit

# This command runs:
# - python scripts/add_copyrights.py
# - pre-commit run --all-files (ruff check and format)
# - doc-builder style trl tests docs/source --max_len 119
```

### CLI Tools
```bash
# Get environment information for bug reports
trl env

# Train using CLI (examples)
trl sft --model_name_or_path Qwen/Qwen2.5-0.5B --dataset_name trl-lib/Capybara --output_dir output/
trl dpo --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct --dataset_name argilla/Capybara-Preferences --output_dir output/
```

## Architecture

### Core Structure

- **`trl/trainer/`**: All trainer implementations and configs
  - Each trainer has a corresponding `*_trainer.py` and `*_config.py` file
  - Trainers extend Hugging Face's Trainer class
  - Key trainers: `sft_trainer.py`, `dpo_trainer.py`, `grpo_trainer.py`, `rloo_trainer.py`, `ppo_trainer.py`, `reward_trainer.py`
  - `base_trainer.py`: Common base functionality
  - `utils.py`: Shared trainer utilities (quantization, PEFT configs, device maps)
  - `callbacks.py`: Custom callbacks (BEMACallback, MergeModelCallback, SyncRefModelCallback, etc.)
  - `judges.py`: Judge implementations for evaluation (OpenAI, HuggingFace, PairRM)

- **`trl/models/`**: Model wrappers with value heads
  - `AutoModelForCausalLMWithValueHead`: Causal LM with value head for RL
  - `AutoModelForSeq2SeqLMWithValueHead`: Seq2Seq models with value head
  - `PreTrainedModelWrapper`: Base wrapper class
  - Utilities for reference models and chat format setup

- **`trl/data_utils.py`**: Dataset processing utilities
  - Chat template application and conversation handling
  - Dataset packing and truncation
  - Preference dataset processing
  - Multimodal message preparation

- **`trl/experimental/`**: Experimental features (subject to change)
  - Unstable/fast-evolving features
  - No stability guarantees between releases

- **`trl/cli.py`**: Command-line interface implementation

- **`examples/`**: Example training scripts organized by method

- **`tests/`**: Test suite organized by trainer type
  - Each trainer has a corresponding `test_*_trainer.py`
  - `conftest.py`: Pytest configuration and fixtures
  - `testing_utils.py`: Common test utilities

### Trainer Pattern

Each post-training method follows a consistent pattern:
1. **Config class** (`*Config`): Inherits from `TrainingArguments`, defines hyperparameters
2. **Trainer class** (`*Trainer`): Implements the training loop, loss computation, and logging
3. **Example script** in `examples/scripts/`: Demonstrates usage
4. **Test file** in `tests/`: Comprehensive test coverage

Example implementations to reference:
- **Paired preference optimization**: `dpo_trainer.py` + `dpo_config.py`
- **RL-based optimization**: `rloo_trainer.py` + `rloo_config.py`
- **Online optimization**: `online_dpo_trainer.py` + `online_dpo_config.py`

## Code Style Guidelines

- **Line length**: 119 characters maximum
- **Linter**: Ruff (configured in `pyproject.toml`)
- **Import order**: Standard library, third-party, first-party (TRL)
- **Type hints**: Required for all function signatures
- **Documentation**: Google-style docstrings with type annotations
  - Use active voice without definite articles ("String to process" not "The string to process")
  - Specify optional parameters with defaults: `(str, *optional*, defaults to "foo")`
  - Include Examples section with executable code snippets

### Documentation Format Example
```python
def my_function(data: list[float], precision: int = 2) -> dict[str, float]:
    r"""
    Calculates basic statistics for a given dataset.

    Args:
        data (`list[float]`):
            A list of numerical values to analyze.
        precision (`int`, *optional*, defaults to `2`):
            Number of decimal places to round the results.

    Returns:
        `dict[str, float]`: A dictionary containing calculated statistics.

    Examples:
    ```python
    >>> my_function([1.0, 2.0, 3.0])
    {"mean": 2.0, "median": 2.0}
    ```
    """
    ...
```

## Adding New Trainers

New post-training methods should meet these criteria:
- **Simplicity**: Achieves similar performance with less complexity
- **Efficiency**: Provides significant improvement in training efficiency

Steps to add a new trainer:
1. Open an issue with paper link, implementation, and trained model weights
2. Implement `*_config.py` inheriting from appropriate config class
3. Implement `*_trainer.py` with training loop and loss computation
4. Add comprehensive tests in `tests/test_*_trainer.py`
5. Create example script in `examples/scripts/`
6. Add documentation following the existing pattern
7. Update `trl/__init__.py` to export the new classes

## Testing Requirements

- All PRs require high-coverage tests
- Tests use pytest with auto-parallelization (`-n auto`)
- Flaky tests use reruns: `--reruns 5 --reruns-delay 1`
- Mark slow tests with `@pytest.mark.slow`
- Mark low-priority tests with `@pytest.mark.low_priority`
- Experimental tests are excluded by default (`-k "not experimental"`)

## Warnings and Deprecation

- Use warnings for supported but not correct operations (deprecated features, suboptimal usage)
- Use logger.info for correct operations that deserve attention
- Use exceptions for unsupported operations
- Deprecation warnings must include:
  - Transition guidance
  - Target removal version
  - Alternative solution

Deprecation timeline:
- Experimental/low-use features: May change between releases
- Widely-used features: ~5 months / 5 minor releases transition period

## Integration Points

- **Transformers**: Base model loading and tokenization
- **Accelerate**: Multi-GPU and distributed training (DDP, DeepSpeed, FSDP)
- **PEFT**: LoRA, QLoRA, and other parameter-efficient fine-tuning methods
- **Datasets**: Dataset loading and processing
- **DeepSpeed**: Memory optimization and ZeRO stages
- **vLLM**: Fast inference for online methods
- **Unsloth**: Optimized kernels for faster training

## File Naming Conventions

- Trainer implementations: `*_trainer.py`
- Configuration classes: `*_config.py`
- Test files: `test_*_trainer.py` or `test_*.py`
- Example scripts: Descriptive names in `examples/scripts/`
- Use relative imports within the trl package
