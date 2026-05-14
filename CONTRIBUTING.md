# Contributing to MLX Coffers

Thank you for contributing to MLX Coffers, an Apple Silicon port of RAM Coffers providing neuromorphic compute routing for M-series Macs.

## Project Overview

MLX Coffers maps transformer layer attention patterns to Apple Silicon's MLX framework for optimized inference on M1/M2/M3 series chips.

## Development Setup

### Prerequisites

- Apple Silicon Mac (M1, M2, M3, or M4)
- macOS 13.0+ (Ventura or later)
- Python 3.11+
- Xcode Command Line Tools

### Environment Setup

```bash
git clone https://github.com/Scottcjn/mlx-coffers.git
cd mlx-coffers

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install MLX (Apple's machine learning framework)
pip install mlx
```

## Code Style

- Python PEP 8 compliant
- Use `black` for formatting: `black .`
- Type hints for all function signatures
- Docstrings in Google style

## Testing

```bash
# Run tests
pytest tests/

# Test on specific model
python -m mlx_coffers.benchmark --model <model_name>
```

## Submitting Changes

1. Fork the repository
2. Create a branch: `git checkout -b feat/your-feature`
3. Make changes with tests
4. Submit a pull request

## Ideas for Contributions

- Additional model architectures
- Memory optimization for larger models
- Benchmarking on additional M-series chips
- Integration with popular LLM frameworks
