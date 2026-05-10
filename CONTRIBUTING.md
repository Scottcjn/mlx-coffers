# Contributing

Thanks for improving MLX Coffers. This project experiments with routing model
work across Apple Silicon CPU and Metal GPU streams, so good contributions
include both code clarity and benchmark evidence.

## Getting Started

1. Read `README.md` for the coffer domains, routing model, and relationship to
   RAM Coffers.
2. Install the Apple Silicon MLX dependencies:

   ```bash
   pip install mlx mlx-lm
   ```

3. Work on a focused branch:

   ```bash
   git checkout -b your-change-name
   ```

## Development Workflow

Keep changes scoped to one area:

- `mlx_coffers.py` for routing primitives and coffer pools.
- `coffer_router.py` for routing policy changes.
- `distributed_matmul.py` for distributed or matrix execution experiments.
- `coffer_demo.py` for CLI demos and benchmarks.
- Research docs such as `PAPER_OUTLINE.md` or `GROKIPEDIA_ARTICLE.md`.

Avoid mixing benchmark changes with unrelated prose cleanup. Routing and stream
changes should be easy to isolate and compare.

## Validation

For quick checks, run:

```bash
python3 coffer_demo.py --routing-only
python3 test_coffers_synthetic.py
```

For performance changes, include:

- Mac model and chip generation.
- macOS, Python, MLX, and mlx-lm versions.
- Model name and quantization level.
- Baseline vs coffer-routed timing or throughput.
- Any Metal GPU utilization observations.

## Code Style

- Keep routing rules explicit and easy to inspect.
- Prefer deterministic synthetic tests for router behavior.
- Document assumptions about CPU/GPU stream placement.
- Do not claim speedups without benchmark evidence.

## Pull Request Checklist

Before opening a PR, include:

- Summary of the coffer domain or routing behavior affected.
- Commands run and outputs.
- Hardware/software environment.
- Benchmark data when performance claims are involved.
- Known limitations or cases that still route poorly.

