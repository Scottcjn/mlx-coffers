# Contributing to MLX Coffers

Thanks for helping improve MLX Coffers. This project maps RAM Coffers routing
ideas onto Apple Silicon CPU and Metal GPU streams, so contributions should keep
examples reproducible and note the Apple hardware used for validation.

## Useful Contributions

- Improve setup notes for MLX, Python, and Apple Silicon devices.
- Add examples for routing additional model layer names to coffer domains.
- Document performance or memory behavior on different M-series chips.
- Improve error messages or fallback behavior when MLX devices are unavailable.
- Add small tests for layer classification and routing decisions.

## Development Workflow

1. Fork the repository and create a focused branch.
2. Keep changes scoped to one behavior, example, or documentation section.
3. Include the Python, MLX, macOS, and chip version used for validation.
4. Avoid committing generated model files, caches, or benchmark outputs.

## Validation

- Documentation-only changes: run `git diff --check`.
- Python changes: run the relevant script or test with the installed MLX
  version and include the command output.
- Performance changes: include chip model, memory size, model name, prompt or
  workload, and before/after timing.

## Pull Request Checklist

- The PR explains why the routing or documentation change is needed.
- Validation commands and Apple Silicon model are included.
- New routing rules include examples of matching layer names.
- Generated model artifacts and caches are excluded.
- Any benchmark claims include enough detail to reproduce them.

## Reporting Issues

Include macOS version, chip model, Python version, MLX version, command run, and
full traceback or performance observation. For routing bugs, include the layer
names that were classified incorrectly.
