# Transformer repository guidance

This repository contains an educational PyTorch transformer with state
capture for analysis and visualization. Start at [README.md](README.md).

- [docs/](docs/README.md) contains maintained user documentation and
  [docs/manuscript/](docs/manuscript/README.md) contains manuscript source.
- `src/` implements the model and supporting tools. `tests/` covers their behavior.
- `examples/`, `scripts/`, and `run.py` provide runnable entry points.
- `outputs/` holds native generated artifacts, including a tracked example GIF
  and system-report snapshot. Manuscript renders use a separate `output/` tree.
- Update `DOC_STRUCTURE` in `src/utils/documentation.py` when moving a maintained
  page that appears in the generated index. The documentation tests check its paths.
