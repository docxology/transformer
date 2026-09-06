# scripts

Thin orchestrator scripts for the First-Principles Transformer. All logic
lives in `src/`; each script only does path bootstrap, logging/argparse, and a
single delegated call. Run from the repo root.

## Inventory

| Script | Purpose | Delegates to | Command |
|--------|---------|--------------|---------|
| `verify_installation.py` | Verify install: model creation, forward pass, state tracking, path analysis, generation | `src.transformer`, `src.visualization` | `python scripts/verify_installation.py` |
| `generate_all_animations.py` | Generate all 9 animation GIFs into `outputs/animations/` | `src.visualization` (flow/path/head animations, incl. `advanced_animations`) | `python scripts/generate_all_animations.py` |
| `generate_architecture_visualizations.py` | Generate architecture diagrams into `outputs/visualizations/` | `src.visualization.architecture_viz.create_all_architecture_visualizations` | `python scripts/generate_architecture_visualizations.py` |
| `generate_complete_documentation.py` | Write `MASTER_INDEX.json`, `FILE_COUNTS.json`, `SYSTEM_REPORT.md` into `outputs/reports/` | `src.utils.documentation.generate_documentation_reports` | `python scripts/generate_complete_documentation.py` |
| `generate_subway_map.py` | CLI: subway-map animation of the model graph (`--config`, `--seq-len`, `--style`, `--output`, ...) | `src.visualization.create_subway_map_animation` | `python scripts/generate_subway_map.py --config small --seq-len 16 --output subway_map.gif` |
| `organize_outputs.py` | Sort stray files in `outputs/` into subdirectories | `src.utils.enhanced_logging` (logging); sorting logic in-script | `python scripts/organize_outputs.py [--dry-run] [--outputs-dir DIR]` |
| `run_comprehensive_validation.py` | Full validation: model forward pass, visualizations, evaluation, structured HTML reports | `src.visualization`, `src.evaluation`, `src.utils.reporting` | `python scripts/run_comprehensive_validation.py` |

## Outputs

Everything lands under `outputs/` (`animations/`, `visualizations/`,
`reports/`, `data/`, `logs/`). Heavy scripts (animations, validation) take
minutes on CPU; `verify_installation.py` is the quick smoke check.

## Tests

Testable behavior lives in `src/` and is covered by the pytest suite in
`tests/` (`uv run pytest -q` from the repo root). No tests live here.
