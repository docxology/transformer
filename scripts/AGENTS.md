# AGENTS.md — `transformer/scripts`

## Scope
- Local-only path under `projects/ongoing/` — matched by the root
  `.gitignore` rule `projects/*` and never committed.
- Parent standard: see `../../AGENTS.md` (ActiveInference lane root).

## Thin-Orchestrator Contract

`scripts/` contains **thin orchestrators only**. Each script may do:

1. Path bootstrap — `project_root = Path(__file__).parent.parent` +
   `sys.path.insert(0, str(project_root))` (never CWD-relative `'.'`).
2. Argparse / logging setup.
3. A single delegated call into a `src/<pkg>/` entrypoint.

Business, data, plot, and analysis logic lives in `src/` where it is
importable and covered by the pytest suite in `tests/`. New script behavior
means new `src/` logic plus a thin call here — not inline code in a script.

## Layout
- `README.md` — inventory table (script | purpose | delegates to | command).
- `verify_installation.py` — install smoke check; delegates to
  `src.transformer` / `src.visualization`.
- `generate_all_animations.py` — all 9 animation GIFs; delegates to
  `src.visualization`.
- `generate_architecture_visualizations.py` — architecture diagrams;
  delegates to `src.visualization.architecture_viz`.
- `generate_complete_documentation.py` — `outputs/reports/` artifacts;
  delegates to `src.utils.documentation.generate_documentation_reports`.
- `generate_subway_map.py` — CLI wrapper around
  `src.visualization.create_subway_map_animation`.
- `organize_outputs.py` — sorts `outputs/` into subdirectories.
- `run_comprehensive_validation.py` — full validation run; delegates to
  `src.visualization`, `src.evaluation`, `src.utils.reporting`.

## Gotchas
- The two bootstrap patterns `sys.path.insert(0, '.')` (CWD-dependent) and
  missing bootstrap (breaks standalone runs) are prohibited; use the
  `Path(__file__)` form above so scripts run from any directory.
- Generated documentation must be deterministic: no `datetime.now()` or
  wall-clock stamps inside artifacts produced by
  `src.utils.documentation`.
- The documentation manifest (`DOC_STRUCTURE` in
  `src/utils/documentation.py`) lists real files only; a test
  (`tests/test_documentation.py`) fails when an entry goes stale — update
  the manifest, not the test.
- `test_*.py` files do not belong here: pytest only collects `tests/`.
  The former `scripts/test_all_features.py` smoke harness was ported to
  `tests/test_features.py` and deleted.
- Heavy scripts (animations, validation) run minutes on CPU and write into
  `outputs/`; run them from the repo root so relative output paths land in
  the right place.
