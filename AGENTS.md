# AGENTS.md — `transformer`

**What this is:** Real directory (not a symlink) in the ActiveInference lane,
hosting **First-Principles Transformer**: a comprehensive, production-quality,
educational transformer implementation with full observability (residual +
K/V information streams), 20+ visualizations, 74 tests, and 20+ pages of
documentation. Built to understand, not just use. See the repo's `README.md`
and `QUICK_START.md`.

Verified against disk 2026-08-30 (`ls`, repo `README.md`).

## Layout

- `src/` — implementation code.
- `tests/` — test suite (74 tests per README).
- `examples/`, `scripts/`, `run.py` — runnable entry points and examples.
- `doc/` and `docs/` — documentation (two documentation trees are present).
- `outputs/` — generated artifacts.
- `requirements.txt`, `CONTRIBUTING.md`, `QUICK_START.md`, `README.md`,
  `AGENTS.md` (this file); plus `SUBWAY_MAP_*` and
  `DOCUMENTATION_VERIFICATION_REPORT.md` reports at the root.

## Gotchas

- Local-only path under `projects/ongoing/` — matched by the root `.gitignore`
  rule `projects/*` and never committed. Parent standard: see
  `../../AGENTS.md` (ActiveInference lane root).
- Plain lowercase name `transformer`; resolve it via this lane, not a
  hyphenated sibling.
- Two documentation directories (`doc/` and `docs/`) coexist — check both
  before claiming documentation is missing.
- Repo payload (source, tests, outputs) is out of scope for doc passes; edit
  only this file and `README.md` when improving lane documentation.
