# tests

Pytest suite for the First-Principles Transformer: 109 tests over 8 modules.

| Module | Tests | Covers |
|--------|-------|--------|
| `test_components.py` | 23 | Embedding, positional encoding, attention, feed-forward, normalization, residual connections |
| `test_model.py` | 13 | Full model forward pass, state tracking, causal masking, generation, gradient flow |
| `test_config.py` | 12 | Presets, config manager, JSON/YAML save and load |
| `test_evaluation.py` | 12 | Metrics, benchmarking, model profiling |
| `test_visualization.py` | 14 | Attention, residual/K-V stream, highway, layer-statistics and animation output |
| `test_subway_map.py` | 22 | Subway-map graph construction, animation frames, fidelity validation |
| `test_features.py` | 7 | 3D visualizations, model comparison, color schemes, HTML/comparison reports, trends |
| `test_documentation.py` | 6 | Documentation manifest, file counting, report generation |

## Running

```bash
pip install -r requirements.txt

pytest tests/ -q                       # whole suite
pytest tests/test_model.py -v          # one module
pytest tests/test_model.py::TestTransformer::test_forward_shape -v
```

Tests compute on real tensors and write figures into pytest `tmp_path` directories; nothing here
writes into the tracked tree. Shared fixtures live in the modules that use them
(`small_config`, `small_state`, ...). Testable behavior belongs in `src/`; `scripts/` holds thin
orchestrators only.
