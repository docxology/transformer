# First-Principles Transformer

A small, inspectable PyTorch transformer for learning how causal attention and
residual updates work. A forward pass can return the tensors used by the
analysis and visualization code alongside vocabulary logits. Examples use
synthetic token IDs and initialized model weights.

- [Documentation](docs/README.md) covers setup, model configuration, analysis,
  visualizations, and verification.
- [Technical manuscript](docs/manuscript/README.md) explains the captured state
  and includes a seeded CPU probe with a figure and numeric checks.
- [Examples](examples/README.md) contains runnable demonstrations.
- [Citation](CITATION.cff) and [license](LICENSE) identify the software.

Run the smoke check and tests from the repository root after following the
[quick start](docs/guides/quickstart.md).

```bash
python scripts/verify_installation.py
MPLBACKEND=Agg OMP_NUM_THREADS=1 python -m pytest -q
```
