# Analyze a captured state

Use the [quick start](quickstart.md) to obtain `model` and `state`. Then
inspect statistics for the first batch element with
[TransformerAnalyzer](../../src/utils/analysis_tools.py).

```python
from src.utils.analysis_tools import TransformerAnalyzer

analyzer = TransformerAnalyzer(model, state)
attention = analyzer.analyze_attention_patterns()
residual = analyzer.analyze_information_flow()
print(attention["layer_stats"])
print(residual["change_magnitudes"])
```

The layer entropy is computed from the attention distribution averaged
over heads and queries, then summarized as Shannon entropy in nats. Sparsity
is the fraction of weights below 0.01 by default, including masked cells.
Residual magnitudes are mean token-wise Euclidean norms for the first
sample. The reported correlation is mean cosine similarity to the
embedding checkpoint. Gradient diagnostics require a separate backward
pass because inference mode does not record gradients.

## Reproduce the causal probe

```bash
MPLBACKEND=Agg OMP_NUM_THREADS=1 python -m examples.manuscript_probe --output-dir output
```

The fixed-seed probe measures future attention mass and row-sum error,
replaces only the final three input tokens, and compares logits for the
original and modified sequence. It also compares a small grid's formula
and enumerator. Its JSON is `output/data/causal_probe.json` and its heatmap is
`output/figures/causal_attention.png`.

The [results](../manuscript/03_results.md) give the values for the documented
CPU run. Use the tolerance checks in the probe when repeating it across
PyTorch versions and platforms.
