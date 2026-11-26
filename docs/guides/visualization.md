# Visualize the trace

Capture state in evaluation mode as shown in the [quick start](quickstart.md).
Use Matplotlib for a static plot.

```python
from pathlib import Path
from src.visualization import visualize_attention

target = Path("outputs/visualizations/attention_layer_0.png")
target.parent.mkdir(parents=True, exist_ok=True)
visualize_attention(state.get_attention_weights(0), layer_idx=0, save_path=str(target))
```

For a headless run, set `MPLBACKEND=Agg`. Use Plotly-returning functions when
you want a browser artifact, and save the returned figure with
`figure.write_html(...)`. The [API table](../api/visualization.md) lists the
input and return type for supported entry points.

## Animations

Use the validated subway-map script with a small preset and explicit
render limits for a fast example.

```bash
MPLBACKEND=Agg OMP_NUM_THREADS=1 python scripts/generate_subway_map.py \
  --config debug --seq-len 6 --duration 2 --fps 3 --resolution 640x360 \
  --output outputs/animations/subway_map.gif
```

The renderer derives its graph from a captured forward pass. Animated
markers illustrate the selected traversal. Use the captured attention
matrices and residual checkpoints when you need numeric measurements.
