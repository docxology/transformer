# Visualization API

Imports below are exposed through [src.visualization](../../src/visualization/__init__.py).
Layer, head, and position indices are zero-based.

| Function | Input and output |
| --- | --- |
| `visualize_attention(attention_weights, layer_idx=0, save_path=None)` | Supply a capture with axes batch, head, query, key. Plots the first sample as Matplotlib heatmaps. |
| `plot_attention_heads(attention_weights, ...)` | Supply one layer's attention weights. Returns an interactive Plotly figure with a panel per head. |
| `visualize_residual_stream(state, ...)` | Supply a captured state. Returns a Matplotlib view of residual checkpoints. |
| `visualize_kv_stream(state, ...)` | Supply a captured state. Returns a Matplotlib view of attention across positions. |
| `visualize_path_grid(n_layers, seq_len, start, end)` | Supply grid dimensions and (layer, position) endpoints. Returns a NumPy array of per-cell path counts. |
| `create_subway_map_animation(state, save_path, ...)` | Supply a captured state and output path. Saves a GIF and returns a Matplotlib animation. |

Use the [visualization guide](../guides/visualization.md) for runnable examples
and the [theory guide](../theory/README.md) for interpretation.
