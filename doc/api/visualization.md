# Visualization API Reference

Complete reference for all visualization functions.

## Basic Attention Visualization

### `visualize_attention`

Create attention heatmaps for all heads in a layer.

```python
visualize_attention(
    attention_weights: torch.Tensor,
    layer_idx: int = 0,
    tokens: Optional[List[str]] = None,
    figsize: tuple = (12, 10),
    save_path: Optional[str] = None
) -> plt.Figure
```

**Parameters:**
- `attention_weights`: Attention weights `[batch, n_heads, seq_len, seq_len]`
- `layer_idx`: Layer index for title
- `tokens`: Token strings for axis labels
- `figsize`: Figure size
- `save_path`: Path to save figure

**Returns:** Matplotlib figure

**Example:**
```python
attn_weights = state.get_attention_weights(0)
fig = visualize_attention(attn_weights, tokens=['The', 'cat', 'sat'])
```

### `plot_attention_heads`

Interactive Plotly visualization of attention heads.

```python
plot_attention_heads(
    attention_weights: torch.Tensor,
    tokens: Optional[List[str]] = None,
    head_indices: Optional[List[int]] = None,
    save_path: Optional[str] = None
) -> go.Figure
```

**Parameters:**
- `attention_weights`: Attention weights tensor
- `tokens`: Token strings for hover labels
- `head_indices`: Specific heads to plot (defaults to all)
- `save_path`: Path to save HTML

**Returns:** Plotly figure

## Information Flow Visualization

### `visualize_residual_stream`

Visualize vertical residual stream flow.

```python
visualize_residual_stream(
    state: TransformerState,
    position_idx: int = 0,
    save_path: Optional[str] = None
) -> plt.Figure
```

**Parameters:**
- `state`: TransformerState object from forward pass
- `position_idx`: Token position to track
- `save_path`: Path to save

**Returns:** Figure showing evolution through layers

### `visualize_kv_stream`

Visualize horizontal K/V stream at a layer.

```python
visualize_kv_stream(
    state: TransformerState,
    layer_idx: int = 0,
    head_idx: int = 0,
    save_path: Optional[str] = None
) -> plt.Figure
```

**Parameters:**
- `state`: TransformerState object
- `layer_idx`: Layer to visualize
- `head_idx`: Attention head
- `save_path`: Path to save

**Returns:** Figure with Q, K, V matrices

### `visualize_information_highways`

Combined view of both information highways.

```python
visualize_information_highways(
    state: TransformerState,
    position_indices: Optional[List[int]] = None,
    save_path: Optional[str] = None
) -> plt.Figure
```

## Advanced Visualizations

### `visualize_gradient_flow`

Visualize gradient magnitudes through network.

```python
visualize_gradient_flow(
    model: torch.nn.Module,
    save_path: Optional[str] = None
) -> plt.Figure
```

**Requirements:** Must call after `loss.backward()`

### `visualize_activation_patterns`

Analyze activation patterns at a layer.

```python
visualize_activation_patterns(
    state: TransformerState,
    layer_idx: int,
    save_path: Optional[str] = None
) -> plt.Figure
```

**Returns:** Multi-panel figure with:
- Activation heatmap
- Distribution histogram
- Per-position norms
- Position correlations

### `visualize_causal_graph`

Show transformer as a causal graph.

```python
visualize_causal_graph(
    n_layers: int,
    seq_len: int,
    highlight_path: Optional[List[Tuple[int, int]]] = None,
    save_path: Optional[str] = None
) -> plt.Figure
```

**Parameters:**
- `n_layers`: Number of layers
- `seq_len`: Sequence length
- `highlight_path`: Path to highlight as list of (layer, position) tuples

### `visualize_token_journey`

Track a single token through the network.

```python
visualize_token_journey(
    state: TransformerState,
    token_position: int,
    save_path: Optional[str] = None
) -> plt.Figure
```

**Shows:**
- Norm evolution
- Representation heatmap
- Sparsity changes
- Attention received

### `visualize_head_specialization`

Analyze attention head specialization.

```python
visualize_head_specialization(
    state: TransformerState,
    save_path: Optional[str] = None
) -> plt.Figure
```

**Metrics computed:**
- Locality (average attention distance)
- Entropy (concentration vs. diffusion)
- Previous token focus
- Uniformity

## Animation Functions

### `animate_information_flow`

Animate information flow through layers.

```python
animate_information_flow(
    state: TransformerState,
    save_path: str = 'flow_animation.gif',
    fps: int = 2,
    position_focus: Optional[int] = None
) -> animation.FuncAnimation
```

**Parameters:**
- `state`: TransformerState object
- `save_path`: Path to save GIF
- `fps`: Frames per second
- `position_focus`: Specific position to highlight

### `animate_attention_evolution`

Animate attention pattern evolution.

```python
animate_attention_evolution(
    state: TransformerState,
    query_position: int,
    save_path: str = 'attention_evolution.gif',
    fps: int = 2
) -> animation.FuncAnimation
```

### `animate_path_trace`

Animate a specific information flow path.

```python
animate_path_trace(
    state: TransformerState,
    start_pos: int,
    end_pos: int,
    path_indices: List[int],
    save_path: str = 'path_trace.gif',
    fps: int = 3
) -> animation.FuncAnimation
```

## Training Visualization

### `TrainingVisualizer`

Class for tracking and visualizing training dynamics.

```python
class TrainingVisualizer:
    def __init__(self):
        """Initialize visualizer."""
        
    def log_metrics(
        self,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        weight_norm: Optional[float] = None,
        attention_entropy: Optional[float] = None
    ):
        """Log metrics for current epoch."""
    
    def plot_training_curves(
        self,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot comprehensive training curves."""
    
    def save_history(self, path: str):
        """Save training history to JSON."""
    
    def load_history(self, path: str):
        """Load training history from JSON."""
```

**Example:**
```python
viz = TrainingVisualizer()

for epoch in range(num_epochs):
    train_loss = train_epoch(model, dataloader, optimizer)
    val_loss = evaluate(model, val_loader)
    
    viz.log_metrics(train_loss=train_loss, val_loss=val_loss)

fig = viz.plot_training_curves()
viz.save_history('history.json')
```

### `visualize_weight_distributions`

Visualize weight distribution evolution.

```python
visualize_weight_distributions(
    model: torch.nn.Module,
    epoch: int = 0,
    save_path: Optional[str] = None
) -> plt.Figure
```

## Path Analysis Functions

### `compute_path_counts`

Compute number of distinct information flow paths.

```python
compute_path_counts(
    layer_displacement: int,
    position_displacement: int
) -> int
```

**Returns:** C(m+n, n) where m=position_displacement, n=layer_displacement

**Example:**
```python
paths = compute_path_counts(6, 10)  # 8008 paths
```

### `trace_information_paths`

Enumerate possible information flow paths.

```python
trace_information_paths(
    start_layer: int,
    start_position: int,
    end_layer: int,
    end_position: int,
    max_paths: int = 1000
) -> List[List[Tuple[int, int]]]
```

**Returns:** List of paths, each path is list of (layer, position) tuples

### `analyze_path_complexity`

Generate human-readable path complexity analysis.

```python
analyze_path_complexity(
    n_layers: int,
    seq_len: int
) -> str
```

**Returns:** Formatted analysis string

### `visualize_path_grid`

Create grid showing path density.

```python
visualize_path_grid(
    n_layers: int,
    seq_len: int,
    start: Tuple[int, int],
    end: Tuple[int, int]
) -> np.ndarray
```

**Returns:** Grid array `[n_layers, seq_len]` with path counts

## Common Patterns

### Basic Workflow

```python
# 1. Forward pass with state tracking
logits, state = model(input_ids, return_state=True)

# 2. Visualize attention
attn = state.get_attention_weights(layer_idx=0)
visualize_attention(attn, save_path='attention.png')

# 3. Visualize information flow
visualize_information_highways(state, save_path='highways.png')

# 4. Create animation
animate_information_flow(state, save_path='flow.gif')
```

### Training Monitoring

```python
viz = TrainingVisualizer()

for epoch in range(num_epochs):
    # Training
    train_loss = train_epoch(model, train_loader, optimizer)
    
    # Evaluation
    val_loss = evaluate(model, val_loader)
    
    # Log
    viz.log_metrics(train_loss=train_loss, val_loss=val_loss)
    
    # Periodic visualization
    if epoch % 10 == 0:
        viz.plot_training_curves(f'curves_epoch_{epoch}.png')
        
        logits, state = model(sample_input, return_state=True)
        visualize_attention(
            state.get_attention_weights(0),
            save_path=f'attention_epoch_{epoch}.png'
        )
```

### Analysis Workflow

```python
# Comprehensive analysis
def analyze_model(model, input_ids):
    # Forward pass
    logits, state = model(input_ids, return_state=True)
    
    # Attention analysis
    for layer_idx in range(model.config.n_layers):
        attn = state.get_attention_weights(layer_idx)
        visualize_attention(attn, layer_idx=layer_idx)
    
    # Head specialization
    visualize_head_specialization(state)
    
    # Token journeys
    for pos in [0, len(input_ids)//2, len(input_ids)-1]:
        visualize_token_journey(state, pos)
    
    # Information highways
    visualize_information_highways(state)
    
    # Path analysis
    analysis = analyze_path_complexity(
        model.config.n_layers,
        input_ids.shape[1]
    )
    print(analysis)
```

## See Also

- [Visualization Guide](../guides/visualization.md) - Detailed usage guide
- [Examples](../examples/visualization.md) - Practical examples
- [Concepts: Information Flow](../concepts/information_flow.md) - Understanding what to visualize


