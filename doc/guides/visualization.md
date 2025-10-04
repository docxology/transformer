# Visualization Guide

Comprehensive guide to visualizing transformer operations.

## Overview

The transformer provides extensive visualization capabilities:

1. **Attention Visualization** - See what the model attends to
2. **Information Flow** - Track residual and K/V streams
3. **Activation Patterns** - Analyze internal representations
4. **Training Dynamics** - Monitor learning progress
5. **Path Analysis** - Understand computational complexity
6. **Animations** - Dynamic visualizations of operations

## Attention Visualization

### Basic Attention Heatmap

```python
from src.visualization import visualize_attention

# Get state from forward pass
logits, state = model(input_ids, return_state=True)

# Visualize specific layer
attn_weights = state.get_attention_weights(layer_idx=0)
fig = visualize_attention(
    attn_weights,
    layer_idx=0,
    tokens=['The', 'cat', 'sat', 'on', 'mat'],
    save_path='outputs/attention_layer_0.png'
)
```

### Interactive Attention Plot

```python
from src.visualization import plot_attention_heads

# Interactive Plotly visualization
fig = plot_attention_heads(
    attn_weights,
    tokens=tokens,
    head_indices=[0, 1, 2, 3],  # Select specific heads
    save_path='outputs/attention_interactive.html'
)
```

### Attention Across All Layers

```python
from src.visualization import visualize_attention_patterns

# Compare attention patterns across layers
fig = visualize_attention_patterns(
    state,
    layer_range=(0, 4),
    save_path='outputs/attention_evolution.png'
)
```

### Attention Flow Sankey Diagram

```python
from src.visualization import visualize_attention_flow_sankey

# Sankey-style flow visualization
fig = visualize_attention_flow_sankey(
    attn_weights,
    position_labels=tokens,
    threshold=0.1,  # Only show weights > 0.1
    save_path='outputs/attention_sankey.png'
)
```

## Information Flow Visualization

### Residual Stream (Vertical)

```python
from src.visualization import visualize_residual_stream

# Track how a position evolves through layers
fig = visualize_residual_stream(
    state,
    position_idx=5,  # Track position 5
    save_path='outputs/residual_stream.png'
)
```

### K/V Stream (Horizontal)

```python
from src.visualization import visualize_kv_stream

# See Q, K, V matrices at a layer
fig = visualize_kv_stream(
    state,
    layer_idx=0,
    head_idx=0,
    save_path='outputs/kv_stream.png'
)
```

### Information Highways

```python
from src.visualization import visualize_information_highways

# Combined view of both highways
fig = visualize_information_highways(
    state,
    position_indices=[0, 5, 10],  # Highlight specific positions
    save_path='outputs/info_highways.png'
)
```

### Layer Statistics

```python
from src.visualization import plot_layer_statistics

# Statistics: norms, entropy, sparsity
fig = plot_layer_statistics(
    state,
    save_path='outputs/layer_stats.png'
)
```

## Advanced Visualizations

### Gradient Flow

```python
from src.visualization import visualize_gradient_flow

# After backward pass, visualize gradients
loss.backward()
fig = visualize_gradient_flow(
    model,
    save_path='outputs/gradient_flow.png'
)
```

### Activation Patterns

```python
from src.visualization import visualize_activation_patterns

# Analyze activations at a layer
fig = visualize_activation_patterns(
    state,
    layer_idx=2,
    save_path='outputs/activations.png'
)
```

### Token Journey

```python
from src.visualization import visualize_token_journey

# Track a single token through the network
fig = visualize_token_journey(
    state,
    token_position=3,
    save_path='outputs/token_journey.png'
)
```

### Head Specialization

```python
from src.visualization import visualize_head_specialization

# Analyze what each attention head does
fig = visualize_head_specialization(
    state,
    save_path='outputs/head_specialization.png'
)
```

### Causal Graph

```python
from src.visualization import visualize_causal_graph

# Show transformer as a causal graph
fig = visualize_causal_graph(
    n_layers=6,
    seq_len=12,
    highlight_path=[(0, 0), (1, 1), (2, 3), (3, 5)],
    save_path='outputs/causal_graph.png'
)
```

## Animations

### Information Flow Animation

```python
from src.visualization import animate_information_flow

# Animate how information flows through layers
anim = animate_information_flow(
    state,
    save_path='outputs/flow_animation.gif',
    fps=2,
    position_focus=5
)
```

### Attention Evolution Animation

```python
from src.visualization import animate_attention_evolution

# Animate attention pattern evolution
anim = animate_attention_evolution(
    state,
    query_position=10,
    save_path='outputs/attention_evolution.gif',
    fps=2
)
```

### Path Trace Animation

```python
from src.visualization import animate_path_trace

# Animate a specific information path
path_indices = [0, 1, 3, 5, 8, 10]  # Positions at each layer
anim = animate_path_trace(
    state,
    start_pos=0,
    end_pos=10,
    path_indices=path_indices,
    save_path='outputs/path_trace.gif',
    fps=3
)
```

## Training Visualization

### Training Visualizer

```python
from src.visualization import TrainingVisualizer

# Create visualizer
viz = TrainingVisualizer()

# Log metrics during training
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss = evaluate(model, val_loader)
    
    viz.log_metrics(
        train_loss=train_loss,
        val_loss=val_loss,
        learning_rate=optimizer.param_groups[0]['lr'],
        gradient_norm=get_gradient_norm(model),
        weight_norm=get_weight_norm(model)
    )

# Plot training curves
fig = viz.plot_training_curves(
    save_path='outputs/training_curves.png'
)

# Save history
viz.save_history('training_history.json')
```

### Weight Distributions

```python
from src.visualization import visualize_weight_distributions

# Visualize weight distributions during training
fig = visualize_weight_distributions(
    model,
    epoch=10,
    save_path='outputs/weights_epoch_10.png'
)
```

### Learning Dynamics

```python
from src.visualization import visualize_learning_dynamics

# Analyze learning dynamics
fig = visualize_learning_dynamics(
    losses=loss_history,
    gradients=gradient_history,
    save_path='outputs/learning_dynamics.png'
)
```

## Path Analysis Visualization

### Path Grid

```python
from src.visualization import visualize_path_grid

# Show path density through the network
grid = visualize_path_grid(
    n_layers=6,
    seq_len=12,
    start=(0, 0),
    end=(5, 11)
)

import matplotlib.pyplot as plt
plt.imshow(grid.T, cmap='hot', aspect='auto')
plt.colorbar(label='Number of Paths')
plt.xlabel('Layer')
plt.ylabel('Position')
plt.title('Path Density')
plt.savefig('outputs/path_grid.png')
```

## Batch Visualization

### Visualize Multiple Samples

```python
# Visualize multiple samples
batch_logits, batch_states = [], []

for sample in samples:
    logits, state = model(sample, return_state=True)
    batch_logits.append(logits)
    batch_states.append(state)

# Create comparison plots
fig, axes = plt.subplots(2, len(samples), figsize=(20, 10))

for idx, state in enumerate(batch_states):
    # Attention
    ax = axes[0, idx]
    attn = state.get_attention_weights(0)[0, 0]  # First head
    ax.imshow(attn.cpu().detach().numpy())
    ax.set_title(f'Sample {idx}')
    
    # Residual norms
    ax = axes[1, idx]
    residuals = state.get_residual_stream()
    norms = [r[0, :, :].norm(dim=-1).mean().item() for r in residuals]
    ax.plot(norms)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Norm')

plt.tight_layout()
plt.savefig('outputs/batch_comparison.png')
```

## Customization

### Custom Color Maps

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Use custom color maps
plt.imshow(attention_weights, cmap='viridis')  # or 'plasma', 'inferno', etc.

# Seaborn color palettes
sns.heatmap(attention_weights, cmap='coolwarm')
```

### Figure Styling

```python
import matplotlib.pyplot as plt

# Set style
plt.style.use('seaborn-v0_8-darkgrid')  # or 'ggplot', 'bmh', etc.

# Customize figure
fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
ax.set_facecolor('#f0f0f0')
fig.patch.set_facecolor('white')
```

## Tips and Best Practices

### 1. Choose Appropriate Visualizations
- **Debugging**: Gradient flow, activation patterns
- **Understanding**: Attention heatmaps, information highways
- **Presentation**: Animations, Sankey diagrams
- **Analysis**: Head specialization, token journeys

### 2. Manage Memory
- Visualize smaller models for exploration
- Use `return_state=False` when not visualizing
- Delete large figure objects after saving

### 3. Batch Processing
- Generate multiple visualizations in a loop
- Use consistent naming schemes
- Create comparison grids

### 4. Interactive Exploration
- Use Plotly for interactive plots
- Enable zoom and hover information
- Export to HTML for sharing

## Example Workflow

```python
# Complete visualization workflow
def visualize_model_analysis(model, input_ids, output_dir='outputs'):
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Forward pass
    logits, state = model(input_ids, return_state=True)
    
    # 1. Attention patterns
    for layer_idx in range(model.config.n_layers):
        attn = state.get_attention_weights(layer_idx)
        visualize_attention(
            attn, 
            layer_idx=layer_idx,
            save_path=f'{output_dir}/attention_layer_{layer_idx}.png'
        )
    
    # 2. Information flow
    visualize_information_highways(
        state,
        save_path=f'{output_dir}/info_highways.png'
    )
    
    # 3. Layer statistics
    plot_layer_statistics(
        state,
        save_path=f'{output_dir}/layer_stats.png'
    )
    
    # 4. Animations
    animate_information_flow(
        state,
        save_path=f'{output_dir}/flow_animation.gif'
    )
    
    print(f"Visualizations saved to {output_dir}/")

# Use it
visualize_model_analysis(model, input_ids)
```


