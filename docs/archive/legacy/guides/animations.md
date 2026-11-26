# Animation Guide

Complete guide to all animation capabilities in the transformer visualization toolkit.

## Table of Contents

1. [Overview](#overview)
2. [Animation Types](#animation-types)
3. [Usage Examples](#usage-examples)
4. [Customization](#customization)
5. [Performance Tips](#performance-tips)

## Overview

The transformer toolkit provides 9+ different animation types to visualize:
- **Information Flow**: How data moves through the network
- **Attention Patterns**: How attention evolves and differs across heads
- **Layer Dynamics**: How representations change through layers
- **Token Interactions**: How tokens communicate via attention

All animations are saved as `.gif` files for easy sharing and embedding.

## Animation Types

### 1. Information Flow Animation

**Purpose**: Shows the dual-highway information flow (residual + K/V streams)

**What it reveals**:
- Vertical flow through layers (residual stream)
- Horizontal flow across positions (K/V stream)
- Computational bottlenecks

**Usage**:
```python
from src.visualization import animate_information_flow

animate_information_flow(
    state,
    save_path='outputs/animations/info_flow.gif',
    fps=2
)
```

**Key insights**:
- Watch how information spreads from early to late positions
- Observe which layers create the most change
- See the path explosion phenomenon in action

---

### 2. Attention Evolution Animation

**Purpose**: Shows how attention patterns evolve through layers for a specific query position

**What it reveals**:
- How attention focus changes with depth
- Which positions become more/less important
- Pattern convergence or divergence

**Usage**:
```python
from src.visualization import animate_attention_evolution

animate_attention_evolution(
    state,
    query_position=11,  # Last position
    save_path='outputs/animations/attn_evolution.gif',
    fps=2
)
```

**Key insights**:
- Early layers often show broader attention
- Later layers may focus on specific positions
- Reveals hierarchical pattern formation

---

### 3. Path Trace Animation

**Purpose**: Traces a specific computational path through the network

**What it reveals**:
- Exact sequence of transformations
- When information moves vertically vs horizontally
- Individual path contribution

**Usage**:
```python
from src.visualization import animate_path_trace

animate_path_trace(
    state,
    start_pos=0,
    end_pos=11,
    path_indices=[0, 2, 4, 6, 8, 10, 11],
    save_path='outputs/animations/path.gif',
    fps=3
)
```

**Key insights**:
- Each path represents one "computational history"
- Different paths can carry different information
- Illustrates the combinatorial explosion of paths

---

### 4. Layer Activations Animation

**Purpose**: Shows how activations change as they pass through layers

**What it reveals**:
- Representation evolution
- Feature emergence and transformation
- Norm dynamics

**Usage**:
```python
from src.visualization.advanced_animations import animate_layer_activations

animate_layer_activations(
    state,
    save_path='outputs/animations/activations.gif',
    fps=3
)
```

**Panels shown**:
- **Activation heatmap**: First 64 features across positions
- **Distribution**: Histogram of activation values
- **Norm evolution**: How magnitude changes through layers

---

### 5. Attention Heads Animation

**Purpose**: Cycles through all attention heads in a layer

**What it reveals**:
- Head specialization and diversity
- Different attention patterns per head
- Head-specific statistics (mean, max, entropy)

**Usage**:
```python
from src.visualization.advanced_animations import animate_attention_heads

animate_attention_heads(
    state,
    layer_idx=0,
    save_path='outputs/animations/heads.gif',
    fps=2
)
```

**Key insights**:
- Different heads attend to different patterns
- Some heads are "specialists" (local, global, previous-token)
- Entropy reveals attention sharpness

---

### 6. Token Interactions Animation

**Purpose**: Shows token-to-token communication via attention

**What it reveals**:
- Which tokens talk to which
- Communication strength
- Network topology of attention

**Usage**:
```python
from src.visualization.advanced_animations import animate_token_interactions

animate_token_interactions(
    state,
    save_path='outputs/animations/interactions.gif',
    fps=3
)
```

**Visualization**:
- Tokens arranged in a circle
- Arrows show attention flow
- Arrow thickness = attention strength

---

### 7. Information Diffusion Animation

**Purpose**: Tracks how information from one position spreads to others

**What it reveals**:
- Influence propagation
- Information reach
- Temporal dynamics of spreading

**Usage**:
```python
from src.visualization.advanced_animations import animate_information_diffusion

animate_information_diffusion(
    state,
    source_position=0,
    save_path='outputs/animations/diffusion.gif',
    fps=3
)
```

**Key insights**:
- See which positions are influenced by the source
- Track cumulative influence over layers
- Understand information locality vs globality

---

### 8. Residual Evolution Animation

**Purpose**: Detailed analysis of residual stream changes

**What it reveals**:
- Fine-grained representation changes
- Correlations between positions
- Statistical properties

**Usage**:
```python
from src.visualization.advanced_animations import animate_residual_evolution

animate_residual_evolution(
    state,
    save_path='outputs/animations/residual.gif',
    fps=2
)
```

**Panels shown**:
1. Current residual state (heatmap)
2. Norm evolution over checkpoints
3. Per-position norms
4. Change from previous checkpoint
5. Feature distribution
6. Position correlation matrix

---

### 9. Head Comparison Animation

**Purpose**: Side-by-side comparison of all heads in a layer

**What it reveals**:
- Head diversity
- Complementary patterns
- Specialization across heads

**Usage**:
```python
from src.visualization.advanced_animations import animate_head_comparison

animate_head_comparison(
    state,
    save_path='outputs/animations/comparison.gif',
    fps=2
)
```

**Display**:
- Grid layout of all heads
- Entropy score for each head
- Layer-by-layer progression

---

## Usage Examples

### Generate All Animations

Use the provided script to generate all animation types at once:

```bash
python generate_all_animations.py
```

This creates:
- 9 different animation types
- Structured output in `outputs/animations/`
- Comprehensive logging in `outputs/logs/`
- Performance metrics

### Custom Animation Pipeline

```python
import torch
from src.transformer import Transformer, TransformerConfig
from src.visualization.advanced_animations import *

# Setup
config = TransformerConfig(d_model=256, n_heads=8, n_layers=4)
model = Transformer(config)
input_ids = torch.randint(0, config.vocab_size, (1, 12))

# Forward pass with state
logits, state = model(input_ids, return_state=True)

# Generate specific animations
animate_layer_activations(state, save_path='custom1.gif')
animate_attention_heads(state, layer_idx=2, save_path='custom2.gif')
animate_token_interactions(state, save_path='custom3.gif')
```

## Customization

### Animation Speed

Control animation speed with the `fps` parameter:

```python
# Slower (more detail)
animate_information_flow(state, fps=1)  # 1 frame per second

# Normal
animate_information_flow(state, fps=3)  # 3 frames per second

# Faster (overview)
animate_information_flow(state, fps=5)  # 5 frames per second
```

### Save Paths

Organize animations by purpose:

```python
# By type
animate_attention_evolution(state, save_path='outputs/animations/attention/evolution.gif')

# By model
animate_layer_activations(state, save_path='outputs/animations/small_model/activations.gif')

# By experiment
animate_token_interactions(state, save_path='outputs/experiment_1/interactions.gif')
```

### Visual Themes

Most animations use colorblind-friendly colormaps:
- `viridis`: Attention weights
- `RdBu_r`: Residual states (diverging)
- `coolwarm`: Correlations

## Performance Tips

### Memory Management

Animations can be memory-intensive for large models:

```python
# Use smaller models for animation
config = get_preset_config('tiny')  # Instead of 'large'

# Limit sequence length
input_ids = input_ids[:, :16]  # Shorter sequences

# Clear memory after each animation
import gc
gc.collect()
torch.cuda.empty_cache()  # If using GPU
```

### Batch Processing

Generate multiple animations efficiently:

```python
animations = [
    ('info_flow', animate_information_flow),
    ('activations', animate_layer_activations),
    ('heads', animate_attention_heads)
]

for name, func in animations:
    print(f"Generating {name}...")
    func(state, save_path=f'outputs/animations/{name}.gif')
    gc.collect()
```

### File Size Optimization

Reduce file sizes:

```python
# Lower FPS (fewer frames)
animate_information_flow(state, fps=1)  # Instead of fps=3

# Shorter animations (fewer layers)
config = TransformerConfig(n_layers=3)  # Instead of 6

# Compress further with external tools
# ffmpeg -i input.gif -vf "fps=2,scale=640:-1" output.gif
```

## Integration with Logging

Combine animations with comprehensive logging:

```python
from src.utils.logging_system import TransformerLogger

logger = TransformerLogger("animation_session")

logger.info("Starting animation generation...")

for i, (name, func) in enumerate(animations):
    logger.info(f"[{i+1}/{len(animations)}] Creating {name}...")
    func(state, save_path=f'outputs/animations/{name}.gif')
    logger.log_metric(f"{name}_created", 1.0, step=i)

logger.save_metrics()
logger.info("All animations complete!")
```

## Troubleshooting

### Issue: Animations are too slow

**Solution**: Reduce FPS or use smaller models
```python
animate_information_flow(state, fps=1)  # Slower but smoother
```

### Issue: Out of memory

**Solution**: Use smaller sequences or models
```python
config = get_preset_config('tiny')
input_ids = input_ids[:, :8]
```

### Issue: Poor visual quality

**Solution**: Increase figure size in source code
```python
# In animation functions, change:
fig, ax = plt.subplots(figsize=(12, 10))  # Larger figure
```

## Next Steps

- Explore [Visualization Guide](visualization.md) for static visualizations
- Check [Analysis Guide](analysis.md) for quantitative metrics
- See [API Reference](../api/visualization.md) for detailed parameters


