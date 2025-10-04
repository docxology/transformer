# Analysis Guide

Comprehensive guide to analyzing transformer behavior using built-in analysis tools.

## Table of Contents

1. [Overview](#overview)
2. [Analysis Tools](#analysis-tools)
3. [Attention Analysis](#attention-analysis)
4. [Information Flow Analysis](#information-flow-analysis)
5. [Layer Importance](#layer-importance)
6. [Representational Geometry](#representational-geometry)
7. [Complete Reports](#complete-reports)

## Overview

The `TransformerAnalyzer` provides deep insights into:
- **Attention patterns** and head specialization
- **Information flow** through the network
- **Layer contributions** and importance
- **Gradient health** (when available)
- **Representational geometry** and dimensionality

## Analysis Tools

### Basic Usage

```python
from src.transformer import Transformer, TransformerConfig
from src.utils.analysis_tools import TransformerAnalyzer
import torch

# Create model and generate state
config = TransformerConfig(d_model=256, n_heads=8, n_layers=4)
model = Transformer(config)
input_ids = torch.randint(0, config.vocab_size, (1, 12))

with torch.no_grad():
    logits, state = model(input_ids, return_state=True)

# Create analyzer
analyzer = TransformerAnalyzer(model, state)

# Run analyses
attention_analysis = analyzer.analyze_attention_patterns()
flow_analysis = analyzer.analyze_information_flow()
importance = analyzer.analyze_layer_importance()
geometry = analyzer.analyze_representational_geometry()

# Generate complete report
report = analyzer.generate_comprehensive_report(
    save_path='outputs/analysis/my_analysis.json'
)
```

## Attention Analysis

### What It Reveals

The attention analysis identifies:
1. **Layer statistics**: Mean/max attention, entropy, sparsity
2. **Head specialization**: Pattern types (local, global, previous-token, mixed)
3. **Attention flow**: How information is routed

### Running the Analysis

```python
attention_results = analyzer.analyze_attention_patterns()

# Results structure:
# {
#   'layer_stats': [
#     {
#       'layer': 0,
#       'mean_attention': 0.083,
#       'max_attention': 0.95,
#       'entropy': 2.45,
#       'sparsity': 0.12
#     },
#     ...
#   ],
#   'head_specialization': [
#     {
#       'layer': 0,
#       'head': 0,
#       'pattern_type': 'local',  # or 'previous', 'uniform', 'mixed'
#       'entropy': 2.1,
#       'locality': 1.5  # Average distance attended to
#     },
#     ...
#   ]
# }
```

### Interpreting Results

**Entropy**:
- High entropy (>2.5): Uniform/diffuse attention
- Low entropy (<1.5): Focused/sparse attention

**Pattern Types**:
- `local`: Attends to nearby tokens (diagonal pattern)
- `previous`: Attends to previous token (next-token prediction)
- `uniform`: Attends uniformly (global context)
- `mixed`: Complex pattern

**Sparsity**:
- High sparsity (>0.8): Very selective attention
- Low sparsity (<0.2): Broad attention

### Example: Find Specialized Heads

```python
# Find heads with specific patterns
local_heads = [
    h for h in attention_results['head_specialization']
    if h['pattern_type'] == 'local'
]

print(f"Found {len(local_heads)} local attention heads")

# Find most focused heads (lowest entropy)
focused_heads = sorted(
    attention_results['head_specialization'],
    key=lambda h: h['entropy']
)[:5]

print("Most focused heads:")
for h in focused_heads:
    print(f"  Layer {h['layer']}, Head {h['head']}: entropy={h['entropy']:.2f}")
```

## Information Flow Analysis

### What It Reveals

The flow analysis tracks:
1. **Norm evolution**: How representation magnitude changes
2. **Change magnitudes**: How much each layer modifies the state
3. **Correlation decay**: How representations drift from embeddings

### Running the Analysis

```python
flow_results = analyzer.analyze_information_flow()

# Results structure:
# {
#   'norm_evolution': [
#     {'checkpoint': 'embedding', 'norm': 15.2},
#     {'checkpoint': 0, 'norm': 16.1},
#     ...
#   ],
#   'change_magnitudes': [
#     {
#       'from_checkpoint': 'embedding',
#       'to_checkpoint': 0,
#       'magnitude': 3.4
#     },
#     ...
#   ],
#   'correlation_decay': [
#     {
#       'checkpoint': 0,
#       'correlation_with_embedding': 0.92
#     },
#     ...
#   ]
# }
```

### Interpreting Results

**Norm Evolution**:
- Increasing norms: Network is amplifying signals
- Stable norms: Good normalization
- Exploding norms (>100): Potential instability

**Change Magnitudes**:
- Large changes: Layer is highly transformative
- Small changes: Layer preserves information
- Compare across layers to find most important ones

**Correlation Decay**:
- High correlation (>0.8): Preserves original information
- Low correlation (<0.5): Heavily transformed
- Gradual decay: Smooth transformation

### Example: Identify Critical Layers

```python
# Find layers with largest contributions
changes = flow_results['change_magnitudes']
critical_layers = sorted(
    changes,
    key=lambda c: c['magnitude'],
    reverse=True
)[:3]

print("Most transformative layers:")
for layer in critical_layers:
    print(f"  Layer {layer['to_checkpoint']}: magnitude={layer['magnitude']:.2f}")
```

## Layer Importance

### What It Reveals

Layer importance analysis measures:
1. **Contribution magnitude**: Absolute change introduced
2. **Relative magnitude**: Change relative to current state
3. **Directional change**: How much the representation direction shifts

### Running the Analysis

```python
importance = analyzer.analyze_layer_importance()

# Results structure:
# {
#   'contribution_magnitudes': [
#     {
#       'layer': 0,
#       'magnitude': 3.4,
#       'relative_magnitude': 0.21,
#       'directional_change': 0.15
#     },
#     ...
#   ]
# }
```

### Interpreting Results

**Absolute Magnitude**:
- High: Layer makes large changes
- Low: Layer is subtle or redundant

**Relative Magnitude**:
- >0.3: Significant contribution
- <0.1: Minor refinement
- Useful for comparing layers of different depths

**Directional Change**:
- High (>0.5): Changes semantic direction
- Low (<0.1): Preserves direction, changes magnitude

### Example: Prune Redundant Layers

```python
# Identify potentially redundant layers
redundant = [
    layer for layer in importance['contribution_magnitudes']
    if layer['relative_magnitude'] < 0.05
]

if redundant:
    print("Potentially redundant layers:")
    for layer in redundant:
        print(f"  Layer {layer['layer']}: contribution={layer['relative_magnitude']:.3f}")
else:
    print("All layers contribute significantly")
```

## Representational Geometry

### What It Reveals

Geometric analysis examines:
1. **Effective dimensionality**: Intrinsic dimensionality of representations
2. **Isotropy**: Uniformity of directions (sphere-like vs clustered)
3. **Singular value spectrum**: Distribution of information

### Running the Analysis

```python
geometry = analyzer.analyze_representational_geometry()

# Results structure:
# {
#   'dimensionality': [
#     {
#       'checkpoint': 'embedding',
#       'effective_dim': 128,
#       'intrinsic_dim_ratio': 0.5  # 128/256
#     },
#     ...
#   ],
#   'isotropy': [
#     {
#       'checkpoint': 'embedding',
#       'isotropy_score': 0.15  # Lower is more isotropic
#     },
#     ...
#   ]
# }
```

### Interpreting Results

**Effective Dimensionality**:
- High ratio (>0.8): Uses most dimensions efficiently
- Low ratio (<0.3): Representations are low-rank
- Decreasing over layers: Information compression

**Isotropy Score**:
- Low (<0.2): Uniform sphere (good)
- High (>0.5): Anisotropic/clustered (may indicate issues)
- Increasing over layers: Representations becoming less uniform

### Example: Track Dimensionality Collapse

```python
dims = geometry['dimensionality']

print("Dimensionality evolution:")
for d in dims:
    cp = d['checkpoint']
    ratio = d['intrinsic_dim_ratio']
    print(f"  {cp}: {ratio:.2%} of dimensions used")

# Check for collapse
if dims[-1]['intrinsic_dim_ratio'] < 0.2:
    print("⚠️  Warning: Severe dimensionality collapse detected!")
```

## Complete Reports

### Generate Comprehensive Report

```python
# Single command for complete analysis
report = analyzer.generate_comprehensive_report(
    save_path='outputs/analysis/full_report.json'
)

# Report contains:
# - Model configuration
# - Attention analysis
# - Information flow
# - Layer importance
# - Gradient health (if available)
# - Representational geometry
```

### Report Structure

```json
{
  "model_config": {
    "n_layers": 4,
    "seq_len": 12,
    "d_model": 256
  },
  "attention_analysis": { ... },
  "information_flow": { ... },
  "layer_importance": { ... },
  "gradient_health": { ... },
  "geometry": { ... }
}
```

### Loading and Using Reports

```python
import json

# Load saved report
with open('outputs/analysis/full_report.json') as f:
    report = json.load(f)

# Extract specific insights
attention_entropy = [
    stat['entropy'] 
    for stat in report['attention_analysis']['layer_stats']
]

print(f"Attention entropy per layer: {attention_entropy}")

# Compare multiple runs
reports = []
for i in range(5):
    # ... run model ...
    analyzer = TransformerAnalyzer(model, state)
    reports.append(analyzer.generate_comprehensive_report(
        save_path=f'outputs/analysis/run_{i}.json'
    ))

# Aggregate statistics
avg_entropy = sum(
    stat['entropy'] 
    for r in reports 
    for stat in r['attention_analysis']['layer_stats']
) / sum(len(r['attention_analysis']['layer_stats']) for r in reports)

print(f"Average entropy across runs: {avg_entropy:.2f}")
```

## Advanced Use Cases

### 1. Monitor Training Health

```python
# During training
for epoch in range(num_epochs):
    # ... training loop ...
    
    with torch.no_grad():
        logits, state = model(val_batch, return_state=True)
    
    analyzer = TransformerAnalyzer(model, state)
    geometry = analyzer.analyze_representational_geometry()
    
    # Check for collapse
    final_dim = geometry['dimensionality'][-1]['intrinsic_dim_ratio']
    if final_dim < 0.1:
        print(f"⚠️  Epoch {epoch}: Dimensionality collapse detected!")
```

### 2. Compare Architectures

```python
configs = [
    TransformerConfig(n_heads=4, n_layers=6),
    TransformerConfig(n_heads=8, n_layers=4),
    TransformerConfig(n_heads=16, n_layers=2)
]

for i, config in enumerate(configs):
    model = Transformer(config)
    # ... forward pass ...
    
    analyzer = TransformerAnalyzer(model, state)
    report = analyzer.generate_comprehensive_report(
        save_path=f'outputs/analysis/arch_{i}.json'
    )
    
    print(f"\nArchitecture {i} ({config.n_heads} heads, {config.n_layers} layers):")
    print(f"  Avg attention entropy: {sum(s['entropy'] for s in report['attention_analysis']['layer_stats']) / len(report['attention_analysis']['layer_stats']):.2f}")
```

### 3. Debug Specific Heads

```python
attention_results = analyzer.analyze_attention_patterns()

# Find problematic heads
problematic = [
    h for h in attention_results['head_specialization']
    if h['entropy'] < 0.5  # Too focused
    or h['entropy'] > 3.5  # Too diffuse
]

if problematic:
    print("Problematic heads detected:")
    for h in problematic:
        print(f"  Layer {h['layer']}, Head {h['head']}: "
              f"entropy={h['entropy']:.2f}, pattern={h['pattern_type']}")
```

## Next Steps

- See [Animation Guide](animations.md) for visual analysis
- Check [Visualization Guide](visualization.md) for static plots
- Review [API Reference](../api/analysis.md) for detailed parameters


