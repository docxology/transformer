# Examples

Comprehensive examples demonstrating the transformer implementation and visualization capabilities.

---

## 📚 Documentation

**Complete Documentation**: See `doc/DOCUMENTATION_INDEX.md` for comprehensive guides, theory, and tutorials.

**Quick Links**:
- **Theory**: `doc/theory/TRANSFORMER_THEORY.md` - Understand how transformers work
- **Implementation**: `doc/guides/IMPLEMENTATION_GUIDE.md` - Build and train models
- **Recipes**: `doc/praxis/PRACTICAL_RECIPES.md` - Quick, actionable solutions
- **FAQ**: `doc/FAQ_COMPREHENSIVE.md` - Common questions answered
- **Glossary**: `doc/GLOSSARY.md` - Terms and concepts explained

---

## Basic Usage

```bash
python3 examples/basic_usage.py
```

Demonstrates:
- Creating a transformer model
- Forward pass with state tracking
- Basic visualizations of attention and information flow
- Extracting residual stream and K/V stream data

**See also**: `doc/guides/quickstart.md` for detailed walkthrough

## Path Analysis

```bash
python3 examples/path_analysis.py
```

Explores the astronomical number of computational paths through the transformer:
- Computes path counts for various layer/position displacements
- Traces specific information flow paths
- Visualizes path density through the network
- Explains implications for transformer cognition

**Key Insight**: A full traverse from (layer 0, position 0) to (layer N-1, position M-1) has C(N+M-2, N-1) distinct paths, which quickly exceeds the number of atoms in the universe.

**Deep dive**: `doc/theory/TRANSFORMER_THEORY.md` §Computational Path Explosion  
**Mathematics**: `doc/theory/MATHEMATICS.md` §13 Path Counting

## Animation Demo

```bash
python3 examples/animation_demo.py
```

Creates animated visualizations:
- Information flow through layers (vertical + horizontal)
- Attention pattern evolution
- Specific path tracing through the network

Generates GIF animations in `outputs/animations/` directory.

**Guide**: `doc/guides/animations.md` - Complete animation reference  
**API**: `doc/api/visualization.md` - All animation functions

## Training Example

```bash
python3 examples/training_example.py
```

Demonstrates:
- Complete training loop setup
- Loss monitoring and optimization
- Visualization of attention patterns during training
- Analysis of learned representations

**Practical guide**: `doc/praxis/PRACTICAL_RECIPES.md` Recipe 1 - Train a Language Model  
**In-depth**: `doc/guides/IMPLEMENTATION_GUIDE.md` §Training  
**Troubleshooting**: `doc/FAQ_COMPREHENSIVE.md` §Training Questions

## Understanding the Outputs

### Attention Visualizations
Show how each attention head distributes weights across positions. Bright spots indicate strong attention between query and key positions.

### Residual Stream
Shows the vertical information highway - how representations evolve through layers at a specific position.

### K/V Stream
Shows the horizontal information highway - Query (where to look), Key (what to offer), Value (what to transmit).

### Information Highways
Combined view of both vertical (residual) and horizontal (attention) information flow.

### Path Analysis
Reveals the massive redundancy and interference patterns that enable robust, nuanced information processing.

## Custom Usage

You can easily adapt these examples for your own use cases:

```python
from src.transformer import Transformer, TransformerConfig
from src.visualization import *

# Configure model
config = TransformerConfig(
    vocab_size=10000,
    d_model=512,
    n_heads=8,
    n_layers=6
)

# Create and use model
model = Transformer(config)
logits, state = model(input_ids, return_state=True)

# Visualize any aspect
visualize_attention(state.get_attention_weights(layer_idx=0))
visualize_residual_stream(state)
animate_information_flow(state)
```

**Full implementation guide**: `doc/guides/IMPLEMENTATION_GUIDE.md`  
**Quick recipes**: `doc/praxis/PRACTICAL_RECIPES.md`  
**API reference**: `doc/api/visualization.md`

---

## 🎓 Learning Paths

### New to Transformers?
1. Read `doc/theory/TRANSFORMER_THEORY.md` (start with §1-4)
2. Run `basic_usage.py` to see it in action
3. Check `doc/GLOSSARY.md` for terminology
4. Ask questions via `doc/FAQ_COMPREHENSIVE.md`

### Want to Build Something?
1. Follow `doc/guides/quickstart.md`
2. Study `training_example.py`
3. Use `doc/praxis/PRACTICAL_RECIPES.md` for common tasks
4. Reference `doc/guides/IMPLEMENTATION_GUIDE.md` for details

### Interested in the Math?
1. Read `doc/theory/MATHEMATICS.md` for rigorous derivations
2. Study `path_analysis.py` for path counting
3. Check `doc/theory/STATISTICS.md` for empirical analysis
4. Explore source code in `src/transformer/`

---

## 📊 Output Organization

All examples generate outputs in structured directories:

```
outputs/
├── visualizations/      # Static plots and heatmaps
│   ├── attention_*.png
│   ├── residual_stream.png
│   ├── kv_stream.png
│   ├── information_highways.png
│   └── training/        # Training-specific visualizations
├── animations/          # Animated GIFs
│   ├── 01_information_flow.gif
│   ├── 02_attention_evolution.gif
│   └── 03_path_trace.gif
├── analysis/           # JSON analysis data
├── logs/              # Training logs
└── reports/           # Generated reports
```

---

## 🔗 Navigation

- **Up**: `../README.md` - Project root
- **Documentation**: `../doc/DOCUMENTATION_INDEX.md` - Complete guide
- **Source Code**: `../src/` - Implementation
- **Tests**: `../tests/` - Test suite

---

**Last Updated**: October 2025  
**For comprehensive documentation**: See `doc/DOCUMENTATION_INDEX.md`


