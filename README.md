# First-Principles Transformer

A **comprehensive, production-quality, educational** transformer implementation with complete visualization, evaluation, and documentation.

[![Tests](https://img.shields.io/badge/tests-109%20passing-brightgreen)]() 
[![Documentation](https://img.shields.io/badge/docs-complete-blue)]()
[![Code Style](https://img.shields.io/badge/code%20style-modular-orange)]()

## What Makes This Different

Unlike typical transformer implementations, this provides:

✅ **Complete Observability** - Track both information highways (residual + K/V streams)  
✅ **20+ Visualizations** - From attention heatmaps to animated flow diagrams  
✅ **109 Tests** - Comprehensive coverage of all components  
✅ **Full Documentation** - 20+ pages of concepts, guides, and API reference  
✅ **Educational Design** - Built to understand, not just use  
✅ **Research-Ready** - Easy to modify and extend  
✅ **Production-Quality** - Clean, tested, documented code  

## Information Flow Architecture

The transformer processes information through **two distinct pathways**:

### 1. Residual Stream (Vertical)
Information flows **UP** through layers at each position, carrying accumulated representation through depth.

### 2. K/V Stream (Horizontal)  
Information flows **RIGHT** across positions via attention, enabling dynamic information sharing.

### Path Explosion
For `n` layers and `m` positions: **C(n+m, n)** distinct computational paths!
- 6 layers × 32 positions = **2.3 million paths**
- 12 layers × 512 positions = **> atoms in universe**

This massive redundancy enables robust, nuanced information processing.

## Quick Start

**⚡ [See QUICK_START.md](QUICK_START.md) for 5-minute setup guide**

### One-Command Setup & Execution

The fastest way to get started - install, verify, and run everything:

```bash
git clone https://github.com/docxology/transformer.git
cd transformer
python3 run.py
```

This will:
- ✅ Auto-detect and use `uv` for fast installation (falls back to pip)
- ✅ Install dependencies
- ✅ Verify installation  
- ✅ Run all examples
- ✅ Generate animations
- ✅ Create comprehensive outputs

**Recommended: Install uv for 10-100x faster dependency installation**
```bash
# Install uv (optional but recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Then run normally - uv will be auto-detected
python3 run.py
```

**Options:**
```bash
python3 run.py --skip-install     # Skip dependency check
python3 run.py --examples-only    # Only run examples
python3 run.py --use-venv         # Create and use virtual environment
```

### Manual Installation

```bash
git clone https://github.com/docxology/transformer.git
cd transformer
pip install -r requirements.txt
```

### Verify Installation

```bash
python3 scripts/verify_installation.py
```

### Basic Usage

```python
from src.transformer import Transformer, TransformerConfig
from src.visualization import visualize_attention

# Create model
config = TransformerConfig(
    vocab_size=10000,
    d_model=512,
    n_heads=8,
    n_layers=6
)
model = Transformer(config)

# Forward pass with state tracking
input_ids = torch.randint(0, 10000, (2, 10))
logits, state = model(input_ids, return_state=True)

# Visualize attention
attn = state.get_attention_weights(layer_idx=0)
visualize_attention(attn, save_path='attention.png')
```

### Use Presets

```python
from src.config import get_preset_config

# Quick configuration
config = get_preset_config('small')  # or 'base', 'large', etc.
model = Transformer(config)
```

## Key Features

### 🎨 Visualization System

**Basic:**
- Attention heatmaps (all heads, all layers)
- Residual stream evolution  
- K/V stream patterns
- Layer statistics

**Advanced:**
- Gradient flow analysis
- Activation patterns
- Causal graph representation
- Attention flow Sankey diagrams
- Token journey tracking
- Head specialization analysis

**Animations:**
- Information flow through layers
- Attention evolution
- Specific path tracing

### 📊 Evaluation & Benchmarking

- Perplexity and accuracy metrics
- Inference speed benchmarking
- Training speed measurement
- Model profiling
- Configuration comparison
- Metrics tracking system

### ⚙️ Configuration Management

- Save/load configs (JSON/YAML)
- 7 preset configurations (debug to xlarge)
- Configuration validation
- Easy modification

### 📈 Training Tools

- `TrainingVisualizer` for monitoring
- Weight distribution tracking
- Learning dynamics analysis
- Gradient visualization

### 🧪 Comprehensive Testing

109 tests covering:
- All components individually
- Full model integration
- Visualization functions
- Evaluation metrics
- Configuration management

## Repository Structure

```
transformer/
├── run.py              # ⭐ One-command setup & execution
├── README.md           # This file
├── requirements.txt    # Dependencies
│
├── src/
│   ├── transformer/        # Core model & components
│   ├── visualization/      # 20+ visualization functions
│   ├── evaluation/        # Metrics & benchmarking
│   ├── config/           # Configuration management
│   └── utils/            # Helper functions
│
├── tests/                # 109 comprehensive tests
├── examples/            # 4 complete examples
├── scripts/             # Utility scripts
│   ├── verify_installation.py
│   ├── generate_all_animations.py
│   ├── generate_architecture_visualizations.py
│   ├── generate_complete_documentation.py
│   ├── generate_subway_map.py
│   ├── organize_outputs.py
│   └── run_comprehensive_validation.py
│
├── doc/                 # Complete documentation
│   ├── concepts/       # Core concepts explained
│   ├── guides/         # User guides
│   ├── api/           # API reference
│   ├── ARCHITECTURE.md     # Architecture deep dive
│   ├── OVERVIEW.md        # Complete overview
│   └── FAQ.md         # 32 questions answered
│
└── outputs/            # Generated outputs (animations, analysis, etc.)
```

## Examples

Run the included examples:

```bash
# Basic usage and visualization
python3 examples/basic_usage.py

# Path analysis and complexity
python3 examples/path_analysis.py

# Create animations
python3 examples/animation_demo.py

# Full training workflow
python3 examples/training_example.py
```

## Documentation

**📚 Start here:** [`doc/DOCUMENTATION_INDEX.md`](doc/DOCUMENTATION_INDEX.md) - Master documentation hub

**Quick Links:**
- [Quick Start Guide](doc/guides/quickstart.md) - Get started in 5 minutes
- [Learning Paths](doc/DOCUMENTATION_INDEX.md#learning-paths) - 6 structured paths (2-6 hours each)
- [FAQ](doc/FAQ_COMPREHENSIVE.md) - 45 questions answered
- [Glossary](doc/GLOSSARY.md) - 77 terms defined
- [Practical Recipes](doc/praxis/PRACTICAL_RECIPES.md) - 12 copy-paste solutions

**Theory & Concepts:**
- [Transformer Theory](doc/theory/TRANSFORMER_THEORY.md) - Complete conceptual foundation
- [Mathematics](doc/theory/MATHEMATICS.md) - Rigorous mathematical derivations
- [Statistics](doc/theory/STATISTICS.md) - Empirical analysis and best practices
- [Information Flow](doc/concepts/information_flow.md) - Dual highway architecture
- [Attention Mechanism](doc/concepts/attention.md) - Q, K, V explained

**Implementation:**
- [Implementation Guide](doc/guides/IMPLEMENTATION_GUIDE.md) - Complete practical guide
- [Visualization Guide](doc/guides/visualization.md) - All 20+ visualization functions
- [Animation Guide](doc/guides/animations.md) - Creating animated visualizations

**Total Documentation**: ~33,500 words across 37 files in `doc/`

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_components.py -v
pytest tests/test_visualization.py -v
pytest tests/test_evaluation.py -v
```

All 109 tests passing ✅

## Configuration Presets

```python
from src.config import get_preset_config, list_presets

# Available presets
presets = list_presets()
# debug, minimal, tiny, small, base, large, xlarge

# Use preset
config = get_preset_config('base')
```

| Preset | d_model | Layers | Params | Use Case |
|--------|---------|--------|---------|----------|
| debug | 64 | 2 | ~40K | Quick testing |
| tiny | 128 | 2 | ~150K | Small experiments |
| small | 256 | 4 | ~1.2M | Most experiments |
| base | 512 | 6 | ~40M | Serious training |
| large | 1024 | 12 | ~300M | Large-scale |

## Visualization Examples

```python
from src.visualization import (
    visualize_attention,
    visualize_information_highways,
    animate_information_flow,
    visualize_head_specialization,
    visualize_token_journey
)

# Get state from forward pass
logits, state = model(input_ids, return_state=True)

# Basic attention visualization
visualize_attention(state.get_attention_weights(0))

# Information highways (both streams)
visualize_information_highways(state)

# Animated flow
animate_information_flow(state, save_path='flow.gif')

# Head analysis
visualize_head_specialization(state)

# Track single token
visualize_token_journey(state, token_position=5)
```

## Training Example

```python
from src.visualization import TrainingVisualizer

viz = TrainingVisualizer()

for epoch in range(num_epochs):
    # Train
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss = evaluate(model, val_loader)
    
    # Log metrics
    viz.log_metrics(
        train_loss=train_loss,
        val_loss=val_loss,
        learning_rate=optimizer.param_groups[0]['lr']
    )
    
    # Periodic visualization
    if epoch % 10 == 0:
        viz.plot_training_curves(f'training_epoch_{epoch}.png')

# Save history
viz.save_history('training_history.json')
```

## Evaluation & Benchmarking

```python
from src.evaluation import evaluate_model, benchmark_inference, profile_model

# Comprehensive evaluation
results = evaluate_model(model, val_loader, device)
print(results)  # Perplexity, accuracy, top-5 accuracy

# Benchmark inference
bench_results = benchmark_inference(
    model, 
    input_shape=(2, 128),
    vocab_size=config.vocab_size,
    device=device
)

# Profile model
profile = profile_model(model, input_shape=(2, 128), vocab_size=config.vocab_size, device=device)
print(f"Parameters: {profile['total_parameters']:,}")
print(f"Forward time: {profile['forward_time_ms']:.2f} ms")
```

## Philosophy

This implementation embodies:

🎯 **Transparency** - No black boxes, everything observable  
📚 **Education** - Built to learn from, not just use  
🔬 **Research** - Easy to modify and experiment  
✨ **Quality** - Production-grade, tested code  
🧩 **Modularity** - Composable, reusable components  
📖 **Documentation** - Comprehensive, clear explanations  

## Use Cases

- **Education**: Learn transformer architecture
- **Research**: Experiment with modifications
- **Prototyping**: Quick testing of ideas
- **Analysis**: Deep model interpretation
- **Visualization**: Create publication-quality figures

## Performance Note

This implementation **prioritizes clarity over speed**:
- ~5-10x slower than optimized implementations
- Full state tracking adds overhead
- Suitable for small-to-medium experiments
- For production: use PyTorch native or HuggingFace

## What's Included

✅ Complete transformer with state tracking  
✅ 20+ visualization functions  
✅ Training & evaluation tools  
✅ Configuration management  
✅ 109 comprehensive tests  
✅ 20+ documentation pages  
✅ 4 complete examples  
✅ Path analysis tools  
✅ Benchmarking system  

## Dependencies

All standard, well-maintained libraries:
- PyTorch ≥2.6.0 (Python 3.10+)
- NumPy, Matplotlib, Seaborn, Plotly
- PyYAML, NetworkX
- Pytest (for testing)

## Contributing

Contributions welcome! See [`CONTRIBUTING.md`](CONTRIBUTING.md)

Areas of interest:
- New visualizations
- Performance improvements
- Documentation improvements
- Example notebooks
- Educational content

## License

[Your chosen license]

## Citation

If you use this in your research, please cite:

```bibtex
@software{first_principles_transformer,
  title={First-Principles Transformer: A Comprehensive Educational Implementation},
  author={docxology},
  year={2025},
  url={https://github.com/docxology/transformer}
}
```

## Acknowledgments

Built on foundational work:
- "Attention Is All You Need" (Vaswani et al., 2017)
- Mechanistic interpretability research
- PyTorch ecosystem

## Project Status & Documentation

- [📊 Documentation Verification Report](DOCUMENTATION_VERIFICATION_REPORT.md) - Documentation audit results
- [📚 Documentation Index](doc/DOCUMENTATION_INDEX.md) - Master documentation hub
- [🔍 Validation Certificate](doc/VALIDATION_CERTIFICATE.md) - Evidence that all methods are real (no mocks)
- [📖 Architecture](doc/ARCHITECTURE.md) - Information flow architecture
- [🎬 Animation Index](doc/ANIMATION_INDEX.md) - Animation reference guide

---

**Built with clarity, tested thoroughly, visualized completely, documented comprehensively.**

