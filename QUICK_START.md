# Quick Start Guide

Get started with the First-Principles Transformer in under 5 minutes.

---

## 1. Clone & Setup (1 minute)

```bash
git clone <repository-url>
cd transformer
```

---

## 2. One-Command Run (2-3 minutes)

```bash
python run.py
```

This automatically:
- ✅ Installs all dependencies (uses `uv` if available for 10-100× faster install)
- ✅ Verifies installation
- ✅ Runs all 4 examples
- ✅ Generates visualizations
- ✅ Creates animations
- ✅ Validates everything works

**Output**: Check `outputs/` directory for all generated files

---

## 3. Basic Usage (< 1 minute)

```python
from src.transformer import Transformer, TransformerConfig

# Create model
config = TransformerConfig(
    vocab_size=10000,
    d_model=512,
    n_heads=8,
    n_layers=6
)
model = Transformer(config)

# Forward pass with state tracking
import torch
input_ids = torch.randint(0, 10000, (2, 128))
logits, state = model(input_ids, return_state=True)

# Visualize
from src.visualization import visualize_attention
attention = state.get_attention_weights(layer_idx=0)
visualize_attention(attention, save_path='my_attention.png')
```

---

## 4. Next Steps

**Learn Theory** (2-3 hours):
- Read [Transformer Theory](doc/theory/TRANSFORMER_THEORY.md)
- Understand the dual information highways
- See computational path explosion

**Implement Something** (1-2 hours):
- Follow [Implementation Guide](doc/guides/IMPLEMENTATION_GUIDE.md)
- Try [Practical Recipes](doc/praxis/PRACTICAL_RECIPES.md)
- Run `examples/training_example.py`

**Explore Visualizations**:
- Check `outputs/visualizations/` - 10+ static plots
- Check `outputs/animations/` - 9 animated GIFs
- Read [Visualization Guide](doc/guides/visualization.md)

**Get Help**:
- [FAQ](doc/FAQ_COMPREHENSIVE.md) - 45 questions answered
- [Glossary](doc/GLOSSARY.md) - 150+ terms defined
- [Documentation Index](doc/DOCUMENTATION_INDEX.md) - Master hub

---

## Common Commands

```bash
# Run all examples
python run.py

# Skip installation (if already installed)
python run.py --skip-install

# Just run examples (skip other generation)
python run.py --examples-only

# Use virtual environment
python run.py --use-venv

# Run specific example
python examples/basic_usage.py
python examples/training_example.py
python examples/animation_demo.py
python examples/path_analysis.py

# Run tests
pytest tests/ -v

# Generate specific animations
python scripts/generate_all_animations.py

# Generate architecture visualizations
python scripts/generate_architecture_visualizations.py
```

---

## What You Get

### Code
- ✅ Complete transformer implementation
- ✅ 20+ visualization functions
- ✅ 74 comprehensive tests
- ✅ 4 complete examples
- ✅ 7 configuration presets

### Documentation
- ✅ ~100,000 words comprehensive docs
- ✅ 6 structured learning paths
- ✅ 45-question FAQ
- ✅ 150-term glossary
- ✅ Complete theory + math + stats

### Visualizations
- ✅ Attention heatmaps
- ✅ Information flow diagrams
- ✅ Architecture diagrams
- ✅ Training visualizations
- ✅ 9 types of animations

---

## Requirements

**Python**: 3.8+  
**Dependencies**: All installed automatically by `run.py`

**Optional but recommended**:
```bash
# Install uv for 10-100× faster dependency installation
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## File Structure

```
transformer/
├── run.py                    # ⭐ Start here
├── README.md                 # Project overview
├── QUICK_START.md           # This file
├── PROJECT_STATUS.md        # Current status
│
├── examples/                # 4 complete examples
├── src/                     # All source code
├── tests/                   # 74 tests
├── doc/                     # Complete documentation
└── outputs/                 # Generated files
```

---

## Troubleshooting

**Issue**: Dependencies won't install  
**Solution**: Try `python run.py --use-venv` to use virtual environment

**Issue**: Examples fail  
**Solution**: Run `pytest tests/` to verify installation

**Issue**: Out of memory  
**Solution**: Use smaller config preset:
```python
from src.config import get_preset_config
config = get_preset_config('small')  # or 'tiny' or 'debug'
```

**Issue**: Need help  
**Solution**: Check [FAQ](doc/FAQ_COMPREHENSIVE.md) or [Glossary](doc/GLOSSARY.md)

---

## What's Next?

### For Learning
1. Read [Transformer Theory](doc/theory/TRANSFORMER_THEORY.md)
2. Follow [Complete Beginner Path](doc/DOCUMENTATION_INDEX.md#path-1-complete-beginner)
3. Explore visualizations in `outputs/`

### For Building
1. Read [Implementation Guide](doc/guides/IMPLEMENTATION_GUIDE.md)
2. Try [Practical Recipes](doc/praxis/PRACTICAL_RECIPES.md)
3. Modify `examples/training_example.py`

### For Research
1. Read [Architecture](doc/ARCHITECTURE.md)
2. Study [Path Analysis](examples/path_analysis.py)
3. Explore source code in `src/`

---

## Full Documentation

**📚 [Complete Documentation Index](doc/DOCUMENTATION_INDEX.md)**

This is your master hub with:
- 6 learning paths for different goals
- Complete topic-based organization
- 400+ cross-references
- Quick reference guide

---

**Ready?** Run `python run.py` and start exploring! 🚀

