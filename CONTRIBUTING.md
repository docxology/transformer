# Contributing to First-Principles Transformer

Thank you for your interest in contributing! This project welcomes contributions from the community.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Code of Conduct](#code-of-conduct)
3. [How to Contribute](#how-to-contribute)
4. [Development Setup](#development-setup)
5. [Code Style](#code-style)
6. [Testing](#testing)
7. [Documentation](#documentation)
8. [Pull Request Process](#pull-request-process)
9. [Areas for Contribution](#areas-for-contribution)

---

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Set up the development environment** (see below)
4. **Create a branch** for your changes
5. **Make your changes** following our guidelines
6. **Test thoroughly** before submitting
7. **Submit a pull request**

---

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers warmly
- Focus on constructive feedback
- Prioritize educational value
- Maintain professional discourse

### Our Values

- **Clarity over cleverness**: Code should be easy to understand
- **Education first**: Help others learn
- **Quality over speed**: Take time to do it right
- **Test everything**: No untested code
- **Document thoroughly**: Explain the "why"

---

## How to Contribute

### Types of Contributions

**Code Contributions**:
- New features
- Bug fixes
- Performance improvements
- Test improvements
- Visualization enhancements

**Documentation Contributions**:
- Fix typos or errors
- Add examples
- Improve clarity
- Add tutorials
- Translate documentation

**Community Contributions**:
- Answer questions
- Report bugs
- Suggest features
- Share use cases
- Write blog posts

---

## Development Setup

### Prerequisites

- Python 3.8+
- git
- (Optional) `uv` for fast dependency installation

### Setup Steps

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/transformer.git
cd transformer

# Set up virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
python run.py --use-venv

# Or manually:
pip install -r requirements.txt

# Run tests to verify
pytest tests/ -v

# Run examples to verify
python examples/basic_usage.py
```

---

## Code Style

### Python Style Guide

We follow **PEP 8** with these specifics:

**General**:
- Use 4 spaces for indentation (no tabs)
- Maximum line length: 100 characters
- Use descriptive variable names
- Add type hints where helpful

**Imports**:
```python
# Standard library
import os
from pathlib import Path

# Third-party
import torch
import numpy as np

# Local
from src.transformer import Transformer
from src.utils import helpers
```

**Comments**:
```python
# Good: Explain WHY, not WHAT
# Use causal masking to prevent information leakage from future tokens
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)

# Bad: Obvious what the code does
# Create a mask with ones above the diagonal
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
```

**Docstrings**:
```python
def visualize_attention(
    attention_weights: torch.Tensor,
    layer_idx: int = 0,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize attention patterns for all heads in a layer.
    
    Creates a grid of heatmaps showing how each attention head
    distributes attention across positions. This reveals what
    patterns each head has learned.
    
    Args:
        attention_weights: Shape (batch, heads, seq_len, seq_len)
        layer_idx: Which layer to visualize
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> attention = state.get_attention_weights(layer_idx=0)
        >>> fig = visualize_attention(attention, save_path='attention.png')
    """
    # Implementation...
```

### .cursorrules Compliance

Follow the project's `.cursorrules`:
- Professional, functional, intelligent code
- Clear "why" comments
- Modular architecture
- Test-driven development
- No mock methods - use real data
- Track both information highways
- Enable state tracking

---

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_model.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_model.py::test_forward_pass -v
```

### Writing Tests

**Location**: Place tests in `tests/test_*.py`

**Structure**:
```python
import pytest
import torch
from src.transformer import Transformer, TransformerConfig

def test_forward_pass_shape():
    """Test that forward pass returns correct shape."""
    config = TransformerConfig(
        vocab_size=1000,
        d_model=64,
        n_heads=4,
        n_layers=2
    )
    model = Transformer(config)
    
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    logits, state = model(input_ids, return_state=True)
    
    assert logits.shape == (batch_size, seq_len, 1000)
    assert state is not None

def test_attention_causality():
    """Test that attention respects causality."""
    # Test implementation...
```

**Requirements**:
- All new features must have tests
- Aim for high coverage (>80%)
- Test edge cases
- Test error conditions
- Use real data, not mocks

---

## Documentation

### Documentation Standards

**When to Document**:
- All new features
- All public functions/classes
- Complex algorithms
- Design decisions
- Breaking changes

**Where to Document**:
- **Docstrings**: For functions/classes
- **Comments**: For complex logic
- **README.md**: For overview
- **doc/**: For detailed guides
- **CHANGELOG.md**: For all changes

### Documentation Structure

```
doc/
├── theory/              # Conceptual explanations
├── guides/              # How-to guides
├── praxis/              # Practical recipes
├── concepts/            # Deep dives
└── api/                 # API reference
```

### Writing Style

- Use clear, simple language
- Show, don't just tell (include examples)
- Progressive disclosure (simple → complex)
- Cross-reference related content
- Use analogies where helpful

---

## Pull Request Process

### Before Submitting

**Checklist**:
- [ ] Code follows style guidelines
- [ ] All tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] No linter errors
- [ ] Commit messages are clear

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
Describe tests added/modified

## Documentation
List documentation changes

## Screenshots (if applicable)
Add screenshots for visual changes

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Code style followed
- [ ] Changelog updated
```

### Review Process

1. **Automated checks** run (tests, linting)
2. **Maintainer review** for code quality
3. **Discussion** on design decisions
4. **Revisions** if needed
5. **Approval** and merge

### Commit Messages

**Format**:
```
Type: Brief description (50 chars max)

Longer explanation if needed (wrap at 72 chars).
Explain WHY the change was made, not just WHAT.

- Bullet points for multiple changes
- Reference issues: Fixes #123
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code restructuring
- `perf`: Performance improvement
- `style`: Formatting

**Examples**:
```
feat: Add gradient flow visualization

Adds visualize_gradient_flow() to help debug training issues.
Creates heatmap of gradient magnitudes across all layers.

- Uses log scale for better visibility
- Includes parameter names on y-axis
- Saves to outputs/visualizations/

Closes #45
```

---

## Areas for Contribution

### High Priority

**Performance**:
- Optimize attention computation
- Add Flash Attention support
- Profile and improve bottlenecks
- Memory usage optimization

**Features**:
- Interactive visualizations (Plotly/Dash)
- Jupyter notebook tutorials
- Model architecture search
- Pre-training recipes

**Documentation**:
- Video tutorials
- Jupyter notebooks
- More practical examples
- Translation to other languages

### Good First Issues

**Easy**:
- Fix typos in documentation
- Add docstrings to functions
- Write additional tests
- Improve error messages
- Add type hints

**Medium**:
- Create new visualization functions
- Add configuration presets
- Write tutorial notebooks
- Implement new metrics

**Hard**:
- Add distributed training support
- Implement sparse attention
- Add model quantization
- Create interactive dashboards

### Feature Requests

See [GitHub Issues](issues) for current feature requests and discussions.

---

## Development Guidelines

### Project Philosophy

This project prioritizes:
1. **Clarity** - Code should be easy to understand
2. **Education** - Help others learn transformers
3. **Transparency** - All information flow visible
4. **Quality** - Professional, tested code
5. **Documentation** - Comprehensive explanations

### Design Principles

**Modularity**:
- Single responsibility per module
- Clear interfaces
- Composable components
- Minimal coupling

**State Tracking**:
- All components return state
- Enable inspection at any layer
- Track both information highways
- Support visualization

**Visualization**:
- Every aspect should be visualizable
- Support both static and animated
- Clear, informative plots
- Consistent color schemes

---

## Questions?

- **Documentation**: See [DOCUMENTATION_INDEX.md](doc/DOCUMENTATION_INDEX.md)
- **FAQ**: See [FAQ_COMPREHENSIVE.md](doc/FAQ_COMPREHENSIVE.md)
- **Glossary**: See [GLOSSARY.md](doc/GLOSSARY.md)
- **Issues**: Open a GitHub issue
- **Discussions**: Use GitHub Discussions

---

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

## Recognition

All contributors are recognized in:
- GitHub contributors page
- CHANGELOG.md for significant contributions
- Project documentation where appropriate

---

**Thank you for contributing to making transformers more understandable!** 🎉

