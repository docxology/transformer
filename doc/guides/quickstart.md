# Quick Start Guide

Get running with the transformer in 5 minutes.

## Installation

```bash
# Clone repository
git clone <repository-url>
cd transformer

# Install dependencies
pip install -r requirements.txt
```

## Your First Transformer

```python
from src.transformer import Transformer, TransformerConfig

# Create configuration
config = TransformerConfig(
    vocab_size=10000,
    d_model=512,
    n_heads=8,
    n_layers=6,
    d_ff=2048
)

# Create model
model = Transformer(config)

# Use model
import torch
input_ids = torch.randint(0, 10000, (2, 10))  # [batch, seq_len]
logits, state = model(input_ids, return_state=True)

print(f"Input shape: {input_ids.shape}")
print(f"Output shape: {logits.shape}")
```

## Basic Visualization

```python
from src.visualization import visualize_attention

# Visualize attention patterns
attn_weights = state.get_attention_weights(layer_idx=0)
visualize_attention(attn_weights, save_path='attention.png')
```

## Training Loop

```python
import torch.nn as nn
import torch.optim as optim

# Setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training step
model.train()
optimizer.zero_grad()

# Forward
logits, _ = model(input_ids)
loss = criterion(logits.view(-1, config.vocab_size), targets.view(-1))

# Backward
loss.backward()
optimizer.step()

print(f"Loss: {loss.item():.4f}")
```

## Generation

```python
# Generate text
model.eval()
prompt = torch.randint(0, 10000, (1, 5))  # Starting tokens

with torch.no_grad():
    generated = model.generate(
        prompt,
        max_new_tokens=20,
        temperature=1.0,
        top_k=50
    )

print(f"Generated {generated.shape[1]} tokens")
```

## Using Presets

```python
from src.config import get_preset_config

# Use preset configurations
config = get_preset_config('small')  # or 'base', 'large', etc.
model = Transformer(config)
```

## State Tracking

```python
# Get complete computational state
logits, state = model(input_ids, return_state=True)

# Access residual stream
residuals = state.get_residual_stream()
print(f"Checkpoints: {len(residuals)}")

# Access attention weights
for layer_idx in range(config.n_layers):
    attn = state.get_attention_weights(layer_idx)
    print(f"Layer {layer_idx} attention shape: {attn.shape}")
```

## Evaluation

```python
from src.evaluation import compute_perplexity, compute_accuracy

# Evaluate model
model.eval()
with torch.no_grad():
    logits, _ = model(input_ids)
    
    perplexity = compute_perplexity(logits, targets)
    accuracy = compute_accuracy(logits, targets)
    
    print(f"Perplexity: {perplexity:.2f}")
    print(f"Accuracy: {accuracy:.4f}")
```

## Saving and Loading

```python
from src.config import save_config, load_config

# Save configuration
save_config(config, 'my_config.json')

# Load configuration
config = load_config('my_config.json')

# Save model
torch.save(model.state_dict(), 'model.pth')

# Load model
model = Transformer(config)
model.load_state_dict(torch.load('model.pth'))
```

## Next Steps

- [Basic Usage Guide](basic_usage.md) - More detailed examples
- [Training Guide](training.md) - Complete training workflow
- [Visualization Guide](visualization.md) - All visualization options
- [API Reference](../api/model.md) - Detailed API documentation

## Verification

Run the verification script to ensure everything works:

```bash
python3 verify_installation.py
```

## Common Issues

### Import Error
```bash
# Make sure you're in the project root
cd /path/to/transformer
python3 -c "from src.transformer import Transformer"
```

### CUDA Out of Memory
```python
# Use smaller model
config = get_preset_config('small')

# Or reduce batch size
batch_size = 2
```

### Slow Performance
```python
# Use smaller sequence length
seq_len = 128  # instead of 512

# Or use CPU for debugging
device = torch.device('cpu')
```

## Examples

See the `examples/` directory for complete working examples:

- `basic_usage.py` - Comprehensive basic example
- `path_analysis.py` - Information flow analysis
- `animation_demo.py` - Creating animations
- `training_example.py` - Full training loop


