# Practical Implementation Guide

## Overview

This guide provides practical guidance for implementing, training, and deploying transformer models using this codebase.

---

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/docxology/transformer.git
cd transformer

# Run complete setup (handles dependencies, examples, validation)
python run.py
```

The `run.py` script automatically:
- Checks for `uv` (fast package installer) or uses `pip`
- Installs all dependencies
- Runs all examples
- Generates visualizations and animations
- Validates the installation

### 2. Basic Usage

```python
from src.transformer import Transformer, TransformerConfig

# Configure model
config = TransformerConfig(
    vocab_size=10000,
    d_model=512,
    n_heads=8,
    n_layers=6,
    d_ff=2048,
    dropout=0.1
)

# Create model
model = Transformer(config)

# Forward pass
import torch
input_ids = torch.randint(0, config.vocab_size, (2, 128))
logits, state = model(input_ids, return_state=True)
```

**See**: `examples/basic_usage.py` for complete example

---

## Configuration

### TransformerConfig Parameters

```python
@dataclass
class TransformerConfig:
    vocab_size: int = 50000      # Vocabulary size
    d_model: int = 512           # Model dimension
    n_heads: int = 8             # Number of attention heads
    n_layers: int = 6            # Number of transformer blocks
    d_ff: int = 2048             # Feed-forward dimension
    dropout: float = 0.1         # Dropout probability
    max_seq_len: int = 5000      # Maximum sequence length
    pad_token_id: int = 0        # Padding token ID
```

### Choosing Hyperparameters

**Small Model** (fast training, limited capacity):
```python
config = TransformerConfig(
    vocab_size=10000,
    d_model=256,
    n_heads=4,
    n_layers=4,
    d_ff=1024,
    dropout=0.1
)
# Parameters: ~10M
```

**Medium Model** (balanced):
```python
config = TransformerConfig(
    vocab_size=30000,
    d_model=512,
    n_heads=8,
    n_layers=6,
    d_ff=2048,
    dropout=0.1
)
# Parameters: ~65M
```

**Large Model** (high capacity, slow training):
```python
config = TransformerConfig(
    vocab_size=50000,
    d_model=768,
    n_heads=12,
    n_layers=12,
    d_ff=3072,
    dropout=0.1
)
# Parameters: ~150M
```

### Rule of Thumb

- **d_ff**: Usually 4× d_model
- **n_heads**: Usually 8, 12, or 16
- **d_model**: Must be divisible by n_heads
- **n_layers**: 6-12 for most tasks, 24+ for very large models

---

## Training

### Basic Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Setup
model = Transformer(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, targets = batch
        
        # Forward
        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = criterion(logits.view(-1, config.vocab_size), targets.view(-1))
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
```

**Complete Example**: `examples/training_example.py`

### Recommended Settings

**Optimizer**: AdamW
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)
```

**Learning Rate Schedule**:
```python
from torch.optim.lr_scheduler import OneCycleLR

scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-3,
    epochs=num_epochs,
    steps_per_epoch=len(dataloader),
    pct_start=0.1  # 10% warmup
)
```

**Gradient Clipping**:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Training Checklist

✓ Use warmup (10% of training)  
✓ Clip gradients to norm 1.0  
✓ Apply weight decay (0.01)  
✓ Monitor validation loss  
✓ Save checkpoints regularly  
✓ Track attention statistics  
✓ Visualize attention periodically  

---

## Data Preparation

### Input Format

Model expects token IDs:
```python
# Shape: (batch_size, sequence_length)
input_ids = torch.tensor([
    [101, 2023, 2003, 1037, 6251, 102],  # Sentence 1
    [101, 2178, 6251, 2003, 2182, 102]   # Sentence 2
])
```

### Tokenization

**Basic approach** (word-level):
```python
vocab = {'<pad>': 0, '<unk>': 1, 'the': 2, 'cat': 3, ...}

def tokenize(text):
    tokens = text.lower().split()
    return [vocab.get(token, vocab['<unk>']) for token in tokens]
```

**Recommended**: Use subword tokenization (BPE, WordPiece)

### Padding

```python
from torch.nn.utils.rnn import pad_sequence

# Pad sequences to same length
padded = pad_sequence(sequences, batch_first=True, padding_value=0)
```

### Attention Mask

```python
# Mask for padding (1 = attend, 0 = ignore)
attention_mask = (input_ids != pad_token_id).float()
```

---

## State Tracking and Visualization

### Enabling State Tracking

```python
logits, state = model(input_ids, return_state=True)
```

The `state` object contains:
- Residual stream at each layer
- Attention weights for all heads
- K/V matrices

### Extracting Information

**Residual Stream** (vertical flow):
```python
residuals = state.get_residual_stream()
# List of tensors: [embedding, layer1, layer2, ..., layerN]
```

**Attention Weights** (horizontal flow):
```python
attention = state.get_attention_weights(layer_idx=0)
# Shape: (batch, n_heads, seq_len, seq_len)
```

**Layer Statistics**:
```python
from src.visualization import plot_layer_statistics
fig = plot_layer_statistics(state, save_path='layer_stats.png')
```

### Visualization Functions

```python
from src.visualization import (
    visualize_attention,
    visualize_residual_stream,
    visualize_kv_stream,
    visualize_information_highways,
    animate_information_flow
)

# Static visualizations
visualize_attention(state.get_attention_weights(0), save_path='attention.png')
visualize_residual_stream(state, position_idx=5, save_path='residual.png')
visualize_information_highways(state, save_path='highways.png')

# Animations
animate_information_flow(state, save_path='flow.gif', fps=2)
```

**See**: `examples/animation_demo.py` for complete animation examples

---

## Advanced Features

### Path Analysis

Analyze computational paths through the network:

```python
from src.visualization import (
    compute_path_counts,
    trace_information_paths,
    analyze_path_complexity
)

# Count paths between positions
count = compute_path_counts(layer_displacement=3, position_displacement=5)

# Trace specific paths
paths = trace_information_paths(
    start_layer=0,
    start_position=0,
    end_layer=2,
    end_position=3,
    max_paths=100
)

# Full analysis
analysis = analyze_path_complexity(n_layers=6, seq_len=32)
print(analysis)
```

**See**: `examples/path_analysis.py`

### Head Specialization Analysis

```python
from src.visualization import visualize_head_specialization

fig = visualize_head_specialization(state, save_path='heads.png')
```

Analyzes and visualizes different attention head behaviors:
- Local vs. global attention
- Previous token focus
- Positional patterns

### Gradient Flow Visualization

```python
from src.visualization import visualize_gradient_flow

# After backward pass
loss.backward()
fig = visualize_gradient_flow(model, save_path='gradients.png')
```

Shows gradient magnitudes through the network.

---

## Performance Optimization

### GPU Acceleration

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
input_ids = input_ids.to(device)
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        logits, _ = model(inputs)
        loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
    
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

**Speedup**: 2-3× faster training with minimal accuracy loss

### Batch Size Optimization

Find largest batch size that fits in memory:

```python
def find_max_batch_size(model, seq_len, device):
    batch_size = 1
    while True:
        try:
            input_ids = torch.randint(0, 100, (batch_size, seq_len)).to(device)
            logits, _ = model(input_ids)
            loss = logits.sum()
            loss.backward()
            optimizer.zero_grad()
            batch_size *= 2
        except RuntimeError:  # Out of memory
            return batch_size // 2
```

### Memory Optimization

**Gradient Checkpointing**:
```python
# Trade computation for memory
model.enable_gradient_checkpointing()
```

**Gradient Accumulation**:
```python
# Simulate larger batches
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    logits, _ = model(inputs)
    loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## Model Checkpointing

### Saving

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'config': config.__dict__
}
torch.save(checkpoint, 'checkpoint.pt')
```

### Loading

```python
checkpoint = torch.load('checkpoint.pt')
config = TransformerConfig(**checkpoint['config'])
model = Transformer(config)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

### Best Practices

- Save every N epochs
- Keep best model (lowest validation loss)
- Save last N checkpoints (recover from corruption)
- Include all information needed to resume training

---

## Inference and Generation

### Basic Inference

```python
model.eval()
with torch.no_grad():
    logits, state = model(input_ids, return_state=True)
    predictions = logits.argmax(dim=-1)
```

### Autoregressive Generation

```python
def generate(model, prompt, max_length=50):
    model.eval()
    generated = prompt.tolist()
    
    for _ in range(max_length):
        input_ids = torch.tensor([generated]).to(device)
        
        with torch.no_grad():
            logits, _ = model(input_ids)
        
        # Get next token
        next_token = logits[0, -1, :].argmax().item()
        generated.append(next_token)
        
        if next_token == eos_token_id:
            break
    
    return generated
```

### Sampling Strategies

**Greedy** (deterministic):
```python
next_token = logits[0, -1, :].argmax()
```

**Top-K Sampling**:
```python
top_k = 50
top_k_logits, top_k_indices = torch.topk(logits[0, -1, :], top_k)
probs = F.softmax(top_k_logits, dim=-1)
next_token = top_k_indices[torch.multinomial(probs, 1)]
```

**Nucleus (Top-P) Sampling**:
```python
top_p = 0.9
sorted_logits, sorted_indices = torch.sort(logits[0, -1, :], descending=True)
cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
filter_mask = cumulative_probs > top_p
filter_mask[0] = False  # Keep at least one token
sorted_logits[filter_mask] = float('-inf')
probs = F.softmax(sorted_logits, dim=-1)
next_token = sorted_indices[torch.multinomial(probs, 1)]
```

**Temperature Scaling**:
```python
temperature = 0.8
logits = logits / temperature
probs = F.softmax(logits[0, -1, :], dim=-1)
next_token = torch.multinomial(probs, 1)
```

---

## Debugging

### Common Issues

**Issue: NaN Loss**

Causes:
- Learning rate too high
- Numerical overflow in softmax
- Invalid data (NaN in inputs)

Solutions:
```python
# 1. Reduce learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# 2. Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

# 3. Check for NaN
assert not torch.isnan(input_ids).any()
assert not torch.isnan(logits).any()
```

**Issue: No Learning**

Causes:
- Learning rate too low
- Vanishing gradients
- Data quality issues

Solutions:
```python
# 1. Increase learning rate with warmup
# 2. Check gradient flow
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")

# 3. Visualize attention patterns
visualize_attention(state.get_attention_weights(0))
```

**Issue: Out of Memory**

Solutions:
```python
# 1. Reduce batch size
# 2. Reduce sequence length
# 3. Use gradient checkpointing
# 4. Use gradient accumulation
```

### Debugging Tools

**Attention Inspection**:
```python
attn = state.get_attention_weights(layer_idx=0)
print(f"Min: {attn.min()}, Max: {attn.max()}, Mean: {attn.mean()}")
print(f"NaN: {torch.isnan(attn).any()}")
```

**Gradient Inspection**:
```python
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm()
        print(f"{name}: {grad_norm:.4f}")
```

**Activation Inspection**:
```python
from src.visualization import visualize_activation_patterns
fig = visualize_activation_patterns(state, layer_idx=0, save_path='activations.png')
```

---

## Testing

### Unit Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_model.py

# Run with coverage
python -m pytest --cov=src tests/
```

### Test Structure

```
tests/
├── test_components.py    # Test individual components
├── test_model.py         # Test full model
├── test_config.py        # Test configuration
└── test_visualization.py # Test visualization functions
```

### Writing Tests

```python
import pytest
import torch
from src.transformer import Transformer, TransformerConfig

def test_forward_pass():
    config = TransformerConfig(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
    model = Transformer(config)
    
    input_ids = torch.randint(0, 100, (2, 10))
    logits, state = model(input_ids, return_state=True)
    
    assert logits.shape == (2, 10, 100)
    assert state is not None
```

---

## Deployment

### Export to ONNX

```python
import torch.onnx

dummy_input = torch.randint(0, config.vocab_size, (1, 128))
torch.onnx.export(
    model,
    dummy_input,
    "transformer.onnx",
    export_params=True,
    opset_version=11,
    input_names=['input_ids'],
    output_names=['logits'],
    dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'}}
)
```

### Model Quantization

```python
import torch.quantization

# Post-training quantization
model.eval()
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# 4× smaller model, ~2× faster inference
```

### Serving

Basic Flask API:
```python
from flask import Flask, request, jsonify

app = Flask(__name__)
model = load_model('checkpoint.pt')

@app.route('/predict', methods=['POST'])
def predict():
    input_ids = torch.tensor(request.json['input_ids'])
    logits, _ = model(input_ids)
    predictions = logits.argmax(dim=-1).tolist()
    return jsonify({'predictions': predictions})
```

---

## Best Practices Summary

### Configuration
✓ Start with small model for prototyping  
✓ Ensure d_model divisible by n_heads  
✓ Use d_ff = 4 × d_model  
✓ Set max_seq_len > training sequences  

### Training
✓ Use AdamW optimizer with weight decay  
✓ Apply learning rate warmup  
✓ Clip gradients to norm 1.0  
✓ Monitor validation loss  
✓ Save checkpoints regularly  
✓ Use mixed precision for speed  

### Data
✓ Use subword tokenization  
✓ Pad to consistent lengths within batch  
✓ Create attention masks for padding  
✓ Shuffle training data  
✓ Validate data quality  

### Debugging
✓ Visualize attention patterns  
✓ Monitor gradient norms  
✓ Check for NaN values  
✓ Track attention statistics  
✓ Use provided visualization tools  

### Production
✓ Quantize for deployment  
✓ Benchmark inference speed  
✓ Monitor memory usage  
✓ Version models and configs  
✓ Document architecture choices  

---

## Additional Resources

### Documentation
- `doc/theory/TRANSFORMER_THEORY.md` - Theoretical foundations
- `doc/theory/MATHEMATICS.md` - Mathematical details
- `doc/theory/STATISTICS.md` - Statistical analysis
- `doc/FAQ.md` - Frequently asked questions

### Examples
- `examples/basic_usage.py` - Simple forward pass
- `examples/training_example.py` - Complete training loop
- `examples/animation_demo.py` - Visualization demos
- `examples/path_analysis.py` - Path counting analysis

### Scripts
- `run.py` - Complete setup and validation
- `scripts/generate_all_animations.py` - Generate all visualizations
- `scripts/verify_installation.py` - Verify dependencies

---

**Last Updated**: October 2025  
**For Support**: See `doc/FAQ.md` or open an issue on GitHub

