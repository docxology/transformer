# Practical Recipes: Transformer Implementation

## Purpose

Quick, actionable recipes for common tasks. Less theory, more "just tell me what to do."

---

## Recipe 1: Train a Language Model

**Goal**: Train a transformer to predict next tokens.

**Ingredients**:
- Text data
- GPU (recommended)
- 30 minutes

**Steps**:

```python
# 1. Prepare data
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, texts, vocab, seq_len=128):
        self.vocab = vocab
        self.seq_len = seq_len
        self.data = [self.tokenize(text) for text in texts]
    
    def tokenize(self, text):
        return [self.vocab.get(word, 0) for word in text.split()]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = self.data[idx][:self.seq_len]
        return torch.tensor(tokens), torch.tensor(tokens[1:] + [0])

# 2. Configure model
from src.transformer import Transformer, TransformerConfig

config = TransformerConfig(
    vocab_size=len(vocab),
    d_model=256,
    n_heads=4,
    n_layers=4,
    d_ff=1024,
    dropout=0.1
)
model = Transformer(config).cuda()

# 3. Set up training
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

# 4. Train
for epoch in range(10):
    for inputs, targets in dataloader:
        inputs, targets = inputs.cuda(), targets.cuda()
        
        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = criterion(logits.view(-1, config.vocab_size), targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 5. Save
torch.save(model.state_dict(), 'language_model.pt')
```

**Done!**

---

## Recipe 2: Fine-tune for Classification

**Goal**: Take pre-trained transformer, add classification head, fine-tune.

**Steps**:

```python
# 1. Load pre-trained model
checkpoint = torch.load('pretrained_model.pt')
config = TransformerConfig(**checkpoint['config'])
model = Transformer(config)
model.load_state_dict(checkpoint['model_state_dict'])

# 2. Add classification head
class TransformerClassifier(nn.Module):
    def __init__(self, transformer, num_classes):
        super().__init__()
        self.transformer = transformer
        self.classifier = nn.Linear(transformer.config.d_model, num_classes)
    
    def forward(self, input_ids):
        logits, _ = self.transformer(input_ids)
        # Use [CLS] token (first position) or mean pool
        cls_output = logits[:, 0, :]  # First token
        return self.classifier(cls_output)

classifier = TransformerClassifier(model, num_classes=10).cuda()

# 3. Freeze transformer, train head only (optional)
for param in classifier.transformer.parameters():
    param.requires_grad = False

# 4. Train classification head
optimizer = torch.optim.Adam(classifier.classifier.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    for inputs, labels in dataloader:
        inputs, labels = inputs.cuda(), labels.cuda()
        
        optimizer.zero_grad()
        outputs = classifier(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 5. Unfreeze and fine-tune full model (optional)
for param in classifier.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-5)
# Train for a few more epochs...
```

---

## Recipe 3: Generate Text

**Goal**: Use trained model to generate text autoregressively.

**Steps**:

```python
def generate_text(model, prompt_ids, max_length=50, temperature=1.0, top_k=50):
    """
    Generate text from prompt.
    
    Args:
        model: Trained transformer
        prompt_ids: Starting tokens (tensor)
        max_length: Maximum generation length
        temperature: Sampling temperature (higher = more random)
        top_k: Sample from top K tokens only
    """
    model.eval()
    generated = prompt_ids.tolist()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get model predictions
            input_tensor = torch.tensor([generated]).cuda()
            logits, _ = model(input_tensor)
            next_token_logits = logits[0, -1, :] / temperature
            
            # Top-K filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            # Add to sequence
            generated.append(next_token)
            
            # Stop on end token (if applicable)
            if next_token == eos_token_id:
                break
    
    return generated

# Usage
prompt = "The transformer architecture"
prompt_ids = tokenize(prompt)
generated_ids = generate_text(model, prompt_ids, temperature=0.8, top_k=40)
generated_text = detokenize(generated_ids)
print(generated_text)
```

---

## Recipe 4: Visualize What Model Learned

**Goal**: Create comprehensive visualizations of trained model.

**Steps**:

```python
from src.visualization import (
    visualize_attention,
    visualize_information_highways,
    visualize_head_specialization,
    animate_information_flow
)

# 1. Get model state
sample_input = torch.randint(0, config.vocab_size, (1, 64)).cuda()
model.eval()
with torch.no_grad():
    logits, state = model(sample_input, return_state=True)

# 2. Attention patterns per layer
for layer_idx in range(config.n_layers):
    attn = state.get_attention_weights(layer_idx)
    visualize_attention(
        attn,
        layer_idx=layer_idx,
        save_path=f'outputs/layer_{layer_idx}_attention.png'
    )

# 3. Information highways
visualize_information_highways(
    state,
    save_path='outputs/information_flow.png'
)

# 4. Head specialization
visualize_head_specialization(
    state,
    save_path='outputs/head_specialization.png'
)

# 5. Animation
animate_information_flow(
    state,
    save_path='outputs/flow_animation.gif',
    fps=2
)

print("Visualizations saved to outputs/")
```

---

## Recipe 5: Debug Training Issues

**Goal**: Systematically diagnose why model isn't learning.

**Checklist**:

```python
def diagnose_training(model, dataloader, optimizer):
    """Run diagnostic checks."""
    
    # 1. Check data
    print("=== Data Check ===")
    batch = next(iter(dataloader))
    inputs, targets = batch
    print(f"Input shape: {inputs.shape}")
    print(f"Input range: [{inputs.min()}, {inputs.max()}]")
    print(f"Target shape: {targets.shape}")
    print(f"NaN in data: {torch.isnan(inputs).any() or torch.isnan(targets).any()}")
    
    # 2. Check forward pass
    print("\n=== Forward Pass ===")
    model.eval()
    with torch.no_grad():
        logits, state = model(inputs.cuda(), return_state=True)
    print(f"Output shape: {logits.shape}")
    print(f"Output range: [{logits.min():.2f}, {logits.max():.2f}]")
    print(f"NaN in output: {torch.isnan(logits).any()}")
    
    # 3. Check loss
    print("\n=== Loss Check ===")
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits.view(-1, model.config.vocab_size), targets.cuda().view(-1))
    print(f"Loss: {loss.item():.4f}")
    print(f"Loss is NaN: {torch.isnan(loss).any()}")
    
    # 4. Check gradients
    print("\n=== Gradient Check ===")
    model.train()
    model.zero_grad()
    loss.backward()
    
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    
    print(f"Gradient norms (first few layers):")
    for name, norm in list(grad_norms.items())[:5]:
        print(f"  {name}: {norm:.4f}")
    
    max_grad = max(grad_norms.values())
    min_grad = min(grad_norms.values())
    print(f"Max gradient: {max_grad:.4f}")
    print(f"Min gradient: {min_grad:.4f}")
    
    if max_grad > 10:
        print("⚠️  WARNING: Exploding gradients! Use gradient clipping.")
    if max_grad < 1e-5:
        print("⚠️  WARNING: Vanishing gradients! Check model or learning rate.")
    
    # 5. Check attention
    print("\n=== Attention Check ===")
    attn = state.get_attention_weights(0)
    print(f"Attention shape: {attn.shape}")
    print(f"Attention range: [{attn.min():.4f}, {attn.max():.4f}]")
    print(f"Row sums (should be 1.0): {attn[0, 0].sum(dim=-1)[:3]}")
    entropy = -(attn * torch.log(attn + 1e-10)).sum(dim=-1).mean()
    print(f"Attention entropy: {entropy:.3f} (typical: 1.8-2.2)")
    
    # 6. Learning rate
    print("\n=== Optimizer Check ===")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    
    print("\n=== Diagnosis Complete ===")

# Run diagnostics
diagnose_training(model, dataloader, optimizer)
```

---

## Recipe 6: Speed Up Training

**Goal**: Make training 2-3× faster with minimal code changes.

**Steps**:

```python
# 1. Enable mixed precision
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# 2. Modify training loop
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        inputs, targets = inputs.cuda(), targets.cuda()
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast():
            logits, _ = model(inputs)
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
        
        # Scaled backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping (unscale first!)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

# 3. Use larger batch size
dataloader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)

# 4. Gradient accumulation (if memory-limited)
accumulation_steps = 4
for i, (inputs, targets) in enumerate(dataloader):
    with autocast():
        logits, _ = model(inputs.cuda())
        loss = criterion(logits.view(-1, vocab_size), targets.cuda().view(-1))
        loss = loss / accumulation_steps
    
    scaler.scale(loss).backward()
    
    if (i + 1) % accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

**Expected**: 2-3× speedup with same accuracy.

---

## Recipe 7: Handle Long Documents

**Goal**: Process documents longer than max sequence length.

**Strategy**: Sliding window approach.

**Steps**:

```python
def process_long_document(model, doc_tokens, window_size=512, stride=256):
    """
    Process long document with sliding window.
    
    Args:
        model: Transformer model
        doc_tokens: Full document tokens (length > window_size)
        window_size: Size of each window
        stride: Step size (< window_size for overlap)
    
    Returns:
        Aggregated representations for full document
    """
    model.eval()
    all_representations = []
    
    with torch.no_grad():
        for start in range(0, len(doc_tokens), stride):
            end = min(start + window_size, len(doc_tokens))
            window = doc_tokens[start:end]
            
            # Pad if needed
            if len(window) < window_size:
                window = window + [0] * (window_size - len(window))
            
            # Process window
            input_tensor = torch.tensor([window]).cuda()
            logits, state = model(input_tensor, return_state=True)
            
            # Extract representations (e.g., mean of non-padding)
            valid_len = min(end - start, window_size)
            repr = logits[0, :valid_len, :].mean(dim=0)
            all_representations.append(repr)
    
    # Aggregate (e.g., mean pool)
    doc_representation = torch.stack(all_representations).mean(dim=0)
    return doc_representation

# Usage
long_doc_tokens = tokenize(long_document)  # e.g., 2000 tokens
doc_repr = process_long_document(model, long_doc_tokens)
```

---

## Recipe 8: Reduce Model Size

**Goal**: Make model 4× smaller for deployment.

**Method**: Quantization.

**Steps**:

```python
import torch.quantization

# 1. Prepare model
model = model.cpu()  # Quantization works on CPU
model.eval()

# 2. Dynamic quantization (easiest)
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # Quantize linear layers
    dtype=torch.qint8    # 8-bit integers
)

# 3. Save
torch.save(quantized_model.state_dict(), 'model_quantized.pt')

# 4. Test
sample_input = torch.randint(0, 1000, (1, 128))
with torch.no_grad():
    original_output, _ = model(sample_input)
    quantized_output, _ = quantized_model(sample_input)

# 5. Compare
diff = (original_output - quantized_output).abs().mean()
print(f"Average difference: {diff:.4f}")  # Should be small

# File size comparison
import os
print(f"Original: {os.path.getsize('model.pt') / 1e6:.1f} MB")
print(f"Quantized: {os.path.getsize('model_quantized.pt') / 1e6:.1f} MB")
```

**Typical results**:
- Size: 4× smaller
- Speed: 2× faster inference
- Accuracy: < 1% loss

---

## Recipe 9: Analyze Information Flow

**Goal**: Understand how information moves through your trained model.

**Steps**:

```python
from src.visualization import (
    visualize_residual_stream,
    visualize_kv_stream,
    compute_path_counts,
    trace_information_paths
)

# 1. Get state from trained model
sample_input = torch.tensor([[10, 45, 23, 67, 12, 89, 34]]).cuda()
model.eval()
with torch.no_grad():
    logits, state = model(sample_input, return_state=True)

# 2. Analyze residual stream (vertical flow)
print("=== Residual Stream ===")
residuals = state.get_residual_stream()
for i, res in enumerate(residuals):
    norm = res[0, 0, :].norm().item()
    print(f"Layer {i}: Norm = {norm:.4f}")

visualize_residual_stream(
    state,
    position_idx=3,  # Focus on token at position 3
    save_path='residual_analysis.png'
)

# 3. Analyze K/V stream (horizontal flow)
print("\n=== K/V Stream ===")
for layer_idx in range(model.config.n_layers):
    attn = state.get_attention_weights(layer_idx)
    sparsity = (attn[0].mean(0) > 0.1).float().mean().item()
    entropy = -(attn[0].mean(0) * torch.log(attn[0].mean(0) + 1e-10)).sum(-1).mean().item()
    print(f"Layer {layer_idx}: Sparsity={sparsity:.3f}, Entropy={entropy:.3f}")

visualize_kv_stream(
    state,
    layer_idx=0,
    save_path='kv_analysis.png'
)

# 4. Path analysis
print("\n=== Path Analysis ===")
n_layers = model.config.n_layers
seq_len = sample_input.shape[1]

# Count paths
total_paths = compute_path_counts(n_layers - 1, seq_len - 1)
print(f"Total paths through network: {total_paths:,}")

# Trace specific paths
paths = trace_information_paths(0, 0, 2, 3, max_paths=10)
print(f"Sample paths from (0,0) to (2,3):")
for i, path in enumerate(paths[:5]):
    print(f"  Path {i+1}: {path}")
```

---

## Recipe 10: Create Publication-Quality Figures

**Goal**: Generate polished visualizations for papers/presentations.

**Steps**:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)

# 1. High-resolution attention heatmap
from src.visualization import visualize_attention

attn = state.get_attention_weights(layer_idx=0)
fig = visualize_attention(
    attn,
    layer_idx=0,
    tokens=['The', 'quick', 'brown', 'fox', 'jumps'],
    figsize=(10, 8),
    save_path='attention_hq.pdf'  # Vector format!
)

# 2. Custom styling
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Training curve
axes[0, 0].plot(train_losses, label='Train', linewidth=2)
axes[0, 0].plot(val_losses, label='Validation', linewidth=2)
axes[0, 0].set_xlabel('Epoch', fontsize=14)
axes[0, 0].set_ylabel('Loss', fontsize=14)
axes[0, 0].set_title('Training Progress', fontsize=16, fontweight='bold')
axes[0, 0].legend(fontsize=12)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Attention entropy by layer
entropies = [compute_entropy(state.get_attention_weights(i)) for i in range(n_layers)]
axes[0, 1].bar(range(n_layers), entropies, color='steelblue')
axes[0, 1].set_xlabel('Layer', fontsize=14)
axes[0, 1].set_ylabel('Attention Entropy', fontsize=14)
axes[0, 1].set_title('Attention Concentration', fontsize=16, fontweight='bold')

# Plot 3: Head specialization
# ... (similar styling)

# Plot 4: Information flow
# ... (similar styling)

plt.tight_layout()
plt.savefig('comprehensive_analysis.pdf', dpi=300, bbox_inches='tight')
plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
```

**Output**: Publication-ready figures in both PDF (vector) and PNG (raster) formats.

---

## Recipe 11: Profile and Optimize

**Goal**: Find and fix performance bottlenecks.

**Steps**:

```python
import torch.profiler

# 1. Profile forward pass
def profile_model(model, input_ids, num_steps=100):
    model.eval()
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            for _ in range(num_steps):
                logits, _ = model(input_ids)
    
    # Print results
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # Export for visualization
    prof.export_chrome_trace("profile_trace.json")
    # Open in chrome://tracing

# 2. Memory profiling
def profile_memory(model, input_ids):
    import torch.cuda
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        logits, state = model(input_ids, return_state=True)
    
    peak_mem = torch.cuda.max_memory_allocated() / 1e9  # GB
    print(f"Peak memory usage: {peak_mem:.2f} GB")
    
    # Breakdown
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    activations = peak_mem - model_size
    print(f"  Model: {model_size:.2f} GB")
    print(f"  Activations: {activations:.2f} GB")

# Run profiling
sample_input = torch.randint(0, 1000, (8, 512)).cuda()
profile_model(model, sample_input)
profile_memory(model, sample_input)
```

---

## Recipe 12: Export for Production

**Goal**: Export model for deployment (ONNX, TorchScript).

**Option A: ONNX** (cross-platform):

```python
# 1. Create dummy input
dummy_input = torch.randint(0, config.vocab_size, (1, 128))

# 2. Export
torch.onnx.export(
    model,
    dummy_input,
    "transformer.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input_ids'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence'},
        'logits': {0: 'batch_size', 1: 'sequence'}
    }
)

# 3. Verify
import onnx
onnx_model = onnx.load("transformer.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model valid!")
```

**Option B: TorchScript** (PyTorch ecosystem):

```python
# 1. Trace model
model.eval()
example_input = torch.randint(0, config.vocab_size, (1, 128))
traced_model = torch.jit.trace(model, example_input)

# 2. Save
traced_model.save("transformer_traced.pt")

# 3. Load and use
loaded_model = torch.jit.load("transformer_traced.pt")
output = loaded_model(example_input)
```

---

## Quick Reference

### Most Common Commands

```bash
# Setup
python run.py

# Run examples
python examples/basic_usage.py
python examples/training_example.py
python examples/animation_demo.py

# Tests
python -m pytest tests/

# Generate all visualizations
python scripts/generate_all_animations.py
```

### Most Common Issues

| Problem | Solution |
|---------|----------|
| NaN loss | Reduce LR, add gradient clipping |
| Out of memory | Reduce batch size or sequence length |
| Slow training | Enable mixed precision, increase batch size |
| No learning | Check data, LR, gradient flow |
| Poor generation | Adjust temperature, use top-k/nucleus |

### Most Useful Visualizations

```python
# Attention patterns
visualize_attention(state.get_attention_weights(0))

# Information flow
visualize_information_highways(state)

# Training progress (during training)
visualize_attention(state.get_attention_weights(0), save_path=f'epoch_{epoch}.png')

# Head behavior
visualize_head_specialization(state)
```

---

## Need More Help?

- **Full examples**: `examples/` directory
- **Detailed guides**: `doc/guides/IMPLEMENTATION_GUIDE.md`
- **Theory**: `doc/theory/TRANSFORMER_THEORY.md`
- **FAQ**: `doc/FAQ_COMPREHENSIVE.md`
- **Glossary**: `doc/GLOSSARY.md`

---

**Last Updated**: October 2025  
**Philosophy**: Show, don't tell. Code first, explanation second.

