# Comprehensive FAQ: Transformer Architecture

## Table of Contents

1. [Conceptual Questions](#conceptual-questions)
2. [Mathematical Questions](#mathematical-questions)
3. [Implementation Questions](#implementation-questions)
4. [Training Questions](#training-questions)
5. [Visualization Questions](#visualization-questions)
6. [Performance Questions](#performance-questions)
7. [Debugging Questions](#debugging-questions)
8. [Architecture Questions](#architecture-questions)

---

## Conceptual Questions

### Q1: What makes transformers different from RNNs and CNNs?

**A**: Three key differences:

1. **Parallelization**: Unlike RNNs, transformers process all positions simultaneously
2. **Long-range dependencies**: Direct connections between all positions (no recurrence bottleneck)
3. **Minimal inductive bias**: No built-in assumptions about locality (CNNs) or sequence (RNNs)

**Trade-off**: More flexible but requires more data and has quadratic complexity in sequence length.

**See**: `doc/theory/TRANSFORMER_THEORY.md` (§Comparison to Other Architectures)

---

### Q2: Why are they called "transformers"?

**A**: The name emphasizes the architecture's ability to **transform** input sequences through multiple layers of attention and processing, without relying on recurrence or convolution. Each layer transforms representations to progressively more abstract and task-relevant forms.

---

### Q3: What are the "two information highways"?

**A**: 

1. **Vertical (Residual Stream)**: Information flows through layers via residual connections
   - Formula: `x_{l+1} = x_l + F(x_l)`
   - Enables layer-by-layer refinement

2. **Horizontal (K/V Stream)**: Information flows across positions via attention
   - Mechanism: Queries attend to Keys/Values at other positions
   - Enables token-to-token communication

This dual-highway architecture is key to transformers' expressiveness.

**Visualization**: `outputs/visualizations/information_highways.png`

---

### Q4: How does attention actually work?

**A**: Think of it like a weighted average:

1. **Query asks**: "What information do I need?"
2. **Keys answer**: "Here's what I can provide"
3. **Similarity scores**: How well each Key matches the Query
4. **Attention weights**: Softmax of similarity scores (sum to 1)
5. **Output**: Weighted sum of Values using attention weights

**Analogy**: Like asking classmates for help on homework - you ask a question (Query), everyone responds with what they know (Keys), you decide who to listen to (attention weights), and gather their knowledge (Values).

**See**: `doc/concepts/attention.md`

---

### Q5: Why do we need multiple attention heads?

**A**: Different heads learn different relationships:

- **Head 1**: Local syntax (nearby words)
- **Head 2**: Long-range dependencies (subject-verb agreement)
- **Head 3**: Semantic relationships (synonyms, related concepts)
- **Head 4**: Positional patterns (beginning/end of sentence)

Multiple heads = multiple perspectives = richer representations.

**Analysis**: `outputs/visualizations/head_specialization.png`

---

### Q6: What's the purpose of positional encoding?

**A**: Attention is **permutation invariant** - it doesn't inherently know token order. Positional encoding adds position information so the model knows "cat sat mat" ≠ "mat sat cat".

**Why sinusoidal?**: 
- Smooth, continuous
- Can extrapolate to longer sequences
- Different frequencies for different dimensions
- Allows learning relative positions

**Implementation**: `src/transformer/components/positional_encoding.py`

---

### Q7: How many computational paths are there through a transformer?

**A**: **Astronomically many**. For L layers and N positions:

```
Paths = C(L+N-2, L-1) = (L+N-2)! / ((L-1)! × (N-1)!)
```

**Examples**:
- 6 layers, 32 positions: **376,992 paths**
- 12 layers, 512 positions: **More than atoms in the universe**

This massive redundancy enables robust, nuanced information processing.

**Analysis**: `examples/path_analysis.py`

---

### Q8: What's the residual connection doing?

**A**: Two purposes:

1. **Gradient flow**: Allows gradients to flow directly backward
   - `∂loss/∂x = ∂loss/∂output × (1 + ∂F/∂x)`
   - The "1" ensures gradients don't vanish

2. **Identity preservation**: Model can learn identity easily (set F(x) = 0)
   - Each layer adds small refinement to existing representation

**Result**: Enables training very deep networks (100+ layers).

---

## Mathematical Questions

### Q9: Why scale attention scores by √d_k?

**A**: **Variance stabilization**.

For random Q, K with variance 1:
```
Var(Q·K) = Σ Var(q_i × k_i) = d_k
```

Large variance → extreme values → softmax saturation → tiny gradients.

**Solution**: Divide by √d_k:
```
Var(Q·K / √d_k) = d_k / d_k = 1
```

Now variance stays ≈ 1 regardless of dimension.

**See**: `doc/theory/MATHEMATICS.md` (§1.3: Why Scaling)

---

### Q10: How is attention computed exactly?

**A**: Step by step:

```python
# 1. Compute similarity scores
scores = Q @ K.T                    # (n, d_k) @ (d_k, m) → (n, m)

# 2. Scale
scores = scores / math.sqrt(d_k)

# 3. Apply softmax (row-wise normalization)
attn_weights = softmax(scores)      # Sum to 1 across keys

# 4. Weighted sum of values
output = attn_weights @ V           # (n, m) @ (m, d_v) → (n, d_v)
```

**See**: `src/transformer/components/attention.py`

---

### Q11: What are the dimensions of each component?

**A**: For batch_size=B, sequence_length=N:

| Component | Shape | Notation |
|-----------|-------|----------|
| Input IDs | (B, N) | Token indices |
| Embeddings | (B, N, d_model) | Dense vectors |
| Q, K, V (per head) | (B, N, d_k) | d_k = d_model/n_heads |
| Attention weights | (B, n_heads, N, N) | Probability matrix |
| Attention output | (B, N, d_model) | After concatenation |
| FFN hidden | (B, N, d_ff) | d_ff = 4×d_model typically |
| FFN output | (B, N, d_model) | Same as input |
| Final logits | (B, N, vocab_size) | Prediction scores |

**See**: `doc/theory/MATHEMATICS.md` (§2: Multi-Head Attention)

---

### Q12: How many parameters does a transformer have?

**A**: Per layer:

```
Multi-Head Attention: 4 × d_model²
Feed-Forward: 2 × d_model × d_ff = 8 × d_model²  (d_ff = 4×d_model)
Layer Norms: 4 × d_model  (negligible)

Total per layer ≈ 12 × d_model²
```

Full model:
```
Transformer = L × 12 × d_model² + 2 × vocab_size × d_model
```

**Example** (d_model=512, L=6, vocab=30000):
```
≈ 6 × 12 × 512² + 2 × 30000 × 512
≈ 18.9M + 30.7M
≈ 50M parameters
```

**See**: `doc/theory/MATHEMATICS.md` (§9: Parameter Count Analysis)

---

### Q13: What's the computational complexity?

**A**: Per layer:

**Attention**: O(n² × d_model)
- Computing scores: n × d_model × n
- Applying to values: n × n × d_model

**Feed-Forward**: O(n × d_model²)
- First layer: n × d_model × d_ff
- Second layer: n × d_ff × d_model

**Bottleneck**: For long sequences (large n), attention dominates.

**See**: `doc/theory/MATHEMATICS.md` (§10: Computational Complexity)

---

### Q14: What's layer normalization doing mathematically?

**A**: For input x ∈ ℝ^d:

```
μ = mean(x) = (1/d) Σᵢ xᵢ
σ = std(x) = √((1/d) Σᵢ (xᵢ - μ)²)

LayerNorm(x) = γ ⊙ (x - μ)/σ + β
```

Where γ, β are learned parameters.

**Effect**: Normalizes activations to mean=0, std=1, then applies learned transformation.

**Why**: Stabilizes training, prevents activation explosion/vanishing.

**See**: `doc/theory/MATHEMATICS.md` (§6: Layer Normalization)

---

## Implementation Questions

### Q15: How do I create a basic transformer?

**A**:

```python
from src.transformer import Transformer, TransformerConfig

# Configure
config = TransformerConfig(
    vocab_size=10000,
    d_model=512,
    n_heads=8,
    n_layers=6,
    d_ff=2048
)

# Create
model = Transformer(config)

# Use
import torch
input_ids = torch.randint(0, 10000, (2, 100))  # (batch, seq_len)
logits, state = model(input_ids, return_state=True)
```

**Complete example**: `examples/basic_usage.py`

---

### Q16: How do I enable state tracking for visualization?

**A**: Set `return_state=True`:

```python
logits, state = model(input_ids, return_state=True)

# Now you can access:
residuals = state.get_residual_stream()          # Vertical flow
attention = state.get_attention_weights(layer)   # Horizontal flow

# And visualize:
from src.visualization import visualize_attention
fig = visualize_attention(attention, layer_idx=0)
```

**See**: `doc/guides/visualization.md`

---

### Q17: What's the difference between pre-norm and post-norm?

**A**:

**Post-Norm** (original paper):
```python
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FFN(x))
```

**Pre-Norm** (modern, recommended):
```python
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

**Why pre-norm?**
- Better gradient flow
- More stable training
- Can train deeper models
- Faster convergence

**This implementation uses**: Pre-norm

---

### Q18: How do I add causal masking?

**A**: Already implemented in attention:

```python
# In src/transformer/components/attention.py
def get_causal_mask(seq_len, device):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

# Applied automatically for autoregressive models
scores = scores + mask  # Before softmax
```

**Effect**: Prevents position i from attending to positions > i.

---

### Q19: How do I save and load models?

**A**:

```python
# Save
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config.__dict__,
    'epoch': epoch,
    'optimizer_state_dict': optimizer.state_dict()
}, 'checkpoint.pt')

# Load
checkpoint = torch.load('checkpoint.pt')
config = TransformerConfig(**checkpoint['config'])
model = Transformer(config)
model.load_state_dict(checkpoint['model_state_dict'])
```

**See**: `doc/guides/IMPLEMENTATION_GUIDE.md` (§Model Checkpointing)

---

## Training Questions

### Q20: What optimizer should I use?

**A**: **AdamW** (Adam with decoupled weight decay)

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,              # Start conservatively
    betas=(0.9, 0.999),   # Standard
    weight_decay=0.01     # Regularization
)
```

**Why AdamW?**
- Adaptive learning rates per parameter
- Weight decay regularization
- Proven effective for transformers
- Handles sparse gradients well

**See**: `examples/training_example.py`

---

### Q21: What learning rate should I use?

**A**: Depends on model size, but general guidelines:

| Model Size | Initial LR | Strategy |
|------------|-----------|----------|
| Small (< 50M) | 1e-3 | With warmup |
| Medium (50-500M) | 1e-4 | With warmup |
| Large (> 500M) | 1e-5 | With warmup |

**Critical**: Use learning rate warmup (1000-4000 steps)

```python
# Typical schedule
warmup_steps = 4000
current_lr = lr * min(step**(-0.5), step * warmup_steps**(-1.5))
```

**See**: `doc/theory/MATHEMATICS.md` (§15: Optimization Landscape)

---

### Q22: Why is my loss NaN?

**A**: Common causes and fixes:

**1. Learning rate too high**
```python
# Solution: Reduce LR
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
```

**2. Exploding gradients**
```python
# Solution: Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**3. Numerical overflow in softmax**
```python
# Solution: Already handled by scaling, but check inputs
assert not torch.isnan(input_ids).any()
```

**See**: `doc/guides/IMPLEMENTATION_GUIDE.md` (§Debugging)

---

### Q23: How long should I train?

**A**: Depends on:
- Dataset size
- Model capacity
- Task complexity

**General guidelines**:

| Dataset Size | Epochs | Notes |
|--------------|--------|-------|
| Small (< 1M) | 50-100 | Watch for overfitting |
| Medium (1-10M) | 10-30 | Standard |
| Large (> 10M) | 3-10 | Diminishing returns |

**Best practice**: Monitor validation loss, stop when it plateaus or increases.

---

### Q24: What batch size should I use?

**A**: As large as fits in memory (with trade-offs):

**Memory-constrained**:
```python
batch_size = 8-16
# Use gradient accumulation to simulate larger batches
```

**Balanced**:
```python
batch_size = 32-64
# Good stability and speed
```

**Large GPU memory**:
```python
batch_size = 128-256
# Faster training, more stable gradients
```

**Rule**: If you double batch size, multiply learning rate by √2.

**See**: `doc/theory/STATISTICS.md` (§12: Batch Size Effects)

---

### Q25: Should I use dropout?

**A**: **Yes, for regularization**:

```python
config = TransformerConfig(
    ...,
    dropout=0.1  # Standard value
)
```

**Guidelines**:
- Small dataset: 0.1-0.2
- Large dataset: 0.1 or even 0.05
- Overfitting: Increase to 0.2-0.3
- Underfitting: Decrease or remove

**Note**: Dropout is automatically disabled during `model.eval()`.

---

## Visualization Questions

### Q26: How do I visualize attention patterns?

**A**:

```python
from src.visualization import visualize_attention

# Get attention weights
logits, state = model(input_ids, return_state=True)
attn = state.get_attention_weights(layer_idx=0)

# Visualize
fig = visualize_attention(
    attn,
    layer_idx=0,
    tokens=['the', 'cat', 'sat'],  # Optional labels
    save_path='attention.png'
)
```

**Output**: Heatmap showing which positions attend to which.

**See**: `outputs/visualizations/attention_layer_0.png`

---

### Q27: How do I create animations?

**A**:

```python
from src.visualization import animate_information_flow

# Create animation
anim = animate_information_flow(
    state,
    save_path='flow.gif',
    fps=2,  # Frames per second
    position_focus=5  # Optional: focus on specific position
)
```

**Types of animations**:
- `animate_information_flow()` - Overall flow through layers
- `animate_attention_evolution()` - How attention changes
- `animate_path_trace()` - Specific information path

**Example**: `examples/animation_demo.py`

---

### Q28: What do the information highway visualizations show?

**A**:

**Residual Stream** (vertical):
- How representation at one position evolves through layers
- Shows norm, magnitude changes layer by layer

**K/V Stream** (horizontal):
- How information flows between positions via attention
- Shows Query, Key, Value contributions

**Combined Highways**:
- Both flows simultaneously
- Illustrates dual-highway architecture

**Generate**:
```bash
python examples/basic_usage.py
# Creates: outputs/visualizations/information_highways.png
```

---

### Q29: How do I analyze attention head specialization?

**A**:

```python
from src.visualization import visualize_head_specialization

fig = visualize_head_specialization(
    state,
    save_path='head_spec.png'
)
```

**Shows**:
- Locality (local vs. global)
- Entropy (focused vs. diffuse)
- Previous token attention
- Positional patterns

**See**: `outputs/visualizations/head_specialization.png`

---

### Q30: Can I visualize training progress?

**A**: **Yes**, save visualizations during training:

```python
# During training loop
if epoch % 3 == 0:
    model.eval()
    with torch.no_grad():
        logits, state = model(sample_input, return_state=True)
    
    # Visualize attention
    attn = state.get_attention_weights(0)
    visualize_attention(attn, save_path=f'epoch_{epoch}_attn.png')
    
    # Visualize statistics
    plot_layer_statistics(state, save_path=f'epoch_{epoch}_stats.png')
    
    model.train()
```

**Example**: `examples/training_example.py` 

**Output**: `outputs/visualizations/training/`

---

## Performance Questions

### Q31: How can I speed up training?

**A**: Multiple strategies:

**1. Mixed Precision**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    logits, _ = model(inputs)
    loss = criterion(...)
scaler.scale(loss).backward()
```
**Speedup**: 2-3× faster

**2. Larger Batch Size**: Better GPU utilization

**3. Gradient Accumulation**: Simulate larger batches

**4. Efficient Data Loading**: Use `num_workers > 0` in DataLoader

**5. Profile and Optimize**: Use PyTorch profiler

**See**: `doc/guides/IMPLEMENTATION_GUIDE.md` (§Performance Optimization)

---

### Q32: How can I reduce memory usage?

**A**:

**1. Gradient Checkpointing**:
```python
# Trade computation for memory
model.enable_gradient_checkpointing()
```
**Savings**: ~30-50% memory, ~20% slower

**2. Smaller Batch Size**: Most direct

**3. Gradient Accumulation**: Maintain effective batch size

**4. Mixed Precision**: Smaller activations

**5. Shorter Sequences**: Quadratic savings

---

### Q33: Can transformers handle long sequences?

**A**: **Standard transformers struggle** due to O(n²) attention complexity.

**For n = 10,000**:
- Attention weights: 10,000 × 10,000 = 100M values per head
- Memory: ~400 MB per head just for weights
- Computation: Prohibitive

**Solutions**:
1. **Sparse Attention**: Attend to subset of positions
2. **Local Windows**: Only attend to nearby tokens
3. **Hierarchical**: Process in chunks, then combine
4. **Linear Attention**: Approximate with O(n) complexity
5. **State Space Models**: Different architecture (e.g., Mamba)

**This implementation**: Standard O(n²) attention, best for n < 2048

---

### Q34: Should I use GPU or CPU?

**A**: **GPU strongly recommended** for any non-trivial model.

**Speed comparison** (typical):
- CPU: 1× (baseline)
- Single GPU: 10-50× faster
- Multi-GPU: 50-400× faster

**Minimum GPU memory guidelines**:
- Small model (50M params): 4 GB
- Medium model (150M params): 8 GB
- Large model (500M params): 16 GB+

**Code**:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

---

## Debugging Questions

### Q35: Model isn't learning, what should I check?

**A**: Systematic checklist:

**1. Data**:
```python
# Sanity check
print(input_ids.min(), input_ids.max())  # In vocab range?
print(targets.min(), targets.max())      # Valid labels?
assert not torch.isnan(input_ids).any()
```

**2. Learning rate**:
```python
# Try different LRs: 1e-3, 1e-4, 1e-5
# Monitor: Loss should decrease after few batches
```

**3. Gradient flow**:
```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm().item():.4f}")
# Should see non-zero, reasonable values
```

**4. Model capacity**:
```python
# Too small? Increase d_model, n_layers
# Too large? May need more data
```

**5. Attention patterns**:
```python
# Visualize - are patterns reasonable?
visualize_attention(state.get_attention_weights(0))
```

---

### Q36: How do I debug attention patterns?

**A**:

```python
# 1. Get attention weights
attn = state.get_attention_weights(layer_idx=0)  # (batch, heads, seq, seq)

# 2. Check statistics
print(f"Min: {attn.min()}, Max: {attn.max()}, Mean: {attn.mean()}")
print(f"NaN: {torch.isnan(attn).any()}")

# 3. Check normalization (should sum to 1)
print(f"Row sums: {attn[0, 0].sum(dim=-1)}")  # Should be all 1.0

# 4. Compute entropy (concentration measure)
entropy = -(attn * torch.log(attn + 1e-10)).sum(dim=-1).mean()
print(f"Entropy: {entropy:.3f}")  # ~2.0 is typical

# 5. Visualize
from src.visualization import visualize_attention
visualize_attention(attn, save_path='debug_attn.png')
```

---

### Q37: What if gradients are exploding?

**A**:

**Symptoms**:
- Loss becomes NaN
- Very large gradient norms (> 10)
- Model parameters become NaN

**Solutions**:

```python
# 1. Gradient clipping (most important)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 2. Reduce learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# 3. Check initialization
# (Should be handled automatically, but verify)

# 4. Check data for extreme values
print(f"Input range: {input_ids.float().mean()}, {input_ids.float().std()}")
```

---

### Q38: How do I verify my implementation is correct?

**A**:

**1. Run provided tests**:
```bash
python -m pytest tests/ -v
```

**2. Run verification script**:
```bash
python scripts/verify_installation.py
```

**3. Check examples work**:
```bash
python examples/basic_usage.py
python examples/training_example.py
```

**4. Compare attention to known patterns**:
- Untrained: Nearly uniform
- Trained: Clear patterns emerge

**5. Verify mathematical properties**:
```python
# Attention weights sum to 1
attn = state.get_attention_weights(0)
assert torch.allclose(attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)))

# Layer norm mean~0, std~1
x_normed = layer_norm(x)
assert x_normed.mean().abs() < 0.01
assert (x_normed.std() - 1.0).abs() < 0.1
```

---

## Architecture Questions

### Q39: What's the difference between encoder and decoder transformers?

**A**:

**Encoder** (BERT-style):
- Bidirectional attention (can see all positions)
- No causal mask
- For understanding tasks (classification, QA)

**Decoder** (GPT-style):
- Causal attention (can only see previous positions)
- Causal mask applied
- For generation tasks (language modeling, text generation)

**Encoder-Decoder** (T5-style):
- Encoder: Bidirectional on source
- Decoder: Causal on target, cross-attention to encoder
- For sequence-to-sequence (translation, summarization)

**This implementation**: Decoder-only (most general, can handle all tasks)

---

### Q40: How do I choose model size?

**A**: Consider:

**Dataset Size**:
- Small (< 1M examples): Small model (50M params)
- Medium (1-10M): Medium model (150M params)
- Large (> 10M): Large model (500M+ params)

**Task Complexity**:
- Simple (classification): Smaller sufficient
- Complex (generation): Larger better

**Compute Budget**:
- Limited: Small model, longer training
- Ample: Large model, faster convergence

**Rule of thumb**: Start small, scale up if underfitting.

---

### Q41: What's d_ff and why 4× d_model?

**A**: `d_ff` is the hidden dimension of the feed-forward network.

**Why 4×?**: Empirical finding from original paper.

**Purpose of expansion**:
- Increases model capacity
- Allows complex non-linear transformations
- Most parameters in transformer (2/3 of total)

**Can you change it?**: Yes, but 4× is well-tested default.

---

### Q42: How many layers should I use?

**A**:

| Model Type | Layers | Use Case |
|------------|--------|----------|
| Small | 4-6 | Experimentation, small data |
| Medium | 6-12 | Standard tasks |
| Large | 12-24 | Complex tasks, large data |
| Very Large | 24-96 | State-of-art models (GPT-3: 96) |

**Trade-off**: More layers = more capacity but slower, harder to train.

**This repo**: Defaults to 6 (good balance)

---

### Q43: What's the maximum sequence length?

**A**: Limited by two factors:

**1. Positional Encodings**: 
```python
config = TransformerConfig(max_seq_len=5000)  # Default
```
Can process sequences up to this length.

**2. Memory/Computation**:
- O(n²) complexity means longer = much more expensive
- Practical limit: 512-2048 for standard models

**Recommendation**: Set `max_seq_len` > longest training sequence, but use actual lengths during training.

---

### Q44: Can I modify the architecture?

**A**: **Yes!** This is a clean, modular implementation.

**Examples**:

**Add relative position encoding**:
```python
# Modify src/transformer/components/positional_encoding.py
```

**Change activation function**:
```python
# In src/transformer/components/feed_forward.py
self.activation = nn.GELU()  # Instead of ReLU
```

**Add cross-attention** (for encoder-decoder):
```python
# Add to TransformerBlock
self.cross_attention = MultiHeadAttention(config)
```

**Sparse attention**:
```python
# Modify attention computation in attention.py
```

---

### Q45: What's the state object and why use it?

**A**: `TransformerState` stores intermediate computations:

```python
logits, state = model(input_ids, return_state=True)

# Contains:
- Residual stream (all layer outputs)
- Attention weights (all layers, all heads)
- K, V matrices
```

**Why?**
- **Visualization**: See what model is doing internally
- **Analysis**: Understand attention patterns, information flow
- **Debugging**: Verify computations, diagnose issues
- **Interpretability**: Gain insights into model behavior

**Cost**: Minimal (just storage of values already computed)

---

## Next Steps

### Still have questions?

1. **Read theory**: `doc/theory/TRANSFORMER_THEORY.md`
2. **Check glossary**: `doc/GLOSSARY.md`
3. **Run examples**: `examples/` directory
4. **Explore code**: `src/transformer/` - well-commented
5. **Open issue**: GitHub repository

### Want to learn more?

- **Mathematics**: `doc/theory/MATHEMATICS.md`
- **Statistics**: `doc/theory/STATISTICS.md`
- **Implementation**: `doc/guides/IMPLEMENTATION_GUIDE.md`
- **Visualizations**: `doc/guides/visualization.md`

### Ready to build?

```bash
python run.py  # Complete setup and examples
python examples/basic_usage.py  # Start simple
python examples/training_example.py  # Build your own
```

---

**Last Updated**: October 2025  
**Version**: 1.0  
**Contributors**: Welcome! Open issues/PRs on GitHub

