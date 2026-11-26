# Mathematical Foundations of Transformers

## Notation

Throughout this document:
- Matrices: **Bold uppercase** (e.g., **W**, **X**)
- Vectors: **Bold lowercase** (e.g., **x**, **h**)
- Scalars: *Italic* (e.g., *d*, *n*)
- Sets: Calligraphic (e.g., 𝒱 for vocabulary)
- Sequences: Subscripted (e.g., x₁, x₂, ..., xₙ)

---

## 1. Scaled Dot-Product Attention

### Definition

Given queries **Q** ∈ ℝ^(n×d_k), keys **K** ∈ ℝ^(m×d_k), and values **V** ∈ ℝ^(m×d_v):

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

### Step-by-Step

1. **Compute similarity scores**: 
   ```
   S = QK^T ∈ ℝ^(n×m)
   S[i,j] = q_i · k_j = Σ_k q_i[k] * k_j[k]
   ```

2. **Scale**:
   ```
   S_scaled = S / √d_k
   ```

3. **Apply softmax** (row-wise):
   ```
   A[i,j] = exp(S_scaled[i,j]) / Σ_j' exp(S_scaled[i,j'])
   ```

4. **Weighted sum of values**:
   ```
   output[i] = Σ_j A[i,j] * v_j
   ```

### Why Scaling by √d_k?

**Problem**: For large d_k, dot products have variance ≈ d_k

**Proof**: If q, k are i.i.d. with mean 0, variance 1:
```
Var(q·k) = Var(Σ_i q_i k_i) = Σ_i Var(q_i k_i) = d_k
```

**Issue**: Large variance pushes softmax into saturation regions (very small gradients)

**Solution**: Divide by √d_k to maintain variance ≈ 1

**Mathematical justification**:
```
Var(q·k / √d_k) = Var(q·k) / d_k = d_k / d_k = 1
```

### Implementation

```python
# From src/transformer/components/attention.py
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
attn_weights = F.softmax(scores, dim=-1)
output = torch.matmul(attn_weights, V)
```

---

## 2. Multi-Head Attention

### Mathematical Formulation

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O

where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### Parameter Matrices

For each head *i* ∈ {1, ..., h}:
- **W_i^Q** ∈ ℝ^(d_model × d_k): Query projection
- **W_i^K** ∈ ℝ^(d_model × d_k): Key projection
- **W_i^V** ∈ ℝ^(d_model × d_v): Value projection

Output projection:
- **W^O** ∈ ℝ^(h·d_v × d_model): Combines all heads

### Dimensions

Typically:
```
h = number of heads (e.g., 8)
d_k = d_v = d_model / h
```

This ensures total dimension matches:
```
h * d_v = d_model
```

### Computational Steps

1. **Project inputs** (for each head):
   ```
   Q_i = Q W_i^Q  ∈ ℝ^(n × d_k)
   K_i = K W_i^K  ∈ ℝ^(m × d_k)
   V_i = V W_i^V  ∈ ℝ^(m × d_v)
   ```

2. **Compute attention** (for each head):
   ```
   head_i = Attention(Q_i, K_i, V_i) ∈ ℝ^(n × d_v)
   ```

3. **Concatenate heads**:
   ```
   Concat = [head_1; head_2; ...; head_h] ∈ ℝ^(n × h·d_v)
   ```

4. **Final projection**:
   ```
   Output = Concat W^O ∈ ℝ^(n × d_model)
   ```

### Parameter Count

Per multi-head attention:
```
Parameters = 4 * d_model * d_model
           = 4 * d_model²
```

Breakdown:
- Q projection: d_model × d_model
- K projection: d_model × d_model  
- V projection: d_model × d_model
- Output projection: d_model × d_model

---

## 3. Masked Attention

### Causal Mask

For autoregressive models, prevent attending to future positions:

```
Mask[i,j] = {  0    if j ≤ i
            { -∞    if j > i
```

Applied before softmax:
```
Attention = softmax((QK^T / √d_k) + Mask) V
```

### Effect

After softmax, attention weights to masked positions become 0:
```
softmax(-∞) = 0
```

### Mathematical Properties

For causal mask matrix **M**:
- Lower triangular (including diagonal)
- M[i,j] = 0 for j ≤ i
- M[i,j] = -∞ for j > i

### Implementation

```python
# From src/transformer/components/attention.py
def get_causal_mask(seq_len, device):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask.to(device)
```

---

## 4. Position Encoding

### Sinusoidal Position Encoding

For position *pos* and dimension *i*:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### Properties

1. **Bounded**: |PE(pos, i)| ≤ 1 for all pos, i

2. **Unique**: Each position has unique encoding

3. **Relative Position**: For any fixed offset *k*:
   ```
   PE(pos+k) = A_k · PE(pos)
   ```
   where A_k is a rotation matrix (can be learned by model)

4. **Extrapolation**: Can handle sequences longer than training

### Frequency Interpretation

Different dimensions encode different frequencies:
- Low dimensions (i ≈ 0): High frequency (wavelength ≈ 2π)
- High dimensions (i ≈ d_model): Low frequency (wavelength ≈ 10000·2π)

### Matrix Form

Position encoding matrix **PE** ∈ ℝ^(max_len × d_model):

```
PE[pos, :] = [sin(pos/θ_1), cos(pos/θ_1), ..., sin(pos/θ_{d/2}), cos(pos/θ_{d/2})]

where θ_i = 10000^(2i/d_model)
```

---

## 5. Feed-Forward Network

### Formula

```
FFN(x) = max(0, xW_1 + b_1) W_2 + b_2
```

Or with GELU activation:
```
FFN(x) = GELU(xW_1 + b_1) W_2 + b_2
```

### Parameters

- **W_1** ∈ ℝ^(d_model × d_ff): Expansion
- **b_1** ∈ ℝ^(d_ff): Bias
- **W_2** ∈ ℝ^(d_ff × d_model): Projection
- **b_2** ∈ ℝ^(d_model): Bias

Typically: d_ff = 4 × d_model

### Parameter Count

```
Parameters = d_model * d_ff + d_ff + d_ff * d_model + d_model
           ≈ 2 * d_model * d_ff
           = 8 * d_model²  (when d_ff = 4 * d_model)
```

### Per-Position Application

Applied independently to each position:
```
output[i] = FFN(input[i])  for i = 1, ..., n
```

Equivalent to two 1×1 convolutions in sequence modeling terms.

---

## 6. Layer Normalization

### Formula

For input **x** ∈ ℝ^d:

```
LayerNorm(x) = γ ⊙ (x - μ) / σ + β
```

Where:
```
μ = (1/d) Σ_i x_i                 (mean)
σ = √((1/d) Σ_i (x_i - μ)²)      (std dev)
```

Parameters:
- **γ** ∈ ℝ^d: Learned scale (initialized to 1)
- **β** ∈ ℝ^d: Learned shift (initialized to 0)
- ⊙ denotes element-wise multiplication

### Batch vs Layer Normalization

**Batch Norm**: Normalize across batch dimension
```
μ_j = (1/B) Σ_b x_{b,j}
```

**Layer Norm**: Normalize across feature dimension
```
μ_i = (1/d) Σ_j x_{i,j}
```

**Why Layer Norm for Transformers?**
- Works with variable-length sequences
- Per-sample normalization (no batch dependency)
- Stable during inference

### Gradient Flow

Layer norm helps gradient flow by:
1. Keeping activations in reasonable range
2. Reducing covariate shift
3. Enabling larger learning rates

---

## 7. Residual Connections

### Formula

```
output = x + F(x)
```

Where F is a sublayer (attention or FFN).

### Gradient Flow Analysis

**Forward**:
```
y = x + F(x)
```

**Backward**:
```
∂L/∂x = ∂L/∂y * (1 + ∂F/∂x)
```

Key insight: Gradient always has component "1", ensuring gradient flow even if ∂F/∂x is small.

### In Transformer Block

Two residual connections per block:

```
x₁ = x₀ + MultiHeadAttention(LayerNorm(x₀))
x₂ = x₁ + FeedForward(LayerNorm(x₁))
```

### Mathematical Properties

1. **Identity shortcut**: Model can learn identity easily (set F(x) = 0)
2. **Gradient highways**: Direct path for gradients to flow backward
3. **Ensemble interpretation**: Each layer adds small refinement

---

## 8. Full Transformer Block

### Mathematical Definition

For input **X** ∈ ℝ^(n × d_model):

**Pre-Norm** (modern):
```
X' = X + MultiHeadAttention(LayerNorm(X))
X_out = X' + FFN(LayerNorm(X'))
```

**Post-Norm** (original):
```
X' = LayerNorm(X + MultiHeadAttention(X))
X_out = LayerNorm(X' + FFN(X'))
```

### Complete Forward Pass

Given input sequence **x** = (x₁, ..., xₙ):

1. **Embedding**:
   ```
   E = TokenEmbedding(x) + PositionEncoding
   ```

2. **Layer l** (for l = 1 to L):
   ```
   H^l = E if l=1 else H^{l-1}
   
   # Self-attention
   Q = H^l W_Q
   K = H^l W_K  
   V = H^l W_V
   A^l = Attention(Q, K, V)
   H^l_1 = LayerNorm(H^l + A^l)
   
   # Feed-forward
   F^l = FFN(H^l_1)
   H^l_2 = LayerNorm(H^l_1 + F^l)
   ```

3. **Output**:
   ```
   logits = H^L_2 W_vocab + b
   ```

---

## 9. Parameter Count Analysis

### Per Layer

**Multi-Head Attention**:
```
P_attn = 4 * d_model²
```

**Feed-Forward**:
```
P_ffn = 2 * d_model * d_ff
       = 8 * d_model²  (when d_ff = 4 * d_model)
```

**Layer Norm** (×2):
```
P_ln = 4 * d_model
```

**Total per layer**:
```
P_layer ≈ 12 * d_model²
```

### Full Model

**All Layers**:
```
P_layers = L * 12 * d_model²
```

**Embeddings**:
```
P_embed = vocab_size * d_model
```

**Output Projection**:
```
P_output = d_model * vocab_size
```

**Total**:
```
P_total ≈ L * 12 * d_model² + 2 * vocab_size * d_model
```

### Example: GPT-2 Small

- L = 12 layers
- d_model = 768
- vocab_size = 50,257

```
P_total ≈ 12 * 12 * 768² + 2 * 50,257 * 768
       ≈ 85M + 77M
       ≈ 162M parameters
```

---

## 10. Computational Complexity

### Per Layer

**Self-Attention**:
```
O(n² * d_model)
```
- Matrix multiplication QK^T: O(n² * d_model)
- Matrix multiplication with V: O(n² * d_model)

**Feed-Forward**:
```
O(n * d_model * d_ff)
= O(n * 4 * d_model²)
```

**Total per Layer**:
```
O(n² * d_model + n * d_model²)
```

### Bottleneck Analysis

For typical values (n ≈ 1000, d_model ≈ 1000):

**If n < d_model**: FFN dominates (O(n * d_model²))
**If n > d_model**: Attention dominates (O(n² * d_model))

For long sequences (e.g., n = 10,000), attention becomes very expensive.

---

## 11. Information Theory Perspective

### Attention as Information Routing

Attention weights A[i,j] can be viewed as:
```
I(query_i → key_j) = A[i,j]
```

Mutual information between query and key positions.

### Entropy of Attention

**Attention Entropy**:
```
H(A_i) = -Σ_j A[i,j] log A[i,j]
```

**Interpretation**:
- Low entropy: Focused attention (peaked distribution)
- High entropy: Diffuse attention (uniform distribution)

**Metrics**:
- Sparsity: Fraction of weights > threshold
- Entropy: Measure of concentration
- Max attention: Peak weight value

**Implementation**: See `examples/basic_usage.py` for analysis

---

## 12. Gradient Flow Analysis

### Vanishing Gradient Problem

In deep networks, gradients can vanish:
```
∂L/∂x₀ = ∂L/∂xₗ * ∏_{i=1}^L ∂x_i/∂x_{i-1}
```

If |∂x_i/∂x_{i-1}| < 1, product → 0 exponentially.

### Transformer Solution

**Residual connections** ensure:
```
∂x_i/∂x_{i-1} = 1 + ∂F/∂x_{i-1}
```

Gradient always has magnitude ≥ 1 component.

**Layer norm** prevents exploding activations, stabilizing gradients.

---

## 13. Path Counting Mathematics

### Combinatorial Analysis

From position (0,0) to (L-1, N-1):
- Must move L-1 times vertically (layers)
- Must move N-1 times horizontally (positions)
- Total moves: (L-1) + (N-1) = L + N - 2

**Number of paths**:
```
C(L+N-2, L-1) = (L+N-2)! / ((L-1)! * (N-1)!)
```

This is the binomial coefficient: "In how many ways can we arrange (L-1) vertical moves among (L+N-2) total moves?"

### Example Calculations

**Small network** (L=3, N=4):
```
Paths = C(3+4-2, 3-1) = C(5, 2) = 10
```

**Medium network** (L=6, N=32):
```
Paths = C(36, 5) = 376,992
```

**Large network** (L=12, N=512):
```
Paths = C(522, 11) ≈ 10^27
```

For comparison, estimated atoms in observable universe ≈ 10^80.

### Implementation

See `src/visualization/path_analysis.py` for detailed path counting and visualization.

---

## 14. Approximation Theory

### Universal Approximation

**Theorem** (informal): A transformer with sufficient depth and width can approximate any continuous function on a compact domain to arbitrary precision.

**Intuition**: 
- Attention can route information arbitrarily
- FFN provides non-linear transformations
- Multiple layers enable hierarchical composition

### Expressiveness Results

**Turing Completeness**: Transformers can simulate arbitrary algorithms (with appropriate construction).

**Computational Universality**: Given enough layers, can implement any computation.

---

## 15. Optimization Landscape

### Loss Function

For language modeling:
```
L = -(1/N) Σ_i log P(x_i | x_{<i})
```

Cross-entropy between predicted and true distributions.

### Gradient Computation

**Backpropagation through attention**:
```
∂L/∂Q = (∂L/∂A) K / √d_k
∂L/∂K = (∂L/∂A)^T Q / √d_k
∂L/∂V = A^T (∂L/∂output)
```

Where:
```
∂L/∂A = softmax_jacobian(scores) · (∂L/∂output V^T)
```

### Adam Optimizer

**Update rule**:
```
m_t = β₁ m_{t-1} + (1-β₁) g_t
v_t = β₂ v_{t-1} + (1-β₂) g_t²
θ_t = θ_{t-1} - α * m_t / (√v_t + ε)
```

**Typical values**:
- β₁ = 0.9 (momentum)
- β₂ = 0.999 (variance)
- α = learning rate (with warmup)
- ε = 10^-8 (numerical stability)

---

## References

### Foundational Papers

1. Vaswani et al. (2017): "Attention is All You Need"
2. Ba et al. (2016): "Layer Normalization"  
3. He et al. (2016): "Deep Residual Learning"

### Further Reading

- `TRANSFORMER_THEORY.md` - Conceptual foundations
- `STATISTICS.md` - Statistical analysis of transformer behavior
- `../guides/quickstart.md` - Implementation guide

---

**Last Updated**: October 2025  
**Implementation**: See `src/transformer/` for all formulas implemented in code

