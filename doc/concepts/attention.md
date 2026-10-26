# Attention Mechanism Deep Dive

Understanding Query, Key, and Value from first principles.

## The Core Idea

Attention is a **content-based routing mechanism** that dynamically determines how information flows between positions.

Unlike fixed connections (like in CNNs) or sequential dependencies (like in RNNs), attention learns **what to retrieve based on what's needed**.

## The Three Matrices: Q, K, V

### Query (Q): "What do I need?"

**Semantic Meaning**: Encodes the current state's information needs

**Computational Role**: Vector that will be matched against Keys

**Analogy**: A search query in a database

**Example**: If processing "The cat sat on the", the query might encode "looking for location/surface information"

### Key (K): "Am I relevant?"

**Semantic Meaning**: Encodes what information this position offers for retrieval

**Computational Role**: Vector that will be matched by future Queries

**Analogy**: Index entries or hashtags in a database

**Example**: The token "mat" might have a Key encoding "I'm a surface/location"

### Value (V): "Here's the actual information"

**Semantic Meaning**: The actual information to transmit if attended to

**Computational Role**: Content that gets weighted and summed

**Analogy**: The actual database records

**Example**: Rich representation of "mat" including semantic, syntactic, and contextual information

## The Attention Computation

```
Step 1: Compute Q, K, V from input
  Q = Input @ W_q
  K = Input @ W_k
  V = Input @ W_v

Step 2: Compute attention scores (similarity)
  Scores = (Q @ K.T) / sqrt(d_k)
  
Step 3: Convert to probabilities
  Weights = softmax(Scores)
  
Step 4: Weighted sum of values
  Output = Weights @ V
```

## Why This Works

### Content-Based Addressing
- Similar Q and K → high attention
- Different Q and K → low attention
- Learned, not fixed

### Dynamic Routing
- Different contexts produce different Q
- Same token can attend differently
- Adaptive to input

### Information Packaging
- V can be different from input
- Learned transformation
- Optimal for downstream tasks

## Multi-Head Attention

Instead of one attention, use **multiple heads in parallel**:

```python
for head in range(n_heads):
    Q_h, K_h, V_h = split_heads(Q, K, V)
    Output_h = attention(Q_h, K_h, V_h)
    
Final_Output = concatenate(all_outputs) @ W_o
```

### Why Multiple Heads?

1. **Different relationship types**: One head might learn syntactic patterns, another semantic
2. **Ensemble effects**: Multiple perspectives reduce variance
3. **Specialization**: Heads specialize for different tasks

## Attention Patterns in Practice

### Induction Heads
- Detect repeated patterns
- "If I saw A followed by B before, and now I see A, predict B"

### Previous Token Heads
- Always attend to the immediately previous position
- Learn sequential dependencies

### Global Aggregation Heads
- Attend uniformly to all positions
- Compute aggregate statistics

### Specialized Heads
- Syntactic attention (subject-verb agreement)
- Semantic attention (related concepts)
- Positional attention (beginning/end of sequence)

## Causal Masking

For autoregressive models, prevent attending to future:

```python
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
scores = scores.masked_fill(mask == 1, float('-inf'))
```

This ensures position `i` can only attend to positions `0` through `i`.

## Attention Entropy

Measures how concentrated attention is:

```python
entropy = -sum(weights * log(weights))
```

- High entropy: Diffuse attention (looking at many positions)
- Low entropy: Focused attention (looking at few positions)

## Visualizing Attention

```python
from src.visualization import visualize_attention

# Heatmap of attention weights
visualize_attention(attention_weights, layer_idx=0)

# Sankey flow diagram
visualize_attention_flow_sankey(attention_weights)

# Head specialization analysis
visualize_head_specialization(state)
```

## Common Misconceptions

### Misconception 1: "Attention is just weighted averaging"
**Reality**: It's content-based routing with learned transformations (Q, K, V projections)

### Misconception 2: "Each head does one thing"
**Reality**: Heads are polysemantic - they do multiple things

### Misconception 3: "Attention weights tell you what the model 'looks at'"
**Reality**: Weights show information flow, but don't reveal why or what's extracted

### Misconception 4: "More attention = more important"
**Reality**: Uniform attention can be informative (aggregation), peaked attention can be noise

## Mathematical Properties

### Permutation Equivariance
Without positional encoding, attention is permutation equivariant:
```
attention(permute(input)) = permute(attention(input))
```

### Complexity
- Time: O(n² · d) where n=seq_len, d=d_model
- Space: O(n²) for attention weights

### Approximations
For long sequences, various approximations exist:
- Sparse attention
- Linformer
- Performer
- Flash Attention

## Implementation Notes

In our implementation:

```python
class MultiHeadAttention(nn.Module):
    def forward(self, x, mask=None, return_state=False):
        # Compute Q, K, V
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        
        # Split into heads
        Q, K, V = split_heads(Q, K, V, self.n_heads)
        
        # Attention
        scores = (Q @ K.T) / sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -inf)
        
        weights = softmax(scores)
        output = weights @ V
        
        # Concatenate heads
        output = concat_heads(output)
        
        return output, AttentionState(Q, K, V, weights, output)
```

## Further Reading

- [Information Flow](information_flow.md) - How attention fits in the architecture
- [K/V Stream](../ARCHITECTURE.md#2-kv-stream-horizontal-flow) - Horizontal information flow
- [Path Explosion](../theory/TRANSFORMER_THEORY.md#computational-path-explosion) - Computational paths through attention


