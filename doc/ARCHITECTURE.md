# Transformer Architecture: Information Flow

This document explains the core architecture and information flow principles underlying this implementation.

## Two Information Highways

The transformer has two distinct information pathways:

### 1. Residual Stream (Vertical Flow)
- Flows **vertically** through layers at each position
- Carries the accumulated representation at each token position
- Modified by attention and feed-forward sublayers
- Preserved via residual connections

```
Position i:
  Embedding
     ↓
  Layer 0 (attn + ffn)
     ↓
  Layer 1 (attn + ffn)
     ↓
    ...
     ↓
  Layer N
     ↓
  Output
```

### 2. K/V Stream (Horizontal Flow)
- Flows **horizontally** across positions at each layer
- Enables information sharing via attention mechanism
- Carries Query, Key, Value representations
- Implements the "memory" and "retrieval" mechanism

```
Layer i:
  Pos 0 → Pos 1 → Pos 2 → ... → Pos M
    ↓       ↓       ↓             ↓
   K/V     K/V     K/V           K/V
    ↓       ↓       ↓             ↓
  Attention computes weighted sum
```

## Information Flow at Each Layer

At layer `i`, position `j`:

1. **Input**: Residual stream from layer `i-1`

2. **K/V Computation**: 
   - Current residual → K, V values
   - These are offered to future positions

3. **Attention**:
   - Current residual → Q values
   - Q queries all K values from current and previous positions
   - Computes attention weights (similarity between Q and K)
   - Weighted sum of V values → attention output

4. **Feed-Forward**:
   - Takes attention output + residual
   - Applies position-wise transformation
   - Output → next layer's residual stream

## Q, K, V Interpretations

### Query (Q): "Where should I look?"
- Encodes: Given the current state, what information do I need?
- Searches: For relevant K values in the past
- Implements: Active retrieval based on current context

### Key (K): "What futures should look here?"
- Encodes: Given the current state, what contexts should retrieve this?
- Offers: Matching signal for future Q values
- Implements: Indexing mechanism for retrieval

### Value (V): "What information to transmit?"
- Encodes: The actual information to pass forward
- Transmits: Rich representation to attending positions
- Implements: Content delivery once attention decides

## Path Explosion

Information from layer `i-1`, position `j-2` to layer `i`, position `j` can travel:

1. Up 1 layer, then right 2 positions
2. Right 1, up 1, right 1
3. Right 2, then up 1

For displacement of `n` layers and `m` positions, there are **C(m+n, n)** distinct paths.

Example: 6 layers, 32 positions, full traverse = C(37, 6) ≈ **2.3 million paths**

This creates:
- **Massive redundancy**: Many ways for information to flow
- **Interference patterns**: Different paths create nuanced encodings
- **Robust transmission**: Lossy transformations don't destroy information
- **Continuous cognition**: Information blends through multiple routes

## Implications for Understanding Transformers

### 1. Memory is Distributed
No single "memory cell" - information spreads through:
- Residual stream (vertical)
- Attention connections (horizontal)
- Multiple simultaneous paths

### 2. Cognition is Interferometric
Like quantum mechanics or wave optics:
- Multiple paths interfere
- Patterns emerge from combinations
- Context shapes retrieval through attention

### 3. Introspection is Possible
The architecture permits looking back at:
- Previous token states (via K/V stream)
- Previous layer states (via residual stream)
- Internal computations (both highways visible)

Contrary to some claims, LLMs **can** introspect on past internal states - the architecture explicitly enables it.

### 4. Time is Multi-Dimensional
Two "time" dimensions:
- **Layer depth**: Computational "time" (vertical)
- **Position**: Sequence "time" (horizontal)

Information flows through both simultaneously.

## Visualization Capabilities

This implementation enables visualization of:

1. **Attention weights**: Which positions attend where
2. **Residual stream**: How representations evolve vertically
3. **K/V stream**: What information flows horizontally
4. **Path traces**: Specific computational routes
5. **Information highways**: Combined view of both flows

## Mathematical Formulation

Given input `x` at layer `l`, position `j`:

```
# Attention sublayer
Q = x @ W_q
K = x @ W_k  
V = x @ W_v

# For position j, gather K, V from positions 0...j
K_past = [K_0, K_1, ..., K_j]
V_past = [V_0, V_1, ..., V_j]

# Compute attention
scores = Q @ K_past.T / sqrt(d_k)
weights = softmax(scores)  # This is the K/V stream flow
attn_out = weights @ V_past

# Residual connection (preserves vertical flow)
x' = x + attn_out

# Feed-forward sublayer
ff_out = FFN(x')
x'' = x' + ff_out  # Residual again

# Output x'' feeds to next layer (continues vertical flow)
```

## Design Principles

This implementation follows these principles:

1. **Modularity**: Each component is independent and testable
2. **Transparency**: All internal states can be tracked and visualized
3. **First-principles**: Built from scratch to understand fundamentals
4. **Visualization-first**: Designed to make information flow visible
5. **Real computation**: No mocks or simplifications

## References

- Original paper: "Attention Is All You Need" (Vaswani et al., 2017)
- Architecture insights based on mechanistic interpretability research
- Information flow analysis inspired by causal graph theory


