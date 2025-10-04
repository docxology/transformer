# Transformer Architecture Glossary

## Purpose

This glossary provides clear definitions of all terms used throughout the codebase and documentation, establishing a coherent ontology for understanding transformers.

---

## Core Architecture

### Transformer
The complete neural network architecture introduced in "Attention is All You Need" (Vaswani et al., 2017). Consists of token embedding, positional encoding, multiple transformer blocks, and output projection.

**Implementation**: `src/transformer/model.py`

### Transformer Block
A single layer containing multi-head self-attention, feed-forward network, layer normalization, and residual connections. Modern transformers stack multiple blocks.

**Implementation**: `src/transformer/components/transformer_block.py`

### Token
A discrete unit of input (e.g., word, subword, character). Represented as an integer index into vocabulary.

### Embedding
Mapping from discrete tokens to continuous vector representations. Converts token IDs to dense vectors in ℝ^d_model.

**Implementation**: `src/transformer/components/embedding.py`

### Vocabulary (Vocab)
Set of all possible tokens the model can process. Size denoted as `vocab_size`.

---

## Attention Mechanisms

### Attention
Mechanism that allows model to focus on different parts of input when producing output. Computes weighted sum of values based on query-key similarity.

**Formula**: `Attention(Q, K, V) = softmax(QK^T / √d_k) V`

**Implementation**: `src/transformer/components/attention.py`

### Self-Attention
Special case where queries, keys, and values all come from the same sequence. Each position can attend to all positions.

### Multi-Head Attention
Parallel attention mechanisms that learn different relationships. Concatenates outputs from multiple attention heads.

**Implementation**: `MultiHeadAttention` in `src/transformer/components/attention.py`

### Attention Head
Single attention mechanism within multi-head attention. Different heads specialize in different patterns (local, global, syntactic, etc.).

### Query (Q)
"What am I looking for?" - Vector representing information request from current position.

**Dimension**: ℝ^d_k

### Key (K)
"What do I have to offer?" - Vector representing information available at each position.

**Dimension**: ℝ^d_k

### Value (V)
"What actual information should be transmitted?" - Vector containing information to be passed if attention is high.

**Dimension**: ℝ^d_v

### Attention Weights
Normalized scores determining how much each position attends to others. Sum to 1 across keys (via softmax).

**Shape**: (batch, n_heads, query_len, key_len)

### Attention Score
Raw similarity measure between query and key before normalization. Computed as Q·K^T / √d_k.

### Causal Mask
Mask preventing attention to future positions. Essential for autoregressive generation (language modeling).

**Implementation**: `get_causal_mask()` in `src/transformer/components/attention.py`

### Attention Mask
Binary mask indicating which positions to attend to (1) or ignore (0). Used for padding and causality.

---

## Information Flow

### Residual Stream
**Vertical information highway** - Flow of representations through layers via residual connections. Information accumulates layer by layer.

**Formula**: `x_{l+1} = x_l + F(x_l)`

**Visualization**: `outputs/visualizations/residual_stream.png`

### K/V Stream
**Horizontal information highway** - Flow of information across positions via attention mechanism. Enables token-to-token communication.

**Visualization**: `outputs/visualizations/kv_stream.png`

### Information Highways
Combined view of both vertical (residual) and horizontal (attention) information flow. Key insight of transformer architecture.

**Visualization**: `outputs/visualizations/information_highways.png`

### Computational Path
Route that information takes from input to output, traversing layers (vertical) and positions (horizontal). Astronomical number of possible paths.

**Analysis**: `examples/path_analysis.py`

---

## Model Components

### Feed-Forward Network (FFN)
Two-layer fully connected network applied position-wise. Provides non-linear transformation independent of other positions.

**Formula**: `FFN(x) = max(0, xW₁ + b₁)W₂ + b₂`

**Implementation**: `src/transformer/components/feed_forward.py`

### Layer Normalization
Normalization applied across features (not batch). Stabilizes training by keeping activations in reasonable range.

**Formula**: `LayerNorm(x) = γ(x - μ)/σ + β`

**Implementation**: `src/transformer/components/layer_norm.py`

### Residual Connection
Skip connection that adds input to output of sublayer. Enables gradient flow and deep networks.

**Formula**: `output = x + F(x)`

### Positional Encoding
Information about position in sequence. Added to embeddings since attention is position-agnostic.

**Implementation**: `src/transformer/components/positional_encoding.py`

### Dropout
Regularization technique that randomly zeroes activations during training. Prevents overfitting.

---

## Dimensions and Hyperparameters

### d_model
Dimension of model representations throughout the network. All embeddings and hidden states have this dimension.

**Typical values**: 256, 512, 768, 1024

### n_heads
Number of parallel attention heads in multi-head attention.

**Typical values**: 4, 8, 12, 16

**Constraint**: d_model must be divisible by n_heads

### d_k
Dimension of queries and keys per attention head.

**Formula**: `d_k = d_model / n_heads`

### d_v
Dimension of values per attention head. Usually equals d_k.

**Formula**: `d_v = d_model / n_heads`

### d_ff
Dimension of feed-forward network's hidden layer.

**Formula**: `d_ff = 4 × d_model` (typically)

### n_layers (L)
Number of transformer blocks stacked in the model.

**Typical values**: 6, 12, 24 (or more for large models)

### seq_len (n)
Length of input sequence. Number of tokens.

### max_seq_len
Maximum sequence length model can handle (limited by positional encodings).

---

## State and Tracking

### State
Object containing intermediate representations, attention weights, and K/V matrices from forward pass. Enables visualization and analysis.

**Type**: `TransformerState`

**Usage**: `logits, state = model(input_ids, return_state=True)`

### Activation
Output values from neurons/layers during forward pass. Used for analysis and visualization.

### Hidden State
Intermediate representation at any layer. Same as "activation" in this context.

---

## Training

### Loss Function
Measure of prediction error. For language modeling, typically cross-entropy.

**Formula**: `L = -Σ log P(x_i | x_{<i})`

### Gradient
Partial derivative of loss with respect to parameters. Used to update weights during training.

### Backpropagation
Algorithm for computing gradients through network via chain rule.

### Optimizer
Algorithm for updating parameters using gradients. Adam and AdamW most common for transformers.

### Learning Rate
Step size for parameter updates. Critical hyperparameter requiring careful tuning.

### Warmup
Period at training start with gradually increasing learning rate. Stabilizes early training.

### Gradient Clipping
Limiting gradient magnitude to prevent exploding gradients.

**Implementation**: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`

### Weight Decay
Regularization technique that penalizes large parameter values. Part of AdamW optimizer.

### Epoch
One complete pass through training dataset.

### Batch
Subset of data processed together. Enables parallel computation.

### Batch Size
Number of sequences processed in parallel.

---

## Generation

### Autoregressive
Generating one token at a time, conditioning on previous tokens. Standard for language models.

### Greedy Decoding
Always selecting most probable next token. Deterministic but may produce repetitive text.

### Sampling
Probabilistically selecting next token from distribution. Introduces randomness.

### Temperature
Parameter controlling randomness in sampling. Higher = more random, lower = more focused.

### Top-K Sampling
Sampling from K most probable tokens only. Balances quality and diversity.

### Nucleus (Top-P) Sampling
Sampling from smallest set of tokens with cumulative probability ≥ p. Adaptive cutoff.

### Beam Search
Maintaining multiple hypotheses during generation. Higher quality but more expensive.

---

## Analysis Metrics

### Sparsity
Fraction of attention weights above threshold. Measures focus vs. diffusion.

**Formula**: `sparsity = (attention > threshold).mean()`

### Entropy
Measure of attention distribution uniformity. Higher = more diffuse, lower = more focused.

**Formula**: `H = -Σ p log p`

### Locality
Average distance of attention. Measures local vs. global focus.

**Implementation**: `src/visualization/advanced_viz.py`

### Norm
Magnitude of vector. Used to track representation strength through layers.

**Formula**: `||x|| = √(Σ x_i²)`

---

## Visualization Concepts

### Heatmap
2D plot showing attention weights as colors. Bright = high attention.

**Tool**: `visualize_attention()` in `src/visualization/attention_viz.py`

### Animation
Temporal visualization showing evolution through layers or training.

**Tools**: `src/visualization/animation.py` and `advanced_animations.py`

### Path Trace
Visualization of specific information flow route through network.

**Tool**: `animate_path_trace()` in `src/visualization/animation.py`

### Flow Diagram
Visualization of information movement (residual and K/V streams).

**Tool**: `visualize_information_highways()` in `src/visualization/flow_viz.py`

---

## Mathematical Notation

### Matrices
Bold uppercase letters: **W**, **X**, **Q**, **K**, **V**

### Vectors
Bold lowercase letters: **x**, **h**, **q**, **k**, **v**

### Scalars
Italic letters: *d*, *n*, *L*

### Subscripts
Position/index: x₁, x₂, ..., xₙ

### Superscripts
Layer number: x⁽ˡ⁾ for layer l

### Operators
- Transpose: X^T
- Element-wise: ⊙
- Matrix multiplication: XY or X·Y
- Summation: Σ
- Norm: ||x||

---

## File Organization Ontology

### Source Code (`src/`)

**`src/transformer/`** - Core architecture
- `model.py` - Main transformer class
- `components/` - Individual components (attention, FFN, etc.)

**`src/visualization/`** - Visualization tools
- `attention_viz.py` - Attention heatmaps
- `flow_viz.py` - Information flow diagrams
- `animation.py` - Basic animations
- `advanced_animations.py` - Advanced animations
- `architecture_viz.py` - Structural diagrams
- `path_analysis.py` - Path counting and tracing

**`src/config/`** - Configuration management
- `config_manager.py` - Configuration utilities
- `presets.py` - Predefined configurations

**`src/utils/`** - Utilities
- `helpers.py` - General utilities
- `logging_system.py` - Structured logging
- `enhanced_logging.py` - Enhanced console output
- `reporting.py` - Report generation
- `analysis_tools.py` - Analysis utilities

**`src/evaluation/`** - Evaluation tools
- `metrics.py` - Performance metrics
- `benchmark.py` - Benchmarking utilities

### Documentation (`doc/`)

**`doc/theory/`** - Theoretical foundations
- `TRANSFORMER_THEORY.md` - Core concepts
- `MATHEMATICS.md` - Mathematical formulations
- `STATISTICS.md` - Statistical analysis

**`doc/guides/`** - Practical guides
- `quickstart.md` - Getting started
- `running.md` - Running examples
- `IMPLEMENTATION_GUIDE.md` - Implementation details
- `visualization.md` - Visualization guide
- `animations.md` - Animation guide
- `analysis.md` - Analysis guide

**`doc/concepts/`** - Deep dives
- `attention.md` - Attention mechanism details
- `information_flow.md` - Information flow analysis

**`doc/api/`** - API documentation
- `visualization.md` - Visualization API reference

### Examples (`examples/`)
- `basic_usage.py` - Simple usage example
- `training_example.py` - Training loop
- `animation_demo.py` - Animation creation
- `path_analysis.py` - Path analysis demonstration

### Outputs (`outputs/`)
- `visualizations/` - Static plots
- `animations/` - GIF animations
- `analysis/` - JSON analysis results
- `reports/` - Markdown reports
- `logs/` - Training logs
- `data/` - Intermediate data
- `configs/` - Saved configurations

### Tests (`tests/`)
- `test_model.py` - Model tests
- `test_components.py` - Component tests
- `test_visualization.py` - Visualization tests
- `test_config.py` - Configuration tests
- `test_evaluation.py` - Evaluation tests

---

## Common Abbreviations

- **FFN**: Feed-Forward Network
- **MHA**: Multi-Head Attention
- **LN**: Layer Normalization
- **Q, K, V**: Query, Key, Value
- **d_model**: Model dimension
- **d_ff**: Feed-forward dimension
- **seq_len**: Sequence length
- **vocab**: Vocabulary
- **attn**: Attention
- **res**: Residual

---

## Conceptual Relationships

### Hierarchy

```
Transformer
├── Embedding
│   ├── Token Embedding
│   └── Positional Encoding
├── Transformer Blocks (×L)
│   ├── Multi-Head Attention
│   │   ├── Attention Heads (×n_heads)
│   │   │   └── Scaled Dot-Product Attention
│   │   └── Output Projection
│   ├── Layer Normalization (×2)
│   ├── Feed-Forward Network
│   │   ├── Expansion Layer (d_model → d_ff)
│   │   ├── Activation (ReLU/GELU)
│   │   └── Projection Layer (d_ff → d_model)
│   └── Residual Connections (×2)
└── Output Projection
```

### Information Flow

```
Input Tokens
    ↓ (Token Embedding)
Token Vectors
    ↓ (+ Positional Encoding)
Initial Representations
    ↓ (Layer 1: Attention + FFN)
Layer 1 Representations ← ← ← Horizontal (K/V) flow
    ↓ Vertical (Residual) flow
Layer 2 Representations ← ← ← Horizontal (K/V) flow
    ↓ Vertical (Residual) flow
    ...
    ↓ Vertical (Residual) flow
Final Representations
    ↓ (Output Projection)
Logits / Predictions
```

### Dependencies

```
Attention depends on: Q, K, V matrices
Q, K, V depend on: Input representations, Weight matrices
Multi-Head combines: Multiple attention heads
Transformer Block needs: Attention, FFN, LayerNorm, Residual
Full Model needs: Embedding, Positional Encoding, Blocks, Output Projection
```

---

## Cross-References

### By Concept

**Attention** → See:
- `doc/theory/TRANSFORMER_THEORY.md` (§2: Attention Mechanism)
- `doc/theory/MATHEMATICS.md` (§1: Scaled Dot-Product Attention)
- `doc/concepts/attention.md`
- `src/transformer/components/attention.py`

**Information Flow** → See:
- `doc/theory/TRANSFORMER_THEORY.md` (§3: Information Flow)
- `doc/concepts/information_flow.md`
- `examples/basic_usage.py`
- `outputs/visualizations/information_highways.png`

**Path Analysis** → See:
- `doc/theory/TRANSFORMER_THEORY.md` (§4: Computational Path Explosion)
- `doc/theory/MATHEMATICS.md` (§13: Path Counting Mathematics)
- `examples/path_analysis.py`
- `src/visualization/path_analysis.py`

**Training** → See:
- `doc/guides/IMPLEMENTATION_GUIDE.md` (§3: Training)
- `examples/training_example.py`
- `doc/theory/STATISTICS.md` (§6: Training Dynamics)

**Visualization** → See:
- `doc/guides/visualization.md`
- `doc/guides/animations.md`
- `examples/animation_demo.py`
- `src/visualization/`

---

## Version Information

**Specification Version**: 1.0  
**Last Updated**: October 2025  
**Based On**: "Attention is All You Need" (Vaswani et al., 2017)  
**Implementation**: This repository

---

## Further Reading

- `doc/FAQ.md` - Common questions and answers
- `doc/theory/TRANSFORMER_THEORY.md` - Comprehensive theory
- `doc/guides/IMPLEMENTATION_GUIDE.md` - Practical implementation
- `README.md` - Repository overview

---

**Note**: This glossary is a living document. Terms may be added as the codebase evolves.

