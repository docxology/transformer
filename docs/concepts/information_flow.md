# Information flow through a forward pass

The [model](../../src/transformer/model.py) maps a batch of integer token IDs
to token embeddings, adds sinusoidal positional encoding, and applies a
stack of [Transformer blocks](../../src/transformer/components/transformer_block.py).
Final layer normalization and a linear projection produce vocabulary logits.

Within a block, queries, keys, and values come from its input residual tensor.
Each head computes scaled query–key scores and uses masked softmax weights
to mix values across positions. The projected attention output is added to
the input, then normalized. A position-wise feed-forward network transforms
that tensor, followed by a second residual addition and normalization.

`TransformerState.get_residual_stream()` returns the embedded input followed
by one checkpoint per block after feed-forward normalization. The block
states also expose attention and feed-forward checkpoints. Each attention
tensor indexes batch, head, query position, and key position. Keeping state
retains intermediate tensors, and each attention matrix grows quadratically
with sequence length. Disable state capture when only logits are needed.

The [theory guide](../theory/README.md) specifies the mask convention and the
separate combinatorial path model. The [CPU probe](../guides/analysis.md)
demonstrates which inferences the recorded tensors support.
