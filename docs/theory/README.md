# Causal attention and grid paths

For a head of width `d_k`, the implementation divides the query–key inner
product by the square root of `d_k`. The softmax normalizes across key
positions. The default causal mask permits key position *j* for query *i*
when *j* ≤ *i*. The implementation replaces disallowed logits with negative
infinity before the softmax. Captured weights are recorded after dropout,
so test their row sums in evaluation mode.

Each attention head mixes its projected values using those weights.
Output projections, other heads, residual connections, and later blocks
also influence the logits. Inspecting a heatmap describes the mixing
coefficients for a particular pass. Changing inputs and comparing logits
answers a different question.

The [path analysis module](../../src/visualization/path_analysis.py) uses a
grid with one-layer steps and one-position steps. For nonnegative
displacements Δlayer and Δposition, it counts their orderings with the
binomial coefficient C(Δlayer + Δposition, Δlayer). For a 2-by-2 displacement,
the six orderings each contain two steps of each kind. The grid is a teaching
model whose edges differ from the tensor program's direct attention links.

See the [technical manuscript](../manuscript/README.md) for source citations
and the numerical check that compares the grid formula with enumeration.
