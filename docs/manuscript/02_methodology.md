# Methods

We instantiate the repository's `debug` preset on the CPU after seeding
PyTorch with 7. It uses two blocks, two heads per block, a hidden width of
64, a feed-forward width of 256, a vocabulary of 100, and zero dropout.
All passes run in evaluation and inference mode. The input contains the
synthetic IDs [2, 7, 11, 3, 7, 5]. The intervention replaces only indices
3 through 5, giving [2, 7, 11, 13, 17, 19].

Inside a block, queries, keys, and values are projections of the same input.
The implementation scales query–key scores by the square root of the head
width and masks future positions before normalizing over keys. It then mixes
values, adds the attention update to the residual, and normalizes. The
position-wise feed-forward sublayer adds a second update and normalizes again.
The captured residual sequence contains the embedding and the output from each
block. Final normalization and vocabulary projection produce the logits.

We take the maximum absolute upper-triangular attention weight and the
maximum absolute row-sum error across both blocks and heads. We calculate
the maximum absolute difference between the original and modified logits over
the prefix and suffix separately. Positive suffix change checks that the
intervention reaches the readout. The row- and prefix-error tolerances
are both one part per million.

For the grid model, the repository's counting routine applies the binomial
coefficient to orderings of one-layer and one-position steps. We compare a
2-by-2 displacement against its independent recursive enumerator.
