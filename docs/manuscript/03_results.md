# Results

The probe produced logits of shape [1, 6, 100], attention tensors of shape
[1, 2, 6, 6] in each block, and three residual checkpoints of shape
[1, 6, 64]. The values below are from Python `3.12.14` with the PyTorch `2.15.0`
CPU build.

| Check | Observed value |
| --- | ---: |
| Maximum future attention weight across both blocks | 0 |
| Maximum absolute attention row-sum error | 0.0000001192 |
| Maximum absolute prefix logit difference | 0 |
| Maximum absolute suffix logit difference | 2.82479 |
| Grid count and enumerator count for displacement (2, 2) | 6 each |

![First block's attention weights for two heads. In both heatmaps, the region above the diagonal has zero weight. Query and key axes show the six input token IDs in sequence order.](../../output/figures/causal_attention.png)
