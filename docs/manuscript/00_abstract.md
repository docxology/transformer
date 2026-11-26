# Abstract

The First-Principles Transformer exposes intermediate tensors from a causal
sequence model implemented in PyTorch. We use one initialized, two-layer CPU
model to check three consequences of its causal mask. Future attention
weights are zero, each attention row sums to one within floating-point
tolerance, and replacing a three-token suffix leaves the prefix logits
unchanged. We separately compare a combinatorial grid count with direct
enumeration. Together these examples distinguish observations from a real
forward pass from a teaching model of possible routes.
