# Introduction

Scaled dot-product attention combines value vectors according to
query–key similarity [@vaswani2017]. A language-model-style causal mask
limits that operation to the current and earlier positions. The project
offers an educational implementation with source-level access to the
embedded input, the block outputs, and the attention weights
[@transformerSoftware].

A heatmap can help locate the causal boundary but only represents a
specific head and layer. The experiment examines the boundary directly
in the recorded weights and tests its effect on logits by changing the
input. A separate grid analogy provides a finite example of route
enumeration.
