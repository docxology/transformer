# Discussion

With the default mask, an unchanged prefix has the same available keys
and values at every layer. Subsequent layers preserve the constraint
because feed-forward updates, normalization, and vocabulary projection act
independently at each position. The zero suffix-to-prefix difference is
consistent with this implementation for the tested input. The suffix
changes enough to show that the comparison used two distinct inputs.

The experiment uses random initialized weights, one seed, and a
six-position input. It establishes basic invariants for that probe. It
does not estimate trained-model accuracy, interpret learned circuits, or
demonstrate the behavior of custom masks. The displayed attention
coefficients describe where a head mixes values. Their values alone do
not quantify each token's effect on an output logit.

The grid routes are orderings of two allowed step types. They do not
enumerate the autograd graph, which also includes direct links across
earlier positions, several heads, residual updates, and nonlinear
operations. The agreement at six paths checks the counting example
within that abstraction.
