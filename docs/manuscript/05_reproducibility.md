# Reproducibility

Run `python -m examples.manuscript_probe --output-dir output` from the
repository root. The probe builds the model from the local preset and uses
the local attention visualizer, grid counter, and grid enumerator. It
writes `output/data/causal_probe.json` and
`output/figures/causal_attention.png`. Its checks enforce the zero future
weight, the prefix invariance and row-sum tolerances, a finite positive
suffix difference, and agreement between the two grid routines.

The command runs without a dataset, model downloads, training, or a GPU.
Floating-point values such as the suffix difference may vary across
versions and CPU kernels. The supplied build instructions pin the Template
renderer separately from the project's Python environment.
