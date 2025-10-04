"""Evaluation and benchmarking tools."""

from .metrics import (
    compute_perplexity,
    compute_accuracy,
    evaluate_model,
    MetricsTracker
)
from .benchmark import (
    benchmark_inference,
    benchmark_training,
    profile_model,
    BenchmarkResults
)

__all__ = [
    'compute_perplexity',
    'compute_accuracy',
    'evaluate_model',
    'MetricsTracker',
    'benchmark_inference',
    'benchmark_training',
    'profile_model',
    'BenchmarkResults'
]


