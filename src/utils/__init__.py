"""Utility functions."""

from .helpers import count_parameters, set_seed, get_device
from .html_reporter import generate_html_report
from .progress_tracker import (
    create_progress_bar,
    track_progress,
    MultiProgressTracker,
    Timer,
    format_time,
    estimate_remaining
)
from .trend_analyzer import (
    analyze_trends,
    compare_trends,
    generate_trend_report,
    detect_performance_regression
)
from .comparison_reporter import (
    generate_comparison_report,
    create_quick_comparison
)

__all__ = [
    'count_parameters', 
    'set_seed', 
    'get_device',
    # NEW: HTML reporting
    'generate_html_report',
    # NEW: Progress tracking
    'create_progress_bar',
    'track_progress',
    'MultiProgressTracker',
    'Timer',
    'format_time',
    'estimate_remaining',
    # NEW: Trend analysis
    'analyze_trends',
    'compare_trends',
    'generate_trend_report',
    'detect_performance_regression',
    # NEW: Comparison reporting
    'generate_comparison_report',
    'create_quick_comparison'
]


