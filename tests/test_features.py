"""Tests for feature visualizations, reporting, and trend analysis.

Covers the src/ surfaces previously exercised only by the manual smoke
harness in scripts/ (3D viz, model comparison, color schemes, HTML and
comparison reports, trend analysis).
"""

import pytest
import torch

from src.transformer import Transformer, TransformerConfig
from src.visualization import (
    visualize_residual_3d,
    visualize_model_comparison,
    get_scheme,
    create_gradient,
)
from src.utils import analyze_trends, generate_html_report, generate_comparison_report
from src.utils.reporting import ReportMetrics


@pytest.fixture
def small_state():
    """Create a small model and capture its forward-pass state."""
    config = TransformerConfig(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
    model = Transformer(config)
    model.eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    with torch.no_grad():
        logits, state = model(input_ids, return_state=True)
    return state, config


class Test3DVisualization:
    """Test 3D residual stream visualization."""

    def test_visualize_residual_3d_static(self, small_state, tmp_path):
        """Static 3D visualization saves a PNG and returns a figure."""
        state, _ = small_state
        save_path = tmp_path / "test_3d.png"

        fig = visualize_residual_3d(
            state, use_pca=True, interactive=False, save_path=str(save_path)
        )

        assert fig is not None
        assert save_path.exists()
        assert save_path.stat().st_size > 0


class TestComparativeVisualization:
    """Test side-by-side model comparison visualization."""

    def test_visualize_model_comparison(self, small_state):
        """Comparison figure has one panel per state per metric."""
        state1, _ = small_state

        config2 = TransformerConfig(vocab_size=100, d_model=128, n_heads=8, n_layers=3)
        model2 = Transformer(config2)
        model2.eval()
        input_ids = torch.randint(0, config2.vocab_size, (1, 8))
        with torch.no_grad():
            _, state2 = model2(input_ids, return_state=True)

        metrics = ['attention_entropy', 'residual_norm']
        fig = visualize_model_comparison(
            [state1, state2],
            ['Model 1 (small)', 'Model 2 (large)'],
            metrics=metrics,
        )

        n_rows, n_cols = fig.axes[0].get_gridspec().get_geometry()
        assert (n_rows, n_cols) == (len(metrics), 2)


class TestColorSchemes:
    """Test color scheme utilities."""

    def test_get_scheme_colorblind_safe(self):
        """Named scheme resolves to a non-empty color dict."""
        scheme = get_scheme('colorblind_safe')
        assert isinstance(scheme, dict)
        assert len(scheme) > 0
        assert all(isinstance(v, str) for v in scheme.values())

    def test_create_gradient_length(self):
        """Gradient has exactly the requested number of stops."""
        gradient = create_gradient('#FF0000', '#0000FF', 10)
        assert len(gradient) == 10


class TestHtmlReport:
    """Test HTML report generation."""

    def test_generate_html_report(self, small_state, tmp_path):
        """Report is written to the requested path."""
        _, config = small_state

        metrics = ReportMetrics(
            timestamp="2025-10-03T10:00:00",
            model_config={
                'vocab_size': config.vocab_size,
                'd_model': config.d_model,
                'n_heads': config.n_heads,
                'n_layers': config.n_layers,
            },
            test_results={'total': 10, 'passed': 10, 'failed': 0},
        )

        path = generate_html_report(
            metrics, output_path=str(tmp_path / "test_report.html")
        )

        assert str(tmp_path / "test_report.html") in str(path)
        assert (tmp_path / "test_report.html").stat().st_size > 0


class TestComparisonReport:
    """Test multi-run comparison report generation."""

    def test_generate_comparison_report(self, tmp_path):
        """Comparison report is written for two runs and two metrics."""
        run1 = ReportMetrics(
            timestamp="2025-10-03T09:00:00",
            model_config={'d_model': 64, 'n_layers': 2},
            test_results={'total': 10, 'passed': 10},
        )
        run2 = ReportMetrics(
            timestamp="2025-10-03T10:00:00",
            model_config={'d_model': 128, 'n_layers': 3},
            test_results={'total': 10, 'passed': 9},
        )

        path = generate_comparison_report(
            [run1, run2],
            ['test_results.total', 'test_results.passed'],
            save_path=str(tmp_path / "test_comparison.html"),
        )

        assert str(tmp_path / "test_comparison.html") in str(path)
        assert (tmp_path / "test_comparison.html").stat().st_size > 0


class TestTrendAnalysis:
    """Test trend analysis over metric history."""

    def test_analyze_trends_direction(self):
        """Steadily decreasing loss is classified as degrading."""
        history = [
            {'timestamp': f'2025-10-{i:02d}', 'loss': 2.0 - i * 0.1}
            for i in range(1, 11)
        ]

        trends = analyze_trends(history, 'loss')

        assert trends['direction'] == 'degrading'
        assert trends['rate_of_change'] < 0
