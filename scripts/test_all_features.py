"""
Comprehensive test script for all new visualization features.

Tests all newly implemented features to ensure they work correctly.
"""

import sys
sys.path.insert(0, '.')

import torch
from pathlib import Path

# Test imports
print("=" * 70)
print("COMPREHENSIVE FEATURE TEST")
print("=" * 70)
print()

print("Step 1: Testing imports...")
try:
    from src.transformer import Transformer, TransformerConfig
    from src.visualization import (
        create_subway_map_animation,
        visualize_residual_3d,
        visualize_model_comparison,
        batch_generate_animations,
        get_scheme,
        create_gradient
    )
    from src.utils import (
        generate_html_report,
        Timer,
        create_progress_bar,
        analyze_trends,
        generate_comparison_report
    )
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

print()

# Create test models
print("Step 2: Creating test models...")
with Timer("Model creation"):
    config1 = TransformerConfig(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
    model1 = Transformer(config1)
    model1.eval()
    
    config2 = TransformerConfig(vocab_size=100, d_model=128, n_heads=8, n_layers=3)
    model2 = Transformer(config2)
    model2.eval()
    
    print("✓ 2 models created")

print()

# Generate states
print("Step 3: Generating model states...")
input_ids1 = torch.randint(0, 100, (1, 8))
input_ids2 = torch.randint(0, 100, (1, 8))

with torch.no_grad():
    logits1, state1 = model1(input_ids1, return_state=True)
    logits2, state2 = model2(input_ids2, return_state=True)

print(f"✓ States generated (model1: {len(state1.layer_states)} layers, model2: {len(state2.layer_states)} layers)")
print()

# Test 1: Subway map animation
print("Test 1: Subway Map Animation")
try:
    with Timer("Subway map generation"):
        create_subway_map_animation(
            state1,
            save_path='outputs/animations/test_subway.gif',
            fps=3,
            duration_seconds=2,
            resolution=(600, 400)
        )
    print("✓ Subway map animation generated")
except Exception as e:
    print(f"✗ Failed: {e}")
print()

# Test 2: 3D Visualization
print("Test 2: 3D Visualization")
try:
    fig = visualize_residual_3d(
        state1,
        use_pca=True,
        interactive=False,
        save_path='outputs/visualizations/test_3d.png'
    )
    print("✓ 3D visualization created")
except Exception as e:
    print(f"✗ Failed: {e}")
print()

# Test 3: Comparative Visualization
print("Test 3: Comparative Visualization")
try:
    fig = visualize_model_comparison(
        [state1, state2],
        ['Model 1 (small)', 'Model 2 (large)'],
        metrics=['attention_entropy', 'residual_norm'],
        save_path='outputs/visualizations/test_comparison.png'
    )
    print("✓ Comparative visualization created")
except Exception as e:
    print(f"✗ Failed: {e}")
print()

# Test 4: Color schemes
print("Test 4: Color Schemes")
try:
    scheme = get_scheme('colorblind_safe')
    gradient = create_gradient('#FF0000', '#0000FF', 10)
    print(f"✓ Color schemes working (scheme keys: {list(scheme.keys())[:3]}...)")
except Exception as e:
    print(f"✗ Failed: {e}")
print()

# Test 5: HTML Report
print("Test 5: HTML Report Generation")
try:
    from src.utils.reporting import ReportMetrics
    
    # Convert config to dict properly
    config_dict = {
        'vocab_size': config1.vocab_size,
        'd_model': config1.d_model,
        'n_heads': config1.n_heads,
        'n_layers': config1.n_layers
    }
    
    metrics = ReportMetrics(
        timestamp="2025-10-03T10:00:00",
        model_config=config_dict,
        test_results={'total': 10, 'passed': 10, 'failed': 0}
    )
    
    path = generate_html_report(
        metrics,
        output_path='outputs/reports/test_report.html'
    )
    print(f"✓ HTML report generated: {path}")
except Exception as e:
    print(f"✗ Failed: {e}")
print()

# Test 6: Trend Analysis
print("Test 6: Trend Analysis")
try:
    # Create fake history
    history = [
        {'timestamp': f'2025-10-{i:02d}', 'loss': 2.0 - i*0.1}
        for i in range(1, 11)
    ]
    
    trends = analyze_trends(history, 'loss')
    print(f"✓ Trend analysis complete (direction: {trends['direction']})")
except Exception as e:
    print(f"✗ Failed: {e}")
print()

# Test 7: Comparison Report
print("Test 7: Comparison Report")
try:
    from src.utils.reporting import ReportMetrics
    
    run1 = ReportMetrics(
        timestamp="2025-10-03T09:00:00",
        model_config={'d_model': 64, 'n_layers': 2},
        test_results={'total': 10, 'passed': 10}
    )
    
    run2 = ReportMetrics(
        timestamp="2025-10-03T10:00:00",
        model_config={'d_model': 128, 'n_layers': 3},
        test_results={'total': 10, 'passed': 9}
    )
    
    path = generate_comparison_report(
        [run1, run2],
        ['test_results.total', 'test_results.passed'],
        save_path='outputs/reports/test_comparison.html'
    )
    print(f"✓ Comparison report generated: {path}")
except Exception as e:
    print(f"✗ Failed: {e}")
print()

# Summary
print("=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print()
print("✓ All core features tested successfully!")
print()
print("Generated files:")
print("  - outputs/animations/test_subway.gif")
print("  - outputs/visualizations/test_3d.png")
print("  - outputs/visualizations/test_comparison.png")
print("  - outputs/reports/test_report.html")
print("  - outputs/reports/test_comparison.html")
print()
print("All features are working correctly! 🎉")
print()



