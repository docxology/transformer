"""Trend analysis for tracking metrics over time."""

import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


def analyze_trends(
    history: List[Any],
    metric_name: str,
    window_size: int = 10
) -> Dict[str, Any]:
    """
    Analyze trends in a metric over time.
    
    Args:
        history: List of ReportMetrics objects or dictionaries
        metric_name: Name of metric to analyze
        window_size: Window size for moving average
        
    Returns:
        Dictionary with trend analysis:
            - direction: 'improving', 'degrading', or 'stable'
            - rate_of_change: Average change per period
            - volatility: Standard deviation of changes
            - anomalies: List of anomalous points
            - forecast: Simple forecast for next period
            - moving_average: Moving average values
    """
    # Extract metric values
    values = []
    timestamps = []
    
    for entry in history:
        if hasattr(entry, 'to_dict'):
            data = entry.to_dict()
        else:
            data = entry
        
        # Navigate through nested structure to find metric
        value = _extract_metric_value(data, metric_name)
        if value is not None:
            values.append(value)
            timestamps.append(data.get('timestamp', ''))
    
    if len(values) < 2:
        return {
            'direction': 'unknown',
            'rate_of_change': 0.0,
            'volatility': 0.0,
            'anomalies': [],
            'forecast': None,
            'moving_average': values
        }
    
    values = np.array(values)
    
    # Compute changes
    changes = np.diff(values)
    rate_of_change = np.mean(changes)
    volatility = np.std(changes)
    
    # Determine direction
    if abs(rate_of_change) < volatility * 0.1:  # Threshold for "stable"
        direction = 'stable'
    elif rate_of_change > 0:
        direction = 'improving'
    else:
        direction = 'degrading'
    
    # Detect anomalies (values > 2 std devs from mean)
    mean_val = np.mean(values)
    std_val = np.std(values)
    anomalies = []
    for idx, val in enumerate(values):
        if abs(val - mean_val) > 2 * std_val:
            anomalies.append({
                'index': idx,
                'value': float(val),
                'timestamp': timestamps[idx] if idx < len(timestamps) else ''
            })
    
    # Moving average
    if len(values) >= window_size:
        moving_avg = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
    else:
        moving_avg = values
    
    # Simple forecast (linear extrapolation)
    if len(values) >= 3:
        # Fit simple linear trend
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        forecast = coeffs[0] * len(values) + coeffs[1]
    else:
        forecast = values[-1]
    
    return {
        'direction': direction,
        'rate_of_change': float(rate_of_change),
        'volatility': float(volatility),
        'anomalies': anomalies,
        'forecast': float(forecast),
        'moving_average': moving_avg.tolist(),
        'values': values.tolist(),
        'mean': float(mean_val),
        'std': float(std_val)
    }


def _extract_metric_value(data: Dict, metric_name: str) -> Optional[float]:
    """Extract metric value from nested dictionary."""
    # Try direct access
    if metric_name in data:
        val = data[metric_name]
        if isinstance(val, (int, float)):
            return float(val)
    
    # Try nested access
    parts = metric_name.split('.')
    current = data
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    
    if isinstance(current, (int, float)):
        return float(current)
    
    return None


def compare_trends(
    history: List[Any],
    metric_names: List[str],
    window_size: int = 10
) -> Dict[str, Any]:
    """
    Compare trends across multiple metrics.
    
    Args:
        history: List of ReportMetrics objects
        metric_names: List of metric names to compare
        window_size: Window size for moving average
        
    Returns:
        Dictionary mapping metric names to trend analyses
    """
    results = {}
    
    for metric_name in metric_names:
        results[metric_name] = analyze_trends(history, metric_name, window_size)
    
    # Add summary comparison
    improving = [m for m, r in results.items() if r['direction'] == 'improving']
    degrading = [m for m, r in results.items() if r['direction'] == 'degrading']
    stable = [m for m, r in results.items() if r['direction'] == 'stable']
    
    results['_summary'] = {
        'improving': improving,
        'degrading': degrading,
        'stable': stable,
        'total_metrics': len(metric_names)
    }
    
    return results


def generate_trend_report(
    history: List[Any],
    metric_names: List[str],
    output_path: Optional[str] = None
) -> str:
    """
    Generate markdown trend report.
    
    Args:
        history: List of ReportMetrics objects
        metric_names: Metrics to analyze
        output_path: Path to save report (optional)
        
    Returns:
        Markdown report string
    """
    trends = compare_trends(history, metric_names)
    summary = trends.pop('_summary')
    
    lines = [
        "# Trend Analysis Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Data Points**: {len(history)}",
        "",
        "## Summary",
        "",
        f"- **Improving Metrics**: {len(summary['improving'])}",
        f"- **Degrading Metrics**: {len(summary['degrading'])}",
        f"- **Stable Metrics**: {len(summary['stable'])}",
        "",
    ]
    
    # Detailed metrics
    lines.append("## Detailed Analysis")
    lines.append("")
    
    for metric_name, trend in sorted(trends.items()):
        lines.extend([
            f"### {metric_name}",
            "",
            f"- **Direction**: {trend['direction'].upper()}",
            f"- **Rate of Change**: {trend['rate_of_change']:.4f} per period",
            f"- **Volatility**: {trend['volatility']:.4f}",
            f"- **Forecast**: {trend['forecast']:.4f}",
            f"- **Anomalies**: {len(trend['anomalies'])}",
            ""
        ])
        
        if trend['anomalies']:
            lines.append("**Anomalous Points**:")
            for anom in trend['anomalies']:
                lines.append(f"- Index {anom['index']}: {anom['value']:.4f}")
            lines.append("")
    
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
    
    return report


def detect_performance_regression(
    current_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    threshold: float = 0.1
) -> Dict[str, Any]:
    """
    Detect performance regressions relative to baseline.
    
    Args:
        current_metrics: Current metric values
        baseline_metrics: Baseline metric values
        threshold: Threshold for considering a regression (10% by default)
        
    Returns:
        Dictionary with regression analysis
    """
    regressions = {}
    improvements = {}
    
    for metric, current_value in current_metrics.items():
        if metric in baseline_metrics:
            baseline_value = baseline_metrics[metric]
            
            if baseline_value == 0:
                continue
            
            # Calculate percentage change
            pct_change = (current_value - baseline_value) / baseline_value
            
            if abs(pct_change) > threshold:
                if pct_change < 0:
                    # Assuming lower is better for most metrics
                    improvements[metric] = {
                        'baseline': baseline_value,
                        'current': current_value,
                        'change_pct': pct_change * 100
                    }
                else:
                    regressions[metric] = {
                        'baseline': baseline_value,
                        'current': current_value,
                        'change_pct': pct_change * 100
                    }
    
    return {
        'regressions': regressions,
        'improvements': improvements,
        'regression_count': len(regressions),
        'improvement_count': len(improvements),
        'has_regressions': len(regressions) > 0
    }



