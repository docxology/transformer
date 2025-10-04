"""HTML report generation with embedded visualizations."""

import base64
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import json


def generate_html_report(
    run_data: Any,
    output_path: str = 'outputs/reports/report.html',
    template: str = 'comprehensive',
    dark_mode: bool = False
) -> Path:
    """
    Generate rich HTML report with embedded visualizations.
    
    Args:
        run_data: ReportMetrics object or dictionary
        output_path: Path to save HTML report
        template: Template style ('comprehensive', 'compact', 'detailed')
        dark_mode: Whether to use dark mode styling
        
    Returns:
        Path to saved report
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert run_data to dict if needed
    if hasattr(run_data, 'to_dict'):
        data = run_data.to_dict()
    else:
        data = run_data
    
    # Generate HTML
    html = _generate_html_content(data, template, dark_mode)
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return output_path


def _generate_html_content(
    data: Dict,
    template: str,
    dark_mode: bool
) -> str:
    """Generate complete HTML content."""
    
    # Get styles
    styles = _get_styles(dark_mode)
    
    # Build sections
    header = _build_header(data)
    summary = _build_summary_section(data)
    config = _build_config_section(data)
    results = _build_results_section(data)
    visualizations = _build_visualization_section(data)
    footer = _build_footer()
    
    # Combine into full HTML
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Transformer Analysis Report</title>
    <style>
        {styles}
    </style>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div class="container">
        {header}
        {summary}
        {config}
        {results}
        {visualizations}
        {footer}
    </div>
    
    <script>
        {_get_javascript()}
    </script>
</body>
</html>
"""
    
    return html


def _get_styles(dark_mode: bool) -> str:
    """Get CSS styles."""
    
    if dark_mode:
        bg_color = '#1a1a1a'
        text_color = '#e0e0e0'
        card_bg = '#2d2d2d'
        border_color = '#444'
        accent_color = '#4CAF50'
    else:
        bg_color = '#f5f5f5'
        text_color = '#333'
        card_bg = '#ffffff'
        border_color = '#ddd'
        accent_color = '#1976D2'
    
    return f"""
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: {bg_color};
            color: {text_color};
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: linear-gradient(135deg, {accent_color}, #2196F3);
            color: white;
            padding: 40px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .section {{
            background-color: {card_bg};
            padding: 30px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border: 1px solid {border_color};
        }}
        
        .section h2 {{
            color: {accent_color};
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid {accent_color};
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .card {{
            background-color: {bg_color};
            padding: 20px;
            border-radius: 6px;
            border: 1px solid {border_color};
        }}
        
        .card h3 {{
            color: {accent_color};
            margin-bottom: 10px;
        }}
        
        .metric {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid {border_color};
        }}
        
        .metric:last-child {{
            border-bottom: none;
        }}
        
        .metric-label {{
            font-weight: 500;
        }}
        
        .metric-value {{
            font-weight: bold;
            color: {accent_color};
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid {border_color};
        }}
        
        th {{
            background-color: {accent_color};
            color: white;
            font-weight: 600;
        }}
        
        tr:hover {{
            background-color: {bg_color};
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        
        .badge-success {{
            background-color: #4CAF50;
            color: white;
        }}
        
        .badge-warning {{
            background-color: #FF9800;
            color: white;
        }}
        
        .badge-error {{
            background-color: #F44336;
            color: white;
        }}
        
        .footer {{
            text-align: center;
            padding: 20px;
            margin-top: 40px;
            border-top: 1px solid {border_color};
            color: #888;
        }}
        
        .collapsible {{
            cursor: pointer;
            padding: 10px;
            background-color: {accent_color};
            color: white;
            border: none;
            text-align: left;
            outline: none;
            font-size: 1em;
            border-radius: 4px;
            margin: 10px 0;
            width: 100%;
        }}
        
        .collapsible:hover {{
            opacity: 0.9;
        }}
        
        .content {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
            background-color: {card_bg};
            border: 1px solid {border_color};
            border-top: none;
            border-radius: 0 0 4px 4px;
        }}
        
        .content-inner {{
            padding: 20px;
        }}
        
        .visualization-container {{
            margin: 20px 0;
            text-align: center;
        }}
        
        .visualization-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid {border_color};
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        code {{
            background-color: {bg_color};
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}
        
        pre {{
            background-color: {bg_color};
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
            border: 1px solid {border_color};
        }}
        
        @media print {{
            .collapsible {{
                display: none;
            }}
            .content {{
                max-height: none !important;
            }}
        }}
    """


def _build_header(data: Dict) -> str:
    """Build header section."""
    timestamp = data.get('timestamp', datetime.now().isoformat())
    
    return f"""
    <div class="header">
        <h1>🤖 Transformer Analysis Report</h1>
        <div class="subtitle">
            Generated: {timestamp}
        </div>
    </div>
    """


def _build_summary_section(data: Dict) -> str:
    """Build summary section."""
    config = data.get('model_config') or {}
    test_results = data.get('test_results') or {}
    perf = data.get('performance_metrics') or {}
    
    return f"""
    <div class="section">
        <h2>📊 Executive Summary</h2>
        <div class="grid">
            <div class="card">
                <h3>Model Configuration</h3>
                <div class="metric">
                    <span class="metric-label">Vocabulary Size:</span>
                    <span class="metric-value">{config.get('vocab_size', 'N/A'):,}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Model Dimension:</span>
                    <span class="metric-value">{config.get('d_model', 'N/A')}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Attention Heads:</span>
                    <span class="metric-value">{config.get('n_heads', 'N/A')}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Layers:</span>
                    <span class="metric-value">{config.get('n_layers', 'N/A')}</span>
                </div>
            </div>
            
            <div class="card">
                <h3>Test Results</h3>
                <div class="metric">
                    <span class="metric-label">Total Tests:</span>
                    <span class="metric-value">{test_results.get('total', 'N/A')}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Passed:</span>
                    <span class="metric-value">
                        <span class="badge badge-success">{test_results.get('passed', 'N/A')}</span>
                    </span>
                </div>
                <div class="metric">
                    <span class="metric-label">Failed:</span>
                    <span class="metric-value">
                        <span class="badge badge-error">{test_results.get('failed', 0)}</span>
                    </span>
                </div>
                <div class="metric">
                    <span class="metric-label">Duration:</span>
                    <span class="metric-value">{test_results.get('duration', 0):.2f}s</span>
                </div>
            </div>
        </div>
    </div>
    """


def _build_config_section(data: Dict) -> str:
    """Build configuration details section."""
    config = data.get('model_config') or {}
    
    if not config:
        return """
        <div class="section">
            <h2>⚙️ Configuration Details</h2>
            <p>No configuration data available.</p>
        </div>
        """
    
    config_rows = ''.join([
        f"<tr><td><code>{key}</code></td><td>{value}</td></tr>"
        for key, value in config.items()
    ])
    
    return f"""
    <div class="section">
        <h2>⚙️ Configuration Details</h2>
        <table>
            <thead>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
                {config_rows}
            </tbody>
        </table>
    </div>
    """


def _build_results_section(data: Dict) -> str:
    """Build results section."""
    perf = data.get('performance_metrics') or {}
    
    perf_cards = ''
    for batch_name, metrics in perf.items():
        if isinstance(metrics, dict):
            perf_cards += f"""
            <div class="card">
                <h3>{batch_name.replace('_', ' ').title()}</h3>
                <div class="metric">
                    <span class="metric-label">Mean Time:</span>
                    <span class="metric-value">{metrics.get('mean_time_ms', 0):.2f} ms</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Throughput:</span>
                    <span class="metric-value">{metrics.get('throughput', 0):.2f} samples/s</span>
                </div>
            </div>
            """
    
    return f"""
    <div class="section">
        <h2>📈 Performance Metrics</h2>
        <div class="grid">
            {perf_cards}
        </div>
    </div>
    """


def _build_visualization_section(data: Dict) -> str:
    """Build visualization section."""
    viz_files = data.get('visualization_files', [])
    
    if not viz_files:
        return ""
    
    viz_html = ""
    for viz_file in viz_files[:6]:  # Show first 6
        try:
            # Try to embed image as base64
            with open(viz_file, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode()
            
            filename = Path(viz_file).name
            viz_html += f"""
            <div class="visualization-container">
                <h3>{filename}</h3>
                <img src="data:image/png;base64,{img_data}" alt="{filename}">
            </div>
            """
        except:
            pass
    
    return f"""
    <div class="section">
        <h2>🎨 Visualizations</h2>
        {viz_html}
    </div>
    """


def _build_footer() -> str:
    """Build footer section."""
    return f"""
    <div class="footer">
        <p>Generated by First-Principles Transformer Analysis System</p>
        <p>Report generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    """


def _get_javascript() -> str:
    """Get JavaScript for interactive elements."""
    return """
        // Collapsible sections
        var coll = document.getElementsByClassName("collapsible");
        for (var i = 0; i < coll.length; i++) {
            coll[i].addEventListener("click", function() {
                this.classList.toggle("active");
                var content = this.nextElementSibling;
                if (content.style.maxHeight) {
                    content.style.maxHeight = null;
                } else {
                    content.style.maxHeight = content.scrollHeight + "px";
                }
            });
        }
        
        // Print functionality
        function printReport() {
            window.print();
        }
    """



