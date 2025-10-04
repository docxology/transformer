"""Comprehensive reporting and auto-summarization utilities."""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import torch


@dataclass
class ReportMetrics:
    """Metrics for a single run."""
    timestamp: str
    model_config: Dict[str, Any]
    test_results: Optional[Dict] = None
    performance_metrics: Optional[Dict] = None
    visualization_files: Optional[List[str]] = None
    errors: Optional[List[str]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self, path: Path):
        """Save as JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class StructuredOutputManager:
    """Manages structured output directory with subfolders."""
    
    def __init__(self, base_dir: str = "outputs"):
        self.base_dir = Path(base_dir)
        self.subdirs = {
            'logs': self.base_dir / 'logs',
            'data': self.base_dir / 'data',
            'analysis': self.base_dir / 'analysis',
            'visualizations': self.base_dir / 'visualizations',
            'animations': self.base_dir / 'animations',
            'reports': self.base_dir / 'reports',
            'configs': self.base_dir / 'configs'
        }
        self._create_structure()
    
    def _create_structure(self):
        """Create all subdirectories."""
        self.base_dir.mkdir(exist_ok=True)
        for subdir in self.subdirs.values():
            subdir.mkdir(exist_ok=True)
    
    def get_path(self, category: str, filename: str) -> Path:
        """Get full path for a file in a category."""
        if category not in self.subdirs:
            raise ValueError(f"Unknown category: {category}")
        return self.subdirs[category] / filename
    
    def list_files(self, category: Optional[str] = None) -> Dict[str, List[str]]:
        """List files in categories."""
        if category:
            return {category: [f.name for f in self.subdirs[category].iterdir()]}
        
        return {
            cat: [f.name for f in subdir.iterdir() if f.is_file()]
            for cat, subdir in self.subdirs.items()
        }
    
    def get_summary(self) -> Dict:
        """Get summary of output structure."""
        summary = {
            'base_dir': str(self.base_dir),
            'created': datetime.now().isoformat(),
            'categories': {}
        }
        
        for category, subdir in self.subdirs.items():
            files = list(subdir.iterdir())
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            
            summary['categories'][category] = {
                'path': str(subdir),
                'file_count': len([f for f in files if f.is_file()]),
                'total_size_mb': round(total_size / 1024 / 1024, 2)
            }
        
        return summary


class AutoSummarizer:
    """Automatically generates comprehensive summaries."""
    
    def __init__(self, output_manager: StructuredOutputManager):
        self.output = output_manager
        self.metrics_history = []
    
    def log_run(self, metrics: ReportMetrics):
        """Log a run's metrics."""
        self.metrics_history.append(metrics)
        
        # Save individual run report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics.to_json(self.output.get_path('reports', f'run_{timestamp}.json'))
    
    def generate_summary(self, save: bool = True) -> str:
        """Generate comprehensive summary."""
        summary_lines = [
            "=" * 70,
            "COMPREHENSIVE EXECUTION SUMMARY",
            "=" * 70,
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Runs: {len(self.metrics_history)}",
            ""
        ]
        
        if self.metrics_history:
            latest = self.metrics_history[-1]
            
            summary_lines.extend([
                "LATEST RUN",
                "-" * 70,
                f"Timestamp: {latest.timestamp}",
                ""
            ])
            
            if latest.model_config:
                summary_lines.extend([
                    "Model Configuration:",
                    f"  Vocabulary: {latest.model_config.get('vocab_size', 'N/A'):,}",
                    f"  Dimension: {latest.model_config.get('d_model', 'N/A')}",
                    f"  Heads: {latest.model_config.get('n_heads', 'N/A')}",
                    f"  Layers: {latest.model_config.get('n_layers', 'N/A')}",
                    ""
                ])
            
            if latest.test_results:
                summary_lines.extend([
                    "Test Results:",
                    f"  Total: {latest.test_results.get('total', 0)}",
                    f"  Passed: {latest.test_results.get('passed', 0)}",
                    f"  Failed: {latest.test_results.get('failed', 0)}",
                    f"  Duration: {latest.test_results.get('duration', 0):.2f}s",
                    ""
                ])
            
            if latest.performance_metrics:
                summary_lines.extend([
                    "Performance Metrics:",
                ])
                for key, value in latest.performance_metrics.items():
                    summary_lines.append(f"  {key}: {value}")
                summary_lines.append("")
            
            if latest.visualization_files:
                summary_lines.extend([
                    "Generated Files:",
                    f"  Visualizations: {len(latest.visualization_files)}",
                    ""
                ])
        
        # Output structure summary
        structure = self.output.get_summary()
        summary_lines.extend([
            "OUTPUT STRUCTURE",
            "-" * 70,
        ])
        
        total_size = 0
        total_files = 0
        for category, info in structure['categories'].items():
            summary_lines.append(
                f"  {category:15s} {info['file_count']:3d} files  "
                f"{info['total_size_mb']:6.2f} MB"
            )
            total_size += info['total_size_mb']
            total_files += info['file_count']
        
        summary_lines.extend([
            "  " + "-" * 40,
            f"  {'TOTAL':15s} {total_files:3d} files  {total_size:6.2f} MB",
            "",
            "=" * 70
        ])
        
        summary_text = "\n".join(summary_lines)
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_path = self.output.get_path('reports', f'summary_{timestamp}.txt')
            with open(summary_path, 'w') as f:
                f.write(summary_text)
        
        return summary_text
    
    def generate_markdown_report(self, title: str = "Execution Report") -> str:
        """Generate markdown-formatted report."""
        lines = [
            f"# {title}",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            ""
        ]
        
        if self.metrics_history:
            latest = self.metrics_history[-1]
            
            lines.extend([
                "### Model Configuration",
                "",
                "| Parameter | Value |",
                "|-----------|-------|"
            ])
            
            if latest.model_config:
                for key, value in latest.model_config.items():
                    lines.append(f"| {key} | {value} |")
            
            lines.append("")
            
            if latest.test_results:
                lines.extend([
                    "### Test Results",
                    "",
                    f"- **Total Tests**: {latest.test_results.get('total', 0)}",
                    f"- **Passed**: {latest.test_results.get('passed', 0)} ✅",
                    f"- **Failed**: {latest.test_results.get('failed', 0)}",
                    f"- **Duration**: {latest.test_results.get('duration', 0):.2f}s",
                    ""
                ])
        
        # Add output structure
        structure = self.output.get_summary()
        lines.extend([
            "## Output Structure",
            "",
            "| Category | Files | Size (MB) |",
            "|----------|-------|-----------|"
        ])
        
        for category, info in structure['categories'].items():
            lines.append(
                f"| {category} | {info['file_count']} | {info['total_size_mb']:.2f} |"
            )
        
        lines.append("")
        
        # Add file listings
        lines.extend([
            "## Generated Files",
            ""
        ])
        
        file_list = self.output.list_files()
        for category, files in file_list.items():
            if files:
                lines.extend([
                    f"### {category.title()}",
                    ""
                ])
                for filename in sorted(files):
                    lines.append(f"- `{filename}`")
                lines.append("")
        
        return "\n".join(lines)
    
    def save_markdown_report(self, filename: str = "REPORT.md"):
        """Save markdown report."""
        report = self.generate_markdown_report()
        path = self.output.get_path('reports', filename)
        with open(path, 'w') as f:
            f.write(report)
        return path


def validate_outputs(output_manager: StructuredOutputManager) -> Dict[str, Any]:
    """Validate all outputs are real (not mock/fake)."""
    validation = {
        'timestamp': datetime.now().isoformat(),
        'checks': {},
        'all_valid': True
    }
    
    # Check visualizations are real image files
    vis_files = list(output_manager.subdirs['visualizations'].glob('*.png'))
    for vis_file in vis_files:
        size = vis_file.stat().st_size
        # Real images should be > 1KB
        is_valid = size > 1024
        validation['checks'][vis_file.name] = {
            'type': 'visualization',
            'size_bytes': size,
            'valid': is_valid,
            'reason': 'Real image file' if is_valid else 'File too small'
        }
        if not is_valid:
            validation['all_valid'] = False
    
    # Check animations
    anim_files = list(output_manager.subdirs['animations'].glob('*.gif'))
    for anim_file in anim_files:
        size = anim_file.stat().st_size
        # Real GIFs should be > 10KB
        is_valid = size > 10240
        validation['checks'][anim_file.name] = {
            'type': 'animation',
            'size_bytes': size,
            'valid': is_valid,
            'reason': 'Real animation' if is_valid else 'File too small'
        }
        if not is_valid:
            validation['all_valid'] = False
    
    # Check data files contain actual data
    data_files = list(output_manager.subdirs['data'].glob('*.json'))
    for data_file in data_files:
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
            is_valid = len(data) > 0
            validation['checks'][data_file.name] = {
                'type': 'data',
                'entries': len(data) if isinstance(data, (list, dict)) else 1,
                'valid': is_valid,
                'reason': 'Contains data' if is_valid else 'Empty file'
            }
            if not is_valid:
                validation['all_valid'] = False
        except Exception as e:
            validation['checks'][data_file.name] = {
                'type': 'data',
                'valid': False,
                'reason': f'Error: {str(e)}'
            }
            validation['all_valid'] = False
    
    return validation


