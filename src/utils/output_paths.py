"""
Centralized output path management.

Ensures all outputs go to correct subdirectories.
"""

from pathlib import Path
from typing import Optional


class OutputPathManager:
    """Manages output directory paths."""
    
    def __init__(self, base_dir: str = "outputs"):
        self.base_dir = Path(base_dir)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create all output subdirectories."""
        subdirs = [
            'animations',
            'visualizations',
            'visualizations/training',
            'analysis',
            'data',
            'configs',
            'reports',
            'logs'
        ]
        
        for subdir in subdirs:
            (self.base_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    def get_animation_path(self, filename: str) -> str:
        """Get path for animation file."""
        return str(self.base_dir / 'animations' / filename)
    
    def get_visualization_path(self, filename: str, training: bool = False) -> str:
        """Get path for visualization file."""
        if training:
            return str(self.base_dir / 'visualizations' / 'training' / filename)
        return str(self.base_dir / 'visualizations' / filename)
    
    def get_analysis_path(self, filename: str) -> str:
        """Get path for analysis file."""
        return str(self.base_dir / 'analysis' / filename)
    
    def get_data_path(self, filename: str) -> str:
        """Get path for data file."""
        return str(self.base_dir / 'data' / filename)
    
    def get_config_path(self, filename: str) -> str:
        """Get path for config file."""
        return str(self.base_dir / 'configs' / filename)
    
    def get_report_path(self, filename: str) -> str:
        """Get path for report file."""
        return str(self.base_dir / 'reports' / filename)
    
    def get_log_path(self, filename: str) -> str:
        """Get path for log file."""
        return str(self.base_dir / 'logs' / filename)


# Global instance for easy access
_global_manager: Optional[OutputPathManager] = None


def get_output_manager(base_dir: str = "outputs") -> OutputPathManager:
    """Get or create global output path manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = OutputPathManager(base_dir)
    return _global_manager


def get_animation_path(filename: str) -> str:
    """Convenience function to get animation path."""
    return get_output_manager().get_animation_path(filename)


def get_visualization_path(filename: str, training: bool = False) -> str:
    """Convenience function to get visualization path."""
    return get_output_manager().get_visualization_path(filename, training)


def get_analysis_path(filename: str) -> str:
    """Convenience function to get analysis path."""
    return get_output_manager().get_analysis_path(filename)


def get_data_path(filename: str) -> str:
    """Convenience function to get data path."""
    return get_output_manager().get_data_path(filename)


def get_config_path(filename: str) -> str:
    """Convenience function to get config path."""
    return get_output_manager().get_config_path(filename)


def get_report_path(filename: str) -> str:
    """Convenience function to get report path."""
    return get_output_manager().get_report_path(filename)


def get_log_path(filename: str) -> str:
    """Convenience function to get log path."""
    return get_output_manager().get_log_path(filename)

