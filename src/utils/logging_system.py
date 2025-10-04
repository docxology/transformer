"""Comprehensive logging system for transformer operations."""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import sys


class TransformerLogger:
    """
    Comprehensive logging system for transformer operations.
    
    Provides structured logging with multiple outputs:
    - Console output (with colors)
    - File logging (timestamped)
    - JSON logging (structured data)
    - Performance metrics logging
    """
    
    def __init__(self, name: str = "transformer", log_dir: str = "outputs/logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for this session
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup loggers
        self._setup_console_logger()
        self._setup_file_logger()
        self._setup_json_logger()
        
        # Performance metrics storage
        self.metrics = []
    
    def _setup_console_logger(self):
        """Setup console logger with formatting."""
        self.console_logger = logging.getLogger(f"{self.name}_console")
        self.console_logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        self.console_logger.addHandler(console_handler)
    
    def _setup_file_logger(self):
        """Setup file logger with detailed logging."""
        self.file_logger = logging.getLogger(f"{self.name}_file")
        self.file_logger.setLevel(logging.DEBUG)
        
        # File handler
        log_file = self.log_dir / f"session_{self.session_id}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Detailed formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
            '  File: %(pathname)s:%(lineno)d',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        self.file_logger.addHandler(file_handler)
    
    def _setup_json_logger(self):
        """Setup JSON logger for structured data."""
        self.json_log_file = self.log_dir / f"session_{self.session_id}.json"
        self.json_logs = []
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.console_logger.info(message)
        self.file_logger.info(message)
        self._log_json("INFO", message, kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.file_logger.debug(message)
        self._log_json("DEBUG", message, kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.console_logger.warning(message)
        self.file_logger.warning(message)
        self._log_json("WARNING", message, kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.console_logger.error(message)
        self.file_logger.error(message)
        self._log_json("ERROR", message, kwargs)
    
    def _log_json(self, level: str, message: str, data: Dict):
        """Log to JSON file."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            'data': data
        }
        self.json_logs.append(entry)
        
        # Write to file
        with open(self.json_log_file, 'w') as f:
            json.dump(self.json_logs, f, indent=2)
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a performance metric."""
        metric = {
            'timestamp': datetime.now().isoformat(),
            'name': name,
            'value': value,
            'step': step
        }
        self.metrics.append(metric)
        
        # Also log as info
        step_str = f" (step {step})" if step is not None else ""
        self.info(f"Metric: {name} = {value}{step_str}")
    
    def log_model_info(self, model: Any):
        """Log model information."""
        try:
            num_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            self.info("Model Information", 
                     total_parameters=num_params,
                     trainable_parameters=trainable_params)
        except Exception as e:
            self.error(f"Failed to log model info: {e}")
    
    def log_config(self, config: Dict):
        """Log configuration."""
        self.info("Configuration", config=config)
    
    def save_metrics(self, path: Optional[Path] = None):
        """Save metrics to file."""
        if path is None:
            path = self.log_dir / f"metrics_{self.session_id}.json"
        
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        self.info(f"Metrics saved to {path}")
    
    def get_session_summary(self) -> Dict:
        """Get summary of current session."""
        return {
            'session_id': self.session_id,
            'total_logs': len(self.json_logs),
            'total_metrics': len(self.metrics),
            'log_levels': {
                level: sum(1 for log in self.json_logs if log['level'] == level)
                for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR']
            }
        }


