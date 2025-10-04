"""Configuration management for transformer models."""

import json
import yaml
from pathlib import Path
from typing import Dict, Optional, Union
from dataclasses import asdict
from src.transformer import TransformerConfig


class ConfigManager:
    """
    Manages configuration saving, loading, and validation.
    
    Supports JSON and YAML formats.
    """
    
    @staticmethod
    def save(
        config: TransformerConfig,
        path: Union[str, Path],
        format: str = 'auto'
    ):
        """
        Save configuration to file.
        
        Args:
            config: TransformerConfig object
            path: Path to save to
            format: 'json', 'yaml', or 'auto' (infer from extension)
        """
        path = Path(path)
        
        # Infer format if auto
        if format == 'auto':
            if path.suffix == '.yaml' or path.suffix == '.yml':
                format = 'yaml'
            else:
                format = 'json'
        
        # Convert to dict
        config_dict = asdict(config)
        
        # Save
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif format == 'yaml':
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    @staticmethod
    def load(
        path: Union[str, Path],
        format: str = 'auto'
    ) -> TransformerConfig:
        """
        Load configuration from file.
        
        Args:
            path: Path to load from
            format: 'json', 'yaml', or 'auto'
            
        Returns:
            TransformerConfig object
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        # Infer format
        if format == 'auto':
            if path.suffix == '.yaml' or path.suffix == '.yml':
                format = 'yaml'
            else:
                format = 'json'
        
        # Load
        if format == 'json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        elif format == 'yaml':
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        # Create config
        return TransformerConfig(**config_dict)
    
    @staticmethod
    def validate(config: TransformerConfig) -> bool:
        """
        Validate configuration.
        
        Args:
            config: TransformerConfig to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Check d_model divisible by n_heads
        if config.d_model % config.n_heads != 0:
            raise ValueError(
                f"d_model ({config.d_model}) must be divisible by "
                f"n_heads ({config.n_heads})"
            )
        
        # Check positive values
        if config.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if config.d_model <= 0:
            raise ValueError("d_model must be positive")
        if config.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if config.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if config.d_ff <= 0:
            raise ValueError("d_ff must be positive")
        
        # Check dropout range
        if not 0 <= config.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        
        return True
    
    @staticmethod
    def to_dict(config: TransformerConfig) -> Dict:
        """Convert config to dictionary."""
        return asdict(config)
    
    @staticmethod
    def from_dict(config_dict: Dict) -> TransformerConfig:
        """Create config from dictionary."""
        return TransformerConfig(**config_dict)


def save_config(config: TransformerConfig, path: Union[str, Path], format: str = 'auto'):
    """Convenience function to save config."""
    ConfigManager.save(config, path, format)


def load_config(path: Union[str, Path], format: str = 'auto') -> TransformerConfig:
    """Convenience function to load config."""
    return ConfigManager.load(path, format)


