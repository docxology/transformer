"""Tests for configuration management."""

import pytest
import tempfile
import os
from pathlib import Path

from src.transformer import TransformerConfig
from src.config import (
    ConfigManager,
    save_config,
    load_config,
    get_preset_config,
    list_presets
)


class TestConfigManager:
    """Test configuration manager."""
    
    def test_save_load_json(self):
        """Test saving and loading JSON config."""
        config = TransformerConfig(
            vocab_size=1000,
            d_model=256,
            n_heads=8,
            n_layers=4
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'config.json'
            
            # Save
            ConfigManager.save(config, path, format='json')
            assert path.exists()
            
            # Load
            loaded_config = ConfigManager.load(path, format='json')
            
            # Compare
            assert loaded_config.vocab_size == config.vocab_size
            assert loaded_config.d_model == config.d_model
            assert loaded_config.n_heads == config.n_heads
            assert loaded_config.n_layers == config.n_layers
    
    def test_save_load_yaml(self):
        """Test saving and loading YAML config."""
        config = TransformerConfig(
            vocab_size=2000,
            d_model=512,
            n_heads=16,
            n_layers=12
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'config.yaml'
            
            # Save
            ConfigManager.save(config, path, format='yaml')
            assert path.exists()
            
            # Load
            loaded_config = ConfigManager.load(path, format='yaml')
            
            assert loaded_config.vocab_size == config.vocab_size
            assert loaded_config.d_model == config.d_model
    
    def test_auto_format_detection(self):
        """Test automatic format detection."""
        config = TransformerConfig(vocab_size=500)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # JSON
            json_path = Path(tmpdir) / 'config.json'
            ConfigManager.save(config, json_path, format='auto')
            loaded = ConfigManager.load(json_path, format='auto')
            assert loaded.vocab_size == config.vocab_size
            
            # YAML
            yaml_path = Path(tmpdir) / 'config.yaml'
            ConfigManager.save(config, yaml_path, format='auto')
            loaded = ConfigManager.load(yaml_path, format='auto')
            assert loaded.vocab_size == config.vocab_size
    
    def test_validate_valid_config(self):
        """Test validation of valid config."""
        config = TransformerConfig(
            vocab_size=1000,
            d_model=512,
            n_heads=8,
            n_layers=6
        )
        
        assert ConfigManager.validate(config) is True
    
    def test_validate_invalid_d_model_n_heads(self):
        """Test validation catches d_model not divisible by n_heads."""
        config = TransformerConfig(
            vocab_size=1000,
            d_model=512,
            n_heads=7,  # 512 not divisible by 7
            n_layers=6
        )
        
        with pytest.raises(ValueError, match="divisible"):
            ConfigManager.validate(config)
    
    def test_validate_negative_values(self):
        """Test validation catches negative values."""
        # Test will fail if we try to create config with negative values
        # since dataclass may validate, but let's test the validator
        config = TransformerConfig(vocab_size=1000)
        config.d_model = -512
        
        with pytest.raises(ValueError, match="positive"):
            ConfigManager.validate(config)
    
    def test_to_dict_from_dict(self):
        """Test conversion to/from dictionary."""
        config = TransformerConfig(
            vocab_size=1000,
            d_model=256,
            n_heads=8
        )
        
        # To dict
        config_dict = ConfigManager.to_dict(config)
        assert isinstance(config_dict, dict)
        assert config_dict['vocab_size'] == 1000
        assert config_dict['d_model'] == 256
        
        # From dict
        new_config = ConfigManager.from_dict(config_dict)
        assert new_config.vocab_size == config.vocab_size
        assert new_config.d_model == config.d_model


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_save_load_convenience(self):
        """Test convenience save/load functions."""
        config = TransformerConfig(vocab_size=1000, d_model=256)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'config.json'
            
            save_config(config, path)
            loaded = load_config(path)
            
            assert loaded.vocab_size == config.vocab_size
            assert loaded.d_model == config.d_model


class TestPresets:
    """Test preset configurations."""
    
    def test_get_preset_config(self):
        """Test getting preset configurations."""
        presets = ['debug', 'tiny', 'small', 'base', 'large']
        
        for preset_name in presets:
            config = get_preset_config(preset_name)
            
            assert isinstance(config, TransformerConfig)
            assert config.vocab_size > 0
            assert config.d_model > 0
            assert config.n_heads > 0
            assert config.n_layers > 0
    
    def test_get_invalid_preset(self):
        """Test getting invalid preset raises error."""
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset_config('nonexistent_preset')
    
    def test_list_presets(self):
        """Test listing all presets."""
        presets = list_presets()
        
        assert isinstance(presets, dict)
        assert 'debug' in presets
        assert 'tiny' in presets
        assert 'small' in presets
        assert 'base' in presets
        
        # All values should be TransformerConfig
        for name, config in presets.items():
            assert isinstance(config, TransformerConfig)
    
    def test_preset_sizes(self):
        """Test that presets are ordered by size."""
        debug = get_preset_config('debug')
        tiny = get_preset_config('tiny')
        small = get_preset_config('small')
        base = get_preset_config('base')
        large = get_preset_config('large')
        
        # Check dimensions increase
        assert debug.d_model < tiny.d_model
        assert tiny.d_model < small.d_model
        assert small.d_model < base.d_model
        assert base.d_model < large.d_model


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


