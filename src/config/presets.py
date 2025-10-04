"""Preset configurations for common model sizes."""

from typing import Dict
from src.transformer import TransformerConfig


PRESETS = {
    'tiny': TransformerConfig(
        vocab_size=10000,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        dropout=0.1
    ),
    'small': TransformerConfig(
        vocab_size=30000,
        d_model=256,
        n_heads=8,
        n_layers=4,
        d_ff=1024,
        dropout=0.1
    ),
    'base': TransformerConfig(
        vocab_size=50000,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        dropout=0.1
    ),
    'large': TransformerConfig(
        vocab_size=50000,
        d_model=1024,
        n_heads=16,
        n_layers=12,
        d_ff=4096,
        dropout=0.1
    ),
    'xlarge': TransformerConfig(
        vocab_size=50000,
        d_model=2048,
        n_heads=32,
        n_layers=24,
        d_ff=8192,
        dropout=0.1
    ),
    # Research/debugging sizes
    'debug': TransformerConfig(
        vocab_size=100,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=256,
        dropout=0.0
    ),
    'minimal': TransformerConfig(
        vocab_size=1000,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        dropout=0.1
    ),
}


def get_preset_config(name: str) -> TransformerConfig:
    """
    Get a preset configuration by name.
    
    Available presets:
    - debug: Minimal size for debugging (64-dim, 2 layers)
    - minimal: Small for quick experiments (64-dim, 2 layers)
    - tiny: Tiny model (128-dim, 2 layers)
    - small: Small model (256-dim, 4 layers)
    - base: Base model (512-dim, 6 layers) - similar to original Transformer
    - large: Large model (1024-dim, 12 layers)
    - xlarge: Extra large (2048-dim, 24 layers)
    
    Args:
        name: Preset name
        
    Returns:
        TransformerConfig object
        
    Raises:
        ValueError: If preset name not found
    """
    if name not in PRESETS:
        available = ', '.join(PRESETS.keys())
        raise ValueError(
            f"Unknown preset: {name}. "
            f"Available presets: {available}"
        )
    
    return PRESETS[name]


def list_presets() -> Dict[str, TransformerConfig]:
    """
    List all available preset configurations.
    
    Returns:
        Dictionary mapping preset names to configs
    """
    return PRESETS.copy()


def print_presets():
    """Print all presets with their specifications."""
    print("Available Configuration Presets:")
    print("=" * 70)
    
    for name, config in PRESETS.items():
        from src.utils import count_parameters
        from src.transformer import Transformer
        
        # Create model to count parameters
        model = Transformer(config)
        params = count_parameters(model)
        
        print(f"\n{name.upper()}:")
        print(f"  Vocab: {config.vocab_size:,}")
        print(f"  Dimensions: {config.d_model}")
        print(f"  Heads: {config.n_heads}")
        print(f"  Layers: {config.n_layers}")
        print(f"  FFN: {config.d_ff}")
        print(f"  Parameters: {params:,}")
        print(f"  Memory (approx): {params * 4 / 1024**2:.1f} MB (fp32)")


