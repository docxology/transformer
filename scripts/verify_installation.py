"""
Quick verification script to test the transformer installation.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.transformer import Transformer, TransformerConfig
from src.visualization import compute_path_counts, analyze_path_complexity


def main():
    print("=" * 70)
    print("First-Principles Transformer - Installation Verification")
    print("=" * 70)
    
    # Test 1: Create model
    print("\n[1/5] Creating transformer model...")
    config = TransformerConfig(
        vocab_size=1000,
        d_model=128,
        n_heads=4,
        n_layers=3,
        d_ff=512,
        dropout=0.1
    )
    model = Transformer(config)
    print("✓ Model created successfully")
    
    # Test 2: Forward pass
    print("\n[2/5] Testing forward pass...")
    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    model.eval()
    with torch.no_grad():
        logits, state = model(input_ids, return_state=True)
    
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {logits.shape}")
    
    # Test 3: State tracking
    print("\n[3/5] Testing state tracking...")
    assert state is not None, "State tracking failed"
    residuals = state.get_residual_stream()
    assert len(residuals) == config.n_layers + 1, "Residual stream extraction failed"
    
    for layer_idx in range(config.n_layers):
        attn_weights = state.get_attention_weights(layer_idx)
        assert attn_weights.shape == (batch_size, config.n_heads, seq_len, seq_len)
    
    print("✓ State tracking working correctly")
    print(f"  Residual stream checkpoints: {len(residuals)}")
    print(f"  Attention weights tracked for {config.n_layers} layers")
    
    # Test 4: Path analysis
    print("\n[4/5] Testing path analysis...")
    path_count = compute_path_counts(layer_displacement=3, position_displacement=5)
    print(f"✓ Path analysis working")
    print(f"  Paths for (3 layers, 5 positions): {path_count:,}")
    
    # Test 5: Generation
    print("\n[5/5] Testing autoregressive generation...")
    initial_seq = torch.randint(0, config.vocab_size, (1, 5))
    with torch.no_grad():
        generated = model.generate(initial_seq, max_new_tokens=3, temperature=1.0)
    
    print(f"✓ Generation working")
    print(f"  Initial length: {initial_seq.shape[1]}")
    print(f"  Generated length: {generated.shape[1]}")
    
    # Summary
    print("\n" + "=" * 70)
    print("All Systems Operational!")
    print("=" * 70)
    
    print("\nKey Insights:")
    print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Information highways: 2 (residual + K/V)")
    print(f"  - Path explosion: C({config.n_layers-1}+{seq_len-1}, {config.n_layers-1})")
    print(f"    = {compute_path_counts(config.n_layers-1, seq_len-1):,} paths")
    
    print("\nNext Steps:")
    print("  1. Run: python3 examples/basic_usage.py")
    print("  2. Run: python3 examples/path_analysis.py")
    print("  3. Run: python3 examples/animation_demo.py")
    print("  4. Explore: jupyter notebook (if installed)")
    
    print("\nDocumentation:")
    print("  - README.md: Overview and installation")
    print("  - ARCHITECTURE.md: Information flow principles")
    print("  - examples/README.md: Usage examples")


if __name__ == '__main__':
    main()


