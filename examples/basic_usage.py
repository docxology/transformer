"""
Basic usage example for the transformer implementation.

Demonstrates:
1. Creating a transformer model
2. Forward pass with state tracking
3. Basic visualization of attention and information flow
"""

import torch
from src.transformer import Transformer, TransformerConfig
from src.visualization import (
    visualize_attention,
    visualize_residual_stream,
    visualize_kv_stream,
    visualize_information_highways
)
from src.utils.output_paths import get_visualization_path


def main():
    print("=" * 60)
    print("First-Principles Transformer - Basic Usage")
    print("=" * 60)
    
    # Configure model
    config = TransformerConfig(
        vocab_size=1000,
        d_model=256,
        n_heads=8,
        n_layers=4,
        d_ff=1024,
        dropout=0.1
    )
    
    print(f"\nModel Configuration:")
    print(f"  Vocabulary Size: {config.vocab_size}")
    print(f"  Model Dimension: {config.d_model}")
    print(f"  Attention Heads: {config.n_heads}")
    print(f"  Layers: {config.n_layers}")
    print(f"  Feed-Forward Dim: {config.d_ff}")
    
    # Create model
    model = Transformer(config)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Parameters: {total_params:,}")
    
    # Create sample input
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"\nInput Shape: {input_ids.shape}")
    
    # Forward pass with state tracking
    print("\nPerforming forward pass with state tracking...")
    with torch.no_grad():
        logits, state = model(input_ids, return_state=True)
    
    print(f"Output Shape: {logits.shape}")
    print(f"Output Range: [{logits.min():.2f}, {logits.max():.2f}]")
    
    # Analyze information flow
    print("\n" + "=" * 60)
    print("Information Flow Analysis")
    print("=" * 60)
    
    # Residual stream
    residuals = state.get_residual_stream()
    print(f"\nResidual Stream (Vertical Flow):")
    print(f"  Number of checkpoints: {len(residuals)}")
    print(f"  Embedding norm: {residuals[0][0, 0, :].norm():.4f}")
    for i, res in enumerate(residuals[1:]):
        norm = res[0, 0, :].norm().item()
        print(f"  Layer {i} norm: {norm:.4f}")
    
    # Attention patterns
    print(f"\nAttention Patterns (Horizontal Flow):")
    for layer_idx in range(config.n_layers):
        attn = state.get_attention_weights(layer_idx)
        sparsity = (attn[0].mean(0) > 0.1).float().mean().item()
        entropy = -(attn[0].mean(0) * torch.log(attn[0].mean(0) + 1e-10)).sum(-1).mean().item()
        print(f"  Layer {layer_idx} - Sparsity: {sparsity:.3f}, Entropy: {entropy:.3f}")
    
    # Visualization
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    # Attention heatmap
    print("\nCreating attention visualization...")
    attn_weights = state.get_attention_weights(layer_idx=0)
    path = get_visualization_path('attention_layer_0.png')
    fig1 = visualize_attention(
        attn_weights,
        layer_idx=0,
        save_path=path
    )
    print(f"  Saved: {path}")
    
    # Residual stream
    print("\nCreating residual stream visualization...")
    path = get_visualization_path('residual_stream.png')
    fig2 = visualize_residual_stream(
        state,
        position_idx=seq_len // 2,
        save_path=path
    )
    print(f"  Saved: {path}")
    
    # K/V stream
    print("\nCreating K/V stream visualization...")
    path = get_visualization_path('kv_stream.png')
    fig3 = visualize_kv_stream(
        state,
        layer_idx=0,
        save_path=path
    )
    print(f"  Saved: {path}")
    
    # Information highways
    print("\nCreating information highways visualization...")
    path = get_visualization_path('information_highways.png')
    fig4 = visualize_information_highways(
        state,
        save_path=path
    )
    print(f"  Saved: {path}")
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == '__main__':
    import os
    os.makedirs('outputs', exist_ok=True)
    main()


