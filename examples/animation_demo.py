"""
Animation demonstration.

Creates animated visualizations of information flow through the transformer.
"""

import torch
from src.transformer import Transformer, TransformerConfig
from src.visualization import (
    animate_information_flow,
    animate_attention_evolution,
    animate_path_trace
)
from src.utils.output_paths import get_animation_path


def main():
    print("=" * 60)
    print("Transformer Animation Demo")
    print("=" * 60)
    
    # Create model
    config = TransformerConfig(
        vocab_size=500,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        dropout=0.0
    )
    
    model = Transformer(config)
    model.eval()
    
    # Generate state
    batch_size = 1
    seq_len = 12
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"\nGenerating transformer state...")
    print(f"  Sequence length: {seq_len}")
    print(f"  Layers: {config.n_layers}")
    
    with torch.no_grad():
        logits, state = model(input_ids, return_state=True)
    
    print("State generated successfully!")
    
    # Animation 1: Information flow through layers
    print("\n" + "=" * 60)
    print("Animation 1: Information Flow Through Layers")
    print("=" * 60)
    print("\nCreating animation...")
    
    path = get_animation_path('information_flow.gif')
    anim1 = animate_information_flow(
        state,
        save_path=path,
        fps=2,
        position_focus=seq_len // 2
    )
    print(f"  Saved: {path}")
    print("  Shows: Residual stream evolution, attention patterns, and state")
    
    # Animation 2: Attention evolution
    print("\n" + "=" * 60)
    print("Animation 2: Attention Pattern Evolution")
    print("=" * 60)
    print("\nCreating animation...")
    
    path = get_animation_path('attention_evolution.gif')
    anim2 = animate_attention_evolution(
        state,
        query_position=seq_len - 1,  # Focus on last position
        save_path=path,
        fps=2
    )
    print(f"  Saved: {path}")
    print("  Shows: How attention for a query evolves through layers")
    
    # Animation 3: Path trace
    print("\n" + "=" * 60)
    print("Animation 3: Information Path Trace")
    print("=" * 60)
    print("\nCreating animation...")
    
    # Create a sample path (diagonal path through network)
    path_indices = []
    for layer in range(config.n_layers + 1):
        pos = min(layer * 2, seq_len - 1)
        path_indices.append(pos)
    
    path = get_animation_path('path_trace.gif')
    anim3 = animate_path_trace(
        state,
        start_pos=0,
        end_pos=seq_len - 1,
        path_indices=path_indices,
        save_path=path,
        fps=3
    )
    print(f"  Saved: {path}")
    print("  Shows: Specific information flow path through layers/positions")
    
    print("\n" + "=" * 60)
    print("All animations created successfully!")
    print("=" * 60)
    
    print("""
The animations illustrate:

1. INFORMATION_FLOW.GIF
   - Vertical flow: Residual stream through layers
   - Horizontal flow: Attention across positions
   - State evolution: How representations change

2. ATTENTION_EVOLUTION.GIF
   - How a single query position attends to others
   - Evolution of attention pattern through layers
   - Concentration vs. diffusion of attention

3. PATH_TRACE.GIF
   - A specific computational path
   - Demonstrates one of many possible routes
   - Shows layer-position coordinate system
    """)


if __name__ == '__main__':
    import os
    os.makedirs('outputs', exist_ok=True)
    main()


