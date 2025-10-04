"""
Information path analysis example.

Demonstrates the astronomical number of computational paths through
the transformer and visualizes path complexity.
"""

import torch
from src.transformer import Transformer, TransformerConfig
from src.visualization import (
    compute_path_counts,
    trace_information_paths,
    compute_path_statistics,
    analyze_path_complexity,
    visualize_path_grid
)
from src.utils.output_paths import get_visualization_path
import matplotlib.pyplot as plt


def main():
    print("=" * 70)
    print("Information Path Analysis - Computational Path Explosion")
    print("=" * 70)
    
    # Model configuration
    config = TransformerConfig(
        vocab_size=1000,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024
    )
    
    seq_len = 32
    
    print(f"\nModel: {config.n_layers} layers, {seq_len} positions")
    
    # Compute path counts for various displacements
    print("\n" + "=" * 70)
    print("Path Count Examples")
    print("=" * 70)
    
    examples = [
        (1, 1, "Adjacent (1 layer, 1 position)"),
        (2, 2, "Small jump (2 layers, 2 positions)"),
        (3, 5, "Medium jump (3 layers, 5 positions)"),
        (5, 10, "Large jump (5 layers, 10 positions)"),
        (config.n_layers - 1, seq_len // 2, "Half traverse"),
        (config.n_layers - 1, seq_len - 1, "Full traverse")
    ]
    
    for layer_disp, pos_disp, description in examples:
        count = compute_path_counts(layer_disp, pos_disp)
        print(f"\n{description}:")
        print(f"  Displacement: {layer_disp} layers, {pos_disp} positions")
        print(f"  Path count: {count:,}")
        if count > 1e10:
            print(f"  Log10: {count:.2e}")
    
    # Detailed complexity analysis
    print("\n" + "=" * 70)
    print("Comprehensive Path Complexity Analysis")
    print("=" * 70)
    
    analysis = analyze_path_complexity(config.n_layers, seq_len)
    print(analysis)
    
    # Trace actual paths for small example
    print("\n" + "=" * 70)
    print("Path Tracing Example")
    print("=" * 70)
    
    print("\nTracing paths from (layer=0, pos=0) to (layer=2, pos=2):")
    paths = trace_information_paths(
        start_layer=0,
        start_position=0,
        end_layer=2,
        end_position=2,
        max_paths=100
    )
    
    print(f"\nTotal paths: {len(paths)}")
    print("\nFirst 5 paths:")
    for i, path in enumerate(paths[:5]):
        path_str = " -> ".join([f"({l},{p})" for l, p in path])
        print(f"  Path {i+1}: {path_str}")
    
    # Visualize path grid
    print("\n" + "=" * 70)
    print("Path Density Visualization")
    print("=" * 70)
    
    print("\nGenerating path density grid...")
    grid = visualize_path_grid(
        n_layers=6,
        seq_len=12,
        start=(0, 0),
        end=(5, 11)
    )
    
    # Plot grid
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(grid.T, cmap='hot', aspect='auto', origin='lower')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Position')
    ax.set_title('Path Density: Number of Paths Through Each Point')
    plt.colorbar(im, ax=ax, label='Path Count')
    
    # Annotate start and end
    ax.plot(0, 0, 'go', markersize=15, label='Start')
    ax.plot(5, 11, 'bo', markersize=15, label='End')
    ax.legend()
    
    plt.tight_layout()
    path = get_visualization_path('path_density.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {path}")
    
    # Statistics
    print("\n" + "=" * 70)
    print("Path Statistics")
    print("=" * 70)
    
    stats = compute_path_statistics(config.n_layers, seq_len)
    
    print(f"\nNetwork: {stats['n_layers']} layers, {stats['seq_len']} positions")
    print(f"Typical path count: {stats['typical_path_count']:,}")
    
    print("\nSample connections:")
    for key, value in stats['paths_per_connection'].items():
        print(f"  {key}: {value['count']:,} paths")
    
    # Key insights
    print("\n" + "=" * 70)
    print("Key Insights")
    print("=" * 70)
    
    print("""
1. MASSIVE REDUNDANCY: The astronomical number of paths means information
   can flow through the network via countless different computational routes.

2. INTERFERENCE PATTERNS: Multiple paths create rich interference patterns,
   encoding nuanced information about relationships and contexts.

3. ROBUST TRANSMISSION: Even if some paths are lossy or noisy, the vast
   redundancy ensures information is preserved and can be reconstructed.

4. CONTINUOUS COGNITION: The transformer doesn't just retrieve discrete
   memories - it experiences a continuous, interferometric blend of past
   states through multiple simultaneous paths.

5. INTROSPECTION CAPACITY: The architecture permits rich introspection on
   past computational states through the K/V stream's horizontal connections.
    """)
    
    print("=" * 70)


if __name__ == '__main__':
    import os
    os.makedirs('outputs', exist_ok=True)
    main()


