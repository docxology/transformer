"""
Standalone script to generate subway map animations.

Usage:
    python scripts/generate_subway_map.py --config small --seq-len 16 --output subway_map.gif
"""

import sys
sys.path.insert(0, '.')

import argparse
import torch
from pathlib import Path

from src.transformer import Transformer, TransformerConfig
from src.config import get_preset_config
from src.visualization import create_subway_map_animation


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate subway map animation of transformer architecture'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='small',
        choices=['debug', 'tiny', 'small', 'medium', 'base', 'large', 'xlarge'],
        help='Model configuration preset'
    )
    
    parser.add_argument(
        '--seq-len',
        type=int,
        default=16,
        help='Sequence length for visualization'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/animations/subway_map.gif',
        help='Output file path'
    )
    
    parser.add_argument(
        '--style',
        type=str,
        default='network',
        choices=['network', 'subway', 'circuit'],
        help='Visual style'
    )
    
    parser.add_argument(
        '--resolution',
        type=str,
        default='1920x1080',
        help='Output resolution (WIDTHxHEIGHT)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='Frames per second'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=10,
        help='Animation duration in seconds'
    )
    
    parser.add_argument(
        '--no-token-flow',
        action='store_true',
        help='Disable animated token flow'
    )
    
    parser.add_argument(
        '--no-attention-flow',
        action='store_true',
        help='Disable attention flow visualization'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except:
        print(f"Error: Invalid resolution format '{args.resolution}'. Use WIDTHxHEIGHT (e.g., 1920x1080)")
        sys.exit(1)
    
    # Print configuration
    print("=" * 70)
    print("SUBWAY MAP ANIMATION GENERATOR")
    print("=" * 70)
    print()
    print(f"Configuration:")
    print(f"  Model Config: {args.config}")
    print(f"  Sequence Length: {args.seq_len}")
    print(f"  Style: {args.style}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {args.fps}")
    print(f"  Duration: {args.duration}s")
    print(f"  Token Flow: {'Disabled' if args.no_token_flow else 'Enabled'}")
    print(f"  Attention Flow: {'Disabled' if args.no_attention_flow else 'Enabled'}")
    print()
    
    # Load model configuration
    print(f"Loading model configuration '{args.config}'...")
    config = get_preset_config(args.config)
    
    # Create model
    print(f"Creating transformer model...")
    model = Transformer(config)
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model created: {param_count:,} parameters")
    print(f"  ✓ Architecture: {config.n_layers} layers, {config.n_heads} heads, d_model={config.d_model}")
    print()
    
    # Generate forward pass to get state
    print(f"Generating model state (seq_len={args.seq_len})...")
    batch_size = 1
    input_ids = torch.randint(0, config.vocab_size, (batch_size, args.seq_len))
    
    with torch.no_grad():
        logits, state = model(input_ids, return_state=True)
    
    print(f"  ✓ State generated: {len(state.layer_states)} layers")
    print()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate animation
    print(f"Generating subway map animation...")
    print(f"  This may take a few minutes for high resolutions...")
    print()
    
    create_subway_map_animation(
        state,
        save_path=str(output_path),
        style=args.style,
        show_token_flow=not args.no_token_flow,
        show_attention_flow=not args.no_attention_flow,
        fps=args.fps,
        duration_seconds=args.duration,
        resolution=resolution
    )
    
    # Summary
    print()
    print("=" * 70)
    print("ANIMATION COMPLETE")
    print("=" * 70)
    print()
    print(f"✓ Animation saved to: {output_path}")
    print(f"✓ File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print()
    print(f"Total frames: {args.fps * args.duration}")
    print(f"Duration: {args.duration} seconds at {args.fps} FPS")
    print()
    print("The subway map animation visualizes:")
    print("  • Network topology with nodes for each layer/position")
    print("  • Vertical (green) edges showing residual stream connections")
    print("  • Horizontal (purple) edges showing attention connections")
    if not args.no_token_flow:
        print("  • Animated tokens flowing through the network")
    print()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)



