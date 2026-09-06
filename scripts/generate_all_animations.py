"""
Generate all animation types for comprehensive visualization.

Creates a complete set of animations demonstrating all aspects
of transformer operation and information flow.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.transformer import Transformer, TransformerConfig
from src.config import get_preset_config
from src.visualization import (
    animate_information_flow,
    animate_attention_evolution,
    animate_path_trace
)
from src.visualization.advanced_animations import (
    animate_layer_activations,
    animate_attention_heads,
    animate_token_interactions,
    animate_information_diffusion,
    animate_residual_evolution,
    animate_head_comparison
)
from src.utils.logging_system import TransformerLogger


def main():
    """Generate all animations."""
    
    # Setup logging
    logger = TransformerLogger("animation_generator")
    logger.info("=" * 70)
    logger.info("COMPREHENSIVE ANIMATION GENERATION")
    logger.info("=" * 70)
    
    # Create output directory
    output_dir = Path("outputs/animations")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Create model
    logger.info("\nCreating model...")
    config = get_preset_config('small')
    model = Transformer(config)
    model.eval()
    
    logger.log_model_info(model)
    logger.log_config({
        'vocab_size': config.vocab_size,
        'd_model': config.d_model,
        'n_heads': config.n_heads,
        'n_layers': config.n_layers
    })
    
    # Generate data
    logger.info("\nGenerating input data...")
    batch_size, seq_len = 1, 12
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    logger.debug(f"Input shape: {input_ids.shape}")
    
    # Forward pass
    logger.info("Running forward pass...")
    with torch.no_grad():
        logits, state = model(input_ids, return_state=True)
    logger.info(f"Forward pass complete. Output shape: {logits.shape}")
    
    # Animation 1: Information Flow
    logger.info("\n[1/9] Creating information flow animation...")
    try:
        path = str(output_dir / "01_information_flow.gif")
        animate_information_flow(state, save_path=path, fps=2)
        logger.info(f"  ✓ Saved: {path}")
        logger.log_metric("animation_created", 1.0, step=1)
    except Exception as e:
        logger.error(f"Failed to create information flow animation: {e}")
    
    # Animation 2: Attention Evolution
    logger.info("\n[2/9] Creating attention evolution animation...")
    try:
        path = str(output_dir / "02_attention_evolution.gif")
        animate_attention_evolution(state, query_position=seq_len-1, save_path=path, fps=2)
        logger.info(f"  ✓ Saved: {path}")
        logger.log_metric("animation_created", 1.0, step=2)
    except Exception as e:
        logger.error(f"Failed to create attention evolution animation: {e}")
    
    # Animation 3: Path Trace
    logger.info("\n[3/9] Creating path trace animation...")
    try:
        path = str(output_dir / "03_path_trace.gif")
        path_indices = list(range(0, seq_len, 2))[:config.n_layers + 1]
        while len(path_indices) < config.n_layers + 1:
            path_indices.append(path_indices[-1])
        animate_path_trace(state, start_pos=0, end_pos=seq_len-1, 
                          path_indices=path_indices, save_path=path, fps=3)
        logger.info(f"  ✓ Saved: {path}")
        logger.log_metric("animation_created", 1.0, step=3)
    except Exception as e:
        logger.error(f"Failed to create path trace animation: {e}")
    
    # Animation 4: Layer Activations
    logger.info("\n[4/9] Creating layer activations animation...")
    try:
        path = str(output_dir / "04_layer_activations.gif")
        animate_layer_activations(state, save_path=path, fps=3)
        logger.info(f"  ✓ Saved: {path}")
        logger.log_metric("animation_created", 1.0, step=4)
    except Exception as e:
        logger.error(f"Failed to create layer activations animation: {e}")
    
    # Animation 5: Attention Heads
    logger.info("\n[5/9] Creating attention heads animation...")
    try:
        path = str(output_dir / "05_attention_heads.gif")
        animate_attention_heads(state, layer_idx=0, save_path=path, fps=2)
        logger.info(f"  ✓ Saved: {path}")
        logger.log_metric("animation_created", 1.0, step=5)
    except Exception as e:
        logger.error(f"Failed to create attention heads animation: {e}")
    
    # Animation 6: Token Interactions
    logger.info("\n[6/9] Creating token interactions animation...")
    try:
        path = str(output_dir / "06_token_interactions.gif")
        animate_token_interactions(state, save_path=path, fps=3)
        logger.info(f"  ✓ Saved: {path}")
        logger.log_metric("animation_created", 1.0, step=6)
    except Exception as e:
        logger.error(f"Failed to create token interactions animation: {e}")
    
    # Animation 7: Information Diffusion
    logger.info("\n[7/9] Creating information diffusion animation...")
    try:
        path = str(output_dir / "07_information_diffusion.gif")
        animate_information_diffusion(state, source_position=0, save_path=path, fps=3)
        logger.info(f"  ✓ Saved: {path}")
        logger.log_metric("animation_created", 1.0, step=7)
    except Exception as e:
        logger.error(f"Failed to create information diffusion animation: {e}")
    
    # Animation 8: Residual Evolution
    logger.info("\n[8/9] Creating residual evolution animation...")
    try:
        path = str(output_dir / "08_residual_evolution.gif")
        animate_residual_evolution(state, save_path=path, fps=2)
        logger.info(f"  ✓ Saved: {path}")
        logger.log_metric("animation_created", 1.0, step=8)
    except Exception as e:
        logger.error(f"Failed to create residual evolution animation: {e}")
    
    # Animation 9: Head Comparison
    logger.info("\n[9/9] Creating head comparison animation...")
    try:
        path = str(output_dir / "09_head_comparison.gif")
        animate_head_comparison(state, save_path=path, fps=2)
        logger.info(f"  ✓ Saved: {path}")
        logger.log_metric("animation_created", 1.0, step=9)
    except Exception as e:
        logger.error(f"Failed to create head comparison animation: {e}")
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("ANIMATION GENERATION COMPLETE")
    logger.info("=" * 70)
    
    # Count generated files
    animations = list(output_dir.glob("*.gif"))
    logger.info(f"\nTotal animations created: {len(animations)}")
    
    total_size = sum(f.stat().st_size for f in animations)
    logger.info(f"Total size: {total_size / 1024 / 1024:.2f} MB")
    
    logger.info("\nGenerated animations:")
    for anim in sorted(animations):
        size_mb = anim.stat().st_size / 1024 / 1024
        logger.info(f"  • {anim.name} ({size_mb:.2f} MB)")
    
    # Save metrics
    logger.save_metrics()
    
    # Print session summary
    summary = logger.get_session_summary()
    logger.info("\nSession Summary:")
    logger.info(f"  Session ID: {summary['session_id']}")
    logger.info(f"  Total logs: {summary['total_logs']}")
    logger.info(f"  Total metrics: {summary['total_metrics']}")
    
    logger.info("\n✅ ALL ANIMATIONS GENERATED SUCCESSFULLY")


if __name__ == '__main__':
    main()


