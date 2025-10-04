"""Generate all architecture visualization diagrams."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.transformer.model import Transformer, TransformerConfig
from src.visualization.architecture_viz import create_all_architecture_visualizations
from src.utils.enhanced_logging import create_enhanced_logger


def main():
    """Generate all architecture visualizations."""
    logger = create_enhanced_logger()
    
    logger.header("Architecture Visualization Generation", "=", "OKBLUE")
    
    # Create model configuration
    logger.section("Creating Model Configuration")
    config = TransformerConfig(
        vocab_size=30000,
        d_model=256,
        n_heads=8,
        n_layers=4,
        d_ff=1024,
        max_seq_len=512,
        dropout=0.1
    )
    
    logger.detail("Vocabulary Size", f"{config.vocab_size:,}", indent=1)
    logger.detail("Model Dimension", config.d_model, indent=1)
    logger.detail("Attention Heads", config.n_heads, indent=1)
    logger.detail("Layers", config.n_layers, indent=1)
    logger.detail("Feed-Forward Dimension", config.d_ff, indent=1)
    
    # Create model
    logger.section("Creating Model")
    logger.info("Instantiating transformer model...")
    model = Transformer(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.success(f"Model created with {total_params:,} parameters")
    
    # Generate all visualizations
    logger.section("Generating Visualizations")
    output_dir = project_root / "outputs" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        saved_files = create_all_architecture_visualizations(
            model, config, output_dir=str(output_dir)
        )
        
        logger.success(f"Generated {len(saved_files)} architecture visualizations")
        
        # List generated files
        logger.section("Generated Files")
        total_size = 0
        for name, path in saved_files.items():
            if path.exists():
                size_mb = path.stat().st_size / 1024 / 1024
                total_size += size_mb
                logger.list_item(
                    f"{path.name} ({size_mb:.2f} MB)", 
                    status="success"
                )
        
        logger.detail("Total Size", f"{total_size:.2f} MB", indent=1)
        logger.detail("Output Directory", output_dir, indent=1)
        
        logger.header("Generation Complete", "=", "OKGREEN")
        logger.elapsed_time()
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to generate visualizations: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

