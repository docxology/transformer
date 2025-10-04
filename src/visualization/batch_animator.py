"""Batch animation generation for efficient processing."""

import torch
from pathlib import Path
from typing import List, Dict, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from .subway_map_animation import create_subway_map_animation
from .animation import (
    animate_information_flow,
    animate_attention_evolution,
    animate_path_trace
)
from .advanced_animations import (
    animate_layer_activations,
    animate_attention_heads,
    animate_token_interactions,
    animate_information_diffusion,
    animate_residual_evolution,
    animate_head_comparison
)
from ..utils.progress_tracker import create_progress_bar


def batch_generate_animations(
    states: List[any],
    animation_types: List[str],
    output_dir: str = 'outputs/animations',
    parallel: bool = True,
    max_workers: Optional[int] = None,
    fps: int = 10
) -> Dict[str, List[Path]]:
    """
    Efficiently generate multiple animations.
    
    Args:
        states: List of TransformerState objects
        animation_types: Types of animations to generate
                        ('subway_map', 'information_flow', 'attention_evolution',
                         'layer_activations', 'token_interactions', etc.)
        output_dir: Directory to save animations
        parallel: Whether to use parallel processing
        max_workers: Maximum number of parallel workers
        fps: Frames per second for animations
        
    Returns:
        Dictionary mapping animation types to lists of generated file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Map animation type names to functions
    animation_functions = {
        'subway_map': create_subway_map_animation,
        'information_flow': animate_information_flow,
        'attention_evolution': animate_attention_evolution,
        'layer_activations': animate_layer_activations,
        'attention_heads': animate_attention_heads,
        'token_interactions': animate_token_interactions,
        'information_diffusion': animate_information_diffusion,
        'residual_evolution': animate_residual_evolution,
        'head_comparison': animate_head_comparison
    }
    
    # Prepare tasks
    tasks = []
    for state_idx, state in enumerate(states):
        for anim_type in animation_types:
            if anim_type in animation_functions:
                filename = f'{anim_type}_state{state_idx}.gif'
                save_path = str(output_dir / filename)
                tasks.append((state, anim_type, save_path, fps))
    
    results = {atype: [] for atype in animation_types}
    
    if parallel and len(tasks) > 1:
        # Parallel processing
        if max_workers is None:
            max_workers = min(mp.cpu_count(), len(tasks))
        
        print(f"Generating {len(tasks)} animations using {max_workers} workers...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = []
            for task in tasks:
                future = executor.submit(_generate_single_animation, task, animation_functions)
                futures.append((future, task[1], task[2]))
            
            # Track progress
            with create_progress_bar(len(futures), desc="Generating animations") as pbar:
                for future, anim_type, save_path in futures:
                    try:
                        result = future.result()
                        if result:
                            results[anim_type].append(Path(save_path))
                    except Exception as e:
                        print(f"Error generating {anim_type}: {e}")
                    finally:
                        pbar.update(1)
    
    else:
        # Sequential processing
        print(f"Generating {len(tasks)} animations sequentially...")
        
        with create_progress_bar(len(tasks), desc="Generating animations") as pbar:
            for task in tasks:
                try:
                    state, anim_type, save_path, fps = task
                    result = _generate_single_animation(task, animation_functions)
                    if result:
                        results[anim_type].append(Path(save_path))
                except Exception as e:
                    print(f"Error generating {anim_type}: {e}")
                finally:
                    pbar.update(1)
    
    # Print summary
    print(f"\n✓ Animation generation complete!")
    for anim_type, paths in results.items():
        if paths:
            print(f"  {anim_type}: {len(paths)} animations")
    
    return results


def _generate_single_animation(
    task: tuple,
    animation_functions: Dict[str, Callable]
) -> bool:
    """
    Generate a single animation (worker function).
    
    Args:
        task: Tuple of (state, anim_type, save_path, fps)
        animation_functions: Dictionary of animation functions
        
    Returns:
        True if successful, False otherwise
    """
    state, anim_type, save_path, fps = task
    
    try:
        func = animation_functions[anim_type]
        
        # Call function with appropriate arguments
        if anim_type == 'subway_map':
            func(state, save_path=save_path, fps=fps)
        elif anim_type == 'attention_evolution':
            func(state, query_position=0, save_path=save_path, fps=fps)
        elif anim_type == 'attention_heads':
            func(state, layer_idx=0, save_path=save_path, fps=fps)
        elif anim_type == 'information_diffusion':
            func(state, source_position=0, save_path=save_path, fps=fps)
        else:
            func(state, save_path=save_path, fps=fps)
        
        return True
    
    except Exception as e:
        print(f"Error in worker for {anim_type}: {e}")
        return False


def batch_generate_for_training(
    training_states: List[any],
    epochs: List[int],
    output_dir: str = 'outputs/animations/training',
    animation_types: Optional[List[str]] = None
) -> Dict[int, Dict[str, Path]]:
    """
    Generate animations for multiple training epochs.
    
    Args:
        training_states: List of states from different epochs
        epochs: List of epoch numbers corresponding to states
        output_dir: Output directory
        animation_types: Types of animations to generate (None = all)
        
    Returns:
        Dictionary mapping epoch numbers to animation paths
    """
    if animation_types is None:
        animation_types = ['subway_map', 'information_flow', 'attention_evolution']
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    print(f"Generating training animations for {len(epochs)} epochs...")
    
    for state, epoch in zip(training_states, epochs):
        epoch_dir = output_dir / f'epoch_{epoch}'
        epoch_dir.mkdir(exist_ok=True)
        
        epoch_results = {}
        
        for anim_type in animation_types:
            filename = f'{anim_type}.gif'
            save_path = str(epoch_dir / filename)
            
            try:
                if anim_type == 'subway_map':
                    create_subway_map_animation(state, save_path=save_path, fps=10)
                elif anim_type == 'information_flow':
                    animate_information_flow(state, save_path=save_path, fps=2)
                elif anim_type == 'attention_evolution':
                    animate_attention_evolution(state, query_position=0, save_path=save_path, fps=2)
                
                epoch_results[anim_type] = Path(save_path)
                print(f"  ✓ Epoch {epoch}: {anim_type}")
            
            except Exception as e:
                print(f"  ✗ Epoch {epoch}: {anim_type} failed - {e}")
        
        results[epoch] = epoch_results
    
    print(f"✓ Training animation generation complete!")
    return results


def create_animation_gallery(
    animation_paths: List[Path],
    output_html: str = 'outputs/animations/gallery.html',
    title: str = "Animation Gallery"
) -> Path:
    """
    Create HTML gallery of animations.
    
    Args:
        animation_paths: List of paths to animation GIFs
        output_html: Path to save HTML gallery
        title: Gallery title
        
    Returns:
        Path to generated HTML file
    """
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
            margin: 0;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        h1 {{
            text-align: center;
            color: #333;
            margin-bottom: 40px;
        }}
        
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
        }}
        
        .animation-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .animation-card h2 {{
            margin: 0 0 15px 0;
            color: #1976D2;
            font-size: 1.2em;
        }}
        
        .animation-card img {{
            width: 100%;
            height: auto;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 {title}</h1>
        <div class="gallery">
    """
    
    for path in animation_paths:
        name = path.stem.replace('_', ' ').title()
        html_content += f"""
            <div class="animation-card">
                <h2>{name}</h2>
                <img src="{path.name}" alt="{name}">
            </div>
        """
    
    html_content += """
        </div>
    </div>
</body>
</html>
    """
    
    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"✓ Animation gallery created: {output_path}")
    return output_path



