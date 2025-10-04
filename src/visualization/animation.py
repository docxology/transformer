"""Animation tools for transformer information flow."""

import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from typing import List, Optional, Tuple
import seaborn as sns


def animate_information_flow(
    state: any,
    save_path: str = 'flow_animation.gif',
    fps: int = 2,
    position_focus: Optional[int] = None
) -> animation.FuncAnimation:
    """
    Animate information flow through transformer layers.
    
    Shows how information evolves at each layer, with both the residual
    stream (vertical) and attention patterns (horizontal).
    
    Args:
        state: TransformerState object
        save_path: Path to save animation
        fps: Frames per second
        position_focus: Specific position to highlight (optional)
        
    Returns:
        Animation object
    """
    n_layers = len(state.layer_states)
    seq_len = state.embeddings.shape[1]
    
    if position_focus is None:
        position_focus = seq_len // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Initialize plots
    residual_line, = axes[0].plot([], [], 'o-', linewidth=2, markersize=8)
    axes[0].set_xlim(-1, n_layers)
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Residual Norm')
    axes[0].set_title(f'Residual Stream (Position {position_focus})')
    axes[0].grid(True, alpha=0.3)
    
    # Attention heatmap
    attn_img = axes[1].imshow(
        np.zeros((seq_len, seq_len)),
        cmap='viridis',
        aspect='auto',
        vmin=0,
        vmax=1
    )
    axes[1].set_title('Attention Pattern')
    axes[1].set_xlabel('Key Position')
    axes[1].set_ylabel('Query Position')
    plt.colorbar(attn_img, ax=axes[1], fraction=0.046)
    
    # Residual stream heatmap
    residual_img = axes[2].imshow(
        np.zeros((seq_len, 64)),
        cmap='RdBu_r',
        aspect='auto'
    )
    axes[2].set_title('Residual Stream State')
    axes[2].set_xlabel('Feature Dim (sampled)')
    axes[2].set_ylabel('Position')
    plt.colorbar(residual_img, ax=axes[2], fraction=0.046)
    
    layer_text = fig.text(0.5, 0.95, '', ha='center', fontsize=14, weight='bold')
    
    def update(frame):
        layer_idx = frame
        
        # Update residual stream evolution
        residuals = state.get_residual_stream()
        norms = [
            residuals[i][0, position_focus, :].norm().item()
            for i in range(layer_idx + 2)  # +2 to include embedding + current layer
        ]
        residual_line.set_data(range(-1, layer_idx + 1), norms)
        
        # Update y-axis limits
        max_norm = max(norms) * 1.1
        axes[0].set_ylim(0, max_norm)
        
        # Update attention pattern
        if layer_idx < n_layers:
            attn_weights = state.get_attention_weights(layer_idx)
            avg_attn = attn_weights[0].mean(dim=0).cpu().detach().numpy()
            attn_img.set_array(avg_attn)
        
        # Update residual stream heatmap
        if layer_idx < n_layers:
            residual = state.layer_states[layer_idx].post_ffn
            # Sample features for visualization
            features = residual[0, :, :64].cpu().detach().numpy()
            residual_img.set_array(features)
            
            # Update color scale
            vmax = np.abs(features).max()
            residual_img.set_clim(-vmax, vmax)
        
        # Update title
        if layer_idx < n_layers:
            layer_text.set_text(f'Layer {layer_idx}')
        else:
            layer_text.set_text('Output')
        
        return residual_line, attn_img, residual_img, layer_text
    
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n_layers,
        interval=1000 // fps,
        blit=True,
        repeat=True
    )
    
    anim.save(save_path, writer='pillow', fps=fps)
    plt.close()
    
    return anim


def animate_path_trace(
    state: any,
    start_pos: int,
    end_pos: int,
    path_indices: List[int],
    save_path: str = 'path_trace.gif',
    fps: int = 3
) -> animation.FuncAnimation:
    """
    Animate a specific information flow path.
    
    Traces how information flows from a starting position through
    layers and eventually reaches an ending position.
    
    Args:
        state: TransformerState object
        start_pos: Starting token position
        end_pos: Ending token position
        path_indices: List of positions to trace at each layer
        save_path: Path to save animation
        fps: Frames per second
        
    Returns:
        Animation object
    """
    n_layers = len(state.layer_states)
    seq_len = state.embeddings.shape[1]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create grid for visualization
    layer_coords = np.arange(n_layers + 1)
    pos_coords = np.arange(seq_len)
    
    def update(frame):
        ax.clear()
        
        # Draw grid
        for layer in layer_coords:
            ax.axhline(y=layer, color='gray', alpha=0.3, linewidth=0.5)
        for pos in pos_coords[::max(1, seq_len // 20)]:  # Show subset of positions
            ax.axvline(x=pos, color='gray', alpha=0.3, linewidth=0.5)
        
        # Draw path up to current frame
        for i in range(min(frame + 1, len(path_indices))):
            if i < len(path_indices) - 1:
                # Draw connection
                ax.plot(
                    [path_indices[i], path_indices[i + 1]],
                    [i, i + 1],
                    'b-',
                    linewidth=2,
                    alpha=0.6
                )
            
            # Draw node
            ax.plot(
                path_indices[i],
                i,
                'ro' if i == frame else 'bo',
                markersize=12 if i == frame else 8
            )
        
        # Highlight start and end
        ax.plot(start_pos, 0, 'go', markersize=15, label='Start')
        ax.plot(end_pos, n_layers, 'mo', markersize=15, label='End')
        
        ax.set_xlim(-1, seq_len)
        ax.set_ylim(-0.5, n_layers + 0.5)
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Layer')
        ax.set_title(f'Information Flow Path (Frame {frame}/{len(path_indices)})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax,
    
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(path_indices),
        interval=1000 // fps,
        blit=False,
        repeat=True
    )
    
    anim.save(save_path, writer='pillow', fps=fps)
    plt.close()
    
    return anim


def animate_attention_evolution(
    state: any,
    query_position: int,
    save_path: str = 'attention_evolution.gif',
    fps: int = 2
) -> animation.FuncAnimation:
    """
    Animate how attention pattern for a query evolves through layers.
    
    Args:
        state: TransformerState object
        query_position: Query position to track
        save_path: Path to save animation
        fps: Frames per second
        
    Returns:
        Animation object
    """
    n_layers = len(state.layer_states)
    seq_len = state.embeddings.shape[1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract attention patterns for the query position across all layers
    attention_evolution = []
    for layer_idx in range(n_layers):
        attn_weights = state.get_attention_weights(layer_idx)
        # Average across heads and extract query position
        attn_pattern = attn_weights[0, :, query_position, :].mean(dim=0)
        attention_evolution.append(attn_pattern.cpu().detach().numpy())
    
    attention_evolution = np.array(attention_evolution)
    
    def update(frame):
        layer_idx = frame
        
        # Left: Current layer attention distribution
        ax1.clear()
        ax1.bar(range(seq_len), attention_evolution[layer_idx])
        ax1.set_xlim(-0.5, seq_len - 0.5)
        ax1.set_ylim(0, attention_evolution.max() * 1.1)
        ax1.set_xlabel('Key Position')
        ax1.set_ylabel('Attention Weight')
        ax1.set_title(f'Layer {layer_idx}: Attention from Position {query_position}')
        ax1.axvline(query_position, color='r', linestyle='--', alpha=0.5, label='Query')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Right: Evolution heatmap
        ax2.clear()
        im = ax2.imshow(
            attention_evolution[:layer_idx + 1],
            cmap='viridis',
            aspect='auto',
            vmin=0,
            vmax=attention_evolution.max()
        )
        ax2.set_xlabel('Key Position')
        ax2.set_ylabel('Layer')
        ax2.set_title('Attention Evolution Through Layers')
        ax2.axvline(query_position, color='r', linestyle='--', alpha=0.5)
        plt.colorbar(im, ax=ax2, fraction=0.046)
        
        return ax1, ax2
    
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n_layers,
        interval=1000 // fps,
        blit=False,
        repeat=True
    )
    
    anim.save(save_path, writer='pillow', fps=fps)
    plt.close()
    
    return anim


