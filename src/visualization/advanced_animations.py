"""Advanced animation methods for transformer visualization."""

import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
import numpy as np
from typing import Optional, List, Tuple
import seaborn as sns


def animate_gradient_flow(
    model: torch.nn.Module,
    loss_values: List[float],
    save_path: str = 'outputs/animations/gradient_flow.gif',
    fps: int = 5
) -> animation.FuncAnimation:
    """
    Animate gradient flow through network during training.
    
    Shows how gradients propagate backwards through layers,
    helping visualize vanishing/exploding gradients.
    
    Args:
        model: Transformer model with gradients
        loss_values: List of loss values over time
        save_path: Path to save animation
        fps: Frames per second
        
    Returns:
        Animation object
    """
    # Collect gradient data
    layer_names = []
    grad_norms = []
    
    for name, param in model.named_parameters():
        if param.grad is not None and 'weight' in name:
            layer_names.append(name.replace('.weight', ''))
            grad_norms.append(param.grad.norm().item())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Gradient magnitudes
    bars = ax1.bar(range(len(layer_names)), [0] * len(layer_names))
    ax1.set_xticks(range(len(layer_names)))
    ax1.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Gradient Norm (log scale)')
    ax1.set_yscale('log')
    ax1.set_title('Gradient Flow Through Layers')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right: Loss curve
    ax2.plot([], [], 'b-', linewidth=2)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss')
    ax2.grid(True, alpha=0.3)
    
    def update(frame):
        # Update gradient bars
        for bar, height in zip(bars, grad_norms):
            bar.set_height(height * (frame + 1) / 10)
        
        # Update loss curve
        if frame < len(loss_values):
            ax2.clear()
            ax2.plot(range(frame + 1), loss_values[:frame + 1], 'b-', linewidth=2)
            ax2.set_xlabel('Training Step')
            ax2.set_ylabel('Loss')
            ax2.set_title('Training Loss')
            ax2.grid(True, alpha=0.3)
        
        return bars
    
    anim = animation.FuncAnimation(
        fig, update, frames=min(20, len(loss_values)),
        interval=1000 // fps, blit=False, repeat=True
    )
    
    anim.save(save_path, writer='pillow', fps=fps)
    plt.close()
    
    return anim


def animate_layer_activations(
    state: any,
    save_path: str = 'outputs/animations/layer_activations.gif',
    fps: int = 3
) -> animation.FuncAnimation:
    """
    Animate how activations change through layers.
    
    Shows the evolution of representations as they pass through
    the network, revealing what each layer computes.
    
    Args:
        state: TransformerState object
        save_path: Path to save animation
        fps: Frames per second
        
    Returns:
        Animation object
    """
    n_layers = len(state.layer_states)
    seq_len = state.embeddings.shape[1]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Prepare data
    residuals = state.get_residual_stream()
    
    def update(frame):
        layer_idx = frame
        
        # Current layer activations
        ax = axes[0]
        ax.clear()
        if layer_idx < len(residuals):
            data = residuals[layer_idx][0, :, :64].cpu().detach().numpy()
            im = ax.imshow(data.T, cmap='RdBu_r', aspect='auto')
            ax.set_title(f'Layer {layer_idx - 1 if layer_idx > 0 else "Embedding"}')
            ax.set_xlabel('Position')
            ax.set_ylabel('Feature (first 64)')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Activation histogram
        ax = axes[1]
        ax.clear()
        if layer_idx < len(residuals):
            data = residuals[layer_idx][0].flatten().cpu().detach().numpy()
            ax.hist(data, bins=50, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Activation Value')
            ax.set_ylabel('Count')
            ax.set_title('Distribution')
            ax.axvline(0, color='red', linestyle='--', linewidth=2)
            ax.grid(True, alpha=0.3)
        
        # Norm evolution
        ax = axes[2]
        ax.clear()
        norms = [r[0].norm(dim=-1).mean().item() for r in residuals[:layer_idx + 1]]
        ax.plot(range(-1, layer_idx), norms, 'o-', linewidth=2, markersize=8)
        ax.set_xlim(-1, n_layers - 1)
        ax.set_ylim(0, max([r[0].norm(dim=-1).mean().item() for r in residuals]) * 1.1)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Average Norm')
        ax.set_title('Norm Evolution')
        ax.grid(True, alpha=0.3)
        
        return axes
    
    anim = animation.FuncAnimation(
        fig, update, frames=len(residuals),
        interval=1000 // fps, blit=False, repeat=True
    )
    
    anim.save(save_path, writer='pillow', fps=fps)
    plt.close()
    
    return anim


def animate_attention_heads(
    state: any,
    layer_idx: int = 0,
    save_path: str = 'outputs/animations/attention_heads.gif',
    fps: int = 2
) -> animation.FuncAnimation:
    """
    Animate attention patterns across all heads in a layer.
    
    Shows how different heads focus on different patterns,
    revealing head specialization.
    
    Args:
        state: TransformerState object
        layer_idx: Layer to visualize
        save_path: Path to save animation
        fps: Frames per second
        
    Returns:
        Animation object
    """
    attn_weights = state.get_attention_weights(layer_idx)[0]  # [n_heads, seq_len, seq_len]
    n_heads = attn_weights.shape[0]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    def update(frame):
        head_idx = frame
        ax.clear()
        
        data = attn_weights[head_idx].cpu().detach().numpy()
        
        im = ax.imshow(data, cmap='viridis', aspect='auto')
        ax.set_title(f'Layer {layer_idx}, Head {head_idx}', fontsize=16, weight='bold')
        ax.set_xlabel('Key Position', fontsize=12)
        ax.set_ylabel('Query Position', fontsize=12)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Add statistics
        mean_attn = data.mean()
        max_attn = data.max()
        entropy = -(data * np.log(data + 1e-10)).sum(axis=-1).mean()
        
        stats_text = f'Mean: {mean_attn:.3f}\nMax: {max_attn:.3f}\nEntropy: {entropy:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        return [ax]
    
    anim = animation.FuncAnimation(
        fig, update, frames=n_heads,
        interval=1000 // fps, blit=False, repeat=True
    )
    
    anim.save(save_path, writer='pillow', fps=fps)
    plt.close()
    
    return anim


def animate_token_interactions(
    state: any,
    save_path: str = 'outputs/animations/token_interactions.gif',
    fps: int = 3
) -> animation.FuncAnimation:
    """
    Animate token-to-token interactions through attention.
    
    Shows how tokens communicate with each other,
    revealing information sharing patterns.
    
    Args:
        state: TransformerState object
        save_path: Path to save animation
        fps: Frames per second
        
    Returns:
        Animation object
    """
    n_layers = len(state.layer_states)
    seq_len = state.embeddings.shape[1]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Position tokens in a circle
    angles = np.linspace(0, 2 * np.pi, seq_len, endpoint=False)
    positions = np.column_stack([np.cos(angles), np.sin(angles)]) * 2
    
    def update(frame):
        layer_idx = frame
        ax.clear()
        
        # Draw tokens
        for i, pos in enumerate(positions):
            circle = Circle(pos, 0.15, color='lightblue', ec='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(pos[0], pos[1], str(i), ha='center', va='center', fontsize=10, weight='bold')
        
        # Draw attention connections
        if layer_idx < n_layers:
            attn = state.get_attention_weights(layer_idx)[0].mean(0)  # Average over heads
            attn = attn.cpu().detach().numpy()
            
            for i in range(seq_len):
                for j in range(seq_len):
                    if i != j and attn[i, j] > 0.1:  # Threshold for visibility
                        arrow = FancyArrowPatch(
                            positions[j], positions[i],
                            arrowstyle='->', mutation_scale=20,
                            linewidth=attn[i, j] * 5,
                            alpha=attn[i, j],
                            color='red'
                        )
                        ax.add_patch(arrow)
        
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'Token Interactions - Layer {layer_idx}', fontsize=16, weight='bold', pad=20)
        
        return [ax]
    
    anim = animation.FuncAnimation(
        fig, update, frames=n_layers,
        interval=1000 // fps, blit=False, repeat=True
    )
    
    anim.save(save_path, writer='pillow', fps=fps)
    plt.close()
    
    return anim


def animate_information_diffusion(
    state: any,
    source_position: int = 0,
    save_path: str = 'outputs/animations/information_diffusion.gif',
    fps: int = 3
) -> animation.FuncAnimation:
    """
    Animate how information from one position diffuses to others.
    
    Shows the spread of influence from a source token through
    the network via attention mechanisms.
    
    Args:
        state: TransformerState object
        source_position: Position to track influence from
        save_path: Path to save animation
        fps: Frames per second
        
    Returns:
        Animation object
    """
    n_layers = len(state.layer_states)
    seq_len = state.embeddings.shape[1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    def update(frame):
        layer_idx = frame
        
        # Left: Heatmap of influence
        ax1.clear()
        if layer_idx < n_layers:
            attn = state.get_attention_weights(layer_idx)[0].mean(0)  # Average over heads
            influence = attn[:, source_position].cpu().detach().numpy()
            
            ax1.bar(range(seq_len), influence, color='steelblue', alpha=0.7, edgecolor='black')
            ax1.axvline(source_position, color='red', linestyle='--', linewidth=2, label=f'Source (pos {source_position})')
            ax1.set_xlabel('Position')
            ax1.set_ylabel('Influence')
            ax1.set_title(f'Layer {layer_idx}: Influence from Position {source_position}')
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')
        
        # Right: Cumulative influence over layers
        ax2.clear()
        cumulative_influence = np.zeros(seq_len)
        
        for l in range(layer_idx + 1):
            if l < n_layers:
                attn = state.get_attention_weights(l)[0].mean(0)
                cumulative_influence += attn[:, source_position].cpu().detach().numpy()
        
        ax2.plot(cumulative_influence, 'o-', linewidth=2, markersize=8, color='darkgreen')
        ax2.axvline(source_position, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Cumulative Influence')
        ax2.set_title('Accumulated Influence')
        ax2.grid(True, alpha=0.3)
        
        return [ax1, ax2]
    
    anim = animation.FuncAnimation(
        fig, update, frames=n_layers,
        interval=1000 // fps, blit=False, repeat=True
    )
    
    anim.save(save_path, writer='pillow', fps=fps)
    plt.close()
    
    return anim


def animate_residual_evolution(
    state: any,
    save_path: str = 'outputs/animations/residual_evolution.gif',
    fps: int = 2
) -> animation.FuncAnimation:
    """
    Animate residual stream evolution with detailed analysis.
    
    Shows how the residual stream changes through layers,
    with norm, direction, and magnitude analysis.
    
    Args:
        state: TransformerState object
        save_path: Path to save animation
        fps: Frames per second
        
    Returns:
        Animation object
    """
    residuals = state.get_residual_stream()
    n_checkpoints = len(residuals)
    seq_len = residuals[0].shape[1]
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    def update(frame):
        checkpoint_idx = frame
        
        # Top left: Current residual heatmap
        ax = fig.add_subplot(gs[0, 0])
        ax.clear()
        data = residuals[checkpoint_idx][0, :, :64].cpu().detach().numpy()
        im = ax.imshow(data.T, cmap='RdBu_r', aspect='auto')
        title = 'Embedding' if checkpoint_idx == 0 else f'Layer {checkpoint_idx - 1}'
        ax.set_title(f'{title} Residual State')
        ax.set_xlabel('Position')
        ax.set_ylabel('Feature (first 64)')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Top middle: Norm evolution
        ax = fig.add_subplot(gs[0, 1])
        ax.clear()
        norms = [r[0].norm(dim=-1).mean().item() for r in residuals[:checkpoint_idx + 1]]
        ax.plot(range(-1, checkpoint_idx), norms, 'o-', linewidth=2, markersize=10, color='blue')
        ax.plot(checkpoint_idx - 1, norms[-1], 'ro', markersize=15)
        ax.set_xlim(-1, n_checkpoints - 2)
        ax.set_xlabel('Checkpoint')
        ax.set_ylabel('Average Norm')
        ax.set_title('Norm Evolution')
        ax.grid(True, alpha=0.3)
        
        # Top right: Per-position norms
        ax = fig.add_subplot(gs[0, 2])
        ax.clear()
        norms_per_pos = residuals[checkpoint_idx][0].norm(dim=-1).cpu().detach().numpy()
        ax.bar(range(seq_len), norms_per_pos, color='green', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Position')
        ax.set_ylabel('Norm')
        ax.set_title('Per-Position Norms')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Bottom left: Change from previous
        ax = fig.add_subplot(gs[1, 0])
        ax.clear()
        if checkpoint_idx > 0:
            change = (residuals[checkpoint_idx] - residuals[checkpoint_idx - 1])[0]
            change_norms = change.norm(dim=-1).cpu().detach().numpy()
            ax.bar(range(seq_len), change_norms, color='orange', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Position')
            ax.set_ylabel('Change Magnitude')
            ax.set_title(f'Change from Previous Checkpoint')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Bottom middle: Feature statistics
        ax = fig.add_subplot(gs[1, 1])
        ax.clear()
        data = residuals[checkpoint_idx][0].flatten().cpu().detach().numpy()
        ax.hist(data, bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.3f}')
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        ax.set_title('Feature Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Bottom right: Correlation matrix
        ax = fig.add_subplot(gs[1, 2])
        ax.clear()
        corr = torch.corrcoef(residuals[checkpoint_idx][0])
        im = ax.imshow(corr.cpu().detach().numpy(), cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xlabel('Position')
        ax.set_ylabel('Position')
        ax.set_title('Position Correlations')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        title = 'Embedding' if checkpoint_idx == 0 else f'After Layer {checkpoint_idx - 1}'
        fig.suptitle(f'Residual Evolution: {title}', fontsize=16, weight='bold')
        
        return fig
    
    anim = animation.FuncAnimation(
        fig, update, frames=n_checkpoints,
        interval=1000 // fps, blit=False, repeat=True
    )
    
    anim.save(save_path, writer='pillow', fps=fps)
    plt.close()
    
    return anim


def animate_head_comparison(
    state: any,
    save_path: str = 'outputs/animations/head_comparison.gif',
    fps: int = 2
) -> animation.FuncAnimation:
    """
    Animate comparison of all attention heads side-by-side.
    
    Shows how heads differ in their attention patterns,
    revealing specialization and complementarity.
    
    Args:
        state: TransformerState object
        save_path: Path to save animation
        fps: Frames per second
        
    Returns:
        Animation object
    """
    n_layers = len(state.layer_states)
    
    fig = plt.figure(figsize=(16, 10))
    
    def update(frame):
        layer_idx = frame
        fig.clear()
        
        attn = state.get_attention_weights(layer_idx)[0]  # [n_heads, seq_len, seq_len]
        n_heads = attn.shape[0]
        
        n_cols = min(4, n_heads)
        n_rows = (n_heads + n_cols - 1) // n_cols
        
        for head_idx in range(n_heads):
            ax = fig.add_subplot(n_rows, n_cols, head_idx + 1)
            
            data = attn[head_idx].cpu().detach().numpy()
            im = ax.imshow(data, cmap='viridis', aspect='auto')
            ax.set_title(f'Head {head_idx}', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add entropy as text
            entropy = -(data * np.log(data + 1e-10)).sum(axis=-1).mean()
            ax.text(0.5, -0.15, f'E: {entropy:.2f}', transform=ax.transAxes,
                   ha='center', fontsize=8)
        
        fig.suptitle(f'Layer {layer_idx}: All Attention Heads', fontsize=16, weight='bold')
        plt.tight_layout()
        
        return [fig]
    
    anim = animation.FuncAnimation(
        fig, update, frames=n_layers,
        interval=1000 // fps, blit=False, repeat=True
    )
    
    anim.save(save_path, writer='pillow', fps=fps)
    plt.close()
    
    return anim


