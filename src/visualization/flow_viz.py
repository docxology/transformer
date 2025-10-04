"""Visualization of residual stream and K/V stream information flow."""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List
import seaborn as sns


def visualize_residual_stream(
    state: any,
    position_idx: int = 0,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize the vertical residual stream flow through layers.
    
    Shows how information at a specific position evolves through layers.
    
    Args:
        state: TransformerState object
        position_idx: Token position to track
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    residuals = state.get_residual_stream()
    n_layers = len(residuals) - 1
    
    fig, axes = plt.subplots(1, len(residuals), figsize=(3 * len(residuals), 4))
    
    for idx, residual in enumerate(residuals):
        ax = axes[idx]
        
        # Extract vector at position
        vec = residual[0, position_idx, :].cpu().detach().numpy()
        
        # Plot as heatmap
        ax.imshow(
            vec.reshape(-1, 1),
            cmap='RdBu_r',
            aspect='auto',
            vmin=-vec.std() * 3,
            vmax=vec.std() * 3
        )
        
        if idx == 0:
            ax.set_title('Embedding')
        else:
            ax.set_title(f'Layer {idx-1}')
        
        ax.set_ylabel('Feature Dim')
        ax.set_xticks([])
    
    plt.suptitle(
        f'Residual Stream Flow (Position {position_idx})',
        fontsize=14,
        y=1.02
    )
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_kv_stream(
    state: any,
    layer_idx: int = 0,
    head_idx: int = 0,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize the horizontal K/V stream flow across positions.
    
    Shows how information flows horizontally via attention at a specific layer.
    
    Args:
        state: TransformerState object
        layer_idx: Layer to visualize
        head_idx: Attention head to visualize
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    attn_state = state.layer_states[layer_idx].attention_state
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Query
    Q = attn_state.query[0].cpu().detach().numpy()
    axes[0].imshow(Q.T, cmap='RdBu_r', aspect='auto')
    axes[0].set_title('Query (Q): "Where should I look?"')
    axes[0].set_xlabel('Token Position')
    axes[0].set_ylabel('Feature Dimension')
    
    # Key
    K = attn_state.key[0].cpu().detach().numpy()
    axes[1].imshow(K.T, cmap='RdBu_r', aspect='auto')
    axes[1].set_title('Key (K): "What futures should look here?"')
    axes[1].set_xlabel('Token Position')
    axes[1].set_ylabel('Feature Dimension')
    
    # Value
    V = attn_state.value[0].cpu().detach().numpy()
    axes[2].imshow(V.T, cmap='RdBu_r', aspect='auto')
    axes[2].set_title('Value (V): "What info to transmit?"')
    axes[2].set_xlabel('Token Position')
    axes[2].set_ylabel('Feature Dimension')
    
    plt.suptitle(
        f'K/V Stream - Layer {layer_idx}, Head {head_idx}',
        fontsize=14,
        y=1.02
    )
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_information_highways(
    state: any,
    position_indices: Optional[List[int]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize both information highways simultaneously.
    
    Shows the dual flow: vertical (residual) and horizontal (attention).
    
    Args:
        state: TransformerState object
        position_indices: Positions to highlight (optional)
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    n_layers = len(state.layer_states)
    seq_len = state.embeddings.shape[1]
    
    if position_indices is None:
        position_indices = [0, seq_len // 2, seq_len - 1]
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, n_layers, hspace=0.3, wspace=0.3)
    
    # Top row: Residual stream for each position
    for pos_idx, pos in enumerate(position_indices):
        ax = fig.add_subplot(gs[0, :])
        
        residuals = []
        for layer_idx in range(n_layers):
            res = state.layer_states[layer_idx].post_ffn
            residuals.append(res[0, pos, :].norm().item())
        
        ax.plot(residuals, marker='o', label=f'Position {pos}')
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Residual Norm')
    ax.set_title('Residual Stream (Vertical Flow)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Middle and bottom rows: Attention patterns per layer
    for layer_idx in range(min(n_layers, 6)):  # Show first 6 layers
        row = 1 if layer_idx < 3 else 2
        col = layer_idx % 3
        
        ax = fig.add_subplot(gs[row, col])
        
        attn_weights = state.get_attention_weights(layer_idx)
        avg_attn = attn_weights[0].mean(dim=0).cpu().detach().numpy()
        
        im = ax.imshow(avg_attn, cmap='viridis', aspect='auto')
        ax.set_title(f'Layer {layer_idx} Attention')
        ax.set_xlabel('Key Pos')
        ax.set_ylabel('Query Pos')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle(
        'Information Highways: Residual (Vertical) + Attention (Horizontal)',
        fontsize=16
    )
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_layer_statistics(
    state: any,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot statistical properties of each layer.
    
    Args:
        state: TransformerState object
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    n_layers = len(state.layer_states)
    
    # Collect statistics
    stats = {
        'residual_norm': [],
        'attn_entropy': [],
        'attn_sparsity': []
    }
    
    for layer_idx in range(n_layers):
        layer_state = state.layer_states[layer_idx]
        
        # Residual norm
        res_norm = layer_state.post_ffn.norm(dim=-1).mean().item()
        stats['residual_norm'].append(res_norm)
        
        # Attention entropy (measure of attention concentration)
        attn = layer_state.attention_state.attention_weights
        attn_flat = attn[0].mean(0).cpu().detach()  # Average over heads
        entropy = -(attn_flat * torch.log(attn_flat + 1e-10)).sum(dim=-1).mean().item()
        stats['attn_entropy'].append(entropy)
        
        # Attention sparsity (% of weights above threshold)
        sparsity = (attn_flat > 0.1).float().mean().item()
        stats['attn_sparsity'].append(sparsity)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(stats['residual_norm'], marker='o')
    axes[0].set_title('Residual Stream Norm')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('L2 Norm')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(stats['attn_entropy'], marker='o', color='orange')
    axes[1].set_title('Attention Entropy')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Entropy')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(stats['attn_sparsity'], marker='o', color='green')
    axes[2].set_title('Attention Sparsity')
    axes[2].set_xlabel('Layer')
    axes[2].set_ylabel('Sparsity')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


