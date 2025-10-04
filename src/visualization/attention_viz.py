"""Attention weight visualization tools."""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def visualize_attention(
    attention_weights: torch.Tensor,
    layer_idx: int = 0,
    tokens: Optional[List[str]] = None,
    figsize: tuple = (12, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize attention weights for all heads in a layer.
    
    Args:
        attention_weights: Attention weights [batch, n_heads, seq_len, seq_len]
        layer_idx: Layer index (for title)
        tokens: Token strings for axis labels (optional)
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    if attention_weights.dim() == 4:
        attention_weights = attention_weights[0]  # Take first batch
    
    n_heads = attention_weights.shape[0]
    seq_len = attention_weights.shape[1]
    
    # Create subplots
    n_cols = min(4, n_heads)
    n_rows = (n_heads + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_heads == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Generate token labels if not provided
    if tokens is None:
        tokens = [f"T{i}" for i in range(seq_len)]
    
    # Plot each head
    for head_idx in range(n_heads):
        ax = axes[head_idx]
        weights = attention_weights[head_idx].cpu().detach().numpy()
        
        sns.heatmap(
            weights,
            ax=ax,
            cmap='viridis',
            xticklabels=tokens,
            yticklabels=tokens,
            cbar=True,
            square=True,
            vmin=0,
            vmax=weights.max()
        )
        ax.set_title(f'Head {head_idx}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
    
    # Remove extra subplots
    for idx in range(n_heads, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle(f'Attention Weights - Layer {layer_idx}', fontsize=16, y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_attention_heads(
    attention_weights: torch.Tensor,
    tokens: Optional[List[str]] = None,
    head_indices: Optional[List[int]] = None,
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Interactive attention visualization using Plotly.
    
    Args:
        attention_weights: Attention weights [batch, n_heads, seq_len, seq_len]
        tokens: Token strings for hover labels (optional)
        head_indices: Specific heads to plot (optional, defaults to all)
        save_path: Path to save HTML (optional)
        
    Returns:
        Plotly figure
    """
    if attention_weights.dim() == 4:
        attention_weights = attention_weights[0]  # Take first batch
    
    n_heads = attention_weights.shape[0]
    seq_len = attention_weights.shape[1]
    
    if head_indices is None:
        head_indices = list(range(n_heads))
    
    if tokens is None:
        tokens = [f"T{i}" for i in range(seq_len)]
    
    # Create subplots
    n_cols = min(4, len(head_indices))
    n_rows = (len(head_indices) + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f'Head {i}' for i in head_indices],
        horizontal_spacing=0.1,
        vertical_spacing=0.1
    )
    
    for idx, head_idx in enumerate(head_indices):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        weights = attention_weights[head_idx].cpu().detach().numpy()
        
        heatmap = go.Heatmap(
            z=weights,
            x=tokens,
            y=tokens,
            colorscale='Viridis',
            showscale=(col == n_cols),
            hovertemplate='Query: %{y}<br>Key: %{x}<br>Weight: %{z:.4f}<extra></extra>'
        )
        
        fig.add_trace(heatmap, row=row, col=col)
    
    fig.update_layout(
        title_text='Interactive Attention Weights',
        height=300 * n_rows,
        width=300 * n_cols
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def visualize_attention_patterns(
    state: any,
    layer_range: Optional[tuple] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize attention patterns across all layers.
    
    Args:
        state: TransformerState object
        layer_range: Tuple of (start_layer, end_layer) (optional)
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    n_layers = len(state.layer_states)
    
    if layer_range is None:
        layer_range = (0, n_layers)
    
    layers_to_plot = range(layer_range[0], layer_range[1])
    n_layers_plot = len(layers_to_plot)
    
    fig, axes = plt.subplots(1, n_layers_plot, figsize=(5 * n_layers_plot, 5))
    if n_layers_plot == 1:
        axes = [axes]
    
    for idx, layer_idx in enumerate(layers_to_plot):
        ax = axes[idx]
        
        # Average attention across heads
        attn_weights = state.get_attention_weights(layer_idx)
        avg_attn = attn_weights[0].mean(dim=0).cpu().detach().numpy()
        
        im = ax.imshow(avg_attn, cmap='viridis', aspect='auto')
        ax.set_title(f'Layer {layer_idx}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Average Attention Patterns Across Layers', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


