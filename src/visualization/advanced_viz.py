"""Advanced visualization methods for deep analysis."""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Dict, Tuple
import seaborn as sns
from matplotlib.patches import FancyArrowPatch
import networkx as nx


def visualize_gradient_flow(
    model: torch.nn.Module,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize gradient magnitudes through the network.
    
    Shows which parameters receive strong vs. weak gradients,
    helping diagnose vanishing/exploding gradients.
    
    Args:
        model: Transformer model with computed gradients
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    layers = []
    avg_grads = []
    max_grads = []
    
    for name, param in model.named_parameters():
        if param.grad is not None and param.requires_grad:
            layers.append(name)
            avg_grads.append(param.grad.abs().mean().item())
            max_grads.append(param.grad.abs().max().item())
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Average gradients
    ax1.bar(range(len(layers)), avg_grads, alpha=0.7, color='skyblue')
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels(layers, rotation=90, ha='right')
    ax1.set_ylabel('Average Gradient Magnitude')
    ax1.set_title('Gradient Flow Through Network (Average)')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Max gradients
    ax2.bar(range(len(layers)), max_grads, alpha=0.7, color='coral')
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels(layers, rotation=90, ha='right')
    ax2.set_ylabel('Max Gradient Magnitude')
    ax2.set_title('Gradient Flow Through Network (Max)')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_activation_patterns(
    state: any,
    layer_idx: int,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize activation patterns at a specific layer.
    
    Shows distribution and patterns of activations across positions
    and features, revealing what the layer is computing.
    
    Args:
        state: TransformerState object
        layer_idx: Layer to visualize
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    layer_output = state.layer_states[layer_idx].post_ffn[0]  # [seq_len, d_model]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Heatmap of activations
    ax = axes[0, 0]
    im = ax.imshow(
        layer_output.T.cpu().detach().numpy(),
        aspect='auto',
        cmap='RdBu_r',
        vmin=-layer_output.std().item() * 3,
        vmax=layer_output.std().item() * 3
    )
    ax.set_xlabel('Position')
    ax.set_ylabel('Feature Dimension')
    ax.set_title('Activation Heatmap')
    plt.colorbar(im, ax=ax)
    
    # Distribution of activations
    ax = axes[0, 1]
    activations_flat = layer_output.flatten().cpu().detach().numpy()
    ax.hist(activations_flat, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax.set_xlabel('Activation Value')
    ax.set_ylabel('Count')
    ax.set_title('Activation Distribution')
    ax.axvline(0, color='red', linestyle='--', label='Zero')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Per-position activation norms
    ax = axes[1, 0]
    norms = layer_output.norm(dim=-1).cpu().detach().numpy()
    ax.plot(norms, marker='o', linewidth=2)
    ax.set_xlabel('Position')
    ax.set_ylabel('L2 Norm')
    ax.set_title('Activation Magnitude per Position')
    ax.grid(True, alpha=0.3)
    
    # Correlation between positions
    ax = axes[1, 1]
    corr_matrix = torch.corrcoef(layer_output)
    im = ax.imshow(corr_matrix.cpu().detach().numpy(), cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xlabel('Position')
    ax.set_ylabel('Position')
    ax.set_title('Position Correlation')
    plt.colorbar(im, ax=ax)
    
    plt.suptitle(f'Activation Patterns - Layer {layer_idx}', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_causal_graph(
    n_layers: int,
    seq_len: int,
    highlight_path: Optional[List[Tuple[int, int]]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize the transformer as a causal graph.
    
    Shows the network as a directed graph with nodes at each (layer, position)
    and edges showing possible information flow paths.
    
    Args:
        n_layers: Number of layers
        seq_len: Sequence length
        highlight_path: Optional path to highlight
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Create grid of nodes
    node_spacing_x = 1.0
    node_spacing_y = 1.0
    
    # Draw nodes
    for layer in range(n_layers + 1):  # +1 for embedding layer
        for pos in range(seq_len):
            x = pos * node_spacing_x
            y = layer * node_spacing_y
            
            # Color based on layer
            if layer == 0:
                color = 'lightgreen'
            elif layer == n_layers:
                color = 'lightcoral'
            else:
                color = 'lightblue'
            
            ax.plot(x, y, 'o', color=color, markersize=8, zorder=3)
    
    # Draw edges (sample to avoid cluttering)
    # Vertical edges (residual connections)
    for layer in range(n_layers):
        for pos in range(0, seq_len, max(1, seq_len // 10)):
            x = pos * node_spacing_x
            y1 = layer * node_spacing_y
            y2 = (layer + 1) * node_spacing_y
            ax.plot([x, x], [y1, y2], 'k-', alpha=0.2, linewidth=1, zorder=1)
    
    # Horizontal edges (attention connections) - sample
    for layer in range(1, n_layers + 1):
        y = layer * node_spacing_y
        for pos in range(1, seq_len, max(1, seq_len // 5)):
            for prev_pos in range(0, pos, max(1, pos // 3)):
                x1 = prev_pos * node_spacing_x
                x2 = pos * node_spacing_x
                ax.plot([x1, x2], [y, y], 'purple', alpha=0.1, linewidth=0.5, zorder=1)
    
    # Highlight specific path if provided
    if highlight_path:
        for i in range(len(highlight_path) - 1):
            layer1, pos1 = highlight_path[i]
            layer2, pos2 = highlight_path[i + 1]
            
            x1 = pos1 * node_spacing_x
            y1 = layer1 * node_spacing_y
            x2 = pos2 * node_spacing_x
            y2 = layer2 * node_spacing_y
            
            ax.plot([x1, x2], [y1, y2], 'r-', linewidth=3, alpha=0.8, zorder=2)
            ax.plot(x2, y2, 'ro', markersize=12, zorder=4)
    
    ax.set_xlabel('Position', fontsize=14)
    ax.set_ylabel('Layer', fontsize=14)
    ax.set_title('Transformer as Causal Graph', fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightgreen', label='Embedding'),
        Patch(facecolor='lightblue', label='Hidden Layers'),
        Patch(facecolor='lightcoral', label='Output'),
        plt.Line2D([0], [0], color='k', alpha=0.5, label='Residual (Vertical)'),
        plt.Line2D([0], [0], color='purple', alpha=0.5, label='Attention (Horizontal)')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_attention_flow_sankey(
    attention_weights: torch.Tensor,
    position_labels: Optional[List[str]] = None,
    threshold: float = 0.1,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create Sankey-style flow diagram for attention.
    
    Shows how information flows from keys to queries as
    flowing streams, with width proportional to attention weight.
    
    Args:
        attention_weights: Attention weights [batch, n_heads, seq_len, seq_len]
        position_labels: Labels for positions (optional)
        threshold: Minimum weight to display
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    if attention_weights.dim() == 4:
        attention_weights = attention_weights[0, 0]  # First batch, first head
    
    seq_len = attention_weights.shape[0]
    
    if position_labels is None:
        position_labels = [f"P{i}" for i in range(seq_len)]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Draw source (key) and target (query) positions
    source_x = 0
    target_x = 10
    y_spacing = 10.0 / max(1, seq_len - 1) if seq_len > 1 else 1
    
    # Source positions (keys)
    for i in range(seq_len):
        y = i * y_spacing
        ax.plot(source_x, y, 'o', color='blue', markersize=15, zorder=3)
        ax.text(source_x - 0.5, y, position_labels[i], ha='right', va='center', fontsize=10)
    
    # Target positions (queries)
    for i in range(seq_len):
        y = i * y_spacing
        ax.plot(target_x, y, 'o', color='red', markersize=15, zorder=3)
        ax.text(target_x + 0.5, y, position_labels[i], ha='left', va='center', fontsize=10)
    
    # Draw flows
    weights = attention_weights.cpu().detach().numpy()
    max_weight = weights.max()
    
    for query_idx in range(seq_len):
        for key_idx in range(seq_len):
            weight = weights[query_idx, key_idx]
            
            if weight > threshold:
                y_source = key_idx * y_spacing
                y_target = query_idx * y_spacing
                
                # Bezier curve for flow
                control_x = (source_x + target_x) / 2
                
                t = np.linspace(0, 1, 100)
                x = (1-t)**2 * source_x + 2*(1-t)*t * control_x + t**2 * target_x
                y = (1-t)**2 * y_source + 2*(1-t)*t * ((y_source + y_target)/2) + t**2 * y_target
                
                alpha = min(1.0, weight / max_weight)
                linewidth = weight / max_weight * 10
                
                ax.plot(x, y, 'purple', alpha=alpha * 0.5, linewidth=linewidth, zorder=1)
    
    ax.set_xlim(-2, 12)
    ax.set_ylim(-1, seq_len * y_spacing + 1)
    ax.set_title('Attention Flow Diagram', fontsize=16)
    ax.text(source_x, -0.5, 'Keys\n(Source)', ha='center', fontsize=12, weight='bold')
    ax.text(target_x, -0.5, 'Queries\n(Target)', ha='center', fontsize=12, weight='bold')
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_token_journey(
    state: any,
    token_position: int,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize a single token's journey through the network.
    
    Tracks how a token's representation evolves through layers,
    showing norm, sparsity, and correlation changes.
    
    Args:
        state: TransformerState object
        token_position: Position of token to track
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    n_layers = len(state.layer_states)
    residuals = state.get_residual_stream()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Norm evolution
    ax = axes[0, 0]
    norms = [res[0, token_position, :].norm().item() for res in residuals]
    ax.plot(range(-1, n_layers), norms, marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Layer (-1 = embedding)')
    ax.set_ylabel('L2 Norm')
    ax.set_title(f'Representation Norm Evolution (Position {token_position})')
    ax.grid(True, alpha=0.3)
    
    # Representation heatmap through layers
    ax = axes[0, 1]
    representations = torch.stack([res[0, token_position, :64] for res in residuals])
    im = ax.imshow(representations.T.cpu().detach().numpy(), aspect='auto', cmap='RdBu_r')
    ax.set_xlabel('Layer (-1 to N-1)')
    ax.set_ylabel('Feature Dim (first 64)')
    ax.set_title('Representation Evolution')
    plt.colorbar(im, ax=ax)
    
    # Sparsity evolution (% of near-zero activations)
    ax = axes[1, 0]
    sparsities = []
    for res in residuals:
        vec = res[0, token_position, :]
        sparsity = (vec.abs() < 0.01).float().mean().item()
        sparsities.append(sparsity)
    ax.plot(range(-1, n_layers), sparsities, marker='s', linewidth=2, markersize=8, color='orange')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Sparsity')
    ax.set_title('Activation Sparsity')
    ax.grid(True, alpha=0.3)
    
    # Attention received (how much other positions attend to this one)
    ax = axes[1, 1]
    attention_received = []
    for layer_idx in range(n_layers):
        attn = state.get_attention_weights(layer_idx)
        # Average attention TO this position across all queries and heads
        attn_to_pos = attn[0, :, :, token_position].mean().item()
        attention_received.append(attn_to_pos)
    
    ax.bar(range(n_layers), attention_received, alpha=0.7, color='green')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Average Attention Received')
    ax.set_title('How Much Others Attend Here')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Token Journey Through Network (Position {token_position})', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_head_specialization(
    state: any,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Analyze and visualize what each attention head specializes in.
    
    Computes statistics for each head to identify specialization patterns:
    - Local vs. global attention
    - Uniform vs. peaked attention
    - Previous token vs. broad context
    
    Args:
        state: TransformerState object
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    n_layers = len(state.layer_states)
    n_heads = state.get_attention_weights(0).shape[1]
    
    # Compute metrics for each head
    metrics = {
        'locality': np.zeros((n_layers, n_heads)),
        'entropy': np.zeros((n_layers, n_heads)),
        'prev_token': np.zeros((n_layers, n_heads)),
        'uniformity': np.zeros((n_layers, n_heads))
    }
    
    for layer_idx in range(n_layers):
        attn = state.get_attention_weights(layer_idx)[0]  # [n_heads, seq_len, seq_len]
        
        for head_idx in range(n_heads):
            head_attn = attn[head_idx]  # [seq_len, seq_len]
            
            # Locality: average distance of attention
            seq_len = head_attn.shape[0]
            distances = torch.arange(seq_len).float().unsqueeze(0) - torch.arange(seq_len).float().unsqueeze(1)
            avg_distance = (head_attn * distances.abs()).sum() / head_attn.sum()
            metrics['locality'][layer_idx, head_idx] = avg_distance.item()
            
            # Entropy: concentration of attention
            entropy = -(head_attn * torch.log(head_attn + 1e-10)).sum(dim=-1).mean()
            metrics['entropy'][layer_idx, head_idx] = entropy.item()
            
            # Previous token attention
            prev_token_attn = torch.diagonal(head_attn, offset=-1).mean()
            metrics['prev_token'][layer_idx, head_idx] = prev_token_attn.item()
            
            # Uniformity: how uniform the attention is
            uniformity = 1.0 / (head_attn.var(dim=-1).mean() + 1e-8)
            metrics['uniformity'][layer_idx, head_idx] = uniformity.item()
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metric_names = ['locality', 'entropy', 'prev_token', 'uniformity']
    titles = [
        'Attention Locality (lower = more local)',
        'Attention Entropy (higher = more diffuse)',
        'Previous Token Focus',
        'Attention Uniformity'
    ]
    
    for idx, (metric_name, title) in enumerate(zip(metric_names, titles)):
        ax = axes[idx // 2, idx % 2]
        
        im = ax.imshow(metrics[metric_name].T, aspect='auto', cmap='viridis')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Head')
        ax.set_title(title)
        ax.set_xticks(range(n_layers))
        ax.set_yticks(range(n_heads))
        plt.colorbar(im, ax=ax)
    
    plt.suptitle('Attention Head Specialization Analysis', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


