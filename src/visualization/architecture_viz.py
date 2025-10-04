"""Visualization of transformer architecture and structure itself."""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
from typing import Optional, Dict, List, Tuple
import seaborn as sns
from pathlib import Path


def visualize_model_architecture(
    config: any,
    save_path: Optional[str] = None,
    show_dimensions: bool = True
) -> plt.Figure:
    """
    Visualize the complete transformer architecture structurally.
    
    Shows all components: embeddings, transformer blocks, attention heads,
    FFN layers, and output projection with their dimensions and connections.
    
    Args:
        config: TransformerConfig object
        save_path: Path to save figure
        show_dimensions: Whether to display dimension annotations
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(16, 20))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 25)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'embedding': '#E8F5E9',
        'attention': '#E3F2FD',
        'ffn': '#FFF3E0',
        'norm': '#F3E5F5',
        'output': '#FFEBEE'
    }
    
    y_pos = 23
    
    # Title
    ax.text(5, y_pos, 'Transformer Architecture', 
            ha='center', fontsize=20, weight='bold')
    y_pos -= 1.5
    
    # Input
    ax.add_patch(FancyBboxPatch((3.5, y_pos), 3, 0.6, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#FFCDD2', edgecolor='black', linewidth=2))
    ax.text(5, y_pos + 0.3, 'Input Tokens', ha='center', va='center', fontsize=12, weight='bold')
    if show_dimensions:
        ax.text(6.8, y_pos + 0.3, f'[B, S]', ha='left', va='center', fontsize=9, style='italic')
    y_pos -= 1
    
    # Arrow
    ax.arrow(5, y_pos + 0.3, 0, -0.2, head_width=0.2, head_length=0.1, fc='black', ec='black')
    y_pos -= 0.5
    
    # Token Embedding
    ax.add_patch(FancyBboxPatch((2, y_pos), 6, 0.8, 
                                boxstyle="round,pad=0.1",
                                facecolor=colors['embedding'], edgecolor='black', linewidth=2))
    ax.text(5, y_pos + 0.4, f'Token Embedding (vocab={config.vocab_size:,})', 
            ha='center', va='center', fontsize=12, weight='bold')
    if show_dimensions:
        ax.text(5, y_pos + 0.1, f'→ d_model={config.d_model}', 
                ha='center', va='center', fontsize=9, style='italic')
    y_pos -= 1.2
    
    # Positional Encoding
    ax.add_patch(FancyBboxPatch((2, y_pos), 6, 0.6, 
                                boxstyle="round,pad=0.1",
                                facecolor=colors['embedding'], edgecolor='black', linewidth=2))
    ax.text(5, y_pos + 0.3, f'+ Positional Encoding', 
            ha='center', va='center', fontsize=11, weight='bold')
    y_pos -= 1
    
    # Arrow
    ax.arrow(5, y_pos + 0.3, 0, -0.2, head_width=0.2, head_length=0.1, fc='black', ec='black')
    y_pos -= 0.5
    
    # Transformer Blocks
    for layer_idx in range(min(config.n_layers, 3)):  # Show first 3 layers
        # Block container
        block_height = 4.5
        ax.add_patch(Rectangle((1.2, y_pos - block_height), 7.6, block_height,
                               facecolor='white', edgecolor='blue', linewidth=2.5,
                               linestyle='--', alpha=0.3))
        ax.text(0.5, y_pos - block_height/2, f'Layer\n{layer_idx}', 
                ha='center', va='center', fontsize=11, weight='bold',
                bbox=dict(boxstyle='circle', facecolor='lightblue', edgecolor='blue', linewidth=2))
        
        # Multi-Head Attention
        ax.add_patch(FancyBboxPatch((1.5, y_pos - 1.2), 7, 1, 
                                    boxstyle="round,pad=0.1",
                                    facecolor=colors['attention'], edgecolor='black', linewidth=2))
        ax.text(5, y_pos - 0.7, f'Multi-Head Self-Attention ({config.n_heads} heads)', 
                ha='center', va='center', fontsize=11, weight='bold')
        if show_dimensions:
            ax.text(5, y_pos - 1.0, f'head_dim = {config.d_model // config.n_heads}', 
                    ha='center', va='center', fontsize=9, style='italic')
        
        # Layer Norm
        y_pos -= 1.5
        ax.add_patch(FancyBboxPatch((3, y_pos), 4, 0.4, 
                                    boxstyle="round,pad=0.05",
                                    facecolor=colors['norm'], edgecolor='black', linewidth=1.5))
        ax.text(5, y_pos + 0.2, 'Layer Norm', ha='center', va='center', fontsize=10)
        
        # Feed-Forward Network
        y_pos -= 0.7
        ax.add_patch(FancyBboxPatch((1.5, y_pos - 1), 7, 0.9, 
                                    boxstyle="round,pad=0.1",
                                    facecolor=colors['ffn'], edgecolor='black', linewidth=2))
        ax.text(5, y_pos - 0.55, f'Feed-Forward Network', 
                ha='center', va='center', fontsize=11, weight='bold')
        if show_dimensions:
            ax.text(5, y_pos - 0.8, f'{config.d_model} → {config.d_ff} → {config.d_model}', 
                    ha='center', va='center', fontsize=9, style='italic')
        
        # Layer Norm
        y_pos -= 1.3
        ax.add_patch(FancyBboxPatch((3, y_pos), 4, 0.4, 
                                    boxstyle="round,pad=0.05",
                                    facecolor=colors['norm'], edgecolor='black', linewidth=1.5))
        ax.text(5, y_pos + 0.2, 'Layer Norm', ha='center', va='center', fontsize=10)
        
        y_pos -= 0.7
        
        # Arrow to next layer
        if layer_idx < min(config.n_layers, 3) - 1:
            ax.arrow(5, y_pos + 0.2, 0, -0.2, head_width=0.2, head_length=0.1, 
                    fc='black', ec='black')
            y_pos -= 0.5
    
    # Show "..." if more layers
    if config.n_layers > 3:
        ax.text(5, y_pos - 0.3, f'... {config.n_layers - 3} more layers ...', 
                ha='center', va='center', fontsize=11, style='italic', color='gray')
        y_pos -= 0.8
    
    # Final Layer Norm
    ax.add_patch(FancyBboxPatch((2.5, y_pos), 5, 0.5, 
                                boxstyle="round,pad=0.1",
                                facecolor=colors['norm'], edgecolor='black', linewidth=2))
    ax.text(5, y_pos + 0.25, 'Final Layer Norm', ha='center', va='center', 
            fontsize=11, weight='bold')
    y_pos -= 0.8
    
    # Output Projection
    ax.add_patch(FancyBboxPatch((2, y_pos), 6, 0.7, 
                                boxstyle="round,pad=0.1",
                                facecolor=colors['output'], edgecolor='black', linewidth=2))
    ax.text(5, y_pos + 0.35, f'Output Projection → vocab', 
            ha='center', va='center', fontsize=12, weight='bold')
    if show_dimensions:
        ax.text(5, y_pos + 0.05, f'{config.d_model} → {config.vocab_size:,}', 
                ha='center', va='center', fontsize=9, style='italic')
    y_pos -= 1
    
    # Output
    ax.add_patch(FancyBboxPatch((3.5, y_pos), 3, 0.6, 
                                boxstyle="round,pad=0.1",
                                facecolor='#FFCDD2', edgecolor='black', linewidth=2))
    ax.text(5, y_pos + 0.3, 'Logits', ha='center', va='center', fontsize=12, weight='bold')
    if show_dimensions:
        ax.text(6.8, y_pos + 0.3, f'[B, S, V]', ha='left', va='center', fontsize=9, style='italic')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=colors['embedding'], edgecolor='black', label='Embedding'),
        mpatches.Patch(facecolor=colors['attention'], edgecolor='black', label='Attention'),
        mpatches.Patch(facecolor=colors['ffn'], edgecolor='black', label='Feed-Forward'),
        mpatches.Patch(facecolor=colors['norm'], edgecolor='black', label='Layer Norm'),
        mpatches.Patch(facecolor=colors['output'], edgecolor='black', label='Output')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


def visualize_attention_mechanism(
    n_heads: int = 8,
    d_model: int = 512,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize the multi-head attention mechanism in detail.
    
    Shows Q, K, V projections, attention computation, and multi-head concat.
    
    Args:
        n_heads: Number of attention heads
        d_model: Model dimension
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    head_dim = d_model // n_heads
    
    # Title
    ax.text(8, 11, 'Multi-Head Self-Attention Mechanism', 
            ha='center', fontsize=18, weight='bold')
    
    # Input
    y_pos = 10
    ax.add_patch(FancyBboxPatch((6, y_pos), 4, 0.6, 
                                boxstyle="round,pad=0.1",
                                facecolor='#E8F5E9', edgecolor='black', linewidth=2))
    ax.text(8, y_pos + 0.3, f'Input: [seq_len, d_model={d_model}]', 
            ha='center', va='center', fontsize=11, weight='bold')
    
    # Split into Q, K, V projections
    y_pos = 8.5
    for i, name in enumerate(['Query (Q)', 'Key (K)', 'Value (V)']):
        x_pos = 2 + i * 5
        ax.add_patch(FancyBboxPatch((x_pos, y_pos), 3, 0.7, 
                                    boxstyle="round,pad=0.1",
                                    facecolor='#BBDEFB', edgecolor='black', linewidth=2))
        ax.text(x_pos + 1.5, y_pos + 0.35, name, ha='center', va='center', 
                fontsize=11, weight='bold')
        
        # Arrows from input
        ax.arrow(8, y_pos + 1.5, x_pos + 1.5 - 8, -0.7, 
                head_width=0.15, head_length=0.1, fc='black', ec='black', alpha=0.6)
    
    # Multi-head split
    y_pos = 6.5
    ax.text(8, y_pos + 0.8, f'Split into {n_heads} heads (head_dim = {head_dim})', 
            ha='center', fontsize=10, style='italic', color='gray')
    
    # Show individual heads
    head_width = 14 / n_heads
    for head_idx in range(min(n_heads, 8)):  # Show max 8 heads
        x_pos = 1 + head_idx * head_width
        
        # Head box
        ax.add_patch(Rectangle((x_pos, y_pos - 1.5), head_width - 0.2, 1.3,
                               facecolor='#FFF9C4', edgecolor='black', 
                               linewidth=1.5, alpha=0.7))
        ax.text(x_pos + head_width/2 - 0.1, y_pos - 0.2, f'H{head_idx}', 
                ha='center', fontsize=8, weight='bold')
        
        # Attention computation (simplified)
        ax.text(x_pos + head_width/2 - 0.1, y_pos - 0.7, 'Q·K', 
                ha='center', fontsize=7)
        ax.text(x_pos + head_width/2 - 0.1, y_pos - 1.1, '×V', 
                ha='center', fontsize=7)
    
    # Concatenation
    y_pos = 4.5
    ax.add_patch(FancyBboxPatch((4, y_pos), 8, 0.6, 
                                boxstyle="round,pad=0.1",
                                facecolor='#C5CAE9', edgecolor='black', linewidth=2))
    ax.text(8, y_pos + 0.3, f'Concatenate Heads → [seq_len, d_model={d_model}]', 
            ha='center', va='center', fontsize=11, weight='bold')
    
    # Output projection
    y_pos = 3
    ax.add_patch(FancyBboxPatch((5, y_pos), 6, 0.6, 
                                boxstyle="round,pad=0.1",
                                facecolor='#B2DFDB', edgecolor='black', linewidth=2))
    ax.text(8, y_pos + 0.3, f'Output Projection W_O', 
            ha='center', va='center', fontsize=11, weight='bold')
    
    # Final output
    y_pos = 1.5
    ax.add_patch(FancyBboxPatch((6, y_pos), 4, 0.6, 
                                boxstyle="round,pad=0.1",
                                facecolor='#C8E6C9', edgecolor='black', linewidth=2))
    ax.text(8, y_pos + 0.3, f'Output: [seq_len, d_model={d_model}]', 
            ha='center', va='center', fontsize=11, weight='bold')
    
    # Add formula annotations
    ax.text(1, 2, 'Attention(Q,K,V) = softmax(QK^T / √d_k) V', 
            fontsize=12, family='monospace', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


def visualize_parameter_breakdown(
    model: nn.Module,
    config: any,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize parameter count breakdown by component.
    
    Shows where parameters are allocated in the model with
    pie chart and bar chart views.
    
    Args:
        model: Transformer model
        config: TransformerConfig
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Count parameters by component
    param_counts = {}
    
    for name, param in model.named_parameters():
        # Categorize by component type
        if 'token_embedding' in name:
            category = 'Token Embedding'
        elif 'positional_encoding' in name:
            category = 'Positional Encoding'
        elif 'attention' in name:
            if 'blocks' in name:
                category = 'Attention Layers'
            else:
                category = 'Other Attention'
        elif 'feed_forward' in name or 'ffn' in name:
            category = 'Feed-Forward Layers'
        elif 'norm' in name:
            category = 'Layer Normalization'
        elif 'output_projection' in name:
            category = 'Output Projection'
        else:
            category = 'Other'
        
        param_count = param.numel()
        param_counts[category] = param_counts.get(category, 0) + param_count
    
    total_params = sum(param_counts.values())
    
    # Create visualization
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Pie chart
    ax1 = fig.add_subplot(gs[0, 0])
    colors = plt.cm.Set3(np.linspace(0, 1, len(param_counts)))
    wedges, texts, autotexts = ax1.pie(
        param_counts.values(),
        labels=param_counts.keys(),
        colors=colors,
        autopct=lambda pct: f'{pct:.1f}%\n({pct * total_params / 100:.0f})',
        startangle=90,
        textprops={'fontsize': 9}
    )
    ax1.set_title(f'Parameter Distribution\n(Total: {total_params:,} parameters)', 
                  fontsize=14, weight='bold')
    
    # Bar chart
    ax2 = fig.add_subplot(gs[0, 1])
    categories = list(param_counts.keys())
    counts = list(param_counts.values())
    bars = ax2.barh(categories, counts, color=colors)
    ax2.set_xlabel('Parameter Count', fontsize=12)
    ax2.set_title('Parameters by Component', fontsize=14, weight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add values on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax2.text(count, i, f' {count:,}', va='center', fontsize=9)
    
    # Detailed breakdown table
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    # Create table data
    table_data = [
        ['Component', 'Parameters', 'Percentage', 'Memory (MB)']
    ]
    
    for category, count in sorted(param_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_params * 100
        memory_mb = count * 4 / 1024 / 1024  # Assuming float32
        table_data.append([
            category,
            f'{count:,}',
            f'{percentage:.2f}%',
            f'{memory_mb:.2f}'
        ])
    
    table_data.append([
        'TOTAL',
        f'{total_params:,}',
        '100.00%',
        f'{total_params * 4 / 1024 / 1024:.2f}'
    ])
    
    table = ax3.table(
        cellText=table_data,
        cellLoc='left',
        loc='center',
        colWidths=[0.3, 0.25, 0.15, 0.15]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style total row
    for i in range(4):
        table[(len(table_data) - 1, i)].set_facecolor('#FFEB3B')
        table[(len(table_data) - 1, i)].set_text_props(weight='bold')
    
    ax3.set_title('Detailed Parameter Breakdown', fontsize=14, weight='bold', pad=20)
    
    plt.suptitle(f'Transformer Model Analysis - {config.n_layers} Layers, '
                f'{config.n_heads} Heads, d_model={config.d_model}',
                fontsize=16, weight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


def visualize_computational_graph(
    config: any,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize the computational graph showing data flow.
    
    Shows how tensors flow through the model with shapes at each step.
    
    Args:
        config: TransformerConfig
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 18))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    # Title
    ax.text(7, 19, 'Computational Graph: Tensor Shapes Through Forward Pass', 
            ha='center', fontsize=16, weight='bold')
    
    y = 17.5
    
    # Helper function to draw node
    def draw_node(x, y, text, shape_text, color='lightblue'):
        ax.add_patch(FancyBboxPatch((x - 2, y - 0.4), 4, 0.7,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color, edgecolor='black', linewidth=1.5))
        ax.text(x, y, text, ha='center', va='center', fontsize=10, weight='bold')
        ax.text(x, y - 0.25, shape_text, ha='center', va='center', 
                fontsize=8, style='italic', color='darkblue')
    
    # Input
    draw_node(7, y, 'Input Tokens', '[B, S]', '#FFE0B2')
    y -= 1.2
    ax.arrow(7, y + 0.8, 0, -0.3, head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    # Token Embedding
    draw_node(7, y, 'Token Embedding', f'[B, S, {config.d_model}]', '#E1BEE7')
    y -= 1.2
    ax.arrow(7, y + 0.8, 0, -0.3, head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    # Positional Encoding (addition)
    draw_node(7, y, '+ Positional Encoding', f'[B, S, {config.d_model}]', '#E1BEE7')
    y -= 1.5
    ax.arrow(7, y + 0.8, 0, -0.3, head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    # Transformer block (show one in detail)
    ax.text(7, y, f'Transformer Block (×{config.n_layers})', 
            ha='center', fontsize=11, weight='bold', style='italic', color='darkred')
    y -= 0.8
    
    # Layer Norm
    draw_node(7, y, 'Layer Norm', f'[B, S, {config.d_model}]', '#F8BBD0')
    y -= 1.2
    
    # Multi-Head Attention
    draw_node(7, y, f'Multi-Head Attention ({config.n_heads}h)', 
             f'[B, S, {config.d_model}]', '#B3E5FC')
    y -= 1.2
    
    # Add & Norm (residual)
    draw_node(7, y, 'Add & Norm (Residual)', f'[B, S, {config.d_model}]', '#C5CAE9')
    y -= 1.2
    
    # Feed-Forward
    draw_node(7, y, f'Feed-Forward ({config.d_ff})', f'[B, S, {config.d_model}]', '#FFCCBC')
    y -= 1.2
    
    # Add & Norm (residual)
    draw_node(7, y, 'Add & Norm (Residual)', f'[B, S, {config.d_model}]', '#C5CAE9')
    y -= 1.5
    
    ax.text(7, y + 0.2, '↓ ↓ ↓', ha='center', fontsize=14, color='gray')
    y -= 0.8
    
    # Final Layer Norm
    draw_node(7, y, 'Final Layer Norm', f'[B, S, {config.d_model}]', '#F8BBD0')
    y -= 1.2
    ax.arrow(7, y + 0.8, 0, -0.3, head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    # Output Projection
    draw_node(7, y, 'Linear (Output Projection)', f'[B, S, {config.vocab_size}]', '#A5D6A7')
    y -= 1.2
    ax.arrow(7, y + 0.8, 0, -0.3, head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    # Logits
    draw_node(7, y, 'Logits', f'[B, S, {config.vocab_size}]', '#FFE0B2')
    
    # Add legend for shapes
    ax.text(1, 1, 'B = Batch size\nS = Sequence length\nV = Vocabulary size', 
            fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


def visualize_information_highways_diagram(
    n_layers: int = 4,
    seq_len: int = 8,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize the two information highways concept.
    
    Shows vertical (residual) and horizontal (attention) information flow
    as a conceptual diagram.
    
    Args:
        n_layers: Number of layers to show
        seq_len: Sequence length to show
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(-1, seq_len + 1)
    ax.set_ylim(-1, n_layers + 2)
    ax.axis('off')
    
    # Title
    ax.text(seq_len / 2, n_layers + 1.5, 
            'Two Information Highways: Residual Stream + K/V Stream',
            ha='center', fontsize=18, weight='bold')
    
    # Draw grid of positions (nodes)
    node_positions = {}
    for layer in range(n_layers + 1):
        for pos in range(seq_len):
            x, y = pos, layer
            node_positions[(layer, pos)] = (x, y)
            
            # Color based on layer
            if layer == 0:
                color = '#C8E6C9'  # Green for input
            elif layer == n_layers:
                color = '#FFCDD2'  # Red for output
            else:
                color = '#BBDEFB'  # Blue for hidden
            
            circle = Circle((x, y), 0.15, facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(circle)
    
    # Draw vertical arrows (residual stream)
    for pos in range(seq_len):
        for layer in range(n_layers):
            x1, y1 = node_positions[(layer, pos)]
            x2, y2 = node_positions[(layer + 1, pos)]
            ax.arrow(x1, y1 + 0.15, 0, y2 - y1 - 0.3,
                    head_width=0.1, head_length=0.08,
                    fc='green', ec='green', alpha=0.6, linewidth=2)
    
    # Draw horizontal arrows (attention/K/V stream) - sample
    for layer in range(1, n_layers + 1):
        for pos in range(1, seq_len):
            # Draw attention from previous positions
            for prev_pos in range(max(0, pos - 3), pos):
                x1, y1 = node_positions[(layer, prev_pos)]
                x2, y2 = node_positions[(layer, pos)]
                
                # Curved arrow
                ax.annotate('', xy=(x2 - 0.15, y2), xytext=(x1 + 0.15, y1),
                           arrowprops=dict(arrowstyle='->', lw=1, alpha=0.3,
                                         color='purple',
                                         connectionstyle='arc3,rad=0.3'))
    
    # Add labels
    ax.text(-0.7, n_layers / 2, 'Layers →', rotation=90, ha='center', va='center',
            fontsize=14, weight='bold')
    ax.text(seq_len / 2, -0.7, 'Token Positions →', ha='center', va='center',
            fontsize=14, weight='bold')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='green', edgecolor='green', label='Residual Stream (Vertical)', alpha=0.6),
        mpatches.Patch(facecolor='purple', edgecolor='purple', label='K/V Stream (Horizontal)', alpha=0.3),
        mpatches.Patch(facecolor='#C8E6C9', edgecolor='black', label='Input Embeddings'),
        mpatches.Patch(facecolor='#BBDEFB', edgecolor='black', label='Hidden Layers'),
        mpatches.Patch(facecolor='#FFCDD2', edgecolor='black', label='Output Layer')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    # Add text annotations
    ax.text(seq_len + 0.5, n_layers - 1, 
            'VERTICAL:\nResidual connections\ncarry information\nthrough layers',
            fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    ax.text(seq_len + 0.5, 2,
            'HORIZONTAL:\nAttention allows\npositions to\ncommunicate',
            fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='plum', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


def create_all_architecture_visualizations(
    model: nn.Module,
    config: any,
    output_dir: str = 'outputs/visualizations'
) -> Dict[str, Path]:
    """
    Create all architecture visualizations and save them.
    
    Args:
        model: Transformer model
        config: TransformerConfig
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary mapping visualization names to saved paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    print("\nGenerating architecture visualizations...")
    
    # 1. Model architecture
    print("  Creating model architecture diagram...")
    path = output_dir / 'architecture_model.png'
    visualize_model_architecture(config, save_path=str(path))
    saved_files['model_architecture'] = path
    print(f"    ✓ Saved: {path}")
    
    # 2. Attention mechanism
    print("  Creating attention mechanism diagram...")
    path = output_dir / 'architecture_attention.png'
    visualize_attention_mechanism(config.n_heads, config.d_model, save_path=str(path))
    saved_files['attention_mechanism'] = path
    print(f"    ✓ Saved: {path}")
    
    # 3. Parameter breakdown
    print("  Creating parameter breakdown...")
    path = output_dir / 'architecture_parameters.png'
    visualize_parameter_breakdown(model, config, save_path=str(path))
    saved_files['parameter_breakdown'] = path
    print(f"    ✓ Saved: {path}")
    
    # 4. Computational graph
    print("  Creating computational graph...")
    path = output_dir / 'architecture_computation.png'
    visualize_computational_graph(config, save_path=str(path))
    saved_files['computational_graph'] = path
    print(f"    ✓ Saved: {path}")
    
    # 5. Information highways
    print("  Creating information highways diagram...")
    path = output_dir / 'architecture_highways.png'
    visualize_information_highways_diagram(config.n_layers, 8, save_path=str(path))
    saved_files['information_highways'] = path
    print(f"    ✓ Saved: {path}")
    
    print(f"\n✓ Created {len(saved_files)} architecture visualizations")
    
    return saved_files

