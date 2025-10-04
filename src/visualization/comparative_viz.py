"""Comparative visualization tools for analyzing multiple models or states."""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Dict
import seaborn as sns


def visualize_model_comparison(
    states: List[any],
    labels: List[str],
    metrics: List[str] = ['attention_entropy', 'residual_norm', 'gradient_flow'],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare multiple transformer states side-by-side.
    
    Args:
        states: List of TransformerState objects to compare
        labels: Labels for each state
        metrics: Which metrics to compare
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    n_states = len(states)
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(n_metrics, n_states, figsize=(5*n_states, 4*n_metrics))
    
    if n_metrics == 1:
        axes = axes.reshape(1, -1)
    if n_states == 1:
        axes = axes.reshape(-1, 1)
    
    for metric_idx, metric in enumerate(metrics):
        for state_idx, (state, label) in enumerate(zip(states, labels)):
            ax = axes[metric_idx, state_idx]
            
            if metric == 'attention_entropy':
                _plot_attention_entropy(state, ax, label)
            elif metric == 'residual_norm':
                _plot_residual_norms(state, ax, label)
            elif metric == 'gradient_flow':
                _plot_gradient_placeholder(ax, label)
            
            if metric_idx == 0:
                ax.set_title(label, fontsize=14, weight='bold')
            if state_idx == 0:
                ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def _plot_attention_entropy(state: any, ax: plt.Axes, label: str):
    """Plot attention entropy across layers."""
    n_layers = len(state.layer_states)
    entropies = []
    
    for layer_idx in range(n_layers):
        attn = state.get_attention_weights(layer_idx)[0]
        # Compute entropy for each head
        attn_np = attn.cpu().detach().numpy()
        layer_entropies = []
        for head in range(attn_np.shape[0]):
            head_attn = attn_np[head]
            entropy = -(head_attn * np.log(head_attn + 1e-10)).sum(axis=-1).mean()
            layer_entropies.append(entropy)
        entropies.append(np.mean(layer_entropies))
    
    ax.plot(range(n_layers), entropies, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Average Entropy')
    ax.grid(True, alpha=0.3)


def _plot_residual_norms(state: any, ax: plt.Axes, label: str):
    """Plot residual stream norms across layers."""
    residuals = state.get_residual_stream()
    norms = [r[0].norm(dim=-1).mean().item() for r in residuals]
    
    ax.plot(range(-1, len(residuals)-1), norms, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Average Norm')
    ax.grid(True, alpha=0.3)


def _plot_gradient_placeholder(ax: plt.Axes, label: str):
    """Placeholder for gradient flow visualization."""
    ax.text(0.5, 0.5, 'Requires model with\ngradients', 
           ha='center', va='center', transform=ax.transAxes,
           fontsize=12, color='gray')
    ax.set_xticks([])
    ax.set_yticks([])


def compare_attention_patterns(
    states: List[any],
    labels: List[str],
    layer_idx: int = 0,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare attention patterns across multiple models at a specific layer.
    
    Args:
        states: List of TransformerState objects
        labels: Labels for each state
        layer_idx: Which layer to compare
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    n_states = len(states)
    
    fig, axes = plt.subplots(1, n_states, figsize=(6*n_states, 5))
    if n_states == 1:
        axes = [axes]
    
    for idx, (state, label) in enumerate(zip(states, labels)):
        ax = axes[idx]
        
        attn = state.get_attention_weights(layer_idx)[0].mean(0)  # Average over heads
        attn_np = attn.cpu().detach().numpy()
        
        im = ax.imshow(attn_np, cmap='viridis', aspect='auto')
        ax.set_title(f'{label}\nLayer {layer_idx}', fontsize=14, weight='bold')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def compare_layer_statistics(
    states: List[any],
    labels: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare statistical properties across layers for multiple models.
    
    Args:
        states: List of TransformerState objects
        labels: Labels for each state
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    # 1. Residual norms
    ax = axes[0]
    for state, label in zip(states, labels):
        residuals = state.get_residual_stream()
        norms = [r[0].norm(dim=-1).mean().item() for r in residuals]
        ax.plot(range(-1, len(residuals)-1), norms, 'o-', linewidth=2, markersize=6, label=label)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Average Norm', fontsize=12)
    ax.set_title('Residual Stream Norms', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Attention entropy
    ax = axes[1]
    for state, label in zip(states, labels):
        n_layers = len(state.layer_states)
        entropies = []
        for layer_idx in range(n_layers):
            attn = state.get_attention_weights(layer_idx)[0]
            attn_np = attn.cpu().detach().numpy()
            entropy = -(attn_np * np.log(attn_np + 1e-10)).sum(axis=-1).mean()
            entropies.append(entropy)
        ax.plot(range(n_layers), entropies, 'o-', linewidth=2, markersize=6, label=label)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Average Entropy', fontsize=12)
    ax.set_title('Attention Entropy by Layer', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Sequence length comparison
    ax = axes[2]
    seq_lengths = [state.embeddings.shape[1] for state in states]
    d_models = [state.embeddings.shape[2] for state in states]
    ax.bar(labels, seq_lengths, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Sequence Length', fontsize=12)
    ax.set_title('Sequence Length Comparison', fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Model dimension comparison
    ax = axes[3]
    ax.bar(labels, d_models, alpha=0.7, edgecolor='black', color='orange')
    ax.set_ylabel('Model Dimension', fontsize=12)
    ax.set_title('Model Dimension Comparison', fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_comparison_dashboard(
    states: List[any],
    labels: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive comparison dashboard.
    
    Args:
        states: List of TransformerState objects
        labels: Labels for each state
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure with multiple comparison views
    """
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Residual norms
    ax = fig.add_subplot(gs[0, :])
    for state, label in zip(states, labels):
        residuals = state.get_residual_stream()
        norms = [r[0].norm(dim=-1).mean().item() for r in residuals]
        ax.plot(range(-1, len(residuals)-1), norms, 'o-', linewidth=2, markersize=8, label=label)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Average Norm', fontsize=12)
    ax.set_title('Residual Stream Evolution Comparison', fontsize=16, weight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Attention patterns for each model
    for idx, (state, label) in enumerate(zip(states[:3], labels[:3])):  # Max 3 models
        ax = fig.add_subplot(gs[1, idx])
        attn = state.get_attention_weights(0)[0].mean(0)
        im = ax.imshow(attn.cpu().detach().numpy(), cmap='viridis', aspect='auto')
        ax.set_title(f'{label}\nLayer 0 Attention', fontsize=12, weight='bold')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Summary statistics
    ax = fig.add_subplot(gs[2, :2])
    summary_data = []
    for state, label in zip(states, labels):
        seq_len = state.embeddings.shape[1]
        d_model = state.embeddings.shape[2]
        n_layers = len(state.layer_states)
        summary_data.append([label, seq_len, d_model, n_layers])
    
    # Create table
    table = ax.table(
        cellText=summary_data,
        colLabels=['Model', 'Seq Len', 'd_model', 'Layers'],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    ax.axis('off')
    ax.set_title('Model Configurations', fontsize=14, weight='bold', pad=20)
    
    # Statistics comparison
    ax = fig.add_subplot(gs[2, 2])
    for state, label in zip(states, labels):
        n_layers = len(state.layer_states)
        entropies = []
        for layer_idx in range(n_layers):
            attn = state.get_attention_weights(layer_idx)[0]
            entropy = -(attn.cpu().detach().numpy() * np.log(attn.cpu().detach().numpy() + 1e-10)).sum(axis=-1).mean()
            entropies.append(entropy)
        ax.plot(range(n_layers), entropies, 'o-', linewidth=2, label=label)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Entropy')
    ax.set_title('Attention Entropy', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('Model Comparison Dashboard', fontsize=18, weight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig



