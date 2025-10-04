"""Visualization tools for training dynamics."""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
import seaborn as sns


class TrainingVisualizer:
    """
    Tracks and visualizes training dynamics.
    
    Monitors:
    - Loss curves
    - Learning rates
    - Gradient statistics
    - Attention pattern evolution
    - Weight distributions
    """
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'gradient_norms': [],
            'weight_norms': [],
            'attention_entropies': []
        }
        self.epoch = 0
    
    def log_metrics(
        self,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        weight_norm: Optional[float] = None,
        attention_entropy: Optional[float] = None
    ):
        """Log metrics for current epoch."""
        if train_loss is not None:
            self.history['train_loss'].append(train_loss)
        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
        if learning_rate is not None:
            self.history['learning_rates'].append(learning_rate)
        if gradient_norm is not None:
            self.history['gradient_norms'].append(gradient_norm)
        if weight_norm is not None:
            self.history['weight_norms'].append(weight_norm)
        if attention_entropy is not None:
            self.history['attention_entropies'].append(attention_entropy)
        
        self.epoch += 1
    
    def plot_training_curves(
        self,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot comprehensive training curves."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Loss curves
        ax = axes[0, 0]
        if self.history['train_loss']:
            ax.plot(self.history['train_loss'], label='Train', linewidth=2)
        if self.history['val_loss']:
            ax.plot(self.history['val_loss'], label='Val', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning rate
        ax = axes[0, 1]
        if self.history['learning_rates']:
            ax.plot(self.history['learning_rates'], linewidth=2, color='orange')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        # Gradient norms
        ax = axes[0, 2]
        if self.history['gradient_norms']:
            ax.plot(self.history['gradient_norms'], linewidth=2, color='green')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Gradient Norm')
            ax.set_title('Gradient Magnitude')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        # Weight norms
        ax = axes[1, 0]
        if self.history['weight_norms']:
            ax.plot(self.history['weight_norms'], linewidth=2, color='purple')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Weight Norm')
            ax.set_title('Model Weight Magnitude')
            ax.grid(True, alpha=0.3)
        
        # Attention entropy
        ax = axes[1, 1]
        if self.history['attention_entropies']:
            ax.plot(self.history['attention_entropies'], linewidth=2, color='red')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Entropy')
            ax.set_title('Attention Entropy (Concentration)')
            ax.grid(True, alpha=0.3)
        
        # Loss vs. gradient norm
        ax = axes[1, 2]
        if self.history['train_loss'] and self.history['gradient_norms']:
            min_len = min(len(self.history['train_loss']), len(self.history['gradient_norms']))
            ax.scatter(
                self.history['gradient_norms'][:min_len],
                self.history['train_loss'][:min_len],
                alpha=0.6
            )
            ax.set_xlabel('Gradient Norm')
            ax.set_ylabel('Train Loss')
            ax.set_title('Loss vs. Gradient Magnitude')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Training Dynamics', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_history(self, path: str):
        """Save training history to file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load_history(self, path: str):
        """Load training history from file."""
        import json
        with open(path, 'r') as f:
            self.history = json.load(f)
        self.epoch = max(
            len(v) for v in self.history.values() if isinstance(v, list)
        )


def visualize_weight_distributions(
    model: torch.nn.Module,
    epoch: int = 0,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize weight distribution evolution during training.
    
    Args:
        model: Transformer model
        epoch: Current epoch number
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    plot_idx = 0
    for name, param in model.named_parameters():
        if plot_idx >= 9:
            break
        
        if param.dim() >= 2:  # Only plot weight matrices
            ax = axes[plot_idx]
            
            weights = param.detach().cpu().numpy().flatten()
            ax.hist(weights, bins=50, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Count')
            ax.set_title(f'{name}\nMean: {weights.mean():.4f}, Std: {weights.std():.4f}')
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
    
    # Remove unused subplots
    for idx in range(plot_idx, 9):
        fig.delaxes(axes[idx])
    
    plt.suptitle(f'Weight Distributions - Epoch {epoch}', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_learning_dynamics(
    losses: List[float],
    gradients: List[Dict[str, float]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize how learning evolves over training steps.
    
    Args:
        losses: List of loss values
        gradients: List of dicts with layer-wise gradient stats
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curve with smoothing
    ax = axes[0, 0]
    steps = range(len(losses))
    ax.plot(steps, losses, alpha=0.3, color='blue', label='Raw')
    
    # Smooth with moving average
    if len(losses) > 10:
        window = min(50, len(losses) // 10)
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax.plot(range(len(smoothed)), smoothed, color='blue', linewidth=2, label='Smoothed')
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Loss change rate
    ax = axes[0, 1]
    if len(losses) > 1:
        loss_changes = np.diff(losses)
        ax.plot(loss_changes, alpha=0.6, color='red')
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss Change')
        ax.set_title('Loss Change Rate (Δ Loss)')
        ax.grid(True, alpha=0.3)
    
    # Gradient statistics over time
    ax = axes[1, 0]
    if gradients:
        # Extract first layer gradient norms as example
        first_layer_key = list(gradients[0].keys())[0]
        grad_norms = [g[first_layer_key] for g in gradients if first_layer_key in g]
        ax.plot(grad_norms, linewidth=2, color='green')
        ax.set_xlabel('Step')
        ax.set_ylabel('Gradient Norm')
        ax.set_title(f'Gradient Evolution ({first_layer_key})')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    # Learning rate vs loss improvement
    ax = axes[1, 1]
    if len(losses) > 10:
        window = 10
        improvements = []
        for i in range(window, len(losses)):
            improvement = losses[i - window] - losses[i]
            improvements.append(improvement)
        
        ax.plot(range(window, len(losses)), improvements, linewidth=2, color='purple')
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Step')
        ax.set_ylabel(f'Loss Improvement (over {window} steps)')
        ax.set_title('Learning Progress')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


