"""Evaluation metrics for transformer models."""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np


def compute_perplexity(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """
    Compute perplexity from logits and targets.
    
    Perplexity = exp(cross_entropy_loss)
    Lower is better. Measures how surprised the model is.
    
    Args:
        logits: Model predictions [batch, seq_len, vocab_size]
        targets: Ground truth [batch, seq_len]
        ignore_index: Index to ignore in loss computation
        
    Returns:
        Perplexity value
    """
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=ignore_index,
        reduction='mean'
    )
    return torch.exp(loss).item()


def compute_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    top_k: int = 1
) -> float:
    """
    Compute top-k accuracy.
    
    Args:
        logits: Model predictions [batch, seq_len, vocab_size]
        targets: Ground truth [batch, seq_len]
        ignore_index: Index to ignore in computation
        top_k: K for top-k accuracy
        
    Returns:
        Accuracy as fraction in [0, 1]
    """
    logits = logits.view(-1, logits.size(-1))
    targets = targets.view(-1)
    
    # Filter out ignored indices
    mask = targets != ignore_index
    logits = logits[mask]
    targets = targets[mask]
    
    if targets.numel() == 0:
        return 0.0
    
    # Get top-k predictions
    _, top_k_preds = torch.topk(logits, k=top_k, dim=-1)
    
    # Check if target is in top-k
    targets_expanded = targets.unsqueeze(-1).expand_as(top_k_preds)
    correct = (top_k_preds == targets_expanded).any(dim=-1)
    
    return correct.float().mean().item()


@dataclass
class EvaluationResults:
    """Results from model evaluation."""
    perplexity: float
    loss: float
    accuracy: float
    top_5_accuracy: float
    num_samples: int
    num_tokens: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'perplexity': self.perplexity,
            'loss': self.loss,
            'accuracy': self.accuracy,
            'top_5_accuracy': self.top_5_accuracy,
            'num_samples': self.num_samples,
            'num_tokens': self.num_tokens
        }
    
    def __str__(self) -> str:
        """String representation."""
        return (
            f"Evaluation Results:\n"
            f"  Loss: {self.loss:.4f}\n"
            f"  Perplexity: {self.perplexity:.4f}\n"
            f"  Accuracy: {self.accuracy:.4f} ({self.accuracy*100:.2f}%)\n"
            f"  Top-5 Accuracy: {self.top_5_accuracy:.4f} ({self.top_5_accuracy*100:.2f}%)\n"
            f"  Samples: {self.num_samples}\n"
            f"  Tokens: {self.num_tokens}"
        )


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    ignore_index: int = -100
) -> EvaluationResults:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Transformer model
        dataloader: Evaluation dataloader
        device: Device to run on
        ignore_index: Index to ignore in metrics
        
    Returns:
        EvaluationResults object
    """
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    total_samples = 0
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            logits, _ = model(inputs)
            
            # Compute loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=ignore_index,
                reduction='sum'
            )
            
            # Track statistics
            mask = targets != ignore_index
            total_loss += loss.item()
            total_tokens += mask.sum().item()
            total_samples += inputs.size(0)
            
            # Store for accuracy computation
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())
    
    # Aggregate
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    # Concatenate all predictions
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute accuracies
    accuracy = compute_accuracy(all_logits, all_targets, ignore_index, top_k=1)
    top_5_accuracy = compute_accuracy(all_logits, all_targets, ignore_index, top_k=5)
    
    return EvaluationResults(
        perplexity=perplexity,
        loss=avg_loss,
        accuracy=accuracy,
        top_5_accuracy=top_5_accuracy,
        num_samples=total_samples,
        num_tokens=total_tokens
    )


class MetricsTracker:
    """
    Track and aggregate metrics over training/evaluation.
    
    Useful for monitoring training progress and computing
    running statistics.
    """
    
    def __init__(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, **kwargs):
        """
        Update metrics with new values.
        
        Args:
            **kwargs: Metric name-value pairs
        """
        for name, value in kwargs.items():
            if name not in self.metrics:
                self.metrics[name] = 0.0
                self.counts[name] = 0
            
            self.metrics[name] += value
            self.counts[name] += 1
    
    def get_average(self, name: str) -> float:
        """Get average value for a metric."""
        if name not in self.metrics:
            return 0.0
        return self.metrics[name] / max(1, self.counts[name])
    
    def get_all_averages(self) -> Dict[str, float]:
        """Get all metric averages."""
        return {
            name: self.get_average(name)
            for name in self.metrics.keys()
        }
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.counts.clear()
    
    def __str__(self) -> str:
        """String representation."""
        averages = self.get_all_averages()
        lines = ["Metrics:"]
        for name, value in averages.items():
            lines.append(f"  {name}: {value:.4f}")
        return "\n".join(lines)


