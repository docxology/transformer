"""Layer normalization component."""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Layer normalization with learnable affine parameters.
    
    Normalizes activations across the feature dimension to maintain
    stable distributions throughout the residual stream, facilitating
    gradient flow and training stability.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Args:
            d_model: Model dimension
            eps: Epsilon for numerical stability
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input [batch, seq_len, d_model]
            
        Returns:
            Normalized output [batch, seq_len, d_model]
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


