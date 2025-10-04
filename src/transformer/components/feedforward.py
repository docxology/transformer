"""Position-wise feed-forward network."""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network (FFN).
    
    Applies the same two-layer network independently to each position.
    Expands to d_ff dimensions (typically 4*d_model) then contracts back.
    
    This non-linear transformation enables complex feature extraction
    after attention has gathered information across positions.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        """
        Args:
            d_model: Model dimension (residual stream width)
            d_ff: Feed-forward dimension (typically 4 * d_model)
            dropout: Dropout probability
            activation: Activation function ('relu' or 'gelu')
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input from residual stream [batch, seq_len, d_model]
            
        Returns:
            Output [batch, seq_len, d_model]
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


