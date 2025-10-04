"""Multi-head self-attention mechanism with explicit Q, K, V tracking."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class AttentionState:
    """
    Captures the complete state of attention computation for visualization.
    
    Tracks both the residual stream (vertical) and K/V stream (horizontal)
    information highways.
    """
    query: torch.Tensor          # [batch, seq_len, d_model]
    key: torch.Tensor            # [batch, seq_len, d_model]
    value: torch.Tensor          # [batch, seq_len, d_model]
    attention_weights: torch.Tensor  # [batch, n_heads, seq_len, seq_len]
    attention_output: torch.Tensor   # [batch, seq_len, d_model]
    
    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Convert to dictionary for easy serialization."""
        return {
            'query': self.query,
            'key': self.key,
            'value': self.value,
            'attention_weights': self.attention_weights,
            'attention_output': self.attention_output
        }


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    
    Core principles:
    - Q: "Given current state, where should I look?" (queries the past)
    - K: "Given current state, what futures should look here?" (offers retrieval cues)
    - V: "Given current state, what information to transmit?" (packages information)
    
    The attention weights form the horizontal K/V stream, while the
    output contributes to the vertical residual stream.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Args:
            d_model: Model dimension (residual stream width)
            n_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to include bias in linear projections
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_state: bool = False
    ) -> Tuple[torch.Tensor, Optional[AttentionState]]:
        """
        Args:
            x: Input from residual stream [batch, seq_len, d_model]
            mask: Attention mask [batch, 1, seq_len, seq_len] (optional)
            return_state: Whether to return attention state for visualization
            
        Returns:
            output: Attention output [batch, seq_len, d_model]
            state: AttentionState if return_state=True, else None
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.W_q(x)  # [batch, seq_len, d_model]
        K = self.W_k(x)  # [batch, seq_len, d_model]
        V = self.W_v(x)  # [batch, seq_len, d_model]
        
        # Split into multiple heads
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # Now: [batch, n_heads, seq_len, d_k]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # [batch, n_heads, seq_len, seq_len]
        
        # Apply mask if provided (for causal attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Attention weights (the K/V stream horizontal flow)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        # [batch, n_heads, seq_len, d_k]
        
        # Concatenate heads and project
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.d_model)
        output = self.W_o(attention_output)
        
        # Prepare state for visualization
        state = None
        if return_state:
            # Combine Q, K, V back from heads for visualization
            Q_vis = self.W_q(x)
            K_vis = self.W_k(x)
            V_vis = self.W_v(x)
            
            state = AttentionState(
                query=Q_vis,
                key=K_vis,
                value=V_vis,
                attention_weights=attention_weights,
                attention_output=output
            )
        
        return output, state
    
    def get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate causal mask to prevent attending to future positions.
        
        Args:
            seq_len: Sequence length
            device: Device to create mask on
            
        Returns:
            Causal mask [1, 1, seq_len, seq_len]
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, 0).masked_fill(mask == 0, 1)
        return mask.unsqueeze(0).unsqueeze(0)


