"""Transformer block combining attention and feed-forward with residual connections."""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

from .attention import MultiHeadAttention, AttentionState
from .feedforward import FeedForward
from .normalization import LayerNorm


@dataclass
class TransformerBlockState:
    """
    Complete state of a transformer block for visualization.
    
    Tracks the residual stream flow and attention state at this layer.
    """
    input_residual: torch.Tensor      # Input to block
    post_attention: torch.Tensor       # After attention + residual
    post_ffn: torch.Tensor            # After FFN + residual (output)
    attention_state: AttentionState    # Attention mechanism state
    
    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for serialization."""
        return {
            'input_residual': self.input_residual,
            'post_attention': self.post_attention,
            'post_ffn': self.post_ffn,
            'attention_state': self.attention_state.to_dict()
        }


class TransformerBlock(nn.Module):
    """
    Single transformer block with attention and feed-forward sub-layers.
    
    Information flow:
    1. Input residual → LayerNorm → Multi-Head Attention → Add to residual
    2. Residual → LayerNorm → Feed-Forward → Add to residual
    
    The residual connections preserve information from previous layers,
    while attention enables horizontal information flow across positions.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            activation: Activation function for FFN
        """
        super().__init__()
        
        # Sub-layers
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)
        
        # Layer normalization (pre-norm architecture)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_state: bool = False
    ) -> Tuple[torch.Tensor, Optional[TransformerBlockState]]:
        """
        Args:
            x: Input from residual stream [batch, seq_len, d_model]
            mask: Attention mask (optional)
            return_state: Whether to return block state for visualization
            
        Returns:
            output: Block output [batch, seq_len, d_model]
            state: TransformerBlockState if return_state=True, else None
        """
        input_residual = x
        
        # Self-attention sub-layer with residual connection
        normed = self.norm1(x)
        attn_output, attn_state = self.attention(
            normed, mask, return_state=return_state
        )
        x = x + self.dropout(attn_output)
        post_attention = x
        
        # Feed-forward sub-layer with residual connection
        normed = self.norm2(x)
        ff_output = self.feed_forward(normed)
        x = x + self.dropout(ff_output)
        post_ffn = x
        
        # Prepare state for visualization
        state = None
        if return_state:
            state = TransformerBlockState(
                input_residual=input_residual,
                post_attention=post_attention,
                post_ffn=post_ffn,
                attention_state=attn_state
            )
        
        return x, state


