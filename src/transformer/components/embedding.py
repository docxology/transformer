"""Token embedding and positional encoding components."""

import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """
    Converts token indices to dense vector representations.
    
    Maps discrete token IDs to continuous d_model-dimensional vectors,
    establishing the foundation of the residual stream.
    """
    
    def __init__(self, vocab_size: int, d_model: int):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Dimension of model (residual stream width)
        """
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token indices [batch, seq_len]
            
        Returns:
            Embedded tokens [batch, seq_len, d_model]
        """
        # Scale embeddings by sqrt(d_model) for training stability
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Injects position information into token embeddings.
    
    Uses sinusoidal functions of varying frequencies to encode
    absolute position information, enabling the model to use
    sequence order despite the position-independent attention mechanism.
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: Dimension of model
            max_seq_len: Maximum sequence length to pre-compute
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Pre-compute positional encodings
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of model state)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token embeddings [batch, seq_len, d_model]
            
        Returns:
            Position-encoded embeddings [batch, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :]
        return self.dropout(x)


