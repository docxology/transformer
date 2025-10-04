"""Complete transformer model with state tracking for visualization."""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field

from .components import (
    TokenEmbedding,
    PositionalEncoding,
    TransformerBlock,
    LayerNorm
)


@dataclass
class TransformerConfig:
    """Configuration for transformer model."""
    vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    max_seq_len: int = 5000
    dropout: float = 0.1
    activation: str = 'relu'
    pad_idx: Optional[int] = None


@dataclass
class TransformerState:
    """
    Complete state of transformer forward pass for visualization.
    
    Captures both information highways:
    - Residual stream (vertical): Through layer_states
    - K/V stream (horizontal): Through attention weights in each layer
    """
    embeddings: torch.Tensor                    # Initial embeddings
    layer_states: List[any]                     # State per layer
    final_output: torch.Tensor                  # Final logits
    
    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for serialization."""
        return {
            'embeddings': self.embeddings,
            'layer_states': [s.to_dict() if s else None for s in self.layer_states],
            'final_output': self.final_output
        }
    
    def get_attention_weights(self, layer_idx: int) -> torch.Tensor:
        """
        Extract attention weights for specific layer.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Attention weights [batch, n_heads, seq_len, seq_len]
        """
        return self.layer_states[layer_idx].attention_state.attention_weights
    
    def get_residual_stream(self) -> List[torch.Tensor]:
        """
        Extract residual stream at each layer.
        
        Returns:
            List of residual tensors, one per layer
        """
        residuals = [self.embeddings]
        for state in self.layer_states:
            residuals.append(state.post_ffn)
        return residuals


class Transformer(nn.Module):
    """
    Complete transformer architecture with full state tracking.
    
    Architecture:
    1. Token Embedding + Positional Encoding
    2. N x Transformer Blocks (Attention + FFN)
    3. Output projection to vocabulary
    
    Information flows through:
    - Vertical (residual stream): Through layers at each position
    - Horizontal (K/V stream): Across positions via attention
    
    The model can track complete computational state for visualization
    and analysis of information flow paths.
    """
    
    def __init__(self, config: TransformerConfig):
        """
        Args:
            config: TransformerConfig object
        """
        super().__init__()
        self.config = config
        
        # Embedding layers
        self.token_embedding = TokenEmbedding(config.vocab_size, config.d_model)
        self.positional_encoding = PositionalEncoding(
            config.d_model,
            config.max_seq_len,
            config.dropout
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.d_model,
                config.n_heads,
                config.d_ff,
                config.dropout,
                config.activation
            )
            for _ in range(config.n_layers)
        ])
        
        # Final layer norm and output projection
        self.final_norm = LayerNorm(config.d_model)
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)
        
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_state: bool = False
    ) -> Tuple[torch.Tensor, Optional[TransformerState]]:
        """
        Forward pass with optional state tracking.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            mask: Attention mask [batch, 1, seq_len, seq_len] (optional)
            return_state: Whether to return full computational state
            
        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
            state: TransformerState if return_state=True, else None
        """
        # Generate causal mask if not provided
        if mask is None:
            seq_len = input_ids.size(1)
            mask = self.blocks[0].attention.get_causal_mask(
                seq_len, input_ids.device
            )
        
        # Embedding + positional encoding (initialize residual stream)
        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x)
        embeddings = x if return_state else None
        
        # Pass through transformer blocks
        layer_states = []
        for block in self.blocks:
            x, block_state = block(x, mask, return_state=return_state)
            if return_state:
                layer_states.append(block_state)
        
        # Final normalization and projection
        x = self.final_norm(x)
        logits = self.output_projection(x)
        
        # Prepare state for visualization
        state = None
        if return_state:
            state = TransformerState(
                embeddings=embeddings,
                layer_states=layer_states,
                final_output=logits
            )
        
        return logits, state
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Autoregressive generation.
        
        Args:
            input_ids: Initial token IDs [batch, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (optional)
            
        Returns:
            Generated token IDs [batch, seq_len + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len \
                       else input_ids[:, -self.config.max_seq_len:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, idx_next], dim=1)
        
        return input_ids


