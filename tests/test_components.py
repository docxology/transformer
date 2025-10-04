"""Tests for transformer components."""

import pytest
import torch
import torch.nn as nn
from src.transformer.components import (
    TokenEmbedding,
    PositionalEncoding,
    MultiHeadAttention,
    FeedForward,
    LayerNorm,
    TransformerBlock
)


class TestTokenEmbedding:
    """Test token embedding layer."""
    
    def test_initialization(self):
        """Test proper initialization."""
        vocab_size, d_model = 1000, 512
        embedding = TokenEmbedding(vocab_size, d_model)
        
        assert embedding.d_model == d_model
        assert embedding.embedding.num_embeddings == vocab_size
        assert embedding.embedding.embedding_dim == d_model
    
    def test_forward_shape(self):
        """Test output shape."""
        vocab_size, d_model = 1000, 512
        batch_size, seq_len = 4, 10
        
        embedding = TokenEmbedding(vocab_size, d_model)
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        output = embedding(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_scaling(self):
        """Test that embeddings are scaled by sqrt(d_model)."""
        vocab_size, d_model = 100, 64
        embedding = TokenEmbedding(vocab_size, d_model)
        
        # Get raw embedding
        x = torch.tensor([[5]])
        output = embedding(x)
        
        # Compare with manually scaled embedding
        raw_emb = embedding.embedding(x)
        expected = raw_emb * (d_model ** 0.5)
        
        assert torch.allclose(output, expected)


class TestPositionalEncoding:
    """Test positional encoding."""
    
    def test_initialization(self):
        """Test proper initialization."""
        d_model, max_seq_len = 512, 1000
        pos_enc = PositionalEncoding(d_model, max_seq_len)
        
        assert pos_enc.pe.shape == (max_seq_len, d_model)
    
    def test_forward_shape(self):
        """Test output shape preservation."""
        d_model = 512
        batch_size, seq_len = 4, 10
        
        pos_enc = PositionalEncoding(d_model, dropout=0.0)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = pos_enc(x)
        
        assert output.shape == x.shape
    
    def test_deterministic_encoding(self):
        """Test that positional encoding is deterministic."""
        d_model = 512
        pos_enc = PositionalEncoding(d_model, dropout=0.0)
        
        x = torch.randn(1, 10, d_model)
        output1 = pos_enc(x)
        output2 = pos_enc(x)
        
        # With dropout=0, should be identical
        assert torch.allclose(output1, output2)
    
    def test_sinusoidal_pattern(self):
        """Test that encoding follows sinusoidal pattern."""
        d_model = 64
        pos_enc = PositionalEncoding(d_model, dropout=0.0)
        
        # Check that even dimensions use sin and odd use cos
        pe = pos_enc.pe
        
        # First position should have specific pattern
        assert pe[0, 0] == pytest.approx(0.0, abs=1e-5)  # sin(0)
        assert pe[0, 1] == pytest.approx(1.0, abs=1e-5)  # cos(0)


class TestMultiHeadAttention:
    """Test multi-head attention mechanism."""
    
    def test_initialization(self):
        """Test proper initialization."""
        d_model, n_heads = 512, 8
        attn = MultiHeadAttention(d_model, n_heads)
        
        assert attn.d_model == d_model
        assert attn.n_heads == n_heads
        assert attn.d_k == d_model // n_heads
    
    def test_forward_shape(self):
        """Test output shape."""
        d_model, n_heads = 512, 8
        batch_size, seq_len = 4, 10
        
        attn = MultiHeadAttention(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output, _ = attn(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_state_tracking(self):
        """Test that state tracking returns correct information."""
        d_model, n_heads = 512, 8
        batch_size, seq_len = 4, 10
        
        attn = MultiHeadAttention(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output, state = attn(x, return_state=True)
        
        assert state is not None
        assert state.query.shape == (batch_size, seq_len, d_model)
        assert state.key.shape == (batch_size, seq_len, d_model)
        assert state.value.shape == (batch_size, seq_len, d_model)
        assert state.attention_weights.shape == (batch_size, n_heads, seq_len, seq_len)
    
    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1 over key dimension."""
        d_model, n_heads = 512, 8
        batch_size, seq_len = 4, 10
        
        attn = MultiHeadAttention(d_model, n_heads, dropout=0.0)
        x = torch.randn(batch_size, seq_len, d_model)
        
        _, state = attn(x, return_state=True)
        
        # Sum over key dimension should be 1
        weights_sum = state.attention_weights.sum(dim=-1)
        assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-5)
    
    def test_causal_mask(self):
        """Test causal masking prevents future attention."""
        d_model, n_heads = 512, 8
        seq_len = 10
        
        attn = MultiHeadAttention(d_model, n_heads, dropout=0.0)
        x = torch.randn(1, seq_len, d_model)
        
        mask = attn.get_causal_mask(seq_len, x.device)
        _, state = attn(x, mask=mask, return_state=True)
        
        # Check that future positions have zero attention
        weights = state.attention_weights[0, 0]  # First head
        
        # Upper triangle (future) should be close to zero
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert weights[i, j].item() < 1e-5


class TestFeedForward:
    """Test feed-forward network."""
    
    def test_initialization(self):
        """Test proper initialization."""
        d_model, d_ff = 512, 2048
        ff = FeedForward(d_model, d_ff)
        
        assert ff.linear1.in_features == d_model
        assert ff.linear1.out_features == d_ff
        assert ff.linear2.in_features == d_ff
        assert ff.linear2.out_features == d_model
    
    def test_forward_shape(self):
        """Test output shape."""
        d_model, d_ff = 512, 2048
        batch_size, seq_len = 4, 10
        
        ff = FeedForward(d_model, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = ff(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_activation_functions(self):
        """Test different activation functions."""
        d_model, d_ff = 512, 2048
        
        # ReLU
        ff_relu = FeedForward(d_model, d_ff, activation='relu')
        assert isinstance(ff_relu.activation, nn.ReLU)
        
        # GELU
        ff_gelu = FeedForward(d_model, d_ff, activation='gelu')
        assert isinstance(ff_gelu.activation, nn.GELU)
    
    def test_nonlinearity(self):
        """Test that network is nonlinear."""
        d_model, d_ff = 64, 256
        ff = FeedForward(d_model, d_ff, dropout=0.0)
        
        x1 = torch.randn(1, 5, d_model)
        x2 = torch.randn(1, 5, d_model)
        
        # f(x1 + x2) != f(x1) + f(x2) for nonlinear f
        y1 = ff(x1)
        y2 = ff(x2)
        y_sum = ff(x1 + x2)
        
        assert not torch.allclose(y_sum, y1 + y2, rtol=1e-2)


class TestLayerNorm:
    """Test layer normalization."""
    
    def test_initialization(self):
        """Test proper initialization."""
        d_model = 512
        norm = LayerNorm(d_model)
        
        assert norm.gamma.shape == (d_model,)
        assert norm.beta.shape == (d_model,)
        assert torch.allclose(norm.gamma, torch.ones(d_model))
        assert torch.allclose(norm.beta, torch.zeros(d_model))
    
    def test_forward_shape(self):
        """Test output shape preservation."""
        d_model = 512
        batch_size, seq_len = 4, 10
        
        norm = LayerNorm(d_model)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = norm(x)
        
        assert output.shape == x.shape
    
    def test_normalization(self):
        """Test that output has mean~0 and std~1."""
        d_model = 512
        norm = LayerNorm(d_model, eps=1e-6)
        
        x = torch.randn(4, 10, d_model) * 10 + 5  # Non-normalized input
        output = norm(x)
        
        # Check mean and std along feature dimension
        mean = output.mean(dim=-1)
        std = output.std(dim=-1)
        
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        assert torch.allclose(std, torch.ones_like(std), atol=1e-1)


class TestTransformerBlock:
    """Test complete transformer block."""
    
    def test_initialization(self):
        """Test proper initialization."""
        d_model, n_heads, d_ff = 512, 8, 2048
        block = TransformerBlock(d_model, n_heads, d_ff)
        
        assert isinstance(block.attention, MultiHeadAttention)
        assert isinstance(block.feed_forward, FeedForward)
        assert isinstance(block.norm1, LayerNorm)
        assert isinstance(block.norm2, LayerNorm)
    
    def test_forward_shape(self):
        """Test output shape."""
        d_model, n_heads, d_ff = 512, 8, 2048
        batch_size, seq_len = 4, 10
        
        block = TransformerBlock(d_model, n_heads, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output, _ = block(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_residual_connections(self):
        """Test that residual connections preserve information."""
        d_model, n_heads, d_ff = 512, 8, 2048
        block = TransformerBlock(d_model, n_heads, d_ff, dropout=0.0)
        
        x = torch.randn(1, 5, d_model)
        output, _ = block(x)
        
        # Output should be different from input but related
        assert not torch.allclose(output, x)
        
        # Residual component should be significant
        diff = (output - x).norm()
        assert diff < output.norm()  # Change is smaller than output magnitude
    
    def test_state_tracking(self):
        """Test state tracking for visualization."""
        d_model, n_heads, d_ff = 512, 8, 2048
        batch_size, seq_len = 4, 10
        
        block = TransformerBlock(d_model, n_heads, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output, state = block(x, return_state=True)
        
        assert state is not None
        assert state.input_residual.shape == (batch_size, seq_len, d_model)
        assert state.post_attention.shape == (batch_size, seq_len, d_model)
        assert state.post_ffn.shape == (batch_size, seq_len, d_model)
        assert state.attention_state is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


