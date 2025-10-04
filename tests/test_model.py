"""Tests for complete transformer model."""

import pytest
import torch
from src.transformer import Transformer, TransformerConfig


class TestTransformerConfig:
    """Test transformer configuration."""
    
    def test_default_initialization(self):
        """Test default configuration values."""
        config = TransformerConfig(vocab_size=10000)
        
        assert config.vocab_size == 10000
        assert config.d_model == 512
        assert config.n_heads == 8
        assert config.n_layers == 6
        assert config.d_ff == 2048
        assert config.max_seq_len == 5000
        assert config.dropout == 0.1
    
    def test_custom_configuration(self):
        """Test custom configuration."""
        config = TransformerConfig(
            vocab_size=5000,
            d_model=256,
            n_heads=4,
            n_layers=4,
            d_ff=1024
        )
        
        assert config.vocab_size == 5000
        assert config.d_model == 256
        assert config.n_heads == 4
        assert config.n_layers == 4
        assert config.d_ff == 1024


class TestTransformer:
    """Test complete transformer model."""
    
    @pytest.fixture
    def small_config(self):
        """Small config for fast testing."""
        return TransformerConfig(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512,
            max_seq_len=100,
            dropout=0.0
        )
    
    def test_initialization(self, small_config):
        """Test proper initialization."""
        model = Transformer(small_config)
        
        assert len(model.blocks) == small_config.n_layers
        assert model.output_projection.out_features == small_config.vocab_size
    
    def test_forward_shape(self, small_config):
        """Test output shape."""
        model = Transformer(small_config)
        batch_size, seq_len = 2, 10
        
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        logits, _ = model(input_ids)
        
        assert logits.shape == (batch_size, seq_len, small_config.vocab_size)
    
    def test_causal_masking(self, small_config):
        """Test that model uses causal masking correctly."""
        model = Transformer(small_config)
        model.eval()
        
        batch_size, seq_len = 1, 5
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        
        # Forward pass
        with torch.no_grad():
            logits, state = model(input_ids, return_state=True)
        
        # Check that attention is causal
        for layer_idx in range(small_config.n_layers):
            attn_weights = state.get_attention_weights(layer_idx)
            
            # Upper triangle should be very small
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    assert attn_weights[0, :, i, j].max().item() < 1e-5
    
    def test_state_tracking(self, small_config):
        """Test state tracking for visualization."""
        model = Transformer(small_config)
        batch_size, seq_len = 2, 10
        
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        logits, state = model(input_ids, return_state=True)
        
        assert state is not None
        assert state.embeddings.shape == (batch_size, seq_len, small_config.d_model)
        assert len(state.layer_states) == small_config.n_layers
        assert state.final_output.shape == logits.shape
    
    def test_residual_stream_extraction(self, small_config):
        """Test extraction of residual stream."""
        model = Transformer(small_config)
        batch_size, seq_len = 2, 10
        
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        _, state = model(input_ids, return_state=True)
        
        residuals = state.get_residual_stream()
        
        assert len(residuals) == small_config.n_layers + 1  # Embedding + layers
        for residual in residuals:
            assert residual.shape == (batch_size, seq_len, small_config.d_model)
    
    def test_attention_weights_extraction(self, small_config):
        """Test extraction of attention weights."""
        model = Transformer(small_config)
        batch_size, seq_len = 2, 10
        
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        _, state = model(input_ids, return_state=True)
        
        for layer_idx in range(small_config.n_layers):
            attn_weights = state.get_attention_weights(layer_idx)
            assert attn_weights.shape == (
                batch_size,
                small_config.n_heads,
                seq_len,
                seq_len
            )
    
    def test_generation(self, small_config):
        """Test autoregressive generation."""
        model = Transformer(small_config)
        model.eval()
        
        batch_size, initial_len = 1, 5
        max_new_tokens = 10
        
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, initial_len))
        
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens)
        
        assert output.shape == (batch_size, initial_len + max_new_tokens)
    
    def test_generation_top_k(self, small_config):
        """Test generation with top-k sampling."""
        model = Transformer(small_config)
        model.eval()
        
        batch_size, initial_len = 1, 5
        max_new_tokens = 5
        top_k = 10
        
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, initial_len))
        
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens, top_k=top_k)
        
        assert output.shape == (batch_size, initial_len + max_new_tokens)
    
    def test_parameter_initialization(self, small_config):
        """Test that parameters are properly initialized."""
        model = Transformer(small_config)
        
        # Check that parameters exist and have reasonable values
        for param in model.parameters():
            if param.dim() > 1:
                # Check that weights are not all zeros or ones
                assert not torch.allclose(param, torch.zeros_like(param))
                assert not torch.allclose(param, torch.ones_like(param))
                
                # Check reasonable variance
                assert param.std().item() > 0.01
                assert param.std().item() < 1.0
    
    def test_gradient_flow(self, small_config):
        """Test that gradients flow through the model."""
        model = Transformer(small_config)
        
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        target_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        
        # Forward pass
        logits, _ = model(input_ids)
        
        # Compute loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, small_config.vocab_size),
            target_ids.view(-1)
        )
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist and are non-zero
        # (allow for some parameters to have very small gradients)
        non_zero_grad_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                if param.grad.abs().max() > 1e-8:
                    non_zero_grad_count += 1
        
        # At least 90% of parameters should have meaningful gradients
        total_params = sum(1 for p in model.parameters() if p.requires_grad)
        assert non_zero_grad_count > total_params * 0.9, \
            f"Only {non_zero_grad_count}/{total_params} parameters have non-zero gradients"


class TestTransformerIntegration:
    """Integration tests for transformer."""
    
    def test_overfitting_single_batch(self):
        """Test that model can overfit a single batch (sanity check)."""
        config = TransformerConfig(
            vocab_size=50,
            d_model=64,
            n_heads=2,
            n_layers=2,
            d_ff=256,
            dropout=0.0
        )
        
        model = Transformer(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Single batch
        batch_size, seq_len = 4, 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        target_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        # Train for several steps
        initial_loss = None
        for step in range(100):
            optimizer.zero_grad()
            
            logits, _ = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, config.vocab_size),
                target_ids.view(-1)
            )
            
            if step == 0:
                initial_loss = loss.item()
            
            loss.backward()
            optimizer.step()
        
        final_loss = loss.item()
        
        # Model should overfit (loss should decrease significantly)
        assert final_loss < initial_loss * 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

