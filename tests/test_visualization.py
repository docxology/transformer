"""Tests for visualization tools."""

import pytest
import torch
import numpy as np
from src.transformer import Transformer, TransformerConfig
from src.visualization import (
    visualize_attention,
    plot_attention_heads,
    visualize_residual_stream,
    visualize_kv_stream,
    compute_path_counts,
    trace_information_paths
)


@pytest.fixture
def model_and_state():
    """Create a small model and get its state."""
    config = TransformerConfig(
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_layers=3,
        d_ff=256,
        dropout=0.0
    )
    
    model = Transformer(config)
    model.eval()
    
    batch_size, seq_len = 1, 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        logits, state = model(input_ids, return_state=True)
    
    return model, state, config


class TestAttentionVisualization:
    """Test attention visualization functions."""
    
    def test_visualize_attention(self, model_and_state):
        """Test basic attention visualization."""
        model, state, config = model_and_state
        
        attn_weights = state.get_attention_weights(0)
        
        fig = visualize_attention(attn_weights, layer_idx=0)
        
        assert fig is not None
        assert len(fig.axes) >= config.n_heads
    
    def test_visualize_attention_with_tokens(self, model_and_state):
        """Test attention visualization with token labels."""
        model, state, config = model_and_state
        
        attn_weights = state.get_attention_weights(0)
        tokens = [f"Token{i}" for i in range(8)]
        
        fig = visualize_attention(attn_weights, tokens=tokens)
        
        assert fig is not None
    
    def test_plot_attention_heads_interactive(self, model_and_state):
        """Test interactive plotly visualization."""
        model, state, config = model_and_state
        
        attn_weights = state.get_attention_weights(0)
        
        fig = plot_attention_heads(attn_weights)
        
        assert fig is not None
        # Plotly figure should have data
        assert len(fig.data) > 0


class TestFlowVisualization:
    """Test information flow visualization."""
    
    def test_visualize_residual_stream(self, model_and_state):
        """Test residual stream visualization."""
        model, state, config = model_and_state
        
        fig = visualize_residual_stream(state, position_idx=0)
        
        assert fig is not None
        # Should have n_layers + 1 subplots (embedding + layers)
        assert len(fig.axes) == config.n_layers + 1
    
    def test_visualize_kv_stream(self, model_and_state):
        """Test K/V stream visualization."""
        model, state, config = model_and_state
        
        fig = visualize_kv_stream(state, layer_idx=0)
        
        assert fig is not None
        # Should have 3 subplots (Q, K, V)
        assert len(fig.axes) == 3
    
    def test_residual_stream_extraction(self, model_and_state):
        """Test that residual stream can be extracted."""
        model, state, config = model_and_state
        
        residuals = state.get_residual_stream()
        
        assert len(residuals) == config.n_layers + 1
        for residual in residuals:
            assert residual.shape[2] == config.d_model


class TestPathAnalysis:
    """Test path analysis functions."""
    
    def test_compute_path_counts_simple(self):
        """Test simple path count computation."""
        # Moving 1 layer up and 1 position right: C(2,1) = 2 paths
        count = compute_path_counts(layer_displacement=1, position_displacement=1)
        assert count == 2
    
    def test_compute_path_counts_symmetric(self):
        """Test that path counts are symmetric."""
        count1 = compute_path_counts(2, 3)
        count2 = compute_path_counts(3, 2)
        assert count1 == count2
    
    def test_compute_path_counts_grows_fast(self):
        """Test that path counts grow astronomically."""
        count_small = compute_path_counts(5, 5)
        count_large = compute_path_counts(10, 10)
        
        # Should grow super-exponentially
        assert count_large > count_small ** 2
    
    def test_trace_information_paths_simple(self):
        """Test path tracing for simple case."""
        paths = trace_information_paths(
            start_layer=0,
            start_position=0,
            end_layer=1,
            end_position=1,
            max_paths=100
        )
        
        assert len(paths) == 2  # Two paths for C(2,1)
        
        # Check path structures
        for path in paths:
            assert path[0] == (0, 0)  # Starts at origin
            assert path[-1] == (1, 1)  # Ends at destination
    
    def test_trace_information_paths_respects_max(self):
        """Test that path tracing respects max limit."""
        max_paths = 10
        paths = trace_information_paths(
            start_layer=0,
            start_position=0,
            end_layer=5,
            end_position=5,
            max_paths=max_paths
        )
        
        assert len(paths) <= max_paths
    
    def test_path_count_matches_trace(self):
        """Test that path count matches actual traced paths."""
        layer_diff = 2
        pos_diff = 2
        
        expected_count = compute_path_counts(layer_diff, pos_diff)
        
        paths = trace_information_paths(
            start_layer=0,
            start_position=0,
            end_layer=layer_diff,
            end_position=pos_diff,
            max_paths=100
        )
        
        assert len(paths) == expected_count


class TestVisualizationIntegration:
    """Integration tests for visualization pipeline."""
    
    def test_full_visualization_pipeline(self, model_and_state):
        """Test that full visualization pipeline works."""
        model, state, config = model_and_state
        
        # Test multiple visualizations
        fig1 = visualize_attention(state.get_attention_weights(0))
        fig2 = visualize_residual_stream(state, position_idx=0)
        fig3 = visualize_kv_stream(state, layer_idx=0)
        
        assert fig1 is not None
        assert fig2 is not None
        assert fig3 is not None
    
    def test_state_to_dict_serialization(self, model_and_state):
        """Test that state can be serialized to dict."""
        model, state, config = model_and_state
        
        state_dict = state.to_dict()
        
        assert 'embeddings' in state_dict
        assert 'layer_states' in state_dict
        assert 'final_output' in state_dict
        assert len(state_dict['layer_states']) == config.n_layers


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


