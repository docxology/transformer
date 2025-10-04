"""
Comprehensive tests for subway map animation.

Tests architectural accuracy, information flow fidelity, and quantitative correctness.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transformer import Transformer, TransformerConfig
from src.visualization.subway_map_animation import create_subway_map_animation, _build_network_graph
from src.visualization.subway_map_validator import (
    SubwayMapValidator,
    validate_subway_map,
    compute_fidelity_metrics
)


@pytest.fixture
def small_model_state():
    """Create a small model and generate state for testing."""
    config = TransformerConfig(
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128
    )
    model = Transformer(config)
    model.eval()
    
    input_ids = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        logits, state = model(input_ids, return_state=True)
    
    return state


@pytest.fixture
def medium_model_state():
    """Create a medium model for more complex tests."""
    config = TransformerConfig(
        vocab_size=1000,
        d_model=128,
        n_heads=8,
        n_layers=4,
        d_ff=512
    )
    model = Transformer(config)
    model.eval()
    
    input_ids = torch.randint(0, 1000, (1, 12))
    with torch.no_grad():
        logits, state = model(input_ids, return_state=True)
    
    return state


class TestGraphConstruction:
    """Test that the graph correctly represents transformer architecture."""
    
    def test_graph_is_dag(self, small_model_state):
        """Verify graph is a directed acyclic graph."""
        import networkx as nx
        
        graph = _build_network_graph(
            len(small_model_state.layer_states),
            small_model_state.embeddings.shape[1],
            small_model_state
        )
        
        assert isinstance(graph, nx.DiGraph), "Graph should be directed"
        assert nx.is_directed_acyclic_graph(graph), "Graph should be acyclic (DAG)"
    
    def test_node_count_matches_architecture(self, small_model_state):
        """Verify correct number of nodes for model architecture."""
        n_layers = len(small_model_state.layer_states)
        seq_len = small_model_state.embeddings.shape[1]
        
        graph = _build_network_graph(n_layers, seq_len, small_model_state)
        
        # Expected: embeddings + 2*layers*positions + output
        expected_nodes = seq_len + (n_layers * 2 * seq_len) + seq_len
        actual_nodes = graph.number_of_nodes()
        
        assert actual_nodes == expected_nodes, \
            f"Expected {expected_nodes} nodes, got {actual_nodes}"
    
    def test_all_layers_represented(self, small_model_state):
        """Verify all layers have nodes."""
        n_layers = len(small_model_state.layer_states)
        seq_len = small_model_state.embeddings.shape[1]
        
        graph = _build_network_graph(n_layers, seq_len, small_model_state)
        
        # Check each layer has attention and FFN nodes
        for layer in range(n_layers):
            for pos in range(seq_len):
                attn_node = f'attn_L{layer}_P{pos}'
                ffn_node = f'ffn_L{layer}_P{pos}'
                
                assert attn_node in graph, f"Missing attention node: {attn_node}"
                assert ffn_node in graph, f"Missing FFN node: {ffn_node}"
    
    def test_node_types_correct(self, small_model_state):
        """Verify node type attributes are correct."""
        n_layers = len(small_model_state.layer_states)
        seq_len = small_model_state.embeddings.shape[1]
        
        graph = _build_network_graph(n_layers, seq_len, small_model_state)
        
        # Count by type
        type_counts = {}
        for node in graph.nodes():
            node_type = graph.nodes[node]['type']
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        
        assert type_counts['embedding'] == seq_len
        assert type_counts['attention'] == n_layers * seq_len
        assert type_counts['ffn'] == n_layers * seq_len
        assert type_counts['output'] == seq_len


class TestEdgeConnectivity:
    """Test that edges correctly represent computational dependencies."""
    
    def test_residual_connections_exist(self, small_model_state):
        """Verify vertical residual stream connections."""
        n_layers = len(small_model_state.layer_states)
        seq_len = small_model_state.embeddings.shape[1]
        
        graph = _build_network_graph(n_layers, seq_len, small_model_state)
        
        for pos in range(seq_len):
            # Embedding to first layer
            assert graph.has_edge(f'embed_{pos}', f'attn_L0_P{pos}')
            
            # Between layers
            for layer in range(n_layers):
                # Attention to FFN
                assert graph.has_edge(f'attn_L{layer}_P{pos}', f'ffn_L{layer}_P{pos}')
                
                # FFN to next layer or output
                if layer < n_layers - 1:
                    assert graph.has_edge(f'ffn_L{layer}_P{pos}', f'attn_L{layer+1}_P{pos}')
                else:
                    assert graph.has_edge(f'ffn_L{layer}_P{pos}', f'output_{pos}')
    
    def test_attention_connections_present(self, small_model_state):
        """Verify horizontal attention connections exist."""
        n_layers = len(small_model_state.layer_states)
        seq_len = small_model_state.embeddings.shape[1]
        
        graph = _build_network_graph(n_layers, seq_len, small_model_state)
        
        # Count attention edges
        attention_edges = sum(1 for u, v in graph.edges()
                             if graph.edges[u, v]['type'] == 'attention')
        
        # Should have some attention edges (at least causal connections)
        assert attention_edges > 0, "Should have attention connections"
    
    def test_no_backward_time_connections(self, small_model_state):
        """Verify causal mask: no attention to future positions."""
        n_layers = len(small_model_state.layer_states)
        seq_len = small_model_state.embeddings.shape[1]
        
        graph = _build_network_graph(n_layers, seq_len, small_model_state)
        
        # Check attention edges respect causality
        for u, v in graph.edges():
            if graph.edges[u, v]['type'] == 'attention':
                # Extract positions
                if '_P' in u and '_P' in v:
                    pos_u = int(u.split('_P')[1].split('_')[0])
                    pos_v = int(v.split('_P')[1].split('_')[0])
                    
                    # Query should not attend to future
                    assert pos_u <= pos_v, \
                        f"Violation of causality: {u} -> {v} (pos {pos_u} -> {pos_v})"
    
    def test_edge_types_labeled(self, small_model_state):
        """Verify all edges have type labels."""
        n_layers = len(small_model_state.layer_states)
        seq_len = small_model_state.embeddings.shape[1]
        
        graph = _build_network_graph(n_layers, seq_len, small_model_state)
        
        for u, v in graph.edges():
            assert 'type' in graph.edges[u, v], \
                f"Edge {u} -> {v} missing type attribute"
            assert graph.edges[u, v]['type'] in ['residual', 'attention'], \
                f"Edge {u} -> {v} has invalid type: {graph.edges[u, v]['type']}"


class TestInformationFlow:
    """Test that information flows match actual transformer computation."""
    
    def test_paths_exist_for_all_positions(self, small_model_state):
        """Verify path exists from embedding to output for each position."""
        import networkx as nx
        
        n_layers = len(small_model_state.layer_states)
        seq_len = small_model_state.embeddings.shape[1]
        
        graph = _build_network_graph(n_layers, seq_len, small_model_state)
        
        for pos in range(seq_len):
            start = f'embed_{pos}'
            end = f'output_{pos}'
            
            assert nx.has_path(graph, start, end), \
                f"No path from {start} to {end}"
    
    def test_path_lengths_correct(self, small_model_state):
        """Verify shortest path lengths match computational depth."""
        import networkx as nx
        
        n_layers = len(small_model_state.layer_states)
        seq_len = small_model_state.embeddings.shape[1]
        
        graph = _build_network_graph(n_layers, seq_len, small_model_state)
        
        # Sample a few positions
        for pos in range(0, seq_len, max(1, seq_len // 2)):
            start = f'embed_{pos}'
            end = f'output_{pos}'
            
            path_length = nx.shortest_path_length(graph, start, end)
            # Correct formula: embed -> attn_L0 -> ffn_L0 -> ... -> attn_L(n-1) -> ffn_L(n-1) -> output
            # That's: 1 (embed->attn) + n_layers*2 (attn->ffn->attn each layer) + 1 (ffn->output) - 1
            # Simplified: 1 + 2*n_layers
            expected = 2 * n_layers + 1
            
            assert path_length == expected, \
                f"Path length {path_length} != expected {expected} for position {pos}"
    
    def test_residual_norms_increase(self, small_model_state):
        """Verify residual stream norms increase through layers (typical)."""
        residuals = small_model_state.get_residual_stream()
        
        norms = [r[0].norm().item() for r in residuals]
        
        # Check generally increasing (allow small decreases)
        increases = sum(1 for i in range(len(norms)-1) if norms[i] < norms[i+1])
        
        # At least 80% should be increasing
        assert increases / (len(norms) - 1) >= 0.8, \
            f"Residual norms should generally increase, got {norms}"
    
    def test_no_nan_or_inf_values(self, small_model_state):
        """Verify no invalid values in state."""
        residuals = small_model_state.get_residual_stream()
        
        for i, residual in enumerate(residuals):
            assert not torch.isnan(residual).any(), \
                f"NaN values in residual stream at checkpoint {i}"
            assert not torch.isinf(residual).any(), \
                f"Inf values in residual stream at checkpoint {i}"


class TestValidator:
    """Test the validation system itself."""
    
    def test_validator_runs_without_error(self, small_model_state):
        """Verify validator can run on valid input."""
        n_layers = len(small_model_state.layer_states)
        seq_len = small_model_state.embeddings.shape[1]
        
        graph = _build_network_graph(n_layers, seq_len, small_model_state)
        
        validator = SubwayMapValidator(small_model_state, graph)
        results = validator.validate_all()
        
        assert len(results) > 0, "Validator should return results"
    
    def test_validator_catches_errors(self, small_model_state):
        """Verify validator detects invalid graphs."""
        import networkx as nx
        
        # Create intentionally invalid graph
        invalid_graph = nx.DiGraph()
        invalid_graph.add_node('fake_node')
        
        validator = SubwayMapValidator(small_model_state, invalid_graph)
        results = validator.validate_all()
        
        # Should have errors
        errors = [r for r in results if not r.passed and r.severity == 'error']
        assert len(errors) > 0, "Validator should detect invalid graph"
    
    def test_validator_report_generation(self, small_model_state):
        """Verify validator generates readable report."""
        n_layers = len(small_model_state.layer_states)
        seq_len = small_model_state.embeddings.shape[1]
        
        graph = _build_network_graph(n_layers, seq_len, small_model_state)
        
        validator = SubwayMapValidator(small_model_state, graph)
        validator.validate_all()
        
        report = validator.generate_report()
        
        assert len(report) > 0, "Report should not be empty"
        assert "VALIDATION REPORT" in report, "Report should have header"
    
    def test_fidelity_metrics_in_range(self, small_model_state):
        """Verify fidelity metrics are between 0 and 1."""
        n_layers = len(small_model_state.layer_states)
        seq_len = small_model_state.embeddings.shape[1]
        
        graph = _build_network_graph(n_layers, seq_len, small_model_state)
        
        metrics = compute_fidelity_metrics(small_model_state, graph)
        
        for name, value in metrics.items():
            assert 0 <= value <= 1, \
                f"Fidelity metric {name} = {value} not in [0, 1]"
    
    def test_high_fidelity_for_correct_graph(self, small_model_state):
        """Verify correct graph gets high fidelity scores."""
        n_layers = len(small_model_state.layer_states)
        seq_len = small_model_state.embeddings.shape[1]
        
        graph = _build_network_graph(n_layers, seq_len, small_model_state)
        
        metrics = compute_fidelity_metrics(small_model_state, graph)
        
        # Overall fidelity should be high (> 0.8)
        assert metrics['overall_fidelity'] > 0.8, \
            f"Overall fidelity should be high, got {metrics['overall_fidelity']}"


class TestAnimationGeneration:
    """Test the full animation generation process."""
    
    def test_animation_generates_successfully(self, small_model_state, tmp_path):
        """Verify animation can be generated without errors."""
        output_path = tmp_path / "test_subway.gif"
        
        try:
            create_subway_map_animation(
                small_model_state,
                save_path=str(output_path),
                fps=3,
                duration_seconds=1,  # Short for testing
                resolution=(400, 300)  # Small for speed
            )
            
            assert output_path.exists(), "Animation file should be created"
            assert output_path.stat().st_size > 0, "Animation file should not be empty"
        
        except Exception as e:
            pytest.fail(f"Animation generation failed: {e}")
    
    def test_animation_with_validation(self, small_model_state, tmp_path):
        """Verify animation generation includes validation."""
        output_path = tmp_path / "test_validated.gif"
        
        # Generate animation
        create_subway_map_animation(
            small_model_state,
            save_path=str(output_path),
            fps=3,
            duration_seconds=1,
            resolution=(400, 300)
        )
        
        # Validate the graph that would be used
        n_layers = len(small_model_state.layer_states)
        seq_len = small_model_state.embeddings.shape[1]
        
        graph = _build_network_graph(n_layers, seq_len, small_model_state)
        is_valid = validate_subway_map(small_model_state, graph, verbose=False)
        
        assert is_valid, "Generated graph should pass validation"


class TestScalability:
    """Test that system works with different model sizes."""
    
    def test_works_with_single_layer(self):
        """Verify works with minimal 1-layer model."""
        config = TransformerConfig(
            vocab_size=50,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64
        )
        model = Transformer(config)
        model.eval()
        
        input_ids = torch.randint(0, 50, (1, 4))
        with torch.no_grad():
            logits, state = model(input_ids, return_state=True)
        
        graph = _build_network_graph(1, 4, state)
        is_valid = validate_subway_map(state, graph, verbose=False)
        
        assert is_valid, "Should work with single-layer model"
    
    def test_works_with_medium_model(self, medium_model_state):
        """Verify works with medium-sized model."""
        n_layers = len(medium_model_state.layer_states)
        seq_len = medium_model_state.embeddings.shape[1]
        
        graph = _build_network_graph(n_layers, seq_len, medium_model_state)
        is_valid = validate_subway_map(medium_model_state, graph, verbose=False)
        
        assert is_valid, "Should work with medium model"
    
    def test_works_with_long_sequence(self):
        """Verify works with longer sequences."""
        config = TransformerConfig(
            vocab_size=100,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=128
        )
        model = Transformer(config)
        model.eval()
        
        input_ids = torch.randint(0, 100, (1, 24))  # Longer sequence
        with torch.no_grad():
            logits, state = model(input_ids, return_state=True)
        
        graph = _build_network_graph(2, 24, state)
        is_valid = validate_subway_map(state, graph, verbose=False)
        
        assert is_valid, "Should work with longer sequences"


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])



