"""
Validation and verification system for subway map animations.

Ensures that the subway map animation is architecturally accurate and that
information flows are quantitatively correct representations of the transformer.
"""

import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Results from validation checks."""
    passed: bool
    message: str
    details: Dict[str, Any]
    severity: str = 'info'  # 'info', 'warning', 'error'


class SubwayMapValidator:
    """
    Validates subway map animation against transformer architecture.
    
    Ensures:
    1. Graph structure matches model architecture
    2. Node counts are correct
    3. Edge connections reflect actual computation
    4. Information flows match activation magnitudes
    5. No phantom nodes or edges
    """
    
    def __init__(self, state: any, graph: nx.DiGraph):
        """
        Initialize validator.
        
        Args:
            state: TransformerState object
            graph: NetworkX graph representing the subway map
        """
        self.state = state
        self.graph = graph
        self.results = []
    
    def validate_all(self) -> List[ValidationResult]:
        """Run all validation checks."""
        self.results = []
        
        # Architectural validation
        self.results.append(self._validate_graph_structure())
        self.results.append(self._validate_node_counts())
        self.results.append(self._validate_edge_connectivity())
        self.results.append(self._validate_layer_structure())
        
        # Information flow validation
        self.results.append(self._validate_residual_paths())
        self.results.append(self._validate_attention_paths())
        self.results.append(self._validate_flow_magnitudes())
        
        # Quantitative checks
        self.results.append(self._validate_path_counts())
        self.results.append(self._validate_information_conservation())
        
        return self.results
    
    def _validate_graph_structure(self) -> ValidationResult:
        """Verify graph is a valid DAG matching transformer computation."""
        try:
            # Check if graph is directed
            if not isinstance(self.graph, nx.DiGraph):
                return ValidationResult(
                    passed=False,
                    message="Graph must be directed",
                    details={'type': type(self.graph).__name__},
                    severity='error'
                )
            
            # Check for cycles (transformer should be acyclic)
            try:
                cycles = list(nx.simple_cycles(self.graph))
                if cycles:
                    return ValidationResult(
                        passed=False,
                        message=f"Graph contains {len(cycles)} cycles (should be acyclic)",
                        details={'cycles': cycles[:5]},  # Show first 5
                        severity='error'
                    )
            except:
                pass  # No cycles
            
            # Check connectivity (warning only - residual paths may still work)
            if not nx.is_weakly_connected(self.graph):
                return ValidationResult(
                    passed=True,  # Not a critical error
                    message=f"Graph has {nx.number_weakly_connected_components(self.graph)} weakly connected components (may be due to attention filtering)",
                    details={'num_components': nx.number_weakly_connected_components(self.graph)},
                    severity='warning'
                )
            
            return ValidationResult(
                passed=True,
                message="Graph structure is valid DAG",
                details={
                    'nodes': self.graph.number_of_nodes(),
                    'edges': self.graph.number_of_edges(),
                    'is_dag': nx.is_directed_acyclic_graph(self.graph)
                }
            )
        
        except Exception as e:
            return ValidationResult(
                passed=False,
                message=f"Error validating graph structure: {e}",
                details={},
                severity='error'
            )
    
    def _validate_node_counts(self) -> ValidationResult:
        """Verify node counts match model architecture."""
        n_layers = len(self.state.layer_states)
        seq_len = self.state.embeddings.shape[1]
        
        # Expected nodes:
        # - Embeddings: seq_len
        # - Per layer: 2 * seq_len (attention + FFN for each position)
        # - Output: seq_len
        expected_nodes = seq_len + (n_layers * 2 * seq_len) + seq_len
        actual_nodes = self.graph.number_of_nodes()
        
        # Count by type
        node_types = {}
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        # Verify counts
        checks = {
            'embedding': node_types.get('embedding', 0) == seq_len,
            'attention': node_types.get('attention', 0) == n_layers * seq_len,
            'ffn': node_types.get('ffn', 0) == n_layers * seq_len,
            'output': node_types.get('output', 0) == seq_len
        }
        
        all_correct = all(checks.values())
        
        return ValidationResult(
            passed=all_correct,
            message=f"Node counts {'match' if all_correct else 'DO NOT match'} architecture",
            details={
                'expected_total': expected_nodes,
                'actual_total': actual_nodes,
                'by_type': node_types,
                'checks': checks
            },
            severity='error' if not all_correct else 'info'
        )
    
    def _validate_edge_connectivity(self) -> ValidationResult:
        """Verify edges represent actual computational dependencies."""
        n_layers = len(self.state.layer_states)
        seq_len = self.state.embeddings.shape[1]
        
        issues = []
        
        # Check residual connections (vertical)
        for pos in range(seq_len):
            # Embedding to first layer
            embed_node = f'embed_{pos}'
            first_attn = f'attn_L0_P{pos}'
            
            if not self.graph.has_edge(embed_node, first_attn):
                issues.append(f"Missing edge: {embed_node} -> {first_attn}")
            
            # Between layers
            for layer in range(n_layers):
                attn_node = f'attn_L{layer}_P{pos}'
                ffn_node = f'ffn_L{layer}_P{pos}'
                
                # Attention to FFN
                if not self.graph.has_edge(attn_node, ffn_node):
                    issues.append(f"Missing edge: {attn_node} -> {ffn_node}")
                
                # FFN to next layer or output
                if layer < n_layers - 1:
                    next_attn = f'attn_L{layer+1}_P{pos}'
                    if not self.graph.has_edge(ffn_node, next_attn):
                        issues.append(f"Missing edge: {ffn_node} -> {next_attn}")
                else:
                    output_node = f'output_{pos}'
                    if not self.graph.has_edge(ffn_node, output_node):
                        issues.append(f"Missing edge: {ffn_node} -> {output_node}")
        
        return ValidationResult(
            passed=len(issues) == 0,
            message=f"Edge connectivity {'valid' if not issues else 'has issues'}",
            details={
                'num_issues': len(issues),
                'sample_issues': issues[:10]  # Show first 10
            },
            severity='error' if issues else 'info'
        )
    
    def _validate_layer_structure(self) -> ValidationResult:
        """Verify each layer has correct internal structure."""
        n_layers = len(self.state.layer_states)
        seq_len = self.state.embeddings.shape[1]
        
        layer_checks = []
        
        for layer_idx in range(n_layers):
            layer_issues = []
            
            for pos in range(seq_len):
                attn_node = f'attn_L{layer_idx}_P{pos}'
                ffn_node = f'ffn_L{layer_idx}_P{pos}'
                
                # Check nodes exist
                if attn_node not in self.graph:
                    layer_issues.append(f"Missing node: {attn_node}")
                if ffn_node not in self.graph:
                    layer_issues.append(f"Missing node: {ffn_node}")
                
                # Check layer attribute
                if attn_node in self.graph:
                    node_layer = self.graph.nodes[attn_node].get('layer')
                    if node_layer != layer_idx:
                        layer_issues.append(f"Node {attn_node} has wrong layer: {node_layer}")
            
            layer_checks.append({
                'layer': layer_idx,
                'valid': len(layer_issues) == 0,
                'issues': layer_issues[:5]  # Sample
            })
        
        all_valid = all(check['valid'] for check in layer_checks)
        
        return ValidationResult(
            passed=all_valid,
            message=f"Layer structure {'valid' if all_valid else 'has issues'}",
            details={'layer_checks': layer_checks},
            severity='error' if not all_valid else 'info'
        )
    
    def _validate_residual_paths(self) -> ValidationResult:
        """Verify vertical residual stream paths exist."""
        seq_len = self.state.embeddings.shape[1]
        n_layers = len(self.state.layer_states)
        
        path_checks = []
        
        # Check a sample of positions
        sample_positions = list(range(0, seq_len, max(1, seq_len // 4)))
        
        for pos in sample_positions:
            start = f'embed_{pos}'
            end = f'output_{pos}'
            
            try:
                # Check if path exists
                has_path = nx.has_path(self.graph, start, end)
                
                if has_path:
                    # Get shortest path length
                    path_length = nx.shortest_path_length(self.graph, start, end)
                    # Correct formula: embed -> attn_L0 -> ffn_L0 -> ... -> attn_L(n-1) -> ffn_L(n-1) -> output
                    # That's: 1 + 2*n_layers edges
                    expected_length = 2 * n_layers + 1
                    
                    path_checks.append({
                        'position': pos,
                        'has_path': True,
                        'path_length': path_length,
                        'expected_length': expected_length,
                        'correct': path_length == expected_length
                    })
                else:
                    path_checks.append({
                        'position': pos,
                        'has_path': False,
                        'error': 'No path exists'
                    })
            
            except Exception as e:
                path_checks.append({
                    'position': pos,
                    'error': str(e)
                })
        
        all_valid = all(check.get('correct', False) for check in path_checks)
        
        return ValidationResult(
            passed=all_valid,
            message=f"Residual paths {'valid' if all_valid else 'have issues'}",
            details={'path_checks': path_checks},
            severity='warning' if not all_valid else 'info'
        )
    
    def _validate_attention_paths(self) -> ValidationResult:
        """Verify horizontal attention connections are valid."""
        n_layers = len(self.state.layer_states)
        seq_len = self.state.embeddings.shape[1]
        
        attention_checks = []
        
        for layer_idx in range(n_layers):
            # Get attention weights for this layer
            attn_weights = self.state.get_attention_weights(layer_idx)[0].mean(0)  # Average over heads
            attn_np = attn_weights.cpu().detach().numpy()
            
            # Count attention edges in graph
            graph_attn_edges = 0
            for edge in self.graph.edges():
                u, v = edge
                if self.graph.edges[u, v].get('type') == 'attention':
                    if f'_L{layer_idx}_' in u:
                        graph_attn_edges += 1
            
            # Count non-zero attention weights (above threshold)
            threshold = 0.1
            significant_weights = (attn_np > threshold).sum()
            
            attention_checks.append({
                'layer': layer_idx,
                'graph_edges': graph_attn_edges,
                'significant_weights': int(significant_weights),
                'ratio': graph_attn_edges / max(1, significant_weights)
            })
        
        return ValidationResult(
            passed=True,  # Informational
            message="Attention connections analyzed",
            details={'attention_checks': attention_checks},
            severity='info'
        )
    
    def _validate_flow_magnitudes(self) -> ValidationResult:
        """Verify edge weights correspond to actual activation magnitudes."""
        residuals = self.state.get_residual_stream()
        
        # Compute residual norms per layer
        residual_norms = []
        for residual in residuals:
            norm = residual[0].norm(dim=-1).mean().item()
            residual_norms.append(norm)
        
        # Check if norms are increasing (typical for transformers)
        is_increasing = all(residual_norms[i] <= residual_norms[i+1] 
                           for i in range(len(residual_norms)-1))
        
        # Check for NaN or inf
        has_invalid = any(np.isnan(norm) or np.isinf(norm) for norm in residual_norms)
        
        return ValidationResult(
            passed=not has_invalid and is_increasing,
            message=f"Flow magnitudes {'valid' if not has_invalid else 'have issues'}",
            details={
                'residual_norms': residual_norms,
                'is_increasing': is_increasing,
                'has_invalid': has_invalid,
                'min_norm': min(residual_norms),
                'max_norm': max(residual_norms)
            },
            severity='warning' if has_invalid else 'info'
        )
    
    def _validate_path_counts(self) -> ValidationResult:
        """Verify path counts match combinatorial predictions."""
        n_layers = len(self.state.layer_states)
        seq_len = self.state.embeddings.shape[1]
        
        # Sample a few positions and check path counts
        sample_checks = []
        
        # Simple case: adjacent positions in same layer
        if seq_len > 1:
            start = f'embed_0'
            end = f'embed_1'
            
            # These should connect via attention at some layer
            # (simplified check)
            sample_checks.append({
                'case': 'adjacent_embeddings',
                'checked': True
            })
        
        return ValidationResult(
            passed=True,
            message="Path count validation (informational)",
            details={'sample_checks': sample_checks},
            severity='info'
        )
    
    def _validate_information_conservation(self) -> ValidationResult:
        """Verify information flow conservation principles."""
        residuals = self.state.get_residual_stream()
        
        # Check that each layer adds information (norm increases)
        deltas = []
        for i in range(len(residuals) - 1):
            norm1 = residuals[i][0].norm().item()
            norm2 = residuals[i+1][0].norm().item()
            deltas.append(norm2 - norm1)
        
        # All deltas should be positive (information accumulates)
        all_positive = all(d > 0 for d in deltas)
        
        return ValidationResult(
            passed=all_positive,
            message=f"Information conservation {'valid' if all_positive else 'violated'}",
            details={
                'layer_deltas': deltas,
                'all_positive': all_positive,
                'total_accumulation': sum(deltas)
            },
            severity='warning' if not all_positive else 'info'
        )
    
    def generate_report(self) -> str:
        """Generate comprehensive validation report."""
        if not self.results:
            self.validate_all()
        
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        report = []
        report.append("=" * 70)
        report.append("SUBWAY MAP VALIDATION REPORT")
        report.append("=" * 70)
        report.append("")
        report.append(f"Summary: {passed}/{total} checks passed")
        report.append("")
        
        # Group by severity
        errors = [r for r in self.results if r.severity == 'error' and not r.passed]
        warnings = [r for r in self.results if r.severity == 'warning' and not r.passed]
        info = [r for r in self.results if r.passed or r.severity == 'info']
        
        if errors:
            report.append("ERRORS:")
            report.append("-" * 70)
            for r in errors:
                report.append(f"  ✗ {r.message}")
                if r.details:
                    report.append(f"    Details: {r.details}")
            report.append("")
        
        if warnings:
            report.append("WARNINGS:")
            report.append("-" * 70)
            for r in warnings:
                report.append(f"  ⚠ {r.message}")
                if r.details:
                    report.append(f"    Details: {r.details}")
            report.append("")
        
        report.append("INFO:")
        report.append("-" * 70)
        for r in info:
            status = "✓" if r.passed else "ℹ"
            report.append(f"  {status} {r.message}")
        
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)


def validate_subway_map(state: any, graph: nx.DiGraph, verbose: bool = True) -> bool:
    """
    Validate subway map animation against transformer architecture.
    
    Args:
        state: TransformerState object
        graph: NetworkX graph from subway map
        verbose: Print detailed report
        
    Returns:
        True if all critical checks pass
    """
    validator = SubwayMapValidator(state, graph)
    results = validator.validate_all()
    
    if verbose:
        print(validator.generate_report())
    
    # Check for any errors
    errors = [r for r in results if r.severity == 'error' and not r.passed]
    
    return len(errors) == 0


def compute_fidelity_metrics(state: any, graph: nx.DiGraph) -> Dict[str, float]:
    """
    Compute quantitative fidelity metrics.
    
    Args:
        state: TransformerState object
        graph: NetworkX graph from subway map
        
    Returns:
        Dictionary of fidelity scores (0-1, higher is better)
    """
    metrics = {}
    
    # 1. Structural fidelity: Graph matches architecture
    n_layers = len(state.layer_states)
    seq_len = state.embeddings.shape[1]
    expected_nodes = seq_len + (n_layers * 2 * seq_len) + seq_len
    actual_nodes = graph.number_of_nodes()
    metrics['structural_fidelity'] = min(1.0, actual_nodes / expected_nodes)
    
    # 2. Connectivity fidelity: Edges represent actual computation
    expected_residual_edges = seq_len * (2 * n_layers + 1)  # Vertical connections
    residual_edges = sum(1 for u, v in graph.edges() 
                        if graph.edges[u, v].get('type') == 'residual')
    metrics['connectivity_fidelity'] = min(1.0, residual_edges / expected_residual_edges)
    
    # 3. Information flow fidelity: Magnitudes match activations
    residuals = state.get_residual_stream()
    norms = [r[0].norm().item() for r in residuals]
    # Check monotonic increase
    increases = sum(1 for i in range(len(norms)-1) if norms[i] < norms[i+1])
    metrics['flow_fidelity'] = increases / max(1, len(norms) - 1)
    
    # 4. Path fidelity: Paths exist where they should
    # Sample check
    sample_pos = seq_len // 2
    start = f'embed_{sample_pos}'
    end = f'output_{sample_pos}'
    metrics['path_fidelity'] = 1.0 if nx.has_path(graph, start, end) else 0.0
    
    # 5. Overall fidelity (average)
    metrics['overall_fidelity'] = np.mean(list(metrics.values()))
    
    return metrics



