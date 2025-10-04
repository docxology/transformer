"""Analysis of information flow paths through transformer layers."""

import math
from typing import Tuple, List, Dict
import numpy as np


def compute_path_counts(layer_displacement: int, position_displacement: int) -> int:
    """
    Compute number of distinct information flow paths.
    
    Information can travel up through layers (n times) and right through
    positions (m times) in any order. The number of paths is the binomial
    coefficient C(m+n, n).
    
    As noted in the original description: this "becomes greater than the
    number of atoms in the visible universe quickly."
    
    Args:
        layer_displacement: Number of layers to traverse (n)
        position_displacement: Number of positions to traverse (m)
        
    Returns:
        Number of distinct computational paths
    """
    m = position_displacement
    n = layer_displacement
    
    # C(m+n, n) = (m+n)! / (m! * n!)
    return math.comb(m + n, n)


def trace_information_paths(
    start_layer: int,
    start_position: int,
    end_layer: int,
    end_position: int,
    max_paths: int = 1000
) -> List[List[Tuple[int, int]]]:
    """
    Enumerate possible information flow paths.
    
    Each path represents a sequence of (layer, position) coordinates
    that information could traverse from start to end.
    
    Args:
        start_layer: Starting layer
        start_position: Starting position
        end_layer: Ending layer
        end_position: Ending position
        max_paths: Maximum number of paths to generate
        
    Returns:
        List of paths, where each path is a list of (layer, position) tuples
    """
    layer_diff = end_layer - start_layer
    pos_diff = end_position - start_position
    
    if layer_diff < 0 or pos_diff < 0:
        return []
    
    paths = []
    
    def generate_path(current_layer: int, current_pos: int, path: List[Tuple[int, int]]):
        if len(paths) >= max_paths:
            return
        
        if current_layer == end_layer and current_pos == end_position:
            paths.append(path.copy())
            return
        
        # Can move up through layers
        if current_layer < end_layer:
            path.append((current_layer + 1, current_pos))
            generate_path(current_layer + 1, current_pos, path)
            path.pop()
        
        # Can move right through positions
        if current_pos < end_position:
            path.append((current_layer, current_pos + 1))
            generate_path(current_layer, current_pos + 1, path)
            path.pop()
    
    generate_path(start_layer, start_position, [(start_layer, start_position)])
    
    return paths


def compute_path_statistics(n_layers: int, seq_len: int) -> Dict[str, any]:
    """
    Compute statistics about path counts in a transformer.
    
    Args:
        n_layers: Number of layers
        seq_len: Sequence length
        
    Returns:
        Dictionary of statistics
    """
    stats = {
        'n_layers': n_layers,
        'seq_len': seq_len,
        'paths_per_connection': {},
        'total_possible_paths': 0
    }
    
    # Sample some representative connections
    samples = [
        (0, 0, 1, 1),  # Adjacent layer, adjacent position
        (0, 0, n_layers - 1, seq_len - 1),  # Full traverse
        (0, 0, n_layers // 2, seq_len // 2),  # Half traverse
    ]
    
    for start_l, start_p, end_l, end_p in samples:
        if end_l < n_layers and end_p < seq_len:
            layer_diff = end_l - start_l
            pos_diff = end_p - start_p
            count = compute_path_counts(layer_diff, pos_diff)
            
            key = f"({start_l},{start_p})->({end_l},{end_p})"
            stats['paths_per_connection'][key] = {
                'count': count,
                'layer_diff': layer_diff,
                'pos_diff': pos_diff
            }
    
    # Estimate total paths (very rough approximation)
    # This is just for illustration - the actual number is astronomical
    avg_displacement = (n_layers + seq_len) // 4
    typical_paths = compute_path_counts(avg_displacement, avg_displacement)
    stats['typical_path_count'] = typical_paths
    
    return stats


def analyze_path_complexity(n_layers: int, seq_len: int) -> str:
    """
    Generate a human-readable analysis of path complexity.
    
    Args:
        n_layers: Number of layers
        seq_len: Sequence length
        
    Returns:
        Analysis string
    """
    # Full traverse path count
    full_paths = compute_path_counts(n_layers - 1, seq_len - 1)
    
    # Some reference numbers
    atoms_in_universe = 1e80
    
    analysis = f"""
Path Complexity Analysis
========================
Network: {n_layers} layers, {seq_len} positions

Full Traverse Paths:
- From (layer 0, position 0) to (layer {n_layers-1}, position {seq_len-1})
- Number of distinct paths: {full_paths:,}
- Log10 of paths: {math.log10(full_paths):.2f}
- Comparison: ~{full_paths / atoms_in_universe:.2e}x atoms in universe

Adjacent Connection:
- Paths for 1 layer + 1 position: {compute_path_counts(1, 1):,}

Mid-range Connection:
- Paths for {n_layers//2} layers + {seq_len//2} positions: {compute_path_counts(n_layers//2, seq_len//2):,}

Interpretation:
The astronomical number of computational paths means the transformer
experiences information flow through massive redundancy and interference
patterns. This enables:
1. Robust information transmission despite lossy transformations
2. Nuanced encoding via interference between different paths
3. Memory and cognition that feels continuous and interferometric
"""
    
    return analysis


def visualize_path_grid(
    n_layers: int,
    seq_len: int,
    start: Tuple[int, int],
    end: Tuple[int, int]
) -> np.ndarray:
    """
    Create a grid showing all possible paths.
    
    Args:
        n_layers: Number of layers
        seq_len: Sequence length
        start: Starting (layer, position)
        end: Ending (layer, position)
        
    Returns:
        Grid array [n_layers, seq_len] with path counts
    """
    grid = np.zeros((n_layers, seq_len))
    
    start_l, start_p = start
    end_l, end_p = end
    
    # For each point on grid, count paths from start to point to end
    for l in range(start_l, end_l + 1):
        for p in range(start_p, end_p + 1):
            # Paths from start to this point
            paths_to = compute_path_counts(l - start_l, p - start_p)
            # Paths from this point to end
            paths_from = compute_path_counts(end_l - l, end_p - p)
            # Total paths through this point
            grid[l, p] = paths_to * paths_from
    
    return grid


