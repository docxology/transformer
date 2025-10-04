"""Advanced analysis tools for transformer introspection."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path


class TransformerAnalyzer:
    """
    Comprehensive analysis toolkit for transformers.
    
    Provides detailed analysis of:
    - Attention patterns and head specialization
    - Information flow and bottlenecks
    - Layer contributions and importance
    - Gradient flow and optimization health
    - Representational geometry
    """
    
    def __init__(self, model, state):
        self.model = model
        self.state = state
        self.n_layers = len(state.layer_states)
        self.seq_len = state.embeddings.shape[1]
    
    def analyze_attention_patterns(self) -> Dict:
        """
        Analyze attention patterns across all layers and heads.
        
        Returns:
            Dictionary with attention statistics and patterns
        """
        results = {
            'layer_stats': [],
            'head_specialization': [],
            'attention_flow': []
        }
        
        for layer_idx in range(self.n_layers):
            attn = self.state.get_attention_weights(layer_idx)[0]
            
            # Layer statistics
            layer_stats = {
                'layer': layer_idx,
                'mean_attention': attn.mean().item(),
                'max_attention': attn.max().item(),
                'entropy': self._compute_attention_entropy(attn),
                'sparsity': self._compute_sparsity(attn)
            }
            results['layer_stats'].append(layer_stats)
            
            # Head specialization
            for head_idx in range(attn.shape[0]):
                head_attn = attn[head_idx]
                specialization = {
                    'layer': layer_idx,
                    'head': head_idx,
                    'pattern_type': self._classify_attention_pattern(head_attn),
                    'entropy': self._compute_attention_entropy(head_attn.unsqueeze(0)),
                    'locality': self._compute_locality(head_attn)
                }
                results['head_specialization'].append(specialization)
        
        return results
    
    def analyze_information_flow(self) -> Dict:
        """
        Analyze how information flows through the network.
        
        Returns:
            Dictionary with flow statistics
        """
        residuals = self.state.get_residual_stream()
        
        results = {
            'norm_evolution': [],
            'change_magnitudes': [],
            'correlation_decay': []
        }
        
        # Norm evolution
        for i, res in enumerate(residuals):
            norm = res[0].norm(dim=-1).mean().item()
            results['norm_evolution'].append({
                'checkpoint': i - 1 if i > 0 else 'embedding',
                'norm': norm
            })
        
        # Change magnitudes
        for i in range(1, len(residuals)):
            change = (residuals[i] - residuals[i-1])[0]
            mag = change.norm(dim=-1).mean().item()
            results['change_magnitudes'].append({
                'from_checkpoint': i - 2 if i > 1 else 'embedding',
                'to_checkpoint': i - 1,
                'magnitude': mag
            })
        
        # Correlation decay
        embedding = residuals[0][0]
        for i in range(1, len(residuals)):
            current = residuals[i][0]
            correlation = torch.nn.functional.cosine_similarity(
                embedding, current, dim=-1
            ).mean().item()
            results['correlation_decay'].append({
                'checkpoint': i - 1,
                'correlation_with_embedding': correlation
            })
        
        return results
    
    def analyze_layer_importance(self) -> Dict:
        """
        Analyze the importance of each layer.
        
        Returns:
            Dictionary with layer importance metrics
        """
        residuals = self.state.get_residual_stream()
        
        results = {
            'contribution_magnitudes': [],
            'information_added': []
        }
        
        for i in range(1, len(residuals)):
            change = (residuals[i] - residuals[i-1])[0]
            
            contribution = {
                'layer': i - 1,
                'magnitude': change.norm().item(),
                'relative_magnitude': (change.norm() / residuals[i][0].norm()).item(),
                'directional_change': self._compute_directional_change(
                    residuals[i-1][0], residuals[i][0]
                )
            }
            results['contribution_magnitudes'].append(contribution)
        
        return results
    
    def analyze_gradient_health(self) -> Dict:
        """
        Analyze gradient flow health (if gradients available).
        
        Returns:
            Dictionary with gradient statistics
        """
        results = {
            'has_gradients': False,
            'layer_gradients': []
        }
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                results['has_gradients'] = True
                
                grad_stats = {
                    'parameter': name,
                    'grad_norm': param.grad.norm().item(),
                    'grad_mean': param.grad.mean().item(),
                    'grad_std': param.grad.std().item(),
                    'num_zeros': (param.grad == 0).sum().item(),
                    'total_elements': param.grad.numel()
                }
                results['layer_gradients'].append(grad_stats)
        
        return results
    
    def analyze_representational_geometry(self) -> Dict:
        """
        Analyze the geometry of representations.
        
        Returns:
            Dictionary with geometric properties
        """
        residuals = self.state.get_residual_stream()
        
        results = {
            'dimensionality': [],
            'isotropy': [],
            'clustering': []
        }
        
        for i, res in enumerate(residuals):
            # Effective dimensionality (via singular values)
            U, S, V = torch.svd(res[0])
            total_var = S.sum()
            cumsum = S.cumsum(0)
            effective_dim = (cumsum < 0.9 * total_var).sum().item()
            
            results['dimensionality'].append({
                'checkpoint': i - 1 if i > 0 else 'embedding',
                'effective_dim': effective_dim,
                'intrinsic_dim_ratio': effective_dim / res.shape[-1]
            })
            
            # Isotropy (uniformity of directions)
            normalized = res[0] / res[0].norm(dim=-1, keepdim=True)
            gram = normalized @ normalized.T
            isotropy_score = (gram - torch.eye(self.seq_len)).abs().mean().item()
            
            results['isotropy'].append({
                'checkpoint': i - 1 if i > 0 else 'embedding',
                'isotropy_score': isotropy_score
            })
        
        return results
    
    def generate_comprehensive_report(self, save_path: str = 'outputs/analysis/comprehensive_analysis.json'):
        """
        Generate comprehensive analysis report.
        
        Args:
            save_path: Path to save the report
            
        Returns:
            Complete analysis dictionary
        """
        report = {
            'model_config': {
                'n_layers': self.n_layers,
                'seq_len': self.seq_len,
                'd_model': self.state.embeddings.shape[-1],
            },
            'attention_analysis': self.analyze_attention_patterns(),
            'information_flow': self.analyze_information_flow(),
            'layer_importance': self.analyze_layer_importance(),
            'gradient_health': self.analyze_gradient_health(),
            'geometry': self.analyze_representational_geometry()
        }
        
        # Save to file
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _compute_attention_entropy(self, attn: torch.Tensor) -> float:
        """Compute entropy of attention distribution."""
        # Average over heads and queries
        avg_attn = attn.mean(dim=tuple(range(attn.dim() - 1)))
        entropy = -(avg_attn * torch.log(avg_attn + 1e-10)).sum()
        return entropy.item()
    
    def _compute_sparsity(self, attn: torch.Tensor, threshold: float = 0.01) -> float:
        """Compute sparsity of attention."""
        return (attn < threshold).float().mean().item()
    
    def _classify_attention_pattern(self, attn: torch.Tensor) -> str:
        """Classify the type of attention pattern."""
        # Diagonal dominance (local attention)
        diag_score = torch.diag(attn).mean()
        
        # Previous token attention
        if attn.shape[0] > 1:
            prev_score = torch.diag(attn, diagonal=-1).mean()
        else:
            prev_score = 0
        
        # Uniform attention
        uniform_score = 1.0 - attn.std()
        
        if diag_score > 0.5:
            return 'local'
        elif prev_score > 0.5:
            return 'previous'
        elif uniform_score > 0.8:
            return 'uniform'
        else:
            return 'mixed'
    
    def _compute_locality(self, attn: torch.Tensor) -> float:
        """Compute how local the attention is."""
        distances = torch.abs(
            torch.arange(attn.shape[0])[:, None] - 
            torch.arange(attn.shape[1])[None, :]
        ).float()
        
        weighted_distance = (attn * distances).sum() / attn.sum()
        return weighted_distance.item()
    
    def _compute_directional_change(self, before: torch.Tensor, after: torch.Tensor) -> float:
        """Compute directional change between representations."""
        before_norm = before / before.norm(dim=-1, keepdim=True)
        after_norm = after / after.norm(dim=-1, keepdim=True)
        
        cosine_sim = (before_norm * after_norm).sum(dim=-1).mean()
        return 1.0 - cosine_sim.item()


