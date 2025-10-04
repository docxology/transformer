"""3D visualizations for transformer information flow."""

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import numpy as np
from typing import Optional, Union, List
from sklearn.decomposition import PCA


def visualize_residual_3d(
    state: any,
    use_pca: bool = True,
    n_components: int = 3,
    save_path: Optional[str] = None,
    interactive: bool = True,
    positions: Optional[List[int]] = None
) -> Union[plt.Figure, go.Figure]:
    """
    Visualize residual stream evolution in 3D space.
    
    Shows how token representations evolve through layers in a 3D feature space
    (using PCA for dimensionality reduction if needed).
    
    Args:
        state: TransformerState object
        use_pca: Whether to use PCA for dimensionality reduction
        n_components: Number of PCA components (2 or 3)
        save_path: Path to save figure
        interactive: Whether to create interactive Plotly plot
        positions: Which positions to visualize (None = all)
        
    Returns:
        Matplotlib or Plotly figure
    """
    residuals = state.get_residual_stream()
    n_layers = len(residuals)
    seq_len = residuals[0].shape[1]
    d_model = residuals[0].shape[2]
    
    if positions is None:
        # Sample positions for clarity
        positions = list(range(0, seq_len, max(1, seq_len // 8)))
    
    # Extract and prepare data
    trajectories = []
    for pos in positions:
        trajectory = []
        for layer_idx, residual in enumerate(residuals):
            vec = residual[0, pos, :].cpu().detach().numpy()
            trajectory.append((layer_idx, vec))
        trajectories.append((pos, trajectory))
    
    # Apply PCA if requested
    if use_pca and d_model > n_components:
        # Collect all vectors
        all_vectors = []
        for _, trajectory in trajectories:
            all_vectors.extend([vec for _, vec in trajectory])
        
        # Fit PCA
        pca = PCA(n_components=n_components)
        all_vectors = np.array(all_vectors)
        pca.fit(all_vectors)
        
        # Transform trajectories
        transformed_trajectories = []
        for pos, trajectory in trajectories:
            transformed = []
            for layer, vec in trajectory:
                vec_transformed = pca.transform(vec.reshape(1, -1))[0]
                transformed.append((layer, vec_transformed))
            transformed_trajectories.append((pos, transformed))
        trajectories = transformed_trajectories
        
        explained_var = pca.explained_variance_ratio_
        title_suffix = f' (PCA: {explained_var.sum()*100:.1f}% variance)'
    else:
        title_suffix = ''
    
    if interactive and n_components == 3:
        # Create interactive Plotly 3D plot
        fig = go.Figure()
        
        for pos, trajectory in trajectories:
            layers = [layer for layer, _ in trajectory]
            x = [vec[0] for _, vec in trajectory]
            y = [vec[1] for _, vec in trajectory]
            z = [vec[2] for _, vec in trajectory]
            
            # Add trajectory line
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines+markers',
                name=f'Position {pos}',
                line=dict(width=3),
                marker=dict(size=5),
                text=[f'Layer {l}' for l in layers],
                hovertemplate='<b>Position %{fullData.name}</b><br>' +
                             'Layer: %{text}<br>' +
                             'PC1: %{x:.3f}<br>' +
                             'PC2: %{y:.3f}<br>' +
                             'PC3: %{z:.3f}<extra></extra>'
            ))
        
        fig.update_layout(
            title=f'3D Residual Stream Evolution{title_suffix}',
            scene=dict(
                xaxis_title='PC1' if use_pca else 'Dim 0',
                yaxis_title='PC2' if use_pca else 'Dim 1',
                zaxis_title='PC3' if use_pca else 'Dim 2',
            ),
            width=900,
            height=700,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path.replace('.png', '.html'))
        
        return fig
    
    else:
        # Create matplotlib 3D plot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(trajectories)))
        
        for idx, (pos, trajectory) in enumerate(trajectories):
            x = [vec[0] for _, vec in trajectory]
            y = [vec[1] for _, vec in trajectory]
            z = [vec[2] if len(vec) > 2 else 0 for _, vec in trajectory]
            
            ax.plot(x, y, z, 'o-', color=colors[idx], 
                   linewidth=2, markersize=6, label=f'Pos {pos}', alpha=0.7)
            
            # Mark start and end
            ax.scatter([x[0]], [y[0]], [z[0]], color=colors[idx], 
                      s=100, marker='s', edgecolors='black', linewidths=2)
            ax.scatter([x[-1]], [y[-1]], [z[-1]], color=colors[idx],
                      s=100, marker='*', edgecolors='black', linewidths=2)
        
        ax.set_xlabel('PC1' if use_pca else 'Dim 0', fontsize=12)
        ax.set_ylabel('PC2' if use_pca else 'Dim 1', fontsize=12)
        ax.set_zlabel('PC3' if use_pca else 'Dim 2', fontsize=12)
        ax.set_title(f'3D Residual Stream Evolution{title_suffix}', fontsize=14, weight='bold')
        ax.legend(loc='upper left', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def visualize_attention_landscape_3d(
    state: any,
    layer_idx: int = 0,
    head_idx: Optional[int] = None,
    save_path: Optional[str] = None,
    interactive: bool = True
) -> Union[plt.Figure, go.Figure]:
    """
    Create 3D surface plot of attention weights.
    
    Visualizes attention as a landscape where height represents attention weight.
    
    Args:
        state: TransformerState object
        layer_idx: Which layer to visualize
        head_idx: Which head to visualize (None = average all heads)
        save_path: Path to save figure
        interactive: Whether to create interactive Plotly plot
        
    Returns:
        Matplotlib or Plotly figure
    """
    attn_weights = state.get_attention_weights(layer_idx)[0]
    
    if head_idx is not None:
        attn_data = attn_weights[head_idx].cpu().detach().numpy()
        title = f'Attention Landscape - Layer {layer_idx}, Head {head_idx}'
    else:
        attn_data = attn_weights.mean(0).cpu().detach().numpy()
        title = f'Attention Landscape - Layer {layer_idx} (Average)'
    
    seq_len = attn_data.shape[0]
    
    # Create meshgrid
    x = np.arange(seq_len)
    y = np.arange(seq_len)
    X, Y = np.meshgrid(x, y)
    Z = attn_data
    
    if interactive:
        # Plotly 3D surface
        fig = go.Figure(data=[go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Viridis',
            colorbar=dict(title='Attention<br>Weight')
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Key Position',
                yaxis_title='Query Position',
                zaxis_title='Attention Weight',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            width=900,
            height=700
        )
        
        if save_path:
            fig.write_html(save_path.replace('.png', '.html'))
        
        return fig
    
    else:
        # Matplotlib 3D surface
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9,
                              linewidth=0, antialiased=True)
        
        ax.set_xlabel('Key Position', fontsize=12)
        ax.set_ylabel('Query Position', fontsize=12)
        ax.set_zlabel('Attention Weight', fontsize=12)
        ax.set_title(title, fontsize=14, weight='bold')
        
        fig.colorbar(surf, ax=ax, fraction=0.03, pad=0.1, label='Attention Weight')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def visualize_token_trajectory_3d(
    state: any,
    token_position: int,
    save_path: Optional[str] = None,
    interactive: bool = True
) -> Union[plt.Figure, go.Figure]:
    """
    Track a single token's representation through layers in 3D.
    
    Shows how one token's representation evolves, with trajectory colored
    by layer number.
    
    Args:
        state: TransformerState object
        token_position: Which token position to track
        save_path: Path to save figure
        interactive: Whether to create interactive Plotly plot
        
    Returns:
        Matplotlib or Plotly figure
    """
    residuals = state.get_residual_stream()
    
    # Extract token trajectory
    trajectory = []
    for residual in residuals:
        vec = residual[0, token_position, :].cpu().detach().numpy()
        trajectory.append(vec)
    
    trajectory = np.array(trajectory)
    
    # Apply PCA to 3D
    pca = PCA(n_components=3)
    trajectory_3d = pca.fit_transform(trajectory)
    
    layers = np.arange(len(trajectory))
    explained_var = pca.explained_variance_ratio_
    
    if interactive:
        # Plotly 3D
        fig = go.Figure()
        
        # Trajectory line
        fig.add_trace(go.Scatter3d(
            x=trajectory_3d[:, 0],
            y=trajectory_3d[:, 1],
            z=trajectory_3d[:, 2],
            mode='lines+markers',
            line=dict(
                color=layers,
                colorscale='Viridis',
                width=5
            ),
            marker=dict(
                size=8,
                color=layers,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Layer')
            ),
            text=[f'Layer {l}' for l in layers],
            hovertemplate='<b>Token Position {}</b><br>'.format(token_position) +
                         'Layer: %{text}<br>' +
                         'PC1: %{x:.3f}<br>' +
                         'PC2: %{y:.3f}<br>' +
                         'PC3: %{z:.3f}<extra></extra>'
        ))
        
        # Mark start
        fig.add_trace(go.Scatter3d(
            x=[trajectory_3d[0, 0]],
            y=[trajectory_3d[0, 1]],
            z=[trajectory_3d[0, 2]],
            mode='markers',
            marker=dict(size=15, color='green', symbol='square'),
            name='Start',
            showlegend=True
        ))
        
        # Mark end
        fig.add_trace(go.Scatter3d(
            x=[trajectory_3d[-1, 0]],
            y=[trajectory_3d[-1, 1]],
            z=[trajectory_3d[-1, 2]],
            mode='markers',
            marker=dict(size=15, color='red', symbol='diamond'),
            name='End',
            showlegend=True
        ))
        
        fig.update_layout(
            title=f'Token {token_position} Trajectory Through Layers<br>' +
                  f'<sub>PCA: {explained_var.sum()*100:.1f}% variance explained</sub>',
            scene=dict(
                xaxis_title='PC1',
                yaxis_title='PC2',
                zaxis_title='PC3',
            ),
            width=900,
            height=700
        )
        
        if save_path:
            fig.write_html(save_path.replace('.png', '.html'))
        
        return fig
    
    else:
        # Matplotlib 3D
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory with color gradient
        points = ax.scatter(
            trajectory_3d[:, 0],
            trajectory_3d[:, 1],
            trajectory_3d[:, 2],
            c=layers,
            cmap='viridis',
            s=100,
            alpha=0.8
        )
        
        # Connect with line
        ax.plot(
            trajectory_3d[:, 0],
            trajectory_3d[:, 1],
            trajectory_3d[:, 2],
            'k-', alpha=0.3, linewidth=1
        )
        
        # Mark start and end
        ax.scatter([trajectory_3d[0, 0]], [trajectory_3d[0, 1]], [trajectory_3d[0, 2]],
                  s=200, color='green', marker='s', edgecolors='black', linewidths=2,
                  label='Start (Embedding)', zorder=10)
        ax.scatter([trajectory_3d[-1, 0]], [trajectory_3d[-1, 1]], [trajectory_3d[-1, 2]],
                  s=200, color='red', marker='D', edgecolors='black', linewidths=2,
                  label='End (Output)', zorder=10)
        
        ax.set_xlabel('PC1', fontsize=12)
        ax.set_ylabel('PC2', fontsize=12)
        ax.set_zlabel('PC3', fontsize=12)
        ax.set_title(f'Token {token_position} Trajectory Through Layers\n' +
                    f'PCA: {explained_var.sum()*100:.1f}% variance explained',
                    fontsize=14, weight='bold')
        
        fig.colorbar(points, ax=ax, fraction=0.03, pad=0.1, label='Layer')
        ax.legend(loc='upper left', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def create_3d_attention_volume(
    state: any,
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Create 3D volume visualization of all attention weights.
    
    Shows attention across all layers, heads, and positions as a 3D volume.
    
    Args:
        state: TransformerState object
        save_path: Path to save HTML file
        
    Returns:
        Plotly figure
    """
    n_layers = len(state.layer_states)
    
    # Collect all attention weights
    all_attn = []
    for layer_idx in range(n_layers):
        attn = state.get_attention_weights(layer_idx)[0]  # [heads, seq, seq]
        attn_mean = attn.mean(0).cpu().detach().numpy()  # [seq, seq]
        all_attn.append(attn_mean)
    
    # Stack to create volume [layers, seq, seq]
    volume = np.array(all_attn)
    
    # Create isosurface plot
    fig = go.Figure(data=go.Isosurface(
        x=np.repeat(np.arange(volume.shape[2]), volume.shape[0] * volume.shape[1]),
        y=np.tile(np.repeat(np.arange(volume.shape[1]), volume.shape[2]), volume.shape[0]),
        z=np.tile(np.arange(volume.shape[0]), volume.shape[1] * volume.shape[2]),
        value=volume.flatten(),
        isomin=0.1,
        isomax=volume.max(),
        opacity=0.6,
        surface_count=4,
        colorscale='Viridis',
        colorbar=dict(title='Attention<br>Weight')
    ))
    
    fig.update_layout(
        title='3D Attention Volume Across All Layers',
        scene=dict(
            xaxis_title='Key Position',
            yaxis_title='Query Position',
            zaxis_title='Layer',
        ),
        width=900,
        height=700
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig



