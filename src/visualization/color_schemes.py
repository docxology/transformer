"""Professional color schemes for visualizations."""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict, Optional
import seaborn as sns


# Define color schemes
SCHEMES = {
    'default': {
        'primary': '#1976D2',
        'secondary': '#FFC107',
        'success': '#4CAF50',
        'danger': '#F44336',
        'info': '#00BCD4',
        'warning': '#FF9800',
        'dark': '#212121',
        'light': '#F5F5F5'
    },
    'professional': {
        'primary': '#2E3B55',
        'secondary': '#E67E22',
        'success': '#27AE60',
        'danger': '#C0392B',
        'info': '#3498DB',
        'warning': '#F39C12',
        'dark': '#1C2833',
        'light': '#ECF0F1'
    },
    'pastel': {
        'primary': '#A8D5E2',
        'secondary': '#FFD3A5',
        'success': '#C4E1C5',
        'danger': '#FFA5A5',
        'info': '#B4C7E7',
        'warning': '#FFE5A5',
        'dark': '#6B7280',
        'light': '#F9FAFB'
    },
    'vibrant': {
        'primary': '#00D9FF',
        'secondary': '#FF006E',
        'success': '#06FFA5',
        'danger': '#FF4500',
        'info': '#9D4EDD',
        'warning': '#FFD60A',
        'dark': '#000814',
        'light': '#F8F9FA'
    },
    'colorblind_safe': {
        # Colorblind-friendly palette (Paul Tol)
        'primary': '#4477AA',
        'secondary': '#EE6677',
        'success': '#228833',
        'danger': '#CC6677',
        'info': '#66CCEE',
        'warning': '#CCBB44',
        'dark': '#332288',
        'light': '#F0F0F0'
    },
    'grayscale': {
        'primary': '#555555',
        'secondary': '#888888',
        'success': '#333333',
        'danger': '#AAAAAA',
        'info': '#666666',
        'warning': '#999999',
        'dark': '#000000',
        'light': '#FFFFFF'
    }
}


def get_scheme(name: str = 'default') -> Dict[str, str]:
    """
    Get a color scheme by name.
    
    Args:
        name: Scheme name ('default', 'professional', 'pastel', 'vibrant',
              'colorblind_safe', 'grayscale')
              
    Returns:
        Dictionary mapping color names to hex codes
    """
    return SCHEMES.get(name, SCHEMES['default'])


def create_gradient(
    start_color: str,
    end_color: str,
    n_colors: int = 256
) -> List[str]:
    """
    Create gradient between two colors.
    
    Args:
        start_color: Starting color (hex or named)
        end_color: Ending color (hex or named)
        n_colors: Number of colors in gradient
        
    Returns:
        List of hex color codes
    """
    from matplotlib.colors import to_rgb, to_hex
    
    start_rgb = to_rgb(start_color)
    end_rgb = to_rgb(end_color)
    
    gradient = []
    for i in range(n_colors):
        t = i / (n_colors - 1)
        rgb = tuple(start_rgb[j] + t * (end_rgb[j] - start_rgb[j]) for j in range(3))
        gradient.append(to_hex(rgb))
    
    return gradient


def create_perceptually_uniform_colormap(
    name: str = 'viridis',
    n_colors: int = 256
) -> plt.cm.ScalarMappable:
    """
    Create perceptually uniform colormap.
    
    Args:
        name: Colormap name ('viridis', 'plasma', 'inferno', 'magma', 'cividis')
        n_colors: Number of discrete colors
        
    Returns:
        Matplotlib colormap
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    
    cmap = cm.get_cmap(name, n_colors)
    norm = mcolors.Normalize(vmin=0, vmax=1)
    
    return cm.ScalarMappable(norm=norm, cmap=cmap)


def get_categorical_colors(
    n_categories: int,
    scheme: str = 'default',
    style: str = 'distinct'
) -> List[str]:
    """
    Get colors for categorical data.
    
    Args:
        n_categories: Number of categories
        scheme: Color scheme to use
        style: 'distinct' for maximally different colors, 'gradient' for smooth progression
        
    Returns:
        List of hex color codes
    """
    if style == 'distinct':
        if n_categories <= 10:
            # Use tab10 for up to 10 categories
            colors = plt.cm.tab10(np.linspace(0, 1, n_categories))
        elif n_categories <= 20:
            # Use tab20 for up to 20 categories
            colors = plt.cm.tab20(np.linspace(0, 1, n_categories))
        else:
            # Use hsv for more categories
            colors = plt.cm.hsv(np.linspace(0, 1, n_categories))
        
        return [f'#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}' 
                for c in colors]
    
    else:  # gradient
        scheme_colors = get_scheme(scheme)
        return create_gradient(
            scheme_colors['primary'],
            scheme_colors['secondary'],
            n_categories
        )


def get_diverging_colormap(
    center_color: str = 'white',
    negative_color: str = 'blue',
    positive_color: str = 'red',
    n_colors: int = 256
) -> List[str]:
    """
    Create diverging colormap (for data centered at zero).
    
    Args:
        center_color: Color for center/zero value
        negative_color: Color for negative values
        positive_color: Color for positive values
        n_colors: Total number of colors
        
    Returns:
        List of hex color codes
    """
    half = n_colors // 2
    
    negative_gradient = create_gradient(negative_color, center_color, half)
    positive_gradient = create_gradient(center_color, positive_color, half)
    
    return negative_gradient + positive_gradient[1:]  # Avoid duplicating center


def apply_alpha(color: str, alpha: float) -> str:
    """
    Apply alpha transparency to color.
    
    Args:
        color: Hex color code
        alpha: Alpha value (0-1)
        
    Returns:
        Hex color code with alpha
    """
    from matplotlib.colors import to_rgba, to_hex
    
    rgba = to_rgba(color)
    rgba_with_alpha = (*rgba[:3], alpha)
    
    return to_hex(rgba_with_alpha, keep_alpha=True)


def get_attention_colormap() -> plt.cm.ScalarMappable:
    """Get specialized colormap for attention weights."""
    import matplotlib.colors as mcolors
    
    # Create custom colormap: white (low attention) to vibrant color (high attention)
    colors = ['#FFFFFF', '#E3F2FD', '#90CAF9', '#42A5F5', '#1976D2', '#0D47A1']
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list('attention', colors, N=n_bins)
    norm = mcolors.Normalize(vmin=0, vmax=1)
    
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)


def get_activation_colormap() -> plt.cm.ScalarMappable:
    """Get specialized colormap for activations (diverging)."""
    import matplotlib.colors as mcolors
    
    # Blue (negative) -> White (zero) -> Red (positive)
    colors = ['#0D47A1', '#42A5F5', '#E3F2FD', '#FFFFFF', 
              '#FFEBEE', '#EF5350', '#B71C1C']
    cmap = mcolors.LinearSegmentedColormap.from_list('activation', colors, N=256)
    
    # Symmetric around zero
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)


def visualize_scheme(scheme_name: str = 'default', save_path: Optional[str] = None):
    """
    Visualize a color scheme.
    
    Args:
        scheme_name: Name of scheme to visualize
        save_path: Path to save visualization
    """
    scheme = get_scheme(scheme_name)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, (name, color) in enumerate(scheme.items()):
        ax = axes[idx]
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=color))
        ax.text(0.5, 0.5, name, ha='center', va='center', 
               fontsize=14, weight='bold',
               color='white' if _is_dark(color) else 'black')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    fig.suptitle(f'Color Scheme: {scheme_name}', fontsize=18, weight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def _is_dark(color: str) -> bool:
    """Check if color is dark (for choosing text color)."""
    from matplotlib.colors import to_rgb
    
    rgb = to_rgb(color)
    # Calculate perceived luminance
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return luminance < 0.5


def get_layer_colors(n_layers: int, style: str = 'gradient') -> List[str]:
    """
    Get colors for visualizing different layers.
    
    Args:
        n_layers: Number of layers
        style: 'gradient' for smooth progression, 'categorical' for distinct colors
        
    Returns:
        List of hex color codes
    """
    if style == 'gradient':
        # Green (input) -> Blue (middle) -> Red (output)
        colors = []
        for i in range(n_layers):
            t = i / max(1, n_layers - 1)
            if t < 0.5:
                # Green to Blue
                t_local = t * 2
                r = int((1 - t_local) * 129 + t_local * 33)
                g = int((1 - t_local) * 199 + t_local * 150)
                b = int((1 - t_local) * 132 + t_local * 243)
            else:
                # Blue to Red
                t_local = (t - 0.5) * 2
                r = int((1 - t_local) * 33 + t_local * 229)
                g = int((1 - t_local) * 150 + t_local * 57)
                b = int((1 - t_local) * 243 + t_local) * 53
            
            colors.append(f'#{r:02x}{g:02x}{b:02x}')
        
        return colors
    
    else:  # categorical
        return get_categorical_colors(n_layers, style='distinct')


def get_head_colors(n_heads: int) -> List[str]:
    """
    Get colors for visualizing different attention heads.
    
    Args:
        n_heads: Number of attention heads
        
    Returns:
        List of hex color codes
    """
    return get_categorical_colors(n_heads, style='distinct')


# Export available schemes
AVAILABLE_SCHEMES = list(SCHEMES.keys())



