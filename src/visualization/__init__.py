"""Visualization and animation tools for transformer information flow."""

from .attention_viz import visualize_attention, plot_attention_heads, visualize_attention_patterns
from .flow_viz import (
    visualize_residual_stream,
    visualize_kv_stream,
    plot_layer_statistics,
    visualize_information_highways
)
from .animation import (
    animate_information_flow,
    animate_path_trace,
    animate_attention_evolution
)
from .path_analysis import (
    compute_path_counts,
    trace_information_paths,
    analyze_path_complexity,
    compute_path_statistics,
    visualize_path_grid
)
from .advanced_viz import (
    visualize_gradient_flow,
    visualize_activation_patterns,
    visualize_causal_graph,
    visualize_attention_flow_sankey,
    visualize_token_journey,
    visualize_head_specialization
)
from .training_viz import (
    TrainingVisualizer,
    visualize_weight_distributions,
    visualize_learning_dynamics
)
from .advanced_animations import (
    animate_gradient_flow,
    animate_layer_activations,
    animate_attention_heads,
    animate_token_interactions,
    animate_information_diffusion,
    animate_residual_evolution,
    animate_head_comparison
)
from .architecture_viz import (
    visualize_model_architecture,
    visualize_attention_mechanism,
    visualize_parameter_breakdown,
    visualize_computational_graph,
    visualize_information_highways_diagram,
    create_all_architecture_visualizations
)

# NEW: Enhanced visualization capabilities
from .subway_map_animation import create_subway_map_animation
from .viz_3d import (
    visualize_residual_3d,
    visualize_attention_landscape_3d,
    visualize_token_trajectory_3d,
    create_3d_attention_volume
)
from .color_schemes import (
    get_scheme,
    create_gradient,
    get_categorical_colors,
    get_diverging_colormap,
    get_attention_colormap,
    get_activation_colormap,
    get_layer_colors,
    get_head_colors
)
from .animation_utils import (
    interpolate_tensors,
    create_bezier_path,
    apply_easing,
    create_progress_bar_overlay,
    add_timestamp_overlay,
    AnimationConfig
)
from .batch_animator import (
    batch_generate_animations,
    batch_generate_for_training,
    create_animation_gallery
)
from .comparative_viz import (
    visualize_model_comparison,
    compare_attention_patterns,
    compare_layer_statistics,
    create_comparison_dashboard
)

__all__ = [
    # Basic attention visualization
    'visualize_attention',
    'plot_attention_heads',
    'visualize_attention_patterns',
    # Flow visualization
    'visualize_residual_stream',
    'visualize_kv_stream',
    'plot_layer_statistics',
    'visualize_information_highways',
    # Basic animation
    'animate_information_flow',
    'animate_path_trace',
    'animate_attention_evolution',
    # Advanced animations
    'animate_gradient_flow',
    'animate_layer_activations',
    'animate_attention_heads',
    'animate_token_interactions',
    'animate_information_diffusion',
    'animate_residual_evolution',
    'animate_head_comparison',
    # Path analysis
    'compute_path_counts',
    'trace_information_paths',
    'analyze_path_complexity',
    'compute_path_statistics',
    'visualize_path_grid',
    # Advanced visualization
    'visualize_gradient_flow',
    'visualize_activation_patterns',
    'visualize_causal_graph',
    'visualize_attention_flow_sankey',
    'visualize_token_journey',
    'visualize_head_specialization',
    # Training visualization
    'TrainingVisualizer',
    'visualize_weight_distributions',
    'visualize_learning_dynamics',
    # Architecture visualization
    'visualize_model_architecture',
    'visualize_attention_mechanism',
    'visualize_parameter_breakdown',
    'visualize_computational_graph',
    'visualize_information_highways_diagram',
    'create_all_architecture_visualizations',
    # NEW: Subway map animation
    'create_subway_map_animation',
    # NEW: 3D visualizations
    'visualize_residual_3d',
    'visualize_attention_landscape_3d',
    'visualize_token_trajectory_3d',
    'create_3d_attention_volume',
    # NEW: Color schemes
    'get_scheme',
    'create_gradient',
    'get_categorical_colors',
    'get_diverging_colormap',
    'get_attention_colormap',
    'get_activation_colormap',
    'get_layer_colors',
    'get_head_colors',
    # NEW: Animation utilities
    'interpolate_tensors',
    'create_bezier_path',
    'apply_easing',
    'create_progress_bar_overlay',
    'add_timestamp_overlay',
    'AnimationConfig',
    # NEW: Batch processing
    'batch_generate_animations',
    'batch_generate_for_training',
    'create_animation_gallery',
    # NEW: Comparative visualization
    'visualize_model_comparison',
    'compare_attention_patterns',
    'compare_layer_statistics',
    'create_comparison_dashboard'
]

