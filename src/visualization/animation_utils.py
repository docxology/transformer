"""Utilities for creating smooth, high-quality animations."""

import torch
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


def interpolate_tensors(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    num_intermediate: int = 5
) -> List[torch.Tensor]:
    """
    Interpolate between two tensors for smooth transitions.
    
    Args:
        tensor1: Starting tensor
        tensor2: Ending tensor
        num_intermediate: Number of intermediate frames
        
    Returns:
        List of interpolated tensors (including start and end)
    """
    assert tensor1.shape == tensor2.shape, "Tensors must have same shape"
    
    interpolated = []
    for i in range(num_intermediate + 2):
        alpha = i / (num_intermediate + 1)
        interp = (1 - alpha) * tensor1 + alpha * tensor2
        interpolated.append(interp)
    
    return interpolated


def create_bezier_path(
    start: Tuple[float, float],
    end: Tuple[float, float],
    num_points: int = 50,
    control_offset: float = 0.3
) -> List[Tuple[float, float]]:
    """
    Create smooth bezier curve between two points.
    
    Args:
        start: Starting (x, y) coordinate
        end: Ending (x, y) coordinate
        num_points: Number of points along curve
        control_offset: Offset for control points
        
    Returns:
        List of (x, y) coordinates along curve
    """
    x1, y1 = start
    x4, y4 = end
    
    # Calculate control points
    dx = x4 - x1
    dy = y4 - y1
    
    x2 = x1 + dx * 0.33 + dy * control_offset
    y2 = y1 + dy * 0.33 - dx * control_offset
    
    x3 = x1 + dx * 0.67 - dy * control_offset
    y3 = y1 + dy * 0.67 + dx * control_offset
    
    # Generate points along bezier curve
    t_values = np.linspace(0, 1, num_points)
    points = []
    
    for t in t_values:
        # Cubic bezier formula
        x = (1-t)**3 * x1 + 3*(1-t)**2*t * x2 + 3*(1-t)*t**2 * x3 + t**3 * x4
        y = (1-t)**3 * y1 + 3*(1-t)**2*t * y2 + 3*(1-t)*t**2 * y3 + t**3 * y4
        points.append((x, y))
    
    return points


def apply_easing(t: float, easing_type: str = 'ease_in_out') -> float:
    """
    Apply easing function to time parameter.
    
    Args:
        t: Time parameter in [0, 1]
        easing_type: Type of easing ('linear', 'ease_in', 'ease_out', 'ease_in_out')
        
    Returns:
        Eased value in [0, 1]
    """
    if easing_type == 'linear':
        return t
    elif easing_type == 'ease_in':
        return t * t
    elif easing_type == 'ease_out':
        return 1 - (1 - t) ** 2
    elif easing_type == 'ease_in_out':
        if t < 0.5:
            return 2 * t * t
        else:
            return 1 - 2 * (1 - t) ** 2
    else:
        return t


def create_progress_bar_overlay(
    ax,
    frame: int,
    total_frames: int,
    position: str = 'bottom'
) -> None:
    """
    Add a progress bar to an animation frame.
    
    Args:
        ax: Matplotlib axes
        frame: Current frame number
        total_frames: Total number of frames
        position: Position of progress bar ('top', 'bottom')
    """
    from matplotlib.patches import Rectangle
    
    progress = frame / total_frames
    
    # Get axes limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Position progress bar
    bar_height = (ylim[1] - ylim[0]) * 0.02
    bar_width = xlim[1] - xlim[0]
    
    if position == 'bottom':
        bar_y = ylim[0]
    else:  # top
        bar_y = ylim[1] - bar_height
    
    # Background
    bg_rect = Rectangle(
        (xlim[0], bar_y), bar_width, bar_height,
        facecolor='lightgray', edgecolor='black', linewidth=1, zorder=1000
    )
    ax.add_patch(bg_rect)
    
    # Progress
    progress_rect = Rectangle(
        (xlim[0], bar_y), bar_width * progress, bar_height,
        facecolor='dodgerblue', edgecolor='none', zorder=1001
    )
    ax.add_patch(progress_rect)
    
    # Text
    percentage_text = f'{progress*100:.0f}%'
    ax.text(
        xlim[0] + bar_width/2, bar_y + bar_height/2,
        percentage_text,
        ha='center', va='center', fontsize=8, weight='bold',
        color='white' if progress > 0.5 else 'black',
        zorder=1002
    )


def add_timestamp_overlay(
    ax,
    frame: int,
    fps: int,
    position: Tuple[float, float] = (0.02, 0.98),
    format_str: str = 'Frame {frame} | Time {time:.2f}s'
) -> None:
    """
    Add timestamp overlay to animation frame.
    
    Args:
        ax: Matplotlib axes
        frame: Current frame number
        fps: Frames per second
        position: Normalized position (0-1, 0-1)
        format_str: Format string for timestamp
    """
    time_seconds = frame / fps
    
    text = format_str.format(frame=frame, time=time_seconds)
    
    ax.text(
        position[0], position[1], text,
        transform=ax.transAxes,
        fontsize=10, family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                 edgecolor='black', alpha=0.8),
        verticalalignment='top'
    )


def compute_smooth_trajectory(
    waypoints: List[Tuple[float, float]],
    smoothness: int = 10
) -> List[Tuple[float, float]]:
    """
    Create smooth trajectory through waypoints using interpolation.
    
    Args:
        waypoints: List of (x, y) waypoints
        smoothness: Number of interpolated points between each waypoint
        
    Returns:
        Smooth trajectory as list of (x, y) coordinates
    """
    if len(waypoints) < 2:
        return waypoints
    
    trajectory = []
    
    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        end = waypoints[i + 1]
        
        # Linear interpolation between waypoints
        for j in range(smoothness):
            t = j / smoothness
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])
            trajectory.append((x, y))
    
    # Add final waypoint
    trajectory.append(waypoints[-1])
    
    return trajectory


@dataclass
class AnimationConfig:
    """Configuration for animation rendering."""
    fps: int = 10
    duration_seconds: int = 10
    resolution: Tuple[int, int] = (1920, 1080)
    dpi: int = 100
    show_progress: bool = True
    show_timestamp: bool = False
    easing: str = 'ease_in_out'
    interpolation_frames: int = 5
    
    @property
    def total_frames(self) -> int:
        """Total number of frames."""
        return self.fps * self.duration_seconds
    
    @property
    def figsize(self) -> Tuple[float, float]:
        """Figure size in inches."""
        return (self.resolution[0] / self.dpi, self.resolution[1] / self.dpi)


def create_fade_transition(
    alpha_start: float,
    alpha_end: float,
    frame: int,
    total_frames: int,
    easing: str = 'linear'
) -> float:
    """
    Create smooth fade transition.
    
    Args:
        alpha_start: Starting alpha value
        alpha_end: Ending alpha value
        frame: Current frame
        total_frames: Total frames in transition
        easing: Easing function to apply
        
    Returns:
        Alpha value for current frame
    """
    t = frame / max(1, total_frames - 1)
    t_eased = apply_easing(t, easing)
    
    alpha = alpha_start + t_eased * (alpha_end - alpha_start)
    return np.clip(alpha, 0, 1)


def pulse_effect(
    frame: int,
    period: int = 30,
    min_value: float = 0.3,
    max_value: float = 1.0
) -> float:
    """
    Create pulsing effect for animations.
    
    Args:
        frame: Current frame number
        period: Number of frames per pulse cycle
        min_value: Minimum value
        max_value: Maximum value
        
    Returns:
        Pulsed value
    """
    phase = (frame % period) / period
    amplitude = (max_value - min_value) / 2
    offset = (max_value + min_value) / 2
    
    value = offset + amplitude * np.sin(2 * np.pi * phase)
    return value


def create_color_gradient(
    start_color: Tuple[float, float, float],
    end_color: Tuple[float, float, float],
    num_steps: int
) -> List[Tuple[float, float, float]]:
    """
    Create smooth color gradient.
    
    Args:
        start_color: Starting RGB color (0-1 values)
        end_color: Ending RGB color (0-1 values)
        num_steps: Number of gradient steps
        
    Returns:
        List of RGB colors
    """
    gradient = []
    
    for i in range(num_steps):
        t = i / max(1, num_steps - 1)
        color = tuple(
            start_color[j] + t * (end_color[j] - start_color[j])
            for j in range(3)
        )
        gradient.append(color)
    
    return gradient



