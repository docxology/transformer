"""Progress tracking utilities for long-running operations."""

from tqdm.auto import tqdm
from typing import Optional, Iterator
import time


def create_progress_bar(
    total: int,
    desc: str = "Processing",
    unit: str = "it",
    color: str = 'green'
) -> tqdm:
    """
    Create a beautiful progress bar.
    
    Args:
        total: Total number of iterations
        desc: Description to display
        unit: Unit name
        color: Progress bar color
        
    Returns:
        tqdm progress bar object
    """
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
        colour=color
    )


def track_progress(iterable, desc: str = "Processing", **kwargs) -> Iterator:
    """
    Wrap an iterable with a progress bar.
    
    Args:
        iterable: Iterable to track
        desc: Description to display
        **kwargs: Additional arguments for tqdm
        
    Returns:
        Iterator with progress tracking
    """
    return tqdm(iterable, desc=desc, **kwargs)


class MultiProgressTracker:
    """Track progress across multiple concurrent operations."""
    
    def __init__(self):
        self.bars = {}
    
    def add_bar(self, name: str, total: int, desc: str = None):
        """Add a new progress bar."""
        if desc is None:
            desc = name
        
        self.bars[name] = tqdm(
            total=total,
            desc=desc,
            position=len(self.bars),
            leave=True
        )
    
    def update(self, name: str, n: int = 1):
        """Update a specific progress bar."""
        if name in self.bars:
            self.bars[name].update(n)
    
    def close(self, name: Optional[str] = None):
        """Close a specific bar or all bars."""
        if name:
            if name in self.bars:
                self.bars[name].close()
                del self.bars[name]
        else:
            for bar in self.bars.values():
                bar.close()
            self.bars.clear()
    
    def set_description(self, name: str, desc: str):
        """Update description of a bar."""
        if name in self.bars:
            self.bars[name].set_description(desc)


class Timer:
    """Simple timer for tracking elapsed time."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Start timer."""
        print(f"⏱️  {self.name} started...")
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer and print elapsed time."""
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        
        if elapsed < 1:
            print(f"✓ {self.name} completed in {elapsed*1000:.2f} ms")
        elif elapsed < 60:
            print(f"✓ {self.name} completed in {elapsed:.2f} seconds")
        else:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            print(f"✓ {self.name} completed in {minutes}m {seconds:.2f}s")
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time."""
        if self.start_time is None:
            return 0.0
        
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string
    """
    if seconds < 1:
        return f"{seconds*1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def estimate_remaining(completed: int, total: int, elapsed: float) -> str:
    """
    Estimate remaining time.
    
    Args:
        completed: Number of items completed
        total: Total number of items
        elapsed: Time elapsed so far (seconds)
        
    Returns:
        Formatted remaining time estimate
    """
    if completed == 0:
        return "Unknown"
    
    rate = elapsed / completed
    remaining_items = total - completed
    remaining_seconds = rate * remaining_items
    
    return format_time(remaining_seconds)



