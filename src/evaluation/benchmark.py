"""Benchmarking and profiling tools."""

import torch
import time
from typing import Dict, Optional, Callable
from dataclasses import dataclass, field
import numpy as np
from contextlib import contextmanager


@dataclass
class BenchmarkResults:
    """Results from benchmarking runs."""
    operation: str
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    throughput: Optional[float] = None
    memory_allocated: Optional[float] = None
    memory_reserved: Optional[float] = None
    num_runs: int = 0
    
    def __str__(self) -> str:
        """String representation."""
        lines = [
            f"Benchmark: {self.operation}",
            f"  Runs: {self.num_runs}",
            f"  Mean time: {self.mean_time*1000:.2f} ms",
            f"  Std time: {self.std_time*1000:.2f} ms",
            f"  Min time: {self.min_time*1000:.2f} ms",
            f"  Max time: {self.max_time*1000:.2f} ms"
        ]
        
        if self.throughput is not None:
            lines.append(f"  Throughput: {self.throughput:.2f} samples/sec")
        
        if self.memory_allocated is not None:
            lines.append(f"  Memory allocated: {self.memory_allocated:.2f} MB")
        
        if self.memory_reserved is not None:
            lines.append(f"  Memory reserved: {self.memory_reserved:.2f} MB")
        
        return "\n".join(lines)


@contextmanager
def timer():
    """Context manager for timing operations."""
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start
    
    
def benchmark_inference(
    model: torch.nn.Module,
    input_shape: tuple,
    vocab_size: int,
    device: torch.device,
    num_warmup: int = 10,
    num_runs: int = 100,
    batch_sizes: Optional[list] = None
) -> Dict[int, BenchmarkResults]:
    """
    Benchmark inference speed at different batch sizes.
    
    Args:
        model: Transformer model
        input_shape: (batch_size, seq_len)
        vocab_size: Vocabulary size
        device: Device to run on
        num_warmup: Number of warmup runs
        num_runs: Number of benchmark runs
        batch_sizes: List of batch sizes to test
        
    Returns:
        Dict mapping batch_size to BenchmarkResults
    """
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16]
    
    model.eval()
    model.to(device)
    results = {}
    
    for batch_size in batch_sizes:
        _, seq_len = input_shape
        
        # Warmup
        for _ in range(num_warmup):
            inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            with torch.no_grad():
                _ = model(inputs)
        
        # Benchmark
        times = []
        torch.cuda.synchronize() if device.type == 'cuda' else None
        
        for _ in range(num_runs):
            inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(inputs)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end = time.perf_counter()
            
            times.append(end - start)
        
        # Memory stats
        if device.type == 'cuda':
            mem_allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
            mem_reserved = torch.cuda.memory_reserved(device) / 1024**2
        else:
            mem_allocated = None
            mem_reserved = None
        
        # Compute statistics
        times = np.array(times)
        throughput = batch_size / np.mean(times)
        
        results[batch_size] = BenchmarkResults(
            operation=f"Inference (batch_size={batch_size})",
            mean_time=np.mean(times),
            std_time=np.std(times),
            min_time=np.min(times),
            max_time=np.max(times),
            throughput=throughput,
            memory_allocated=mem_allocated,
            memory_reserved=mem_reserved,
            num_runs=num_runs
        )
    
    return results


def benchmark_training(
    model: torch.nn.Module,
    input_shape: tuple,
    vocab_size: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_warmup: int = 10,
    num_runs: int = 50
) -> BenchmarkResults:
    """
    Benchmark training speed (forward + backward + optimizer step).
    
    Args:
        model: Transformer model
        input_shape: (batch_size, seq_len)
        vocab_size: Vocabulary size
        optimizer: Optimizer
        device: Device to run on
        num_warmup: Number of warmup runs
        num_runs: Number of benchmark runs
        
    Returns:
        BenchmarkResults
    """
    model.train()
    model.to(device)
    batch_size, seq_len = input_shape
    
    # Warmup
    for _ in range(num_warmup):
        inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
        loss.backward()
        optimizer.step()
    
    # Benchmark
    times = []
    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    for _ in range(num_runs):
        inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        start = time.perf_counter()
        
        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
        loss.backward()
        optimizer.step()
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end = time.perf_counter()
        
        times.append(end - start)
    
    # Memory stats
    if device.type == 'cuda':
        mem_allocated = torch.cuda.memory_allocated(device) / 1024**2
        mem_reserved = torch.cuda.memory_reserved(device) / 1024**2
    else:
        mem_allocated = None
        mem_reserved = None
    
    times = np.array(times)
    throughput = batch_size / np.mean(times)
    
    return BenchmarkResults(
        operation=f"Training (batch_size={batch_size})",
        mean_time=np.mean(times),
        std_time=np.std(times),
        min_time=np.min(times),
        max_time=np.max(times),
        throughput=throughput,
        memory_allocated=mem_allocated,
        memory_reserved=mem_reserved,
        num_runs=num_runs
    )


def profile_model(
    model: torch.nn.Module,
    input_shape: tuple,
    vocab_size: int,
    device: torch.device
) -> Dict[str, any]:
    """
    Profile model to understand computational breakdown.
    
    Args:
        model: Transformer model
        input_shape: (batch_size, seq_len)
        vocab_size: Vocabulary size
        device: Device to run on
        
    Returns:
        Dictionary with profiling results
    """
    from src.utils import count_parameters
    
    model.eval()
    model.to(device)
    batch_size, seq_len = input_shape
    
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Count parameters
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    # Measure forward pass time
    with torch.no_grad():
        start = time.perf_counter()
        logits, state = model(inputs, return_state=True)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        forward_time = time.perf_counter() - start
    
    # Compute FLOPs (approximate)
    # For transformer: ~12 * n_layers * d_model^2 * seq_len operations
    d_model = model.config.d_model
    n_layers = model.config.n_layers
    approximate_flops = 12 * n_layers * (d_model ** 2) * seq_len * batch_size
    
    results = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'forward_time_ms': forward_time * 1000,
        'approximate_flops': approximate_flops,
        'flops_per_second': approximate_flops / forward_time,
        'memory_mb': torch.cuda.memory_allocated(device) / 1024**2 if device.type == 'cuda' else None,
        'config': {
            'vocab_size': model.config.vocab_size,
            'd_model': model.config.d_model,
            'n_heads': model.config.n_heads,
            'n_layers': model.config.n_layers,
            'd_ff': model.config.d_ff
        }
    }
    
    return results


def compare_configurations(
    configs: list,
    input_shape: tuple,
    device: torch.device
) -> Dict:
    """
    Compare different model configurations.
    
    Args:
        configs: List of TransformerConfig objects
        input_shape: (batch_size, seq_len)
        device: Device to run on
        
    Returns:
        Comparison results
    """
    from src.transformer import Transformer
    
    results = []
    
    for config in configs:
        model = Transformer(config)
        profile = profile_model(model, input_shape, config.vocab_size, device)
        
        results.append({
            'config_name': f"L{config.n_layers}_D{config.d_model}_H{config.n_heads}",
            'parameters': profile['total_parameters'],
            'forward_time_ms': profile['forward_time_ms'],
            'memory_mb': profile['memory_mb'],
            'flops': profile['approximate_flops']
        })
    
    return results


