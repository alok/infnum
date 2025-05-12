"""Benchmarks for Levi-Civita field operations.

This module compares the performance of Levi-Civita field operations with standard
PyTorch autograd for various functions, including discontinuous ones.
"""

import torch
import torch.nn.functional as F
from typing import Callable, Dict, List, Tuple, Union
import time
from dataclasses import dataclass
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

from infnum.torch_sparse import SparseLCTensor
from infnum.torch_autograd import ngrad

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    batch_sizes: List[int]
    lc_times: List[float]  # Time for Levi-Civita method
    autograd_times: List[float]  # Time for standard autograd
    lc_memory: List[float]  # Peak memory for Levi-Civita method (MB)
    autograd_memory: List[float]  # Peak memory for standard autograd (MB)

def time_fn(fn: Callable, *args, **kwargs) -> Tuple[float, float]:
    """Time a function call and measure peak memory usage."""
    torch.cuda.empty_cache()
    start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    start_time = time.perf_counter()
    fn(*args, **kwargs)
    end_time = time.perf_counter()
    
    peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    peak_mem_mb = (peak_mem - start_mem) / 1024 / 1024  # Convert to MB
    
    return end_time - start_time, peak_mem_mb

def benchmark_function(
    fn: Callable[[Union[torch.Tensor, SparseLCTensor]], Union[torch.Tensor, SparseLCTensor]],
    batch_sizes: List[int],
    device: torch.device,
    n_runs: int = 5
) -> BenchmarkResult:
    """Benchmark a function using both Levi-Civita and standard autograd.
    
    Parameters
    ----------
    fn : callable
        Function to benchmark. Should accept a tensor or SparseLCTensor and return
        a tensor or SparseLCTensor.
    batch_sizes : list of int
        Batch sizes to test.
    device : torch.device
        Device to run benchmarks on.
    n_runs : int, optional
        Number of runs to average over.
        
    Returns
    -------
    BenchmarkResult
        Benchmark results including timing and memory usage.
    """
    lc_times = []
    autograd_times = []
    lc_memory = []
    autograd_memory = []
    
    for batch_size in batch_sizes:
        # Levi-Civita method
        x = torch.randn(batch_size, requires_grad=True, device=device)
        lc_time = 0
        lc_mem = 0
        for _ in range(n_runs):
            t, m = time_fn(lambda: ngrad(fn, x))
            lc_time += t
            lc_mem += m
        lc_times.append(lc_time / n_runs)
        lc_memory.append(lc_mem / n_runs)
        
        # Standard autograd
        x = torch.randn(batch_size, requires_grad=True, device=device)
        autograd_time = 0
        autograd_mem = 0
        for _ in range(n_runs):
            def run_autograd():
                y = fn(x)
                if isinstance(y, SparseLCTensor):
                    # Extract constant terms for gradient computation
                    const_terms = torch.zeros(y.batch_size, device=y.values_coeffs.device)
                    for i in range(y.batch_size):
                        start, end = y.row_ptr[i], y.row_ptr[i+1]
                        eps_idx = (y.values_exps[start:end] == 0).nonzero()
                        if len(eps_idx) > 0:
                            const_terms[i] = y.values_coeffs[start + eps_idx[0]]
                    grad, = torch.autograd.grad(const_terms.sum(), [x])
                else:
                    grad, = torch.autograd.grad(y.sum(), [x])
                return grad
            t, m = time_fn(run_autograd)
            autograd_time += t
            autograd_mem += m
        autograd_times.append(autograd_time / n_runs)
        autograd_memory.append(autograd_mem / n_runs)
    
    return BenchmarkResult(
        name=fn.__name__,
        batch_sizes=batch_sizes,
        lc_times=lc_times,
        autograd_times=autograd_times,
        lc_memory=lc_memory,
        autograd_memory=autograd_memory
    )

def plot_results(results: List[BenchmarkResult], output_dir: Path):
    """Plot benchmark results using plotly."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Time comparison
    fig = go.Figure()
    for result in results:
        fig.add_trace(go.Scatter(
            x=result.batch_sizes,
            y=result.lc_times,
            name=f"{result.name} (LC)",
            mode='lines+markers'
        ))
        fig.add_trace(go.Scatter(
            x=result.batch_sizes,
            y=result.autograd_times,
            name=f"{result.name} (Autograd)",
            mode='lines+markers'
        ))
    
    fig.update_layout(
        title="Time Comparison: Levi-Civita vs Standard Autograd",
        xaxis_title="Batch Size",
        yaxis_title="Time (seconds)",
        xaxis_type="log",
        yaxis_type="log"
    )
    fig.write_html(output_dir / "time_comparison.html")
    
    # Memory comparison
    fig = go.Figure()
    for result in results:
        fig.add_trace(go.Scatter(
            x=result.batch_sizes,
            y=result.lc_memory,
            name=f"{result.name} (LC)",
            mode='lines+markers'
        ))
        fig.add_trace(go.Scatter(
            x=result.batch_sizes,
            y=result.autograd_memory,
            name=f"{result.name} (Autograd)",
            mode='lines+markers'
        ))
    
    fig.update_layout(
        title="Memory Usage: Levi-Civita vs Standard Autograd",
        xaxis_title="Batch Size",
        yaxis_title="Memory (MB)",
        xaxis_type="log",
        yaxis_type="log"
    )
    fig.write_html(output_dir / "memory_comparison.html")

def abs_fn(x: Union[torch.Tensor, SparseLCTensor]) -> Union[torch.Tensor, SparseLCTensor]:
    """Absolute value function."""
    if isinstance(x, SparseLCTensor):
        return x.abs()
    return torch.abs(x)

def step_fn(x: Union[torch.Tensor, SparseLCTensor]) -> Union[torch.Tensor, SparseLCTensor]:
    """Step function."""
    if isinstance(x, SparseLCTensor):
        # For Levi-Civita numbers, we need to handle both the constant and ε terms
        # The step function is discontinuous, but we can still compute its derivative
        # at non-zero points
        const_terms = torch.zeros(x.batch_size, device=x.values_coeffs.device)
        for i in range(x.batch_size):
            start, end = x.row_ptr[i], x.row_ptr[i+1]
            eps_idx = (x.values_exps[start:end] == 0).nonzero()
            if len(eps_idx) > 0:
                const_terms[i] = x.values_coeffs[start + eps_idx[0]]
        
        # The step function is 0 for x < 0 and 1 for x > 0
        # The derivative is infinite at x = 0, but we can approximate it
        # with a large value
        result = (const_terms > 0).float()
        return SparseLCTensor.from_real(result)
    return (x > 0).float()

def round_fn(x: Union[torch.Tensor, SparseLCTensor]) -> Union[torch.Tensor, SparseLCTensor]:
    """Round function."""
    if isinstance(x, SparseLCTensor):
        # For Levi-Civita numbers, we need to handle both the constant and ε terms
        # The round function is discontinuous at half-integers
        const_terms = torch.zeros(x.batch_size, device=x.values_coeffs.device)
        for i in range(x.batch_size):
            start, end = x.row_ptr[i], x.row_ptr[i+1]
            eps_idx = (x.values_exps[start:end] == 0).nonzero()
            if len(eps_idx) > 0:
                const_terms[i] = x.values_coeffs[start + eps_idx[0]]
        
        result = torch.round(const_terms)
        return SparseLCTensor.from_real(result)
    return torch.round(x)

if __name__ == "__main__":
    # Set up benchmark parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_sizes = [2**i for i in range(8, 20)]  # From 256 to 524288
    output_dir = Path("benchmark_results")
    
    # Functions to benchmark
    functions = [abs_fn, step_fn, round_fn]
    
    # Run benchmarks
    results = []
    for fn in functions:
        print(f"Benchmarking {fn.__name__}...")
        result = benchmark_function(fn, batch_sizes, device)
        results.append(result)
    
    # Plot results
    plot_results(results, output_dir) 