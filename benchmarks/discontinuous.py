"""Benchmark discontinuous function differentiation.

This script compares the performance of Levi-Civita numbers vs standard autograd
for differentiating discontinuous functions like abs(), step(), and round().
"""

import torch
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from typing import Callable, Dict, List, Tuple
import time

from infnum.torch_sparse import SparseLCTensor
from infnum.torch_autograd import ngrad

def benchmark_function(
    f: Callable[[torch.Tensor], torch.Tensor],
    f_lc: Callable[[SparseLCTensor], SparseLCTensor],
    batch_sizes: List[int],
    device: torch.device = torch.device('cpu'),
    n_trials: int = 10
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """Benchmark a function using both standard autograd and Levi-Civita.
    
    Parameters
    ----------
    f : callable
        Standard PyTorch function
    f_lc : callable
        Levi-Civita version of the function
    batch_sizes : list of int
        Batch sizes to test
    device : torch.device
        Device to run on
    n_trials : int
        Number of trials to average over
        
    Returns
    -------
    times : dict
        Dictionary with 'standard' and 'lc' keys containing lists of times
    memory : dict
        Dictionary with 'standard' and 'lc' keys containing lists of memory usage
    """
    times = {'standard': [], 'lc': []}
    memory = {'standard': [], 'lc': []}
    
    for batch_size in batch_sizes:
        # Standard autograd
        std_times = []
        std_memory = []
        for _ in range(n_trials):
            x = torch.randn(batch_size, device=device, requires_grad=True)
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                start_mem = torch.cuda.memory_allocated()
            else:
                start_mem = 0
            
            start = time.perf_counter()
            y = f(x)
            if y.requires_grad:  # Only compute gradients if y requires grad
                y.backward(torch.ones_like(y))
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            
            if device.type == 'cuda':
                end_mem = torch.cuda.memory_allocated()
            else:
                end_mem = 0
            std_times.append(end - start)
            std_memory.append(end_mem - start_mem)
            
        # Levi-Civita
        lc_times = []
        lc_memory = []
        for _ in range(n_trials):
            x = torch.randn(batch_size, device=device, requires_grad=True)
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                start_mem = torch.cuda.memory_allocated()
            else:
                start_mem = 0
            
            start = time.perf_counter()
            lc_x = SparseLCTensor.from_real(x)
            y = f_lc(lc_x)
            std_part = y.standard_part()
            if std_part.requires_grad:  # Only compute gradients if y requires grad
                std_part.backward(torch.ones(batch_size, device=device))
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            
            if device.type == 'cuda':
                end_mem = torch.cuda.memory_allocated()
            else:
                end_mem = 0
            lc_times.append(end - start)
            lc_memory.append(end_mem - start_mem)
            
        times['standard'].append(np.mean(std_times))
        times['lc'].append(np.mean(lc_times))
        memory['standard'].append(np.mean(std_memory))
        memory['lc'].append(np.mean(lc_memory))
        
    return times, memory

def plot_results(
    batch_sizes: List[int],
    times: Dict[str, List[float]],
    memory: Dict[str, List[float]],
    title: str,
    output_dir: Path
):
    """Plot timing and memory usage results."""
    # Time plot
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(
        x=batch_sizes,
        y=times['standard'],
        name='Standard Autograd',
        mode='lines+markers'
    ))
    fig_time.add_trace(go.Scatter(
        x=batch_sizes,
        y=times['lc'],
        name='Levi-Civita',
        mode='lines+markers'
    ))
    fig_time.update_layout(
        title=f'{title} - Time',
        xaxis_title='Batch Size',
        yaxis_title='Time (s)',
        xaxis_type='log',
        yaxis_type='log'
    )
    fig_time.write_html(output_dir / f'{title.lower()}_time.html')
    fig_time.write_image(output_dir / f'{title.lower()}_time.png')
    
    # Memory plot (only if CUDA is available)
    if any(m != 0 for m in memory['standard'] + memory['lc']):
        fig_mem = go.Figure()
        fig_mem.add_trace(go.Scatter(
            x=batch_sizes,
            y=[m/1024/1024 for m in memory['standard']],
            name='Standard Autograd',
            mode='lines+markers'
        ))
        fig_mem.add_trace(go.Scatter(
            x=batch_sizes,
            y=[m/1024/1024 for m in memory['lc']],
            name='Levi-Civita',
            mode='lines+markers'
        ))
        fig_mem.update_layout(
            title=f'{title} - Memory',
            xaxis_title='Batch Size',
            yaxis_title='Memory (MB)',
            xaxis_type='log',
            yaxis_type='log'
        )
        fig_mem.write_html(output_dir / f'{title.lower()}_memory.html')
        fig_mem.write_image(output_dir / f'{title.lower()}_memory.png')

def main():
    """Run benchmarks for abs, step, and round functions."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running benchmarks on {device}")
    batch_sizes = [2**i for i in range(5, 15)]  # 32 to 16384
    output_dir = Path(__file__).parent.parent
    
    # Absolute value
    def f_abs(x): return torch.abs(x)
    def f_abs_lc(x): return x.abs()
    print("Benchmarking absolute value...")
    times, memory = benchmark_function(f_abs, f_abs_lc, batch_sizes, device)
    plot_results(batch_sizes, times, memory, 'Absolute', output_dir)
    
    # Step function
    def f_step(x): return (x >= 0).to(x.dtype)
    def f_step_lc(x): return x.step()
    print("Benchmarking step function...")
    times, memory = benchmark_function(f_step, f_step_lc, batch_sizes, device)
    plot_results(batch_sizes, times, memory, 'Step', output_dir)
    
    # Round function
    def f_round(x): return torch.round(x)
    def f_round_lc(x): return x.round()
    print("Benchmarking round function...")
    times, memory = benchmark_function(f_round, f_round_lc, batch_sizes, device)
    plot_results(batch_sizes, times, memory, 'Round', output_dir)

if __name__ == '__main__':
    main() 