"""Example demonstrating differentiation of discontinuous functions.

This script shows how the Levi-Civita field approach can differentiate
discontinuous functions like step, abs, and round, producing meaningful
derivatives that match the distributional/generalized derivatives.
"""

from __future__ import annotations

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from infnum.torch_sparse import SparseLCTensor
from infnum.torch_autograd import ngrad

def plot_function_and_derivative(f, x_range: tuple[float, float],
                               n_points: int = 201,
                               title: str = "",
                               name: str = "") -> tuple[go.Figure, plt.Figure]:
    """Plot a function and its non-standard derivative using both plotly and matplotlib.
    
    Parameters
    ----------
    f : callable
        Function to plot
    x_range : tuple[float, float]
        Range of x values to plot
    n_points : int
        Number of points to evaluate
    title : str
        Plot title
    name : str
        Base name for saving files
    
    Returns
    -------
    tuple[go.Figure, plt.Figure]
        Plotly and Matplotlib figures
    """
    x = torch.linspace(x_range[0], x_range[1], n_points,
                      dtype=torch.float64, requires_grad=True)
    
    # Compute function values
    if isinstance(f(x[0]), SparseLCTensor):
        y = torch.tensor([f(xi).standard_part().item() for xi in x])
    else:
        y = f(x)
    
    # Compute non-standard derivative
    dx = ngrad(f, x)
    
    # Convert to numpy for plotting (detach first)
    x_np = x.detach().numpy()
    y_np = y.detach().numpy()
    dx_np = dx.detach().numpy()
    
    # Create plotly figure
    fig_plotly = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add function trace
    fig_plotly.add_trace(
        go.Scatter(x=x_np, y=y_np,
                  name="f(x)", 
                  line=dict(color="blue", width=2)),
        secondary_y=False
    )
    
    # Add derivative trace
    fig_plotly.add_trace(
        go.Scatter(x=x_np, y=dx_np,
                  name="df/dx", 
                  line=dict(color="red", width=2)),
        secondary_y=True
    )
    
    # Update plotly layout
    fig_plotly.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        xaxis=dict(
            title="x",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='gray'
        ),
        plot_bgcolor='white',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        width=800,
        height=500
    )
    
    fig_plotly.update_yaxes(
        title_text="f(x)", 
        secondary_y=False,
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    fig_plotly.update_yaxes(
        title_text="df/dx", 
        secondary_y=True,
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    
    # Create matplotlib figure
    fig_mpl, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[1, 1])
    plt.suptitle(title, fontsize=16)
    
    # Plot function
    ax1.plot(x_np, y_np, 'b-', linewidth=2, label='f(x)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylabel('f(x)')
    ax1.legend()
    
    # Plot derivative
    ax2.plot(x_np, dx_np, 'r-', linewidth=2, label='df/dx')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xlabel('x')
    ax2.set_ylabel('df/dx')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save figures
    if name:
        fig_plotly.write_html(f"{name}.html")
        fig_mpl.savefig(f"{name}.png", dpi=300, bbox_inches='tight')
        plt.close(fig_mpl)  # Close matplotlib figure to free memory
    
    return fig_plotly, fig_mpl

def main():
    """Create plots for step, abs, and round functions."""
    # Heaviside step function
    def step(x):
        """Heaviside step function."""
        if isinstance(x, SparseLCTensor):
            return (x.standard_part() >= 0).to(dtype=torch.float64)
        return (x >= 0).to(dtype=torch.float64)
    
    fig_step = plot_function_and_derivative(
        step, (-2, 2),
        title="Heaviside Step Function and its Dirac Delta Derivative",
        name="step_function"
    )
    
    # Absolute value function
    def abs_fn(x):
        """Absolute value function."""
        if isinstance(x, SparseLCTensor):
            return (x * x).standard_part().sqrt()
        return torch.abs(x)
    
    fig_abs = plot_function_and_derivative(
        abs_fn, (-2, 2),
        title="Absolute Value Function and its Sign Function Derivative",
        name="abs_function"
    )
    
    # Round function
    def round_fn(x):
        """Round to nearest integer."""
        if isinstance(x, SparseLCTensor):
            std_part = x.standard_part()
            return torch.round(std_part)
        return torch.round(x)
    
    fig_round = plot_function_and_derivative(
        round_fn, (-2, 2),
        title="Round Function and its Dirac Comb Derivative",
        name="round_function"
    )

if __name__ == "__main__":
    main() 