"""Tests for autograd functionality with Levi-Civita numbers."""

from __future__ import annotations

import torch
import pytest
from hypothesis import given, strategies as st
import numpy as np

from infnum.torch_sparse import SparseLCTensor
from infnum.torch_autograd import ngrad

def test_step_function():
    """Test differentiation of the Heaviside step function.
    
    The non-standard derivative should be a Dirac delta at x = 0.
    """
    def step(x):
        """Heaviside step function.

        For Levi-Civita inputs we rely on the built-in `.step()` method which
        preserves ε-terms so that the derivative can be extracted via the ε¹
        coefficient instead of finite differences.
        """
        if isinstance(x, SparseLCTensor):
            return x.step()
        return (x >= 0).to(dtype=torch.float64)
    
    # Test points around x = 0
    x = torch.linspace(-1, 1, 201, dtype=torch.float64, requires_grad=True)
    dx = ngrad(step, x)
    
    # The derivative should be zero everywhere except at x = 0
    assert torch.allclose(dx[x < -0.1], torch.zeros_like(dx[x < -0.1]))
    assert torch.allclose(dx[x > 0.1], torch.zeros_like(dx[x > 0.1]))

    # At x = 0 we expect a non-zero spike capturing the distributional delta
    zero_idx = torch.argmin(torch.abs(x))
    assert dx[zero_idx] != 0

def test_abs_function():
    """Test differentiation of the absolute value function.
    
    The non-standard derivative should be sign(x), which is -1 for x < 0
    and +1 for x > 0, with a "jump" at x = 0.
    """
    def abs_fn(x):
        """Absolute value function."""
        if isinstance(x, SparseLCTensor):
            return abs(x)  # uses SparseLCTensor.__abs__
        return torch.abs(x)
    
    # Test points around x = 0
    x = torch.linspace(-1, 1, 201, dtype=torch.float64, requires_grad=True)
    dx = ngrad(abs_fn, x)
    
    # The derivative should be -1 for x < 0 and +1 for x > 0
    assert torch.allclose(dx[x < -0.1], -torch.ones_like(dx[x < -0.1]))
    assert torch.allclose(dx[x > 0.1], torch.ones_like(dx[x > 0.1]))
    
    # At x = 0, we expect a smooth transition between -1 and 1
    transition = dx[(x >= -0.1) & (x <= 0.1)]
    # transition should monotonically increase from -1 towards +1
    assert torch.all(torch.diff(transition) >= 0)

def test_round_function():
    """Test differentiation of the round function.
    
    The non-standard derivative should be a sum of Dirac deltas at
    half-integers (x = ..., -1.5, -0.5, 0.5, 1.5, ...).
    """
    def round_fn(x):
        """Round to nearest integer."""
        if isinstance(x, SparseLCTensor):
            return x.round()
        return torch.round(x)
    
    # Test points around x = 0.5
    x = torch.linspace(0, 1, 201, dtype=torch.float64, requires_grad=True)
    dx = ngrad(round_fn, x)
    
    # The derivative should be zero except near x = 0.5
    assert torch.allclose(dx[x < 0.4], torch.zeros_like(dx[x < 0.4]))
    assert torch.allclose(dx[x > 0.6], torch.zeros_like(dx[x > 0.6]))

    # Spike at the half-integer
    spike_idx = torch.argmin(torch.abs(x - 0.5))
    assert dx[spike_idx] != 0

def test_composition():
    """Test differentiation of composed discontinuous functions."""
    def f(x):
        """Composition of step and abs."""
        if isinstance(x, SparseLCTensor):
            return abs(x) * x.step()
        return torch.abs(x) * (x >= 0).to(dtype=x.dtype)
    
    # Test points around x = 0
    x = torch.linspace(-1, 1, 201, dtype=torch.float64, requires_grad=True)
    dx = ngrad(f, x)
    
    # The derivative should be 0 for x < 0
    assert torch.allclose(dx[x < -0.1], torch.zeros_like(dx[x < -0.1]))
    
    # For x > 0, should match derivative of abs
    assert torch.allclose(dx[x > 0.1], torch.ones_like(dx[x > 0.1]))
    
    # At x = 0, we expect a delta-like spike
    zero_idx = torch.argmin(torch.abs(x))
    assert dx[zero_idx] != 0

@given(st.floats(min_value=-10.0, max_value=10.0,
                 allow_infinity=False, allow_nan=False))
def test_smooth_function_agreement(x0):
    """Test that non-standard derivative agrees with regular autograd
    for smooth functions.
    """
    x = torch.tensor([x0], dtype=torch.float64, requires_grad=True)
    
    def smooth_fn(x):
        """A smooth function: x³ + 2x² - x + 1"""
        if isinstance(x, SparseLCTensor):
            x2 = x * x  # x²
            x3 = x2 * x  # x³
            x2_scaled = x2 * SparseLCTensor.from_real(
                torch.tensor([2.0], dtype=torch.float64, device=x.device))
            one = SparseLCTensor.from_real(
                torch.tensor([1.0], dtype=torch.float64, device=x.device))
            # Build up the polynomial term by term
            result = x3 + x2_scaled  # x³ + 2x²
            result = result + (x * SparseLCTensor.from_real(
                torch.tensor([-1.0], dtype=torch.float64, device=x.device)))  # -x
            result = result + one  # +1
            return result
        return x**3 + 2*x**2 - x + 1
    
    # Compute regular gradient
    y = smooth_fn(x)
    y.backward()
    regular_grad = x.grad
    assert regular_grad is not None  # for type checker
    x.grad = None
    
    # Compute non-standard gradient
    nstd_grad = ngrad(smooth_fn, x)
    
    # They should agree
    assert torch.allclose(regular_grad, nstd_grad) 