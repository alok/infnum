"""Custom autograd functions for Levi-Civita numbers.

This module implements the autograd machinery needed to differentiate through
Levi-Civita numbers. The key insight is that we can extract the non-standard
derivative by looking at the coefficient of ε¹ in the result.

For example, to compute df/dx at x₀:
1. Evaluate f(x₀ + ε)
2. The coefficient of ε¹ gives us df/dx
"""

from __future__ import annotations

import torch
from torch import Tensor
import torch.autograd.function as F
from typing import Tuple, List, Optional

from .torch_sparse import SparseLCTensor
from .config import encode_exp, decode_exp, EXP_DENOM

__all__ = ["LeviCivitaFunction", "ngrad"]

class LeviCivitaFunction(F.Function):
    """Custom autograd function for Levi-Civita numbers.
    
    This function handles the forward and backward passes through operations
    on Levi-Civita numbers, ensuring that gradients are computed correctly
    with respect to the coefficients.
    
    The key is that we track both the coefficients and their exponents through
    the computation graph, but only the coefficients participate in gradient
    computation.
    """
    
    @staticmethod
    def forward(ctx, values_exps: Tensor, values_coeffs: Tensor, 
                row_ptr: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass - just store the inputs for backward."""
        ctx.save_for_backward(values_exps, values_coeffs, row_ptr)
        return values_exps, values_coeffs, row_ptr
    
    @staticmethod
    def backward(ctx, grad_exps: Tensor, grad_coeffs: Tensor,
                 grad_row_ptr: Tensor) -> Tuple[Optional[Tensor], ...]:
        """Backward pass - propagate gradients only through coefficients."""
        values_exps, values_coeffs, row_ptr = ctx.saved_tensors
        
        # We don't compute gradients for exponents or row pointers
        # since they are discrete indices
        return None, grad_coeffs, None

def ngrad(f, x: Tensor) -> Tensor:
    """Compute the non-standard derivative of f at x.
    
    This is done by:
    1. Converting x to x + ε
    2. Evaluating f(x + ε)
    3. Extracting the coefficient of ε¹ from the result
    
    Parameters
    ----------
    f : callable
        Function to differentiate. Should accept a Tensor and return a Tensor
        or SparseLCTensor.
    x : Tensor
        Point at which to evaluate the derivative.
        
    Returns
    -------
    Tensor
        The non-standard derivative df/dx.
    """
    # Convert x to x + ε
    lc_x = SparseLCTensor.from_real(x, order=1)
    
    # Apply the function
    y = f(lc_x)
    
    # Handle both SparseLCTensor and regular Tensor returns
    if isinstance(y, SparseLCTensor):
        batch_size = y.batch_size
        result = torch.zeros_like(x)
        
        for i in range(batch_size):
            start, end = y.row_ptr[i], y.row_ptr[i+1]
            # Find index of ε¹ term if it exists
            eps_idx = (y.values_exps[start:end] == encode_exp(1)).nonzero()
            if len(eps_idx) > 0:
                result[i] = y.values_coeffs[start + eps_idx[0]]
    else:
        # For regular tensors, just return zeros since they don't have ε terms
        result = torch.zeros_like(x)
            
    return result

def register_lc_grad(cls: type[SparseLCTensor]):
    """Register the LeviCivitaFunction with SparseLCTensor.
    
    This allows us to use the custom autograd function whenever a SparseLCTensor
    is created or modified.
    """
    cls._create_with_grad = staticmethod(LeviCivitaFunction.apply)
    
    # Monkey patch the constructor to use the grad function
    old_init = cls.__init__
    def new_init(self, values_exps: Tensor, values_coeffs: Tensor,
                 row_ptr: Tensor):
        if values_coeffs.requires_grad:
            values_exps, values_coeffs, row_ptr = cls._create_with_grad(
                values_exps, values_coeffs, row_ptr)
        old_init(self, values_exps, values_coeffs, row_ptr)
    cls.__init__ = new_init
    
    return cls

# Register the autograd function with SparseLCTensor
register_lc_grad(SparseLCTensor) 