"""Sparse representation of LeviCivitaTensor using CSR format.

This module implements an efficient sparse representation for batches of Levi-Civita numbers
using a CSR-like format with two main tensors:
- values_exps: IntTensor storing scaled integer exponents
- values_coeffs: Tensor storing scalar coefficients
- row_ptr: IntTensor storing CSR segment boundaries

The representation is kept sorted by exponents for efficient operations.
"""

from __future__ import annotations

import torch
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass

from .config import encode_exp, decode_exp, EXP_DENOM

__all__ = ["SparseLCTensor"]

@dataclass
class SparseLCTensor:
    """Sparse representation of a batch of Levi-Civita numbers.
    
    Each number is represented as a sum of terms c_i * ε^(e_i) where:
    - c_i are scalar coefficients
    - e_i are rational exponents (stored as scaled integers)
    - terms are sorted by increasing exponent
    
    The batch is stored efficiently using a CSR-like format:
    - values_exps: (total_nnz,) IntTensor of scaled exponents
    - values_coeffs: (total_nnz,) Tensor of coefficients
    - row_ptr: (batch_size + 1,) IntTensor of segment boundaries
    """
    
    values_exps: torch.Tensor  # (total_nnz,) IntTensor
    values_coeffs: torch.Tensor  # (total_nnz,) Tensor
    row_ptr: torch.Tensor  # (batch_size + 1,) IntTensor
    
    # Class attribute for autograd function
    _create_with_grad: Optional[Callable] = None
    
    def __post_init__(self):
        """Validate tensor shapes and types."""
        if not (self.values_exps.dtype == torch.int64 and
                self.row_ptr.dtype == torch.int64):
            raise TypeError("Exponents and row pointers must be int64")
        
        if not (self.values_exps.ndim == 1 and 
                self.values_coeffs.ndim == 1 and
                self.row_ptr.ndim == 1):
            raise ValueError("All tensors must be 1-dimensional")
            
        if not (len(self.values_exps) == len(self.values_coeffs)):
            raise ValueError("Number of exponents must match coefficients")
            
        # Verify row_ptr is monotonic
        if not torch.all(self.row_ptr[1:] >= self.row_ptr[:-1]):
            raise ValueError("row_ptr must be monotonically increasing")
            
        # Verify exponents are sorted within each segment
        for i in range(len(self.row_ptr) - 1):
            start, end = self.row_ptr[i], self.row_ptr[i+1]
            if not torch.all(self.values_exps[start:end][1:] > 
                           self.values_exps[start:end][:-1]):
                raise ValueError(f"Exponents must be strictly increasing in segment {i}")
    
    @property
    def batch_size(self) -> int:
        """Number of Levi-Civita numbers in the batch."""
        return len(self.row_ptr) - 1
    
    @property
    def device(self) -> torch.device:
        """Device of the underlying tensors."""
        return self.values_coeffs.device
    
    @property
    def dtype(self) -> torch.dtype:
        """Data type of the coefficients."""
        return self.values_coeffs.dtype
    
    def to(self, device: Optional[torch.device] = None, 
           dtype: Optional[torch.dtype] = None) -> SparseLCTensor:
        """Move tensors to specified device and/or cast coefficients to dtype."""
        values_exps = self.values_exps.to(device=device)
        values_coeffs = self.values_coeffs.to(device=device, dtype=dtype)
        row_ptr = self.row_ptr.to(device=device)
        return SparseLCTensor(values_exps, values_coeffs, row_ptr)
    
    def clone(self) -> SparseLCTensor:
        """Create a deep copy."""
        return SparseLCTensor(
            self.values_exps.clone(),
            self.values_coeffs.clone(),
            self.row_ptr.clone()
        )
    
    @classmethod
    def from_real(cls, x: torch.Tensor, order: int = 1) -> SparseLCTensor:
        """Create x + ε^order from a real tensor x."""
        batch_size = x.numel()
        device = x.device
        
        # Flatten input
        x_flat = x.reshape(-1)
        
        # Create two terms per number: constant + ε^order
        values_exps = torch.tensor([0, encode_exp(order)], 
                                 dtype=torch.int64, 
                                 device=device).repeat(batch_size)
        values_coeffs = torch.stack([x_flat, torch.ones_like(x_flat)]).t().reshape(-1)
        
        # Each number has exactly 2 terms
        row_ptr = torch.arange(0, 2*batch_size + 1, 2, 
                             dtype=torch.int64, 
                             device=device)
        
        return cls(values_exps, values_coeffs, row_ptr)
    
    def standard_part(self) -> torch.Tensor:
        """Extract the constant term (ε⁰ coefficient) from each number."""
        result = torch.zeros(self.batch_size, 
                           dtype=self.dtype, 
                           device=self.device)
        
        for i in range(self.batch_size):
            start, end = self.row_ptr[i], self.row_ptr[i+1]
            # Find index of ε⁰ term if it exists
            zero_idx = (self.values_exps[start:end] == 0).nonzero()
            if len(zero_idx) > 0:
                result[i] = self.values_coeffs[start + zero_idx[0]]
                
        return result
    
    def __add__(self, other: SparseLCTensor) -> SparseLCTensor:
        """Add two sparse LC tensors."""
        if not isinstance(other, SparseLCTensor):
            return NotImplemented
            
        if self.batch_size != other.batch_size:
            raise ValueError("Batch sizes must match")
            
        device = self.device
        dtype = self.dtype
        batch_size = self.batch_size
        
        # Pre-allocate maximum possible size
        max_terms = max(len(self.values_exps), len(other.values_exps))
        values_exps = torch.empty(max_terms, dtype=torch.int64, device=device)
        values_coeffs = torch.empty(max_terms, dtype=dtype, device=device)
        row_ptr = torch.empty(batch_size + 1, dtype=torch.int64, device=device)
        row_ptr[0] = 0
        
        curr_pos = 0
        
        # Add corresponding segments
        for i in range(batch_size):
            s1_start, s1_end = self.row_ptr[i], self.row_ptr[i+1]
            s2_start, s2_end = other.row_ptr[i], other.row_ptr[i+1]
            
            # Get segments to merge
            s1_exps = self.values_exps[s1_start:s1_end]
            s1_coeffs = self.values_coeffs[s1_start:s1_end]
            s2_exps = other.values_exps[s2_start:s2_end]
            s2_coeffs = other.values_coeffs[s2_start:s2_end]
            
            # Merge sorted segments
            p1 = p2 = 0
            while p1 < len(s1_exps) and p2 < len(s2_exps):
                if s1_exps[p1] < s2_exps[p2]:
                    values_exps[curr_pos] = s1_exps[p1]
                    values_coeffs[curr_pos] = s1_coeffs[p1]
                    p1 += 1
                elif s1_exps[p1] > s2_exps[p2]:
                    values_exps[curr_pos] = s2_exps[p2]
                    values_coeffs[curr_pos] = s2_coeffs[p2]
                    p2 += 1
                else:  # Equal exponents
                    values_exps[curr_pos] = s1_exps[p1]
                    values_coeffs[curr_pos] = s1_coeffs[p1] + s2_coeffs[p2]
                    p1 += 1
                    p2 += 1
                curr_pos += 1
            
            # Copy remaining terms
            while p1 < len(s1_exps):
                values_exps[curr_pos] = s1_exps[p1]
                values_coeffs[curr_pos] = s1_coeffs[p1]
                p1 += 1
                curr_pos += 1
            while p2 < len(s2_exps):
                values_exps[curr_pos] = s2_exps[p2]
                values_coeffs[curr_pos] = s2_coeffs[p2]
                p2 += 1
                curr_pos += 1
                
            row_ptr[i+1] = curr_pos
            
        # Trim to actual size
        values_exps = values_exps[:curr_pos]
        values_coeffs = values_coeffs[:curr_pos]
        
        return SparseLCTensor(values_exps, values_coeffs, row_ptr)
    
    def __mul__(self, other: SparseLCTensor) -> SparseLCTensor:
        """Multiply two sparse LC tensors.
        
        For each pair of segments, we:
        1. Compute outer sum of exponents and outer product of coefficients
        2. Flatten the results
        3. Sort by exponent and combine duplicate terms
        """
        if not isinstance(other, SparseLCTensor):
            return NotImplemented
            
        if self.batch_size != other.batch_size:
            raise ValueError("Batch sizes must match")
            
        device = self.device
        dtype = self.dtype
        batch_size = self.batch_size
        
        # Pre-allocate maximum possible size (outer product of terms)
        terms_per_seg: List[int] = []
        for i in range(batch_size):
            n1 = int(self.row_ptr[i+1] - self.row_ptr[i])
            n2 = int(other.row_ptr[i+1] - other.row_ptr[i])
            terms_per_seg.append(n1 * n2)
        max_terms_per_seg = max(terms_per_seg) if terms_per_seg else 0
        max_terms = max_terms_per_seg * batch_size
        
        values_exps = torch.empty(max_terms, dtype=torch.int64, device=device)
        values_coeffs = torch.empty(max_terms, dtype=dtype, device=device)
        row_ptr = torch.empty(batch_size + 1, dtype=torch.int64, device=device)
        row_ptr[0] = 0
        
        curr_pos = 0
        
        # Multiply corresponding segments
        for i in range(batch_size):
            s1_start, s1_end = self.row_ptr[i], self.row_ptr[i+1]
            s2_start, s2_end = other.row_ptr[i], other.row_ptr[i+1]
            
            # Get segments to multiply
            s1_exps = self.values_exps[s1_start:s1_end]
            s1_coeffs = self.values_coeffs[s1_start:s1_end]
            s2_exps = other.values_exps[s2_start:s2_end]
            s2_coeffs = other.values_coeffs[s2_start:s2_end]
            
            # Compute outer sums of exponents
            e1 = s1_exps.unsqueeze(1)  # (n1, 1)
            e2 = s2_exps.unsqueeze(0)  # (1, n2)
            out_exps = e1 + e2  # (n1, n2)
            
            # Compute outer products of coefficients
            c1 = s1_coeffs.unsqueeze(1)  # (n1, 1)
            c2 = s2_coeffs.unsqueeze(0)  # (1, n2)
            out_coeffs = c1 * c2  # (n1, n2)
            
            # Flatten and sort by exponent
            flat_exps = out_exps.reshape(-1)
            flat_coeffs = out_coeffs.reshape(-1)
            
            # Sort by exponent
            sorted_idx = torch.argsort(flat_exps)
            flat_exps = flat_exps[sorted_idx]
            flat_coeffs = flat_coeffs[sorted_idx]
            
            # Combine terms with same exponent using unique_consecutive
            unique_exps, inverse_indices = torch.unique_consecutive(
                flat_exps, return_inverse=True)
            
            # Sum coefficients for duplicate exponents
            unique_coeffs = torch.zeros(
                len(unique_exps), dtype=dtype, device=device)
            unique_coeffs.scatter_add_(0, inverse_indices, flat_coeffs)
            
            # Copy to output arrays
            n_terms = len(unique_exps)
            values_exps[curr_pos:curr_pos + n_terms] = unique_exps
            values_coeffs[curr_pos:curr_pos + n_terms] = unique_coeffs
            curr_pos += n_terms
            row_ptr[i+1] = curr_pos
            
        # Trim to actual size
        values_exps = values_exps[:curr_pos]
        values_coeffs = values_coeffs[:curr_pos]
        
        return SparseLCTensor(values_exps, values_coeffs, row_ptr)
    
    def __str__(self) -> str:
        """Human-readable representation showing terms for each number."""
        lines = []
        for i in range(self.batch_size):
            start, end = self.row_ptr[i], self.row_ptr[i+1]
            terms = []
            for j in range(start, end):
                exp = decode_exp(int(self.values_exps[j].item()))
                coeff = self.values_coeffs[j].item()
                if exp == 0:
                    terms.append(f"{coeff:g}")
                else:
                    terms.append(f"{coeff:g}ε^{exp:g}")
            lines.append(" + ".join(terms) if terms else "0")
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"SparseLCTensor(batch_size={self.batch_size}, device={self.device}, dtype={self.dtype})" 