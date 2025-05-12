"""Tests for sparse Levi-Civita tensor implementation."""

from __future__ import annotations

import torch
import pytest
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np

from infnum.torch_sparse import SparseLCTensor
from infnum.config import encode_exp, decode_exp, EXP_DENOM

# Helper strategies for generating test data
@st.composite
def sparse_lc_tensors(draw, 
                     batch_size: int | None = None,
                     max_terms: int = 5,
                     min_exp: float = -2.0,
                     max_exp: float = 2.0,
                     device: str = "cpu") -> SparseLCTensor:
    """Generate random sparse LC tensors for testing."""
    actual_batch_size: int
    if batch_size is None:
        actual_batch_size = draw(st.integers(min_value=1, max_value=10))
    else:
        actual_batch_size = batch_size
        
    # Generate number of terms for each number in batch
    n_terms = draw(st.lists(
        st.integers(min_value=1, max_value=max_terms),
        min_size=actual_batch_size,
        max_size=actual_batch_size
    ))
    
    total_terms = sum(n_terms)
    
    # Generate random exponents and coefficients
    exps = draw(arrays(
        dtype=np.float64,
        shape=total_terms,
        elements=st.floats(min_value=min_exp, max_value=max_exp)
    ))
    coeffs = draw(arrays(
        dtype=np.float64,
        shape=total_terms,
        elements=st.floats(min_value=-10.0, max_value=10.0,
                          allow_infinity=False, allow_nan=False)
    ))
    
    # Convert to tensors and encode exponents
    values_exps = torch.tensor([encode_exp(e) for e in exps], 
                             dtype=torch.int64,
                             device=device)
    values_coeffs = torch.tensor(coeffs, dtype=torch.float64, device=device)
    
    # Create row pointers
    row_ptr = torch.zeros(actual_batch_size + 1, dtype=torch.int64, device=device)
    row_ptr[1:] = torch.cumsum(torch.tensor(n_terms), dim=0)
    
    # Sort exponents within each segment
    for i in range(actual_batch_size):
        start, end = row_ptr[i], row_ptr[i+1]
        if start < end:  # Skip empty segments
            # Sort segment by exponent
            seg_exps = values_exps[start:end]
            seg_coeffs = values_coeffs[start:end]
            sorted_idx = torch.argsort(seg_exps)
            values_exps[start:end] = seg_exps[sorted_idx]
            values_coeffs[start:end] = seg_coeffs[sorted_idx]
    
    return SparseLCTensor(values_exps, values_coeffs, row_ptr)

# Basic functionality tests
def test_creation_and_validation():
    """Test basic tensor creation and validation."""
    # Valid creation
    values_exps = torch.tensor([0, 16, 0, 16], dtype=torch.int64)  # ε⁰, ε¹
    values_coeffs = torch.tensor([1.0, 1.0, 2.0, 1.0], dtype=torch.float64)
    row_ptr = torch.tensor([0, 2, 4], dtype=torch.int64)
    
    lc = SparseLCTensor(values_exps, values_coeffs, row_ptr)
    assert lc.batch_size == 2
    
    # Invalid: unsorted exponents
    with pytest.raises(ValueError):
        SparseLCTensor(
            torch.tensor([16, 0], dtype=torch.int64),  # ε¹, ε⁰ (wrong order)
            torch.tensor([1.0, 1.0], dtype=torch.float64),
            torch.tensor([0, 2], dtype=torch.int64)
        )
    
    # Invalid: mismatched lengths
    with pytest.raises(ValueError):
        SparseLCTensor(
            torch.tensor([0, 16], dtype=torch.int64),
            torch.tensor([1.0], dtype=torch.float64),  # Too short
            torch.tensor([0, 2], dtype=torch.int64)
        )

@given(sparse_lc_tensors())
def test_addition_properties(lc: SparseLCTensor):
    """Test algebraic properties of addition."""
    # Identity: a + 0 = a
    zero = SparseLCTensor(
        torch.empty(0, dtype=torch.int64, device=lc.device),
        torch.empty(0, dtype=lc.dtype, device=lc.device),
        torch.zeros(lc.batch_size + 1, dtype=torch.int64, device=lc.device)
    )
    sum_with_zero = lc + zero
    
    # Check exponents and coefficients match
    assert torch.equal(sum_with_zero.values_exps, lc.values_exps)
    assert torch.equal(sum_with_zero.values_coeffs, lc.values_coeffs)
    
    # Commutativity: a + b = b + a
    b = sparse_lc_tensors(batch_size=lc.batch_size).example()
    assert torch.equal((lc + b).values_exps, (b + lc).values_exps)
    assert torch.allclose((lc + b).values_coeffs, (b + lc).values_coeffs)

@given(sparse_lc_tensors())
def test_multiplication_properties(lc: SparseLCTensor):
    """Test algebraic properties of multiplication."""
    # Identity: a * 1 = a
    one = SparseLCTensor(
        torch.tensor([0], dtype=torch.int64, device=lc.device).repeat(lc.batch_size),
        torch.ones(lc.batch_size, dtype=lc.dtype, device=lc.device),
        torch.arange(0, lc.batch_size + 1, dtype=torch.int64, device=lc.device)
    )
    prod_with_one = lc * one
    
    # Check same number of terms and matching values
    assert torch.equal(prod_with_one.values_exps, lc.values_exps)
    assert torch.allclose(prod_with_one.values_coeffs, lc.values_coeffs)
    
    # Commutativity: a * b = b * a
    b = sparse_lc_tensors(batch_size=lc.batch_size).example()
    assert torch.equal((lc * b).values_exps, (b * lc).values_exps)
    assert torch.allclose((lc * b).values_coeffs, (b * lc).values_coeffs)

def test_from_real():
    """Test creation from real tensors."""
    x = torch.tensor([1.0, 2.0, 3.0])
    lc = SparseLCTensor.from_real(x, order=1)
    
    # Each number should have two terms: x + ε
    assert lc.batch_size == 3
    assert len(lc.values_exps) == 6  # 2 terms per number
    
    # Check standard parts
    assert torch.equal(lc.standard_part(), x)
    
    # Check ε terms are all 1
    for i in range(3):
        start, end = lc.row_ptr[i], lc.row_ptr[i+1]
        eps_idx = (lc.values_exps[start:end] == encode_exp(1)).nonzero()[0]
        assert lc.values_coeffs[start + eps_idx] == 1.0

def test_string_representation():
    """Test string formatting of LC numbers."""
    lc = SparseLCTensor(
        torch.tensor([0, 16, 32], dtype=torch.int64),  # ε⁰, ε¹, ε²
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
        torch.tensor([0, 3], dtype=torch.int64)
    )
    
    expected = "1 + 2ε^1 + 3ε^2"
    assert str(lc) == expected 