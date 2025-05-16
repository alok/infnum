import pytest
import torch

from infnum.torch_sparse import SparseLCTensor
from infnum.torch_autograd import ngrad
from math import sqrt

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def assert_close(a: torch.Tensor, b: torch.Tensor, tol: float = 1e-6):
    """Utility to compare two tensors irrespective of dtype."""
    assert torch.allclose(a.to(torch.float64), b.to(torch.float64), atol=tol)


# -----------------------------------------------------------------------------
# Core algebraic identities
# -----------------------------------------------------------------------------


def test_reciprocal_roundtrip():
    x = torch.tensor([2.0, -3.5, 0.7])
    lc_x = SparseLCTensor.from_real(x)
    lc_inv = lc_x.reciprocal()
    prod = lc_x * lc_inv
    assert_close(prod.standard_part(), torch.ones_like(x))


def test_division_roundtrip():
    x = torch.tensor([2.0, -1.25, 0.8])
    y = torch.tensor([1.1, -0.5, 3.2])
    lc_x = SparseLCTensor.from_real(x)
    lc_y = SparseLCTensor.from_real(y)
    lc_div = lc_x / lc_y
    reconstructed = (lc_div * lc_y).standard_part()
    assert_close(reconstructed, x)


def test_pow_integer():
    x = torch.tensor([1.2, -0.7, 2.0])
    n = 3
    lc_x = SparseLCTensor.from_real(x)
    lc_pow = lc_x ** n
    assert_close(lc_pow.standard_part(), x ** n)


def test_pow_negative_integer():
    x = torch.tensor([2.0, -4.0])
    n = -2
    lc_x = SparseLCTensor.from_real(x)
    lc_pow = lc_x ** n
    expected = x ** n
    assert_close(lc_pow.standard_part(), expected)


def test_pow_fractional_pure():
    coeff = 3.0
    values_exps = torch.tensor([0], dtype=torch.int64)
    values_coeffs = torch.tensor([coeff])
    row_ptr = torch.tensor([0, 1], dtype=torch.int64)
    lc_pure = SparseLCTensor(values_exps, values_coeffs, row_ptr)
    lc_half = lc_pure ** 0.5
    assert_close(lc_half.standard_part(), torch.tensor([sqrt(coeff)]))

# -----------------------------------------------------------------------------
# Non-standard derivatives
# -----------------------------------------------------------------------------


def test_ngrad_abs():
    for val, expected in [(-2.0, -1.0), (3.5, 1.0)]:
        x = torch.tensor(val, requires_grad=True)
        grad = ngrad(lambda z: z.abs(), x)
        assert_close(grad, torch.tensor(expected))


def test_ngrad_step():
    # x != 0  -> derivative 0
    for val in [-1.0, 2.0]:
        x = torch.tensor(val, requires_grad=True)
        grad = ngrad(lambda z: z.step(), x)
        assert_close(grad, torch.tensor(0.0))
    # x == 0 -> derivative 1
    x0 = torch.tensor(0.0, requires_grad=True)
    grad0 = ngrad(lambda z: z.step(), x0)
    assert_close(grad0, torch.tensor(1.0)) 