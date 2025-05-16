import torch
from hypothesis import given, strategies as st
import math

from infnum.torch_sparse import SparseLCTensor
from infnum.torch_autograd import ngrad

# -----------------------------------------------------------------------------
# Helper – sin that accepts both Tensor and SparseLCTensor
# -----------------------------------------------------------------------------

def sin_fn(x):
    if isinstance(x, SparseLCTensor):
        return torch.sin(x.standard_part())
    return torch.sin(x)


# -----------------------------------------------------------------------------
# Scalar inputs – autograd path should match analytic derivative exactly
# -----------------------------------------------------------------------------

@given(
    st.floats(
        min_value=-10 * math.pi,
        max_value=10 * math.pi,
        allow_nan=False,
        allow_infinity=False,
    )
)
def test_ngrad_sin_scalar_matches_cos(x0: float):
    """For scalar inputs  ngrad(sin) == cos  (within FP error)."""
    x = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
    grad = ngrad(sin_fn, x)
    expected = torch.tensor(math.cos(x0), dtype=torch.float64)
    assert torch.allclose(grad, expected, atol=1e-10, rtol=1e-10)


# -----------------------------------------------------------------------------
# Vector inputs – finite-difference path should approximate the derivative
# -----------------------------------------------------------------------------

def test_ngrad_sin_vector_approx_cos():
    """Central-difference derivative approximates cos(x) on a dense grid."""
    x = torch.linspace(-math.pi, math.pi, 401, dtype=torch.float64, requires_grad=True)
    dx_est = ngrad(sin_fn, x)
    dx_true = torch.cos(x)

    # Skip the first/last two points where the stencil is one-sided
    interior = slice(2, -2)
    max_err = (dx_est[interior] - dx_true[interior]).abs().max().item()
    assert max_err < 1e-3


# -----------------------------------------------------------------------------
# LC callback returning LC numbers – ensure ε¹ extraction works correctly
# -----------------------------------------------------------------------------

def square_lc(x):
    if isinstance(x, SparseLCTensor):
        return x * x
    return x * x


def test_ngrad_square_lc_scalar():
    x0 = 2.5
    x = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
    grad = ngrad(square_lc, x)
    expected = torch.tensor(2 * x0, dtype=torch.float64)
    assert torch.allclose(grad, expected) 