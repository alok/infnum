import torch
import pytest
from hypothesis import given, strategies as st

from infnum.torch_sparse import SparseLCTensor
from infnum.torch_autograd import ngrad

# -----------------------------------------------------------------------------
# Sign function derivative – expects 2·δ(x) spike at the origin
# -----------------------------------------------------------------------------

def sign_fn(x):
    """Sign function implemented via 2·step(x) − 1 so that jump = 2.

    We deliberately avoid `torch.sign` because its value at the origin is 0,
    which splits the jump into two smaller steps (−1→0 and 0→1).  The variant
    below keeps the full  −1→+1  discontinuity *at* the origin which makes the
    distributional derivative equal to *two* Dirac deltas (area ≈ 2).
    """
    if isinstance(x, SparseLCTensor):
        # sign(x) = 2·step(x) − 1 with proper ε-propagation
        step_x = x.step()
        two = SparseLCTensor.from_real(torch.full((step_x.batch_size,), 2.0, dtype=step_x.dtype, device=step_x.device))
        neg_one = SparseLCTensor.from_real(torch.full((step_x.batch_size,), -1.0, dtype=step_x.dtype, device=step_x.device))
        return (step_x * two) + neg_one
    return (x >= 0).to(dtype=torch.float64) * 2 - 1


def test_sign_function_derivative_dirac():
    """Finite-difference derivative integrates to 2 at the origin."""
    x = torch.linspace(-1, 1, 201, dtype=torch.float64, requires_grad=True)
    dx = ngrad(sign_fn, x)

    # Zero derivative away from the discontinuity
    assert torch.allclose(dx[x < -0.1], torch.zeros_like(dx[x < -0.1]))
    assert torch.allclose(dx[x > 0.1], torch.zeros_like(dx[x > 0.1]))

    # Spike magnitude at the origin should reflect jump height (≈2)
    zero_idx = torch.argmin(torch.abs(x))
    assert torch.isclose(dx[zero_idx], torch.tensor(2.0, dtype=torch.float64))


# -----------------------------------------------------------------------------
# Reciprocal  f(x) = 1 / x  – smooth except at the origin
# -----------------------------------------------------------------------------


def reciprocal_fn(x):
    if isinstance(x, SparseLCTensor):
        # Use built-in reciprocal for LC numbers and take the real part.
        return x.reciprocal().standard_part()
    return 1.0 / x


@given(st.floats(min_value=-10.0, max_value=10.0, allow_infinity=False, allow_nan=False).filter(lambda v: abs(v) > 0.1))
def test_reciprocal_derivative_matches_classical(x0: float):
    """ngrad should agree with −1 / x² for the smooth reciprocal function."""
    x = torch.tensor([x0], dtype=torch.float64, requires_grad=True)
    grad_est = ngrad(reciprocal_fn, x)
    analytic = torch.tensor([-1.0 / (x0 ** 2)], dtype=torch.float64)
    assert torch.allclose(grad_est, analytic, atol=1e-4, rtol=1e-4) 