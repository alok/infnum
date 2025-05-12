import torch
from infnum.torch_nonstd import ngrad


def test_quadratic_derivative_matches_classical():
    x_val = 3.0
    x = torch.tensor(x_val, requires_grad=True)

    def f(x):  # f(x) = x^2
        return x ** 2

    grad = ngrad(f, x)
    assert torch.allclose(grad, torch.tensor(2 * x_val))


def test_abs_derivative_generalised():
    x_val = 0.0  # nondifferentiable point
    x = torch.tensor(x_val, requires_grad=True)

    def f(x):
        # |x| implemented via piecewise definition using Levi-Civita comparisons
        return (x >= 0) * x + (x < 0) * (-x)

    grad = ngrad(f, x)
    # Clarke derivative of |x| at 0 is in [-1, 1]; non-standard derivative picks 0 here
    assert torch.allclose(grad, torch.tensor(0.0)) 