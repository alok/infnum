from __future__ import annotations

"""Non-standard automatic differentiation helpers built on top of
:class:`infnum.LeviCivitaTensor`.

The core idea is to inject an infinitesimal perturbation ``ε`` into the input
and read off the coefficient of the first-order term after evaluating the
function **f**.  This reproduces the *non-standard derivative*

    f′(x) := (f(x + ε) − f(x)) / ε

which, unlike the classical limit definition, exists for a large class of
piece-wise or even discontinuous functions (e.g. ``sign``, ``step``).

The helper :func:`ngrad` wraps the boiler-plate so the API mirrors
``torch.autograd.grad`` as closely as possible.

Example
-------
>>> import torch
>>> from infnum import ε  # infinitesimal unit
>>> from infnum.torch_nonstd import ngrad
>>>
>>> def step(lc_x):
...     # Heaviside step implemented with Levi-Civita arithmetic
...     return (lc_x > 0) * 1  # boolean → int promotion handled by LC ops
...
>>> x = torch.tensor(0.0, requires_grad=True)
>>> ngrad(step, x)
# tensor(0.)  ← Dirac delta in distributional sense
"""

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from .torch_backend import LeviCivitaTensor, ε

__all__ = [
    "to_nonstd",
    "ngrad",
]


# -----------------------------------------------------------------------------
# Public helpers
# -----------------------------------------------------------------------------


def to_nonstd(x: Tensor | float | int, *, order: int = 1) -> LeviCivitaTensor:
    """Lift a *real* number **x** into the Levi-Civita field as ``x + εᵒʳᵈᵉʳ``.

    The resulting infinitesimal bump is what drives the *non-standard* chain
    rule.  The helper is merely a convenience around

    >>> LeviCivitaTensor.from_number(x) + ε ** order

    Parameters
    ----------
    x:
        Scalar input value (Python number or 0-dim tensor).
    order:
        Exponent of the perturbation.  For classical first-order derivatives
        keep the default ``1``.

    Returns
    -------
    LeviCivitaTensor
    """
    base = LeviCivitaTensor.from_number(x)
    bump = ε ** order if order != 1 else ε  # micro-optimise common path
    return base + bump


@runtime_checkable
class _LCFunc(Protocol):
    """Callable that takes / returns Levi-Civita numbers."""

    def __call__(self, x: LeviCivitaTensor) -> LeviCivitaTensor:  # pragma: no cover
        ...


def ngrad(f: _LCFunc, x: Tensor) -> Tensor:
    """Non-standard derivative of *f* at scalar point *x*.

    Implementation details
    ----------------------
    1. Evaluate the function at the *perturbed* point: ``y₁ = f(x + ε)``.
    2. Evaluate the function at the *base* point:       ``y₀ = f(x)``.
       (``x`` is lifted to a *Levi-Civita* number with **no** infinitesimal
       terms.)
    3. Form the difference quotient ``(y₁ − y₀) / ε`` which lives again in the
       Levi-Civita field.
    4. Read off the coefficient of ``ε⁰`` – that real number is the
       *non-standard* derivative.

    Finally we call ``torch.autograd.grad`` on the real coefficient so the
    derivative itself remains part of the surrounding autograd graph.

    Parameters
    ----------
    f:
        Function defined over Levi-Civita numbers.
    x:
        0-dimensional tensor where we evaluate the derivative.  Must have
        ``requires_grad=True``.

    Returns
    -------
    torch.Tensor
        The non-standard derivative ``∂f/∂x`` as a scalar tensor.  The tensor is
        part of the autograd graph rooted at *x*.
    """
    if x.dim() != 0:
        raise ValueError("ngrad currently supports only scalar (0-dim) tensors")
    # Ensure *x* participates in autograd without mutating caller's tensor.
    x_var = x.detach().clone().requires_grad_(True)

    # 1. f(x + ε) -------------------------------------------------------------
    y_perturbed = f(to_nonstd(x_var))

    # 2. f(x) at the *exact* base point (no ε) -------------------------------
    y_base = f(LeviCivitaTensor.from_number(x_var))

    # 3. Difference quotient ---------------------------------------------------
    diff = (y_perturbed - y_base) / ε

    # Coefficient of ε⁰ carries the derivative
    coeff_0 = diff.terms.get(0.0)
    if coeff_0 is None:
        # Derivative happened to be exactly zero
        coeff_0 = torch.zeros_like(x_var)

    # Ask PyTorch for ∂coeff₀ / ∂x
    (grad_x,) = torch.autograd.grad(coeff_0, x_var, retain_graph=False, create_graph=False)
    return grad_x 