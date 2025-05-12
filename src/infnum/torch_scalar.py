from __future__ import annotations

"""Experimental building block: a single-term Levi-Civita number.

A *scalar* is the smallest indivisible unit of the Levi-Civita field.  It
represents exactly one term   c · εᵉ  with

* exponent **e**   – python ``float`` kept immutable
* coefficient **c** – 0-dim PyTorch tensor that carries autograd history

The class purposefully lives in its own module because it will eventually host
custom ``torch.autograd.Function`` overrides once we hit CUDA / Triton.  For the
CPU reference implementation we can delegate to plain tensor algebra – this is
already enough for autograd to work.

The API mirrors a subset of ``Tensor`` so most downstream code can treat
``LeviCivitaScalar`` and ``LeviCivitaTensor`` interchangeably.  The tensor class
will soon use these scalars as its internal representation.
"""

from dataclasses import dataclass
from typing import Self, Union

import torch
from torch import Tensor

# Public re-export of the infinitesimal – *defined later* to avoid circular deps.
__all__ = ["LeviCivitaScalar", "εs"]

CoeffLike = Union[int, float, Tensor]


@dataclass(frozen=True, slots=True)
class LeviCivitaScalar:
    exponent: float
    coeff: Tensor

    # ------------------------------------------------------------------
    # Constructors & helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_tensor(x: CoeffLike) -> Tensor:
        return x if isinstance(x, Tensor) else torch.as_tensor(float(x), dtype=torch.get_default_dtype())

    @classmethod
    def from_number(cls, x: CoeffLike) -> "LeviCivitaScalar":
        return cls(0.0, cls._to_tensor(x))

    # ------------------------------------------------------------------
    # Basic arithmetic (only the handful we need immediately)
    # ------------------------------------------------------------------

    def __add__(self, other: Union[CoeffLike, "LeviCivitaScalar"]) -> "LeviCivitaScalar":
        if not isinstance(other, LeviCivitaScalar):
            other = LeviCivitaScalar.from_number(other)
        if self.exponent == other.exponent:
            return LeviCivitaScalar(self.exponent, self.coeff + other.coeff)
        # different exponents – return a *Tensor*-level representation (LeviCivitaTensor) later
        raise NotImplementedError("Scalar addition with mismatching exponents not yet implemented")

    def __mul__(self, other: Union[CoeffLike, "LeviCivitaScalar"]) -> "LeviCivitaScalar":
        if not isinstance(other, LeviCivitaScalar):
            other = LeviCivitaScalar.from_number(other)
        return LeviCivitaScalar(self.exponent + other.exponent, self.coeff * other.coeff)

    __rmul__ = __mul__

    def __neg__(self):
        return LeviCivitaScalar(self.exponent, -self.coeff)

    def __sub__(self, other: Union[CoeffLike, Self]):
        return self + (-other)

    # ------------------------------------------------------------------
    # Inversion (needed for division)
    # ------------------------------------------------------------------

    def __invert__(self):
        if torch.allclose(self.coeff, torch.zeros_like(self.coeff)):
            raise ZeroDivisionError("Attempted to invert zero Levi-Civita scalar")
        return LeviCivitaScalar(-self.exponent, 1.0 / self.coeff)

    def __truediv__(self, other: Union[CoeffLike, "LeviCivitaScalar"]):
        if not isinstance(other, LeviCivitaScalar):
            other = LeviCivitaScalar.from_number(other)
        return self * ~other

    __rtruediv__ = __truediv__

    # ------------------------------------------------------------------
    # Convenience predicates
    # ------------------------------------------------------------------

    @property
    def is_infinitesimal(self) -> bool:
        return self.exponent > 0

    @property
    def is_infinite(self) -> bool:
        return self.exponent < 0

    @property
    def standard_part(self) -> Tensor:
        return self.coeff if self.exponent == 0 else torch.zeros_like(self.coeff)

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __str__(self) -> str:  # pragma: no cover – cosmetics only
        if torch.allclose(self.coeff, torch.zeros_like(self.coeff)):
            return "0"
        coeff_str = str(float(self.coeff.item()))
        if self.exponent == 0:
            return coeff_str
        if self.exponent == 1:
            return f"{coeff_str}ε"
        return f"{coeff_str}ε^{self.exponent}"


# Infinitesimal unit singleton ---------------------------------------------------
εs = LeviCivitaScalar(1.0, torch.ones((), dtype=torch.get_default_dtype())) 