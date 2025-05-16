from __future__ import annotations

"""
PyTorch backend for the Levi-Civita field.

The guiding principle is identical to the JAX implementation: represent a
Levi-Civita number as a sparse mapping from exponents (float) to coefficients
(`torch.Tensor`, usually scalar).  The code purposely mirrors the public
interface of :class:`infnum.LeviCivitaNumber` so that existing examples can be
ported with a simple

  ```python
  from infnum.torch_backend import LeviCivitaTensor as LC
  ε = LC.eps()
  ```

This first revision focuses on the basic ring/field operations that we will
need for experimenting with automatic differentiation of discontinuous
functions.  The implementation tries to stay purely functional – every
operation returns a *new* object and never mutates internal state.

NOTE: For now we deliberately do **not** use `torch.compile` / TorchScript or
any other acceleration mechanism.  Correctness comes first and the sparse
representation is anyway dominated by Python-side bookkeeping.
"""

from dataclasses import dataclass, field
from functools import cached_property
import math
from operator import itemgetter
from typing import Dict, Mapping, Self, Union, TypeAlias, overload, Any

import torch
from torch import Tensor

# -- Type aliases -----------------------------------------------------------------
Exponent: TypeAlias = float  # ε^Exponent        (PEP-695 style would be nice but keep simple)
Coefficient: TypeAlias = Tensor  # always a (0-dim) tensor so autograd can flow

CoeffLike = Union[int, float, "LeviCivitaTensor"]


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _as_tensor(x: Union[int, float, Tensor]) -> Tensor:
    """Ensure *x* is a scalar tensor on the same device as the default tensor."""
    if isinstance(x, Tensor):
        return x if x.dim() == 0 else x.squeeze()
    return torch.as_tensor(float(x), dtype=torch.get_default_dtype())


def _is_close(a: Union[int, float, Tensor], b: Union[int, float, Tensor], *, atol: float = 1e-6) -> bool:  # type: ignore[override]
    """Return *True* if two scalar-like inputs are close within *atol*.

    The helper widens the accepted argument types to include plain Python
    scalars which silences static type warnings while still performing the
    calculation on tensors internally.
    """
    at = _as_tensor(a)
    bt = _as_tensor(b)
    return bool(torch.isclose(at, bt, atol=atol))


# -----------------------------------------------------------------------------
# Main data class
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LeviCivitaTensor:
    """Sparse representation of a Levi-Civita number with *torch* scalars.

    The mapping *terms* associates each *exponent* (float) ↦ *coefficient*
    (scalar tensor).  All zeros are eagerly removed so we never store empty
    coefficients.
    """

    terms: Mapping[Exponent, Coefficient] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean(terms: Mapping[Exponent, Union[float, int, Tensor]]) -> Dict[Exponent, Coefficient]:
        """Convert coefficients → Tensor and drop zeros."""
        cleaned: Dict[Exponent, Coefficient] = {}
        for exp, coeff in terms.items():
            tensor_coeff = _as_tensor(coeff)
            if not _is_close(tensor_coeff, 0.0):
                cleaned[float(exp)] = tensor_coeff
        return cleaned

    # the dataclass is frozen – we must use __setattr__ in __post_init__
    def __post_init__(self):
        cleaned = self._clean(self.terms)
        object.__setattr__(self, "terms", cleaned)

    # ------------------------------------------------------------------
    # Smart constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_number(cls, n: Union[int, float, Tensor]) -> "LeviCivitaTensor":
        return cls({0.0: _as_tensor(n)}) if not _is_close(n, 0.0) else cls()

    @classmethod
    def zero(cls) -> "LeviCivitaTensor":
        return cls()

    @classmethod
    def one(cls) -> "LeviCivitaTensor":
        return cls.from_number(1.0)

    @classmethod
    def eps(cls) -> "LeviCivitaTensor":
        return cls({1.0: _as_tensor(1.0)})

    @classmethod
    def ε(cls) -> "LeviCivitaTensor":  # unicode convenience alias
        return cls.eps()

    @classmethod
    def H(cls) -> "LeviCivitaTensor":
        """Infinite unit – 1/ε."""
        return cls({-1.0: _as_tensor(1.0)})

    # ------------------------------------------------------------------
    # Basic dunder helpers
    # ------------------------------------------------------------------

    def _binary_op(self, other: CoeffLike, op):
        if isinstance(other, (int, float)):
            other = self.from_number(other)
        if isinstance(other, Tensor):
            other = self.from_number(other)
        if not isinstance(other, LeviCivitaTensor):
            return NotImplemented
        return op(self, other)

    # ------------------------------------------------------------------
    # Algebraic operations
    # ------------------------------------------------------------------

    def __add__(self, other: CoeffLike) -> "LeviCivitaTensor":
        return self._binary_op(other, self._add)

    def __radd__(self, other: CoeffLike):
        return self.__add__(other)

    @staticmethod
    def _add(a: "LeviCivitaTensor", b: "LeviCivitaTensor") -> "LeviCivitaTensor":
        all_exp = set(a.terms) | set(b.terms)
        result: Dict[Exponent, Coefficient] = {}
        for e in all_exp:
            coeff = _as_tensor(a.terms.get(e, 0.0)) + _as_tensor(b.terms.get(e, 0.0))
            if not _is_close(coeff, 0.0):
                result[e] = coeff
        return LeviCivitaTensor(result)

    # ------------------------------------------------------------------
    def __neg__(self) -> "LeviCivitaTensor":
        return LeviCivitaTensor({exp: -coeff for exp, coeff in self.terms.items()})

    def __sub__(self, other: CoeffLike):
        return self + (-other)

    def __rsub__(self, other: CoeffLike):
        return (-self) + other

    # ------------------------------------------------------------------
    def __mul__(self, other: CoeffLike):
        return self._binary_op(other, self._mul)

    def __rmul__(self, other: CoeffLike):
        return self.__mul__(other)

    @staticmethod
    def _mul(a: "LeviCivitaTensor", b: "LeviCivitaTensor") -> "LeviCivitaTensor":
        result: Dict[Exponent, Coefficient] = {}
        for e1, c1 in a.terms.items():
            for e2, c2 in b.terms.items():
                exp = e1 + e2
                coeff = c1 * c2
                result[exp] = result.get(exp, _as_tensor(0.0)) + coeff
        # prune zeros
        cleaned = {e: c for e, c in result.items() if not _is_close(c, 0.0)}
        return LeviCivitaTensor(cleaned)

    # ------------------------------------------------------------------
    def __truediv__(self, other: CoeffLike):
        return self._binary_op(other, self._div)

    def __rtruediv__(self, other: CoeffLike):
        if isinstance(other, (int, float, Tensor)):
            other = self.from_number(other)
        if isinstance(other, LeviCivitaTensor):
            return other / self
        return NotImplemented

    @staticmethod
    def _div(a: "LeviCivitaTensor", b: "LeviCivitaTensor") -> "LeviCivitaTensor":
        if b.is_zero:
            raise ZeroDivisionError("Division by zero in Levi-Civita field")
        # If *b* is a pure term ε^k * c – divide each exponent and coeff directly.
        if b.only_term is not None:
            exp_b, coeff_b = b.only_term
            result = {exp - exp_b: coeff / coeff_b for exp, coeff in a.terms.items()}
            return LeviCivitaTensor(result)
        # general case: multiply by reciprocal
        return a * ~b

    # ------------------------------------------------------------------
    def __pow__(self, n: int | float):
        # fractional exponents only defined for pure terms.
        if isinstance(n, float) and not n.is_integer():
            if self.only_term is None:
                raise NotImplementedError("Fractional powers only implemented for pure terms")
            exp, coeff = self.only_term
            return LeviCivitaTensor({exp * n: coeff ** n})
        # integer power – exponentiation by squaring
        k = int(n)
        if k == 0:
            return self.one()
        if k < 0:
            return (~self) ** (-k)
        result = self.one()
        base = self
        while k > 0:
            if k & 1:
                result = result * base
            base = base * base
            k >>= 1
        return result

    # ------------------------------------------------------------------
    def __invert__(self, num_terms: int = 8):
        """Multiplicative inverse using a geometric-series-like formula."""
        if self.is_zero:
            raise ZeroDivisionError("Division by zero (attempting to invert 0)")
        # Pure term – easy.
        if self.only_term is not None:
            exp, coeff = self.only_term
            return LeviCivitaTensor({-exp: 1.0 / coeff})
        # Factor out largest term so remaining part is 1 + εₓ with εₓ infinitesimal.
        largest_exp = min(self.terms.keys())
        largest_coeff = self.terms[largest_exp]
        largest = LeviCivitaTensor({largest_exp: largest_coeff})
        rest = self / largest  # == 1 + εₓ
        eps_x = rest - 1
        # Build (1 + εₓ)^{-1} ≈ Σₖ (-εₓ)^k
        series: LeviCivitaTensor = self.one()
        term = self.one()
        for _ in range(1, num_terms):
            term = (-eps_x) * term  # (-εₓ)^k iteratively
            series = series + term
        return (~largest) * series

    # ------------------------------------------------------------------
    # Properties & utility methods
    # ------------------------------------------------------------------

    @property
    def is_zero(self) -> bool:
        """True if the Levi-Civita number is exactly 0."""
        return len(self.terms) == 0

    @property
    def only_term(self) -> tuple[Exponent, Coefficient] | None:
        """Return (exp, coeff) when number has exactly one non-zero term."""
        if len(self.terms) == 1:
            return next(iter(self.terms.items()))
        if self.is_zero:
            return (0.0, _as_tensor(0.0))
        return None

    # convenience boolean predicates ------------------------------------------------
    @property
    def is_infinite(self) -> bool:
        return any(e < 0 for e in self.terms)

    @property
    def is_infinitesimal(self) -> bool:
        return bool(self.terms) and all(e > 0 for e in self.terms)

    @property
    def standard_part(self) -> "LeviCivitaTensor":
        return LeviCivitaTensor({e: c for e, c in self.terms.items() if e == 0})

    # ------------------------------------------------------------------
    # String representation – matching the JAX version
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        if self.is_zero:
            return "0"
        SUP = str.maketrans("0123456789-.", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻·")

        def coeff_to_str(coeff: Tensor, is_std: bool) -> str:
            val = float(coeff.item())
            sign = "-" if val < 0 else ""
            abs_val = abs(val)
            if math.isclose(abs_val, 1.0):
                return ("-1" if sign and is_std else "-" if sign else "1" if is_std else "")
            s = str(int(abs_val) if abs_val.is_integer() else abs_val)
            return f"{sign}{s}"

        def term_to_str(exp: float, coeff_str: str) -> str:
            if exp == 0:
                return coeff_str
            exp_part = "ε" if exp == 1 else f"ε{str(int(exp) if exp.is_integer() else exp).translate(SUP)}"
            return f"{coeff_str}{exp_part}"

        parts = []
        for i, (e, c) in enumerate(sorted(self.terms.items(), key=itemgetter(0))):
            cs = coeff_to_str(c, is_std=(e == 0))
            term = term_to_str(e, cs)
            if i == 0:
                parts.append(term)
            else:
                parts.append(term if term.startswith("-") else f"+ {term}")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Equality & comparison (lexicographic on exponents)
    # ------------------------------------------------------------------

    def is_close_to(self, other: CoeffLike | Any, *, atol: float = 1e-6) -> bool:  # type: ignore[override]
        other = other if isinstance(other, LeviCivitaTensor) else self.from_number(other)
        all_exp = set(self.terms) | set(other.terms)
        return all(
            _is_close(self.terms.get(e, 0.0), other.terms.get(e, 0.0), atol=atol) for e in all_exp
        )

    def __eq__(self, other: object):  # type: ignore[override]
        if not isinstance(other, (int, float, Tensor, LeviCivitaTensor)):
            return NotImplemented
        return self.is_close_to(other)

    # same lexicographic ordering as JAX version
    def __lt__(self, other: CoeffLike):
        other = other if isinstance(other, LeviCivitaTensor) else self.from_number(other)
        all_exp = sorted(set(self.terms) | set(other.terms))
        self_list = [float(_as_tensor(self.terms.get(e, 0.0)).item()) for e in all_exp]
        other_list = [float(_as_tensor(other.terms.get(e, 0.0)).item()) for e in all_exp]
        return self_list < other_list

    def __le__(self, other: CoeffLike):
        return self < other or self == other

    def __gt__(self, other: CoeffLike):
        return not self <= other

    def __ge__(self, other: CoeffLike):
        return self > other or self == other

    # ------------------------------------------------------------------
    # Truncation helper (useful in tests)
    # ------------------------------------------------------------------

    def truncate(self, *, min_order: float = float("-inf"), max_order: float = float("inf")) -> "LeviCivitaTensor":
        subset = {e: c for e, c in self.terms.items() if min_order <= e <= max_order}
        return LeviCivitaTensor(subset)

    # debugging convenience -------------------------------------------------------
    def terms_as_python(self) -> Dict[float, float]:
        """Convert coefficients to plain python floats for readability in tests."""
        return {e: float(c.item()) for e, c in self.terms.items()}


# Convenience module-level constants ------------------------------------------------
ε = LeviCivitaTensor.eps()
H = LeviCivitaTensor.H() 