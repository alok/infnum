# SPDX-License-Identifier: MIT
"""infnum – Infinite & infinitesimal numbers built on top of PyTorch.

This package provides a Levi-Civita field implementation (`LeviCivitaTensor`) that
behaves like an ordered real-closed field extended with infinitesimals (ε) and
infinities (H = ε⁻¹).  All coefficients are 0-dim PyTorch tensors, so automatic
 differentiation flows through algebraic expressions out-of-the-box.
"""

from __future__ import annotations

from .torch_backend import (
    LeviCivitaTensor,  # public class
    ε,  # infinitesimal unit
    H,  # infinite unit
)

__all__ = [
    "LeviCivitaTensor",
    "ε",
    "H",
]
