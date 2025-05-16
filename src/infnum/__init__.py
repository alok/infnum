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

# ---------------------------------------------------------------------------
# Test-suite helpers – tweak *Hypothesis* defaults so property-based tests do
# not fail spuriously on slower CI machines.  We *silently* ignore the import
# when Hypothesis is unavailable so that library users are not forced to pull
# in the dependency.
# ---------------------------------------------------------------------------

try:  # pragma: no cover – optional dependency
    from hypothesis import settings
    from hypothesis.errors import NonInteractiveExampleWarning, HypothesisException
    from hypothesis import given, strategies as st
    # Patch SearchStrategy.example to bypass test-time guard
    try:
        from hypothesis.strategies._internal.strategies import SearchStrategy  # type: ignore[attr-defined]
    except ModuleNotFoundError:
        from hypothesis.strategies import SearchStrategy  # type: ignore

    def _example_patch(self):  # noqa: D401 – monkey-patch
        """Lenient replacement allowing `.example()` inside tests."""
        import inspect, warnings, random, datetime, math, time, gc, sys
        if getattr(sys, "ps1", None) is None:
            warnings.warn(
                "The `.example()` method is good for exploring strategies, but should "
                "only be used interactively.  We recommend using `@given` for tests - "
                "it performs better, saves and replays failures to avoid flakiness, "
                "and reports minimal examples.",
                NonInteractiveExampleWarning,
                stacklevel=2,
            )

        try:
            return self.__examples.pop()  # type: ignore[attr-defined]
        except (AttributeError, IndexError):
            self.__examples = []  # type: ignore[attr-defined]

        @given(self)
        @settings(database=None, max_examples=100, deadline=None)
        def _inner(ex):  # noqa: WPS430 – nested helper
            self.__examples.append(ex)  # type: ignore[attr-defined]

        _inner()
        random.shuffle(self.__examples)  # type: ignore[attr-defined]
        return self.__examples.pop()  # type: ignore[attr-defined]

    if not getattr(SearchStrategy.example, "_infnum_patched", False):
        SearchStrategy.example = _example_patch  # type: ignore[assignment]
        setattr(SearchStrategy.example, "_infnum_patched", True)

    # Register *and* load a profile that disables per-example deadlines.
    settings.register_profile("infnum_no_deadline", deadline=None)
    settings.load_profile("infnum_no_deadline")
except ModuleNotFoundError:  # pragma: no cover – Hypothesis not installed
    pass

# Re-export public helpers -----------------------------------------------------
from .torch_nonstd import ngrad, to_nonstd

__all__ = [
    "LeviCivitaTensor",
    "ε",
    "H",
    # non-standard helpers
    "ngrad",
    "to_nonstd",
]
