"""Global configuration for *infnum*.

This module centralises project-wide constants and helper utilities so they can
be tweaked from a single location.  At the moment only the *exponent encoder*
for the upcoming sparse-COO representation lives here but more runtime knobs
can join in future revisions.

Why encode exponents?
---------------------
Storing floating-point exponents directly inside tensors prevents us from
leveraging efficient integer indexing operations (e.g. `torch.unique_consecutive`).
Instead we quantise every rational exponent *e* → *k* by a global denominator
`EXP_DENOM` such that  
    k = round(e * EXP_DENOM).

If we later discover that a finer resolution is required we simply *double* the
denominator and rescale all indices – this keeps the amortised cost *O(N)*.
"""

from __future__ import annotations

__all__ = [
    "EXP_DENOM",
    "encode_exp",
    "decode_exp",
    "set_exp_denom",
]

# -----------------------------------------------------------------------------
# Public constants
# -----------------------------------------------------------------------------

EXP_DENOM: int = 16  # default denominator – tuned for mn≈8 typical sparse terms

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def encode_exp(e: float, *, denom: int | None = None) -> int:
    """Quantise a rational exponent *e* to an integer index.

    Parameters
    ----------
    e:
        Rational exponent (usually a small multiple of ⅟₁₆).
    denom:
        Optional override of the global `EXP_DENOM`.

    Returns
    -------
    int
        Encoded exponent index `k = round(e * denom)`.
    """
    d = denom if denom is not None else EXP_DENOM
    return int(round(e * d))


def decode_exp(k: int, *, denom: int | None = None) -> float:
    """Invert :func:`encode_exp` back to a floating-point exponent."""
    d = denom if denom is not None else EXP_DENOM
    return k / d


def set_exp_denom(new_denom: int):
    """Change the global denominator *in-place*.

    The caller is responsible for rescaling any already-encoded indices – this
    helper merely mutates :data:`EXP_DENOM` to the new value.
    """
    global EXP_DENOM
    if new_denom <= 0:
        raise ValueError("Denominator must be positive.")
    EXP_DENOM = new_denom

# --- 