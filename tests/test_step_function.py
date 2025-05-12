import unittest

from infnum import ε, H, LeviCivitaTensor

# --------------------------- helpers -------------------------------------------

def step(x):
    """Heaviside step function with convention step(0) = 0."""
    return 1 if x > 0 else 0


class TestStepFunctionNonstandard(unittest.TestCase):
    """Validate non-standard analysis tricks using the Levi-Civita field."""

    def test_step_derivative_dirac(self):
        """(step(ε) - step(0)) / ε  == H (≈ Dirac at 0)."""
        derivative = (step(ε) - step(0)) / ε  # type: ignore[operator]
        self.assertEqual(derivative, H, msg=f"Expected Dirac-like infinity H, got {derivative}")


if __name__ == "__main__":
    unittest.main() 