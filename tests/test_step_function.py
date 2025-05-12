import unittest

from infnum import ε as ε_jax, H as H_jax, LeviCivitaNumber

# Optional torch backend
try:
    from infnum.torch_backend import ε as ε_torch, H as H_torch, LeviCivitaTensor
except ModuleNotFoundError:  # pragma: no cover
    LeviCivitaTensor = None  # type: ignore


# --------------------------- helpers -------------------------------------------

def step(x):
    """Heaviside step function with convention step(0) = 0."""
    return 1 if x > 0 else 0


class TestStepFunctionNonstandard(unittest.TestCase):
    """Validate non-standard analysis tricks using the Levi-Civita field."""

    # JAX backend ----------------------------------------------------------------
    def test_step_derivative_dirac_jax(self):
        """(step(ε) - step(0)) / ε  == H (≈ Dirac at 0)."""
        derivative = (step(ε_jax) - step(0)) / ε_jax  # type: ignore[operator]
        self.assertEqual(derivative, H_jax, msg=f"Expected Dirac-like infinity H, got {derivative}")

    # Torch backend --------------------------------------------------------------
    def test_step_derivative_dirac_torch(self):
        if LeviCivitaTensor is None:
            self.skipTest("PyTorch not available")
        derivative = (step(ε_torch) - step(0)) / ε_torch  # type: ignore[operator]
        self.assertEqual(derivative, H_torch, msg=f"Expected Dirac-like infinity H, got {derivative}")


if __name__ == "__main__":
    unittest.main() 