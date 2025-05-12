import unittest

try:
    from infnum.torch_backend import (
        LeviCivitaTensor as LC,
        ε,
        H,
    )
except ModuleNotFoundError:  # pragma: no cover
    LC = None  # type: ignore


class TestTorchDivision(unittest.TestCase):
    def setUp(self):
        if LC is None:
            self.skipTest("PyTorch backend not available")

    def test_division_by_scalar(self):
        x = 1 + 2 * ε + 3 * ε**2
        y = x / 2
        self.assertEqual(y.terms_as_python(), {0: 0.5, 1: 1.0, 2: 1.5})  # type: ignore[attr-defined]

    def test_division_infinitesimal(self):
        self.assertEqual((ε / ε**2).terms_as_python(), {-1: 1.0})  # type: ignore[attr-defined]  # ε / ε² = ε⁻¹

    def test_division_composition(self):
        a = 1 + ε + ε**2
        b = 1 - ε + ε**2
        self.assertEqual(((a / b) * b).truncate(max_order=4), a.truncate(max_order=4))

    def test_division_by_infinite(self):
        self.assertEqual((1 / H).terms_as_python(), {1: 1.0})  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main() 