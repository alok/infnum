from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Float
import unittest

N = 4  # Order up to which we represent the series
SUPERSCRIPT_MAP = str.maketrans("0123456789.-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻·")

def _to_superscript(n: float | int, superscript_map: dict[int, int] = SUPERSCRIPT_MAP) -> str:
    """Convert a number to its superscript representation.
    
    Handles integers, decimals, and negative numbers.
    """
    return str(n).translate(superscript_map)

@dataclass
class LeviCivitaNumber:
    coefficients: Float[Array, "N + 1"]

    def __post_init__(self):
        # Ensure coefficients is a JAX array
        self.coefficients = jnp.asarray(self.coefficients)

    def __add__(self, other: "LeviCivitaNumber") -> "LeviCivitaNumber":
        return LeviCivitaNumber(self.coefficients + other.coefficients)

    def __mul__(self, other: "LeviCivitaNumber") -> "LeviCivitaNumber":
        # Perform convolution up to order N
        full_conv = jnp.convolve(self.coefficients, other.coefficients)
        conv_coeffs = full_conv[:N + 1]
        return LeviCivitaNumber(conv_coeffs)

    def __neg__(self) -> "LeviCivitaNumber":
        return LeviCivitaNumber(-self.coefficients)

    def __sub__(self, other: "LeviCivitaNumber") -> "LeviCivitaNumber":
        return self + (-other)

    def __repr__(self) -> str:
        terms = []
        for i, coeff in enumerate(self.coefficients):
            if coeff != 0:
                if i == 0:
                    terms.append(f"{coeff}")
                elif i == 1:
                    terms.append(f"{coeff}ε")
                else:
                    # Handle the coefficient formatting
                    if float(coeff).is_integer():
                        coeff_str = f"{int(coeff)}"
                    else:
                        coeff_str = f"{coeff}"
                    terms.append(f"{coeff_str}ε{_to_superscript(i)}")
        return " + ".join(terms) if terms else "0"

    def __eq__(self, other: "LeviCivitaNumber") -> bool:
        return bool(jnp.all(self.coefficients == other.coefficients).item())
    
    def __lt__(self, other: "LeviCivitaNumber") -> bool:
        # Compare coefficients lexicographically
        # Find first non-equal coefficient
        for s, o in zip(self.coefficients, other.coefficients):
            if not (s == o).item():  # Convert JAX boolean to Python boolean
                return bool((s < o).item())
        return False
    
    def __le__(self, other: "LeviCivitaNumber") -> bool:
        return self < other or self == other
    
    def __gt__(self, other: "LeviCivitaNumber") -> bool:
        return not self <= other
    
    def __ge__(self, other: "LeviCivitaNumber") -> bool:
        return not self < other

class TestLeviCivitaNumber(unittest.TestCase):
    def setUp(self):
        # Common test numbers
        self.zero = LeviCivitaNumber(jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]))
        self.one = LeviCivitaNumber(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0]))
        self.eps = LeviCivitaNumber(jnp.array([0.0, 1.0, 0.0, 0.0, 0.0]))
        self.eps_squared = LeviCivitaNumber(jnp.array([0.0, 0.0, 1.0, 0.0, 0.0]))
        self.complex_num = LeviCivitaNumber(jnp.array([1.0, 2.0, 3.0, 0.0, 0.0]))

    def test_addition(self):
        # Test basic addition
        result = self.one + self.eps
        self.assertTrue(jnp.allclose(result.coefficients, jnp.array([1.0, 1.0, 0.0, 0.0, 0.0])))

        # Test commutativity
        self.assertTrue(jnp.allclose((self.one + self.eps).coefficients, 
                                   (self.eps + self.one).coefficients))

        # Test zero identity
        self.assertTrue(jnp.allclose((self.complex_num + self.zero).coefficients, 
                                   self.complex_num.coefficients))

    def test_multiplication(self):
        # Test basic multiplication
        result = self.eps * self.eps
        self.assertTrue(jnp.allclose(result.coefficients, self.eps_squared.coefficients))

        # Test with more complex numbers
        result = self.complex_num * self.eps
        expected = LeviCivitaNumber(jnp.array([0.0, 1.0, 2.0, 3.0, 0.0]))
        self.assertTrue(jnp.allclose(result.coefficients, expected.coefficients))

        # Test one identity
        self.assertTrue(jnp.allclose((self.complex_num * self.one).coefficients, 
                                   self.complex_num.coefficients))

    def test_comparison(self):
        # Test equality
        self.assertEqual(self.one, LeviCivitaNumber(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0])))
        
        # Test less than
        self.assertTrue(self.zero < self.one)
        self.assertTrue(self.eps < self.one)
        
        # Test greater than
        self.assertTrue(self.one > self.zero)
        self.assertTrue(self.complex_num > self.one)
        
        # Test lexicographic ordering
        a = LeviCivitaNumber(jnp.array([1.0, 2.0, 0.0, 0.0, 0.0]))
        b = LeviCivitaNumber(jnp.array([1.0, 3.0, 0.0, 0.0, 0.0]))
        self.assertTrue(a < b)

    def test_negation_and_subtraction(self):
        # Test negation
        self.assertTrue(jnp.allclose((-self.one).coefficients, 
                                   jnp.array([-1.0, 0.0, 0.0, 0.0, 0.0])))
        
        # Test subtraction
        result = self.complex_num - self.one
        expected = LeviCivitaNumber(jnp.array([0.0, 2.0, 3.0, 0.0, 0.0]))
        self.assertTrue(jnp.allclose(result.coefficients, expected.coefficients))

    def test_string_representation(self):
        # Test basic representation
        self.assertEqual(str(self.zero), "0")
        self.assertEqual(str(self.one), "1.0")
        self.assertEqual(str(self.eps), "1.0ε")
        self.assertEqual(str(self.eps_squared), "1.0ε²")
        
        # Test complex number representation
        self.assertEqual(str(self.complex_num), "1.0 + 2.0ε + 3.0ε²")

if __name__ == "__main__":
    unittest.main()
