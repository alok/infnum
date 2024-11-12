import math
import unittest
from unittest import result
from infnum import LeviCivitaNumber, ε, H
from hypothesis import given, assume, strategies as st
from hypothesis.strategies import composite
from _pytest.assertion import truncate

# Prevent pytest from cutting off input.
truncate.DEFAULT_MAX_LINES = 9999
truncate.DEFAULT_MAX_CHARS = 9999


@composite
def levi_civita_numbers(
    draw,
    min_exponent: int = -5,
    max_exponent: int = 5,
    min_coefficient: float = -10.0,
    max_coefficient: float = 10.0,
    max_terms: int = 5,
) -> LeviCivitaNumber:
    """Composite strategy for generating random Levi-Civita numbers."""
    use_special_case = draw(st.booleans())
    if use_special_case:
        special_cases = [
            LeviCivitaNumber.zero(),
            LeviCivitaNumber.one(),
            LeviCivitaNumber.eps(),
            LeviCivitaNumber.H(),
        ]
        special_case = draw(st.sampled_from(special_cases))
        return special_case

    num_terms = draw(st.integers(min_value=1, max_value=max_terms))

    # Generate unique exponents
    exponents = draw(
        st.lists(
            st.integers(min_value=min_exponent, max_value=max_exponent),
            min_size=num_terms,
            max_size=num_terms,
            unique=True,
        )
    )

    # Generate coefficients corresponding to the exponents
    coefficients = draw(
        st.lists(
            st.floats(
                min_value=min_coefficient,
                max_value=max_coefficient,
                allow_nan=False,
                allow_infinity=False,
                exclude_min=True,
                exclude_max=True,
                width=32,
            ),
            min_size=num_terms,
            max_size=num_terms,
        )
    )

    # Combine exponents and coefficients into a terms dictionary
    terms = dict(zip(exponents, coefficients))

    return LeviCivitaNumber(terms)


class TestLeviCivitaNumber(unittest.TestCase):
    def test_addition(self):
        # Test basic addition
        result = 1 + ε
        self.assertEqual(result.terms_fromjax(), {0: 1.0, 1: 1.0}, msg=f"{result = }")

        # Test for high-order terms
        result_high = 1 + ε**100
        self.assertEqual(
            result_high.terms_fromjax(), {0: 1.0, 100: 1.0}, msg=f"{result_high = }"
        )

    def test_multiplication(self):
        # Test basic multiplication
        self.assertEqual((ε * ε).terms_fromjax(), {2: 1.0})
        self.assertEqual(ε * H, 1, msg=f"{ε * H = }")
        self.assertEqual((1 + 2 * ε + 3 * ε**2) * ε, ε + 2 * ε**2 + 3 * ε**3)
        self.assertEqual((ε * ε**100).terms_fromjax(), {101: 1.0})

    def test_comparison(self):

        self.assertTrue(1 - ε < 1 < 1 + ε)
        self.assertTrue(1 + ε >= 1 + ε >= 1)
        self.assertEqual(1, LeviCivitaNumber({0: 1.0}))
        self.assertTrue(0 < 1)
        self.assertTrue(ε < 1)
        self.assertTrue(ε**100 < 999 * ε**99)
        self.assertTrue(
            -H + 1 + 2 * ε + 3 * ε**2 < 1, msg=f"{-H + 1 + 2 * ε + 3 * ε**2 = }"
        )

        # Test lexicographic ordering
        a = 1 + 2 * ε + 3 * ε**2
        b = 1 + 3 * ε - 99 * ε**2
        self.assertTrue(a < b, msg=f"{a = }, {b = }")

    def test_negation_and_subtraction(self):
        # Test negation
        self.assertEqual(-(1 + H), -1 - ε**-1, msg=f"{-(1 + H) = }")

        # Test subtraction
        self.assertEqual(
            (1 + 2 * ε + 3 * ε**2) - 1,
            2 * ε + 3 * ε**2,
            msg=f"{(1 + 2 * ε + 3 * ε**2) - 1 = }",
        )

    def test_string_representation(self):
        # Test basic representation
        self.assertEqual(str(0), "0")
        self.assertEqual(str(1), "1")
        self.assertEqual(str(ε), "ε")
        self.assertEqual(str(ε**2), "ε²")

        # Test complex number representation
        self.assertEqual(str(1 + 2 * ε + 3 * ε**2 - ε**-3.3), "-ε⁻³·³ + 1 + 2ε + 3ε²")

    def test_division(self):
        """Test division of Levi-Civita numbers."""
        # Test division by a scalar
        result = LeviCivitaNumber.one() / 2

        self.assertEqual(result.terms_fromjax(), {0: 0.5})

        # Test division resulting in negative exponents
        result_neg_exp = ε / ε**2

        self.assertEqual(result_neg_exp.terms_fromjax(), {-1: 1.0})  # ε / ε² = ε⁻¹

        # Test division resulting in positive exponents
        result_pos_exp = ε**2 / ε

        self.assertEqual(result_pos_exp.terms_fromjax(), {1: 1.0})  # ε² / ε = ε
        # Test division of non-pure terms
        # TODO: Add test cases for non-pure terms

    def test_power(self):
        """Test exponentiation of Levi-Civita numbers."""

        # Test zero exponent
        self.assertEqual((ε**0), 1)  # ε⁰ = 1

        # Test negative exponent
        self.assertEqual(ε**-1, H)  # ε⁻¹

        self.assertEqual((1 + ε) ** 2, 1 + 2 * ε + ε**2)  # (1 + ε)² = 1 + 2ε + ε²

    def test_inversion(self):
        """Test inversion of Levi-Civita numbers."""
        # Test inversion of a pure infinitesimal
        self.assertEqual(~ε, H, msg=f"{~ε = }")

        # Test inversion of a standard number
        self.assertEqual(~LeviCivitaNumber.one(), 1, msg=f"{~1 = }")

        # Test inversion with series expansion (truncate?)
        expected_terms = 1 + ε + ε**2 + ε**3 + ε**4 + ε**5 + ε**6 + ε**7
        self.assertEqual(
            (1 / (1 - ε)).truncate(max_order=7),
            expected_terms,
            msg=f"{1 / (1 - ε) = }, {expected_terms = }",
        )

    def test_equality(self):
        """Test equality and inequality of Levi-Civita numbers."""
        self.assertEqual(1 + 1e-10 * ε, 1)  # Should be equal within tolerance
        self.assertNotEqual(ε, 0)
        self.assertEqual(0, LeviCivitaNumber({}))

    @given(x=levi_civita_numbers())
    def test_subtraction_self_cancellation(self, x: LeviCivitaNumber) -> None:
        """Test that x - x = 0"""
        self.assertEqual(x - x, 0, f"{x} - {x} = 0")

    @given(x=levi_civita_numbers())
    def test_subtraction_associativity(self, x: LeviCivitaNumber) -> None:
        """Test that x - x + x = x"""
        self.assertEqual((x - x) + x, x, f"{x} - {x} + {x} = {x}")

    @given(x=levi_civita_numbers(), y=levi_civita_numbers())
    def test_subtraction_negation(
        self, x: LeviCivitaNumber, y: LeviCivitaNumber
    ) -> None:
        """Test that x - y = -(y - x)"""
        self.assertEqual(x - y, -(y - x), f"{x} - {y} = -({y} - {x})")

    def test_mixed_operations(self):
        """Test mixed operations combining addition, multiplication, etc."""
        # (1 + ε) * (1 - ε) = 1 - ε²
        self.assertEqual((1 + ε) * (1 - ε), 1 - ε**2)

        # (ε + ε²) / ε = 1 + ε
        self.assertEqual((ε + ε**2) / ε, 1 + ε)

        self.assertEqual((1 + ε) ** 3, 1 + 3 * ε + 3 * ε**2 + ε**3)

    def test_repr(self):
        """Test the string representation of Levi-Civita numbers."""
        self.assertEqual(str(H), "ε⁻¹")
        self.assertEqual(str(-H - ε**2), "-ε⁻¹ - ε²")

    def test_non_pure_operations(self):
        """Test operations on non-pure Levi-Civita numbers."""

        # Approximate square root of num = 1 + ε
        approx_sqrt = 1 + 0.5 * ε - 0.125 * ε**2 + 0.0625 * ε**3 - 0.0390625 * ε**4

        self.assertEqual(
            1 + ε,
            (approx_sqrt**2).truncate(max_order=1),
            msg=f"{str(1 + ε)  }, {str(approx_sqrt**2)  }",
        )

        # Inverse of the square root approximation
        self.assertEqual(
            (~approx_sqrt).truncate(max_order=3),
            1 - 0.5 * ε + 0.375 * ε**2 - 0.3125 * ε**3,
            msg=f"{str((~approx_sqrt).truncate(max_order=3))  }",
        )

    def test_precision_handling(self):
        """Test calculations involving small coefficients and higher-order terms."""

        self.assertEqual(
            1 + 1e-10 * ε, 1, msg=f"{1 + 1e-10 * ε - 1 = }"
        )  # Coefficient is negligible

        # Test addition with negligible terms
        self.assertEqual(1 + 1e-15 * ε**10, 1, msg=f"{1 + 1e-15 * ε**10 - 1 = }")

    def test_edge_cases(self):
        """Test edge cases such as division by infinitesimal and zero."""
        # Division by infinitesimal
        with self.assertRaises(ZeroDivisionError):
            _ = 1 / LeviCivitaNumber.zero()
        # Raising zero to a negative exponent
        with self.assertRaises(ZeroDivisionError):
            _ = LeviCivitaNumber.zero() ** -1

    def test_complex_multiplication(self) -> None:
        """Test complex multiplication expressions."""
        self.assertEqual(
            (1 + ε + ε**2) * (1 - ε + ε**2),
            1 + ε**2 + ε**4,
            msg=f"{(1 + ε + ε**2) * (1 - ε + ε**2) = }",
        )

    def test_complex_division(self) -> None:
        self.assertEqual(
            ((1 + ε) / (1 - ε)).truncate(max_order=7),
            1 + 2 * (ε**1 + ε**2 + ε**3 + ε**4 + ε**5 + ε**6 + ε**7),
            msg=f"{(1 + ε) / (1 - ε) = }",
        )

    def test_exponential(self) -> None:
        # exp(ε) ≈ 1 + ε, elaborated by Euler in [Chapter 7 of Introductio in Analysin Infinitorum](https://www.17centurymaths.com/contents/euler/introductiontoanalysisvolone/ch7vol1.pdf)
        self.assertEqual(
            ε.exp(num_terms=2), 1 + ε, msg=f"{ε.exp(num_terms=2) - (1 + ε) = }"
        )

    @given(x=levi_civita_numbers(), y=levi_civita_numbers())
    def test_addition_commutativity(
        self, x: LeviCivitaNumber, y: LeviCivitaNumber
    ) -> None:
        self.assertEqual(x + y, y + x, f"Addition is not commutative for {x} and {y}")

    @given(x=levi_civita_numbers())
    def test_double_negation(self, x: LeviCivitaNumber) -> None:
        self.assertEqual(-(-x), x, f"Double negation failed for {x}")

    @given(x=levi_civita_numbers().filter(lambda n: not n.is_zero))
    def test_double_reciprocal(self, x: LeviCivitaNumber) -> None:
        print(f"x = {x}")
        print(f"~x = {~x}")
        print(f"~(~x) = {~(~x)}")
        self.assertEqual(
            (~(~x)).truncate(max_order=4),
            x.truncate(max_order=4),
            f"Double reciprocal failed for {x}",
        )

    @given(
        x=levi_civita_numbers(), y=levi_civita_numbers().filter(lambda n: not n.is_zero)
    )
    def test_division_multiplication(
        self, x: LeviCivitaNumber, y: LeviCivitaNumber
    ) -> None:
        print(f"x = {x}")
        print(f"y = {y}")
        print(f"(x / y) = {x / y}")
        print(f"((x / y) * y) = {((x / y) * y).truncate(max_order=8)}")
        self.assertEqual(
            ((x / y) * y).truncate(max_order=8),
            x.truncate(max_order=8),
            f"Division and multiplication failed for {x} and {y}",
        )

    @given(x=levi_civita_numbers())
    def test_zero_addition(self, x: LeviCivitaNumber) -> None:
        self.assertEqual(
            x + LeviCivitaNumber.zero(), x, msg=f"Zero addition failed for {x}"
        )

    @given(x=levi_civita_numbers())
    def test_zero_multiplication(self, x: LeviCivitaNumber) -> None:
        self.assertEqual(
            x * LeviCivitaNumber.zero(),
            LeviCivitaNumber.zero(),
            msg=f"Zero multiplication failed for {x}",
        )

    @given(x=levi_civita_numbers())
    def test_one_multiplication(self, x: LeviCivitaNumber) -> None:
        self.assertEqual(
            x * LeviCivitaNumber.one(), x, msg=f"One multiplication failed for {x}"
        )


if __name__ == "__main__":
    unittest.main()
