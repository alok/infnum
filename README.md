# infnum

**infnum** is a Python library that provides infinite and infinitesimal numbers using the [Levi-Civita Field](https://en.wikipedia.org/wiki/Levi-Civita_field). It uses [JAX](https://github.com/google/jax) under the hood.

## Installation

You can install **infnum** via `pip`:

```bash
pip install infnum
```

## Usage

### Creating Levi-Civita Numbers

The main constructor expects a dictionary with numerical keys and values. The keys represent the exponents and the values the coefficients. In the following example, note that ε⁻³ is an *infinite* number.

```python
from infnum import LeviCivitaNumber

# Create a Levi-Civita number from a real number
num = LeviCivitaNumber.from_number(5.0) # 5.0
print(num)  # Output: 5.0
# Create a Levi-Civita number with infinitesimal parts
num = LeviCivitaNumber({0: 1.0, 1: 2.0, 2: 3.0, -3: 4.0}) # -4.0ε⁻³ + 1.0 + 2.0ε + 3.0ε² 
print(num)  # Output: -4.0ε⁻³ + 1.0 + 2.0ε + 3.0ε² 
```

### Operations

Levi-Civita numbers support arithmetic operations like addition, subtraction, multiplication, and division. Division is truncated to 8 terms by default since it may result in an infinite series, but the higher terms are infinitesimal with respect to the lower terms.


You may compare Levi-Civita numbers using the `<, <=, >, >=, ==, !=` operators, which considers the lexicographical order of the coefficients.

### Testing

Levi-Civita numbers come with unit tests to ensure correctness. You can run the tests using:

```bash
python -m unittest discover tests
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache 2 License.

## TODO

- [ ] add complex number support