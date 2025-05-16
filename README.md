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

## TODO

- [ ] add complex number support
- [ ] support non-integer powers for non-pure terms
- [ ] support log

# Differentiating Through Discontinuities with Levi-Civita Fields

This package provides a PyTorch implementation of Levi-Civita fields that enables automatic differentiation through discontinuous functions like absolute value, step functions, and rounding operations.

## Features

- Sparse representation of Levi-Civita numbers
- Seamless integration with PyTorch's autograd system
- Support for common discontinuous operations:
  - Absolute value
  - Step function
  - Round function
- Efficient batch processing
- CPU and CUDA support

## Installation

```bash
uv pip install infnum
```

## Usage

```python
import torch
from infnum.torch_sparse import SparseLCTensor
from infnum.torch_autograd import ngrad

# Create a tensor with gradients
x = torch.tensor([1.0, -1.0, 0.0], requires_grad=True)

# Convert to Levi-Civita tensor
lc_x = SparseLCTensor.from_real(x)

# Apply discontinuous functions
y_abs = lc_x.abs()
y_step = lc_x.step()
y_round = lc_x.round()

# Extract standard part and compute gradients
y_abs.standard_part().backward()
print(x.grad)  # Shows correct derivatives even at discontinuities
```

## Performance

We provide benchmarks comparing our Levi-Civita implementation with standard PyTorch autograd:

- [Absolute Value Function](absolute_time.html)
- [Step Function](step_time.html)
- [Round Function](round_time.html)

The benchmarks show that our implementation maintains reasonable computational overhead while providing exact derivatives at discontinuities.

## How It Works

The Levi-Civita field extends the real numbers with infinitesimals that can detect and properly handle discontinuities. For a discontinuous function f, we:

1. Convert input x to x + εy
2. Evaluate f(x + εy) = f(x) + εyf'(x)
3. Extract f'(x) from the coefficient of ε

This gives us the correct derivative even at points of discontinuity.

## Implementation Details

We use a sparse CSR-like format to efficiently represent Levi-Civita numbers:

- values_exps: Integer tensor storing scaled exponents
- values_coeffs: Tensor storing coefficients
- row_ptr: Integer tensor for batch segmentation

This representation enables efficient batch processing and minimal memory overhead.

## Development

To set up the development environment:

```bash
# Clone the repository
git clone https://github.com/alok/infnum.git
cd infnum

# Install dependencies (including dev extras)
uv sync --extra dev

# Run tests
just test  # or `just test -- -k abs_function` to filter

# Run full benchmark suite (generates HTML plots under `benchmark_results/`)
just bench

# Quick, lightweight benchmark (smaller batch sizes)
just bench:quick
```

## Citation

If you use this package in your research, please cite:

```bibtex
@article{singh2025differentiating,
  title={Differentiating Through Discontinuities: A PyTorch Implementation of Levi-Civita Fields},
  author={Singh, Alok},
  journal={arXiv preprint arXiv:2025.xxxxx},
  year={2025}
}
```

## License

MIT License
