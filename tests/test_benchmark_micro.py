from __future__ import annotations

"""Micro-benchmarks for critical Levi-Civita operations.

These tests rely on the ``pytest-benchmark`` plugin and are **extremely** light –
inputs are tiny (≤1 k) so they do not slow down the regular CI pipeline.  They
provide a quick sanity-check when tuning kernels locally: run with

```
pytest infnum/tests/test_benchmark_micro.py --benchmark-only
```

which prints a summarised table.  The numbers are **not** asserted; they are
purely informative so the test always passes.
"""

import pytest
pytest.importorskip("pytest_benchmark")

import torch
from infnum.torch_sparse import SparseLCTensor
from infnum.torch_autograd import ngrad

BATCH_SIZE = 512  # keep very small for CI

def _sample(batch_size: int = BATCH_SIZE) -> SparseLCTensor:
    """Create a random Levi-Civita tensor on CPU."""
    x = torch.randn(batch_size, dtype=torch.float64, requires_grad=True)
    return SparseLCTensor.from_real(x)


def test_forward_addition(benchmark):
    """Benchmark addition of two LC tensors and extraction of the standard part."""
    a = _sample()
    b = _sample()

    def _run():
        c = a + b
        # materialise result to prevent lazy short-cuts
        _ = c.standard_part()

    benchmark(_run)


def test_ngrad_abs_fn(benchmark):
    """Benchmark `ngrad` on the absolute value function for a small batch."""

    def abs_fn(x):
        if isinstance(x, SparseLCTensor):
            return x.abs()
        return torch.abs(x)

    x = torch.linspace(-1, 1, BATCH_SIZE, dtype=torch.float64, requires_grad=True)

    benchmark(lambda: ngrad(abs_fn, x)) 