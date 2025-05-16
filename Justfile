default:
    just --list

# Run the complete test-suite
# Usage: just test
# Optional arguments after `--` are forwarded to pytest.
# Example: just test -- -k "abs_function" -q
#
# The quiet flag reduces pytest noise so benchmarking output is more readable.
test *ARGS:
    pytest -q {{ARGS}}

# Execute the primary benchmark harness which compares
# forward-mode Levi-Civita differentiation with standard
# PyTorch ``autograd``.  Generates interactive HTML plots
# under ``benchmark_results/``.
bench:
    python benchmarks/benchmark_lc.py

# Quick, lightweight benchmarks (smaller batch-sizes)
bench:quick:
    python benchmarks/discontinuous.py

# Synchronise the development environment (installs `[dev]` extras)
sync-dev:
    uv sync --extra dev

profile:
    python benchmarks/profile_sparse_lc.py

bench:nn:
    python benchmarks/nn_benchmark_lc.py

bench:micro:
    python benchmarks/micro_bench.py
