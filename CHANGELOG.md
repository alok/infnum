# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Forward-mode `ngrad` now supports *vector* inputs by extracting the ε¹-coefficient directly from the Levi-Civita expansion.
- **Benchmarking harness**: introduced a `Justfile` with a `bench` recipe and added `pytest-benchmark` to the development dependencies for lightweight micro-benchmarks.
- Two command aliases `just bench` and `just bench:quick` generate interactive HTML + PNG plots under `benchmark_results/` for rapid performance profiling.
- *Deep profiling*: new script `benchmarks/profile_sparse_lc.py` and a `just profile` recipe generate **cProfile** stats for the core `SparseLCTensor` kernels.
- *Neural-net benchmark*: `benchmarks/nn_benchmark_lc.py` plus `just bench:nn` measure forward-mode LC differentiation on a toy MLP and report speed-ups against reverse-mode autograd.
- *Micro-benchmarks*: new `benchmarks/micro_bench.py` and `just bench:micro` provide per-kernel timings via `torch.utils.benchmark`.

### Changed
- `torch_autograd.ngrad` no longer falls back to central finite differences when the callback is implemented with `SparseLCTensor` – leading to exact distributional derivatives for discontinuous functions.
- **Performance**
   1. `SparseLCTensor.standard_part` remains fully vectorised (no Python loops).
   2. NEW: `abs`, `step`, and `round` now ship *entirely* tensor-wise kernels, eliminating the old per-row Python loops.
      On an M3-Max laptop (CPU, PyTorch 2.2):
      • `abs`  – 50 k samples: **1.19 s → 0.50 s**  (~2.4 × faster)
      • `step` – 50 k samples: **≈1.2 s* → 0.52 s**  (~2.3 × faster)
      • `round` – 50 k samples: **0.98 s → 0.56 s** (~1.8 × faster)

      *baseline for `step` inferred from the previous `abs` timing as both
      shared the same loop-based implementation.  All timings include
      derivative extraction via `ngrad` and exclude I/O.*

### Fixed
- Ensured ε¹-coefficient extraction retains autograd connectivity by avoiding in-place writes.

## [0.4.0] - 2024-03-28

### Added
- Implemented step function for discontinuous derivatives
- Implemented round function for discontinuous derivatives
- Added benchmarking suite for comparing with standard autograd
- Added performance plots for absolute value, step, and round functions
- Added NeurIPS extended abstract

### Changed
- Improved gradient handling in SparseLCTensor
- Modified step and round functions to use torch.where for better gradient propagation
- Updated README with usage examples and benchmarks

### Fixed
- Fixed gradient propagation in discontinuous functions
- Fixed memory tracking in benchmarks for CPU-only systems

## [0.3.0] - 2024-03-27

### Added
- Initial implementation of Levi-Civita field
- Sparse representation using CSR format
- PyTorch autograd integration
- Absolute value function implementation

### Changed
- Switched to uv for package management
- Updated project structure for better organization

### Fixed
- Fixed memory leaks in tensor operations
- Fixed gradient computation in absolute value function 