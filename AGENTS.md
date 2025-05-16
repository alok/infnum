# Non-Standard Autograd Agent (Levi-CivitaTensor ↔ torch.autograd)

## Motivation
Conventional autodiff fails for discontinuous functions (e.g. `step`, `sign`,
`round`).  Our Levi-Civita field sidesteps the problem: evaluating

```python
df_dx = (f(x + ε) - f(x).standard_part) / ε
```
extracts the *non-standard derivative* that coincides with the distributional
/ Clarke generalized derivative.  We want this workflow to be as ergonomic as
regular `torch.autograd.grad`.

## High-Level Design
1. **LeviCivitaScalar** – thin wrapper around a 0-dim tensor that *also* stores an
   exponent; essentially one term of a `LeviCivitaTensor`.  It will register a
   custom `torch.autograd.Function` so PyTorch can back-prop through the scalar
   coefficient *and* propagate the exponent algebraically.

2. **LeviCivitaTensor** becomes a `typing.Sequence[LeviCivitaScalar]` under the
   hood.  All current algebra stays, but exponent arithmetic delegates to
   vectorised ops over the scalar list.

3. **NonStandardGrad** helper
   ```python
   def ngrad(f, x):
       lc_x = LeviCivitaTensor.from_real(x, order=1)  # x + ε
       y = f(lc_x)
       # ε¹-coefficient equals ∂f/∂x (forward-mode)
       return y.coeff_eps1()
   ```
   Under the hood we call `torch.autograd.grad` on the *coefficients* of the
   ε¹ term.

4. **Integration points**
   • `tensor.nonstd()` – returns `x + ε` convenience.
   • Drop-in replacement for `torch.nn.Module`: override `forward` to accept
     Levi-CivitaTensor as first-class citizen.

5. **Testing matrix**
   | Function | Classical grad | Non-std grad (ours) |
   | -------- | -------------- | ------------------- |
   | `abs`    | undefined at 0 | sign distribution   |
   | `step`   | 0 everywhere   | Dirac delta at 0    |
   | `round`  | 0              | Dirac comb          |

## Efficient Representation & Storage Strategy

The dict-of-tensors prototype is easy to reason about but does **not** scale.  We will migrate to a *sorted-COO* representation which keeps two flat tensors per batch:

```
values_exps   : IntTensor   # scaled-integer exponents  (total_nnz,)
values_coeffs : Tensor      # scalar coefficients      (total_nnz,)
row_ptr       : IntTensor   # CSR segment boundaries   (batch + 1,)
```

Key design decisions
--------------------
1. **Exponent encoding** – choose a global denominator `D` (default 16). Every
   rational exponent e is stored as `k = int(round(e * D))`. If a future
   operation needs finer resolution, we *double* `D` and rescale all indices –
   amortised O(N).
2. **Addition** – merge two sorted exponent arrays (`n≈8` per value):
   `torch.cat` + `scatter_add` on GPU ⇒ `O(n₁+n₂)`.
3. **Multiplication** – outer-broadcast exponents & coefficients, flatten, then
   reduce duplicates via `torch.unique_consecutive` and `scatter_add`.
4. **Batching** – `row_ptr` CSR layout makes every LC number just a slice inside
   one big blob; avoids per-value tensor allocations.
5. **Kernels** – first implemented in plain Torch; swap to Triton/CPP CUDA once
   benchmarks show a win.

Why not `torch.sparse_coo_tensor`?
• requires *integer* indices per sparse tensor – high overhead; jagged CSR gives
  a single contiguous buffer for the whole batch.

Upcoming milestones
-------------------
Milestone tracker (✔ = done)
---------------------------
✔ `infnum/config.py` with `EXP_DENOM` and helper encoders.
✔ **CPU sparse struct** – CSR (`values_exps`, `values_coeffs`, `row_ptr`).
✔ **CPU kernels** – addition / multiplication with autograd tests.
✔ **Autograd integration** – `SparseLCTensor.with_grad`, ε¹-extraction.
✔ **Utility helpers** – `.standard_part()`, `SparseLCTensor.from_real`.
▢ **GPU addition kernel** (Triton prototype).
▢ **GPU multiplication kernel**.
▢ **High-level API polish** – `.to(device)`, `.clone()`, `vmap` semantics.
▢ **Experiments & benchmarks** – MNIST discontinuous activations, RL hard-argmax.
▢ **Docs & paper prep** – README/CHANGELOG refresh, Lean proof, `@main.tex` NeurIPS abstract.
▢ **Stretch** – CUDA C++ kernels, higher-order ε, visualisation tools.

This plan satisfies both autograd compatibility **and** the NeurIPS experiment
batches without blowing up memory.

## Implementation Stages
1. **MVP**: single-term LeviCivitaTensor (`{0: x, 1: 1}`) + autograd Function
   wrapping basic ops (add, mul, div, pow).  Works on CPU.
2. **Vectorized**: support arbitrary sparse terms, batch exponents via
   `torch.stack`.
3. **CUDA**: re-implement inner loops in Triton for speed.
4. **API polish**: `torch.func.vmap` compatibility, `.to(device)` semantics.
5. **Paper experiments**: discontinuous activations in MNIST classifier + RL
   hard-argmax.

## Deliverables Before NeurIPS Deadline
- `torch_nonstd.py` containing `LeviCivitaScalarAutograd` + helpers.
- Updated README with `ngrad` examples.
- Benchmarks comparing convergence against STE & soft-plus.
- Formal Lean equivalence of `ngrad` with distributional derivative for piecewise linear functions.

## Next Steps (v0.4.x Roadmap — 2024-03-28)

Below is the authoritative checklist distilled from our latest design review.  Each
item maps to a focused git commit (atomic, single-concern) and will be crossed
off once merged into `main`.

### A  Correctness / Type Hygiene (🔴 highest priority)
✔ Factory‐based `SparseLCTensor.with_grad` (no monkey-patch).
✔ `_create_with_grad` non-`None` guard + typing.
✔ `gradcheck` coverage for `abs`, `step`, `round`, compositions.
✔ Forward-mode `ngrad` for vectors (no finite diff).

### B  Feature Gaps
✔ Division / reciprocal implementation.
✔ `pow` with non-integer exponents.
✔ Forward-mode `ngrad` for vectors (no finite diff).
▢ Higher-order derivatives: retain ε-series up to arbitrary order.
▢ GPU validation + optional Triton kernels.

### C  Performance / Benchmarking
- [ ] Merge `benchmarks/benchmark_lc.py` into `benchmarks/discontinuous.py`.
- [ ] CPU memory tracking via `tracemalloc` when CUDA absent.
- [ ] Micro-benchmarks for `add`, `mul`, sparse merge.
- [ ] One summary HTML with plots for the paper.

### D  Documentation & Packaging
- [ ] Add `CHANGELOG.md` *Unreleased* section and bump version once A is green.
- [ ] `Justfile` with `just test  |  bench  |  docs  |  release` recipes.
- [ ] Publish Test-PyPI pre-release (`uv publish`).
- [ ] GitHub Actions workflow: lint → test → build wheel → render plots.

### E  NeurIPS Submission Material
- [ ] Insert timing/memory plots + quantitative table into `@main.tex`.
- [ ] Add related-work paragraph (JAX, STE, hyper-dual numbers).
- [ ] Prepare 2-page supplemental with code snippets.

> *Progress tracking*: mark each bullet `- [x]` when merged; CI will fail if an
> item is checked but its code diff is missing. 

✔ `gradcheck` coverage for `abs`, `step`, `round`, compositions.
✔ Remove stale *JAX prototype* from README. 

<!-- Newly proposed optimisation tasks (2025-04-27) ---------------------------------->

Low-hanging fruit (≤ 1 week each)
---------------------------------
* **`torch.compile` acceleration** – wrap the hot kernels (`abs`, `step`, `round`,
  `standard_part`) in a JIT guard and enable on-demand via the environment flag
  `INFNUM_COMPILE=1`.  Empirically yields *20–30 %* speed-up on M3-Max CPU.
* **Vectorised micro timers** – adopt `torch.utils.benchmark.Timer` and hook the
  JSON output into *pytest-benchmark* so perf regressions trigger CI failures.
* **`int32` exponents** – store `values_exps` as *int32* when
  `EXP_DENOM · max(|e|) < 2³¹`; halves memory footprint and improves cache hits.
* **FP32 default coefficients** – keep `values_coeffs` in *float32* by default
  while exposing a `dtype=` knob for scientific users that need float64.

Medium horizon (prototype → integration)
---------------------------------------
* **Dense NaN-boxing scalar** – fast path for *single-term* LC numbers: pack the
  exponent into the mantissa of a signalling NaN.  Early micro-bench shows
  ~2× speed on scalar workloads.  Requires custom `__torch_dispatch__` shim to
  fall back to sparse path for multi-term tensors.
* **Auto-dispatch dense↔sparse** – if a batch row has only one term keep it in
  the dense representation; upgrade to CSR on demand.

Stretch goals (paper-quality numbers)
------------------------------------
* **Triton CUDA kernels** for addition & multiplication (outer-product +
  reduction) – estimates indicate another *3–5×* on A100.
* **End-to-end compiled graph** – investigate `functorch.compile` once stable
  to fuse LC arithmetic inside larger PyTorch models. 