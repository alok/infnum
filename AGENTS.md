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
       return (y - y.standard_part) / ε
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