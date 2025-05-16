from __future__ import annotations

"""Benchmark a *tiny* one-hidden-layer neural network with and without
Levi-Civita numbers.

The goal is *not* state-of-the-art accuracy but to exercise typical
operations (affine transform + ReLU) that appear in modern ML models and
profile their execution through the *forward-mode* LC machinery.

The script measures *wall-clock* time (CPU / CUDA) for the **gradient of
network output with respect to the inputs** computed via

1.   standard reverse-mode *torch.autograd*
2.   forward-mode using :pyfunc:`infnum.torch_autograd.ngrad`

Results are printed as a table and optionally saved to *CSV* for further
analysis.

Run via

    $ just bench:nn

or directly

    $ python benchmarks/nn_benchmark_lc.py --help
"""

import csv
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import List, Tuple

import torch
import tyro  # type: ignore  # project guideline – prefer tyro over argparse

from infnum.torch_sparse import SparseLCTensor
from infnum.torch_autograd import ngrad

# -----------------------------------------------------------------------------
# Model definition -------------------------------------------------------------
# -----------------------------------------------------------------------------


def relu_lc(x: SparseLCTensor) -> SparseLCTensor:  # noqa: D401 – helper
    """ReLU implemented via Heaviside mask (works for LC numbers)."""
    return x * x.step()


def mlp_torch(x: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:  # noqa: D401
    """Scalar MLP:   y = w2 * ReLU(w1 * x + b1) + b2"""
    return w2 * torch.relu(w1 * x + b1) + b2


def mlp_lc(x: SparseLCTensor, w1: SparseLCTensor, b1: SparseLCTensor, w2: SparseLCTensor, b2: SparseLCTensor) -> SparseLCTensor:  # noqa: D401
    """LC-aware MLP with the *same* algebraic form as :pyfunc:`mlp_torch`."""
    return w2 * relu_lc(w1 * x + b1) + b2


# -----------------------------------------------------------------------------
# Benchmark harness ------------------------------------------------------------
# -----------------------------------------------------------------------------


def _to_lc_scalar(val: torch.Tensor, batch: int) -> SparseLCTensor:
    """Broadcast a scalar weight to LC with matching *batch* length."""
    assert val.numel() == 1, "Only scalar weights supported for now"
    rep = val.expand(batch).clone()
    return SparseLCTensor.from_real(rep)


class Mode(str, Enum):
    torch = "torch"
    lc = "lc"


class Config(tyro.conf.FlagConversionOff):  # type: ignore[misc]
    """CLI parameters."""

    batch_sizes: List[int] = [2 ** k for k in range(8, 15)]  # 256 … 16384
    n_runs: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_csv: Path | None = None  # write raw results when set


# Each tuple: (batch_size, torch_time, lc_time) -------------------------------
Result = Tuple[int, float, float]


def _bench_once(batch_size: int, device: torch.device) -> Tuple[float, float]:
    """Return (torch_time, lc_time)"""

    # Random **scalar** parameters -------------------------------------------
    rng = torch.Generator(device=device)
    w1 = torch.randn(1, generator=rng, device=device)
    b1 = torch.randn(1, generator=rng, device=device)
    w2 = torch.randn(1, generator=rng, device=device)
    b2 = torch.randn(1, generator=rng, device=device)

    # ------------------------------------------------------------------
    # 1. Reverse-mode benchmark ----------------------------------------
    # ------------------------------------------------------------------
    t0 = perf_counter()
    x = torch.randn(batch_size, device=device, requires_grad=True)
    y = mlp_torch(x, w1, b1, w2, b2)
    y.sum().backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = perf_counter()
    torch_time = t1 - t0

    # ------------------------------------------------------------------
    # 2. Forward-mode via Levi-Civita ----------------------------------
    # ------------------------------------------------------------------
    # Lift *all* scalars to LC so subsequent ops stay in LC domain
    wl1 = _to_lc_scalar(w1, batch_size)
    bl1 = _to_lc_scalar(b1, batch_size)
    wl2 = _to_lc_scalar(w2, batch_size)
    bl2 = _to_lc_scalar(b2, batch_size)

    def _f(z: SparseLCTensor):  # noqa: D401 – local helper
        return mlp_lc(z, wl1, bl1, wl2, bl2)

    t2 = perf_counter()
    z = torch.randn(batch_size, device=device)
    grad = ngrad(_f, z)
    _ = grad.sum().item()  # force realisation / copy-back for fair timing
    if device.type == "cuda":
        torch.cuda.synchronize()
    t3 = perf_counter()
    lc_time = t3 - t2

    return torch_time, lc_time


def main(cfg: Config) -> None:  # noqa: D401 – CLI entry
    device = torch.device(cfg.device)
    print(f"[nn bench] device={device}")

    results: List[Result] = []

    for batch in cfg.batch_sizes:
        torch_t = lc_t = 0.0
        for _ in range(cfg.n_runs):
            t, l = _bench_once(batch, device)
            torch_t += t
            lc_t += l
        torch_t /= cfg.n_runs
        lc_t /= cfg.n_runs
        results.append((batch, torch_t, lc_t))
        print(f"  batch={batch:6d} | torch={torch_t:8.4f}s | lc={lc_t:8.4f}s | ×speedup={torch_t / lc_t:5.2f}")

    # Optional CSV output ------------------------------------------------------
    if cfg.out_csv is not None:
        cfg.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with cfg.out_csv.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["batch", "torch_time", "lc_time"])
            writer.writerows(results)
        print(f"[nn bench] wrote raw timings → {cfg.out_csv.relative_to(Path.cwd())}")


if __name__ == "__main__":
    tyro.cli(main) 