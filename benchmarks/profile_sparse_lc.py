from __future__ import annotations

"""Deep profiling for core `SparseLCTensor` operations.

This script constructs a synthetic workload that stresses the most
performance-critical kernels (addition, multiplication, ``abs``,
`standard_part`) and emits a *cProfile* statistics file.

Usage (from project root)
-------------------------

    $ just profile

The command streams a human-readable table to *stdout* **and** dumps the
profiling database to ``profile_sparse_lc.prof`` which can be inspected
with GUI tools such as *snakeviz* or *tuna*:

    $ snakeviz profile_sparse_lc.prof

The CLI is powered by *tyro* – run with ``--help`` for all options.
"""

import cProfile
import io
import pstats
from pathlib import Path
from time import perf_counter

import torch
import tyro  # type: ignore – Third-party CLI library (preferred over argparse)

from infnum.torch_sparse import SparseLCTensor


# -----------------------------------------------------------------------------
# Synthetic workload -----------------------------------------------------------
# -----------------------------------------------------------------------------

def _workload(batch_size: int, depth: int, device: torch.device) -> None:
    """Compute-bound micro-benchmark exercising the LC kernels."""

    # Base real tensors -------------------------------------------------------
    x = torch.randn(batch_size, device=device)
    y = torch.randn(batch_size, device=device)

    # Lift to Levi-Civita numbers (x + ε) -------------------------------------
    xl = SparseLCTensor.from_real(x, order=1)
    yl = SparseLCTensor.from_real(y, order=1)

    z = xl
    for _ in range(depth):
        # Mix of +, *, |·| that appear in real workloads ----------------------
        z = (z * yl + z).abs()

    # Force use of the result so JIT / DCE cannot prune the loop --------------
    _ = z.standard_part()


# -----------------------------------------------------------------------------
# Command-line interface -------------------------------------------------------
# -----------------------------------------------------------------------------

class Config(tyro.conf.FlagConversionOff):  # type: ignore[misc]
    """Parameters for the profiler."""

    batch_size: int = 2 ** 14  # 16k elements – balances speed vs resolution
    depth: int = 32            # Repetition count of the LC block
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Output ``.prof`` file (cProfile binary format) --------------------------
    output: Path = Path("profile_sparse_lc.prof")


def main(cfg: Config) -> None:  # noqa: D401 – CLI entry-point
    device = torch.device(cfg.device)

    print(f"[profile] device={device}, batch={cfg.batch_size}, depth={cfg.depth}")

    pr = cProfile.Profile()
    pr.enable()

    t0 = perf_counter()
    _workload(cfg.batch_size, cfg.depth, device)
    t1 = perf_counter()

    pr.disable()

    # 1. Human-readable summary (top-20 by cumulative time) -------------------
    sio = io.StringIO()
    stats = pstats.Stats(pr, stream=sio).sort_stats("cumulative")
    stats.print_stats(20)
    print(sio.getvalue())

    # 2. Dump raw stats for post-mortem inspection ----------------------------
    pr.dump_stats(cfg.output)
    print(f"[profile] written {cfg.output.relative_to(Path.cwd())}")
    print(f"[profile] wall-clock: {t1 - t0:.3f} s")


if __name__ == "__main__":
    tyro.cli(main) 