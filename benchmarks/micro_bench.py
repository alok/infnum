from __future__ import annotations

"""Micro benchmarks for individual `SparseLCTensor` kernels.

This harness relies on *torch.utils.benchmark.Timer* which gives
 statistically sound timings (median, inter-quartile range) and integrates
 nicely with *pytest-benchmark* when executed via `pytest -q`.

Run standalone:

    $ just bench:micro         # prints ASCII table

Run inside pytest (records to history & compares):

    $ pytest -q tests/test_microbench.py
"""

from dataclasses import dataclass
from typing import List

import torch
import tyro  # type: ignore – project guideline
from torch.utils.benchmark import Timer

from infnum.torch_sparse import SparseLCTensor


# -----------------------------------------------------------------------------
# Benchmark entries ------------------------------------------------------------
# -----------------------------------------------------------------------------

@dataclass
class Entry:
    name: str
    stmt: str


def _setup(batch: int, device: torch.device) -> dict[str, object]:
    """Return a dict to be injected into Timer globals."""
    x = torch.randn(batch, device=device)
    y = torch.randn(batch, device=device)

    xl = SparseLCTensor.from_real(x, order=1)
    yl = SparseLCTensor.from_real(y, order=1)

    return dict(torch=torch, SparseLCTensor=SparseLCTensor, x=x, y=y, xl=xl, yl=yl)


BENCHES: List[Entry] = [
    Entry("from_real", "SparseLCTensor.from_real(x)"),
    Entry("standard_part", "xl.standard_part()"),
    Entry("abs", "xl.abs()"),
    Entry("add", "xl + yl"),
    Entry("mul", "xl * yl"),
]


@dataclass
class Config(tyro.conf.FlagConversionOff):  # type: ignore[misc]
    batch: int = 2 ** 12  # 4096 elements
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    repeats: int = 50


def main(cfg: Config) -> None:  # noqa: D401 – CLI entry
    device = torch.device(cfg.device)
    print(f"[micro] batch={cfg.batch}  device={device}  repeats={cfg.repeats}\n")

    setup_globals = _setup(cfg.batch, device)

    rows: List[tuple[str, float]] = []

    for entry in BENCHES:
        t = Timer(stmt=entry.stmt, globals=setup_globals, num_threads=torch.get_num_threads())
        median = t.timeit(cfg.repeats).median
        rows.append((entry.name, median))

    # Pretty print ------------------------------------------------------------
    name_w = max(len(r[0]) for r in rows)
    print("Operation".ljust(name_w), "|  median time (s)")
    print("-" * (name_w + 20))
    for name, med in rows:
        print(name.ljust(name_w), f"|  {med:9.6f}")


if __name__ == "__main__":
    tyro.cli(main) 