"""Self-measurement for stages: wall time and peak Python heap, recorded on every run."""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from proof3d_pipeline.models import StageMetrics


@dataclass
class _Meter:
    wall_s: float = 0.0
    peak_mem_mb: float = 0.0
    gpu_s: float = 0.0

    def metrics(self, cost_usd: float | None = None) -> StageMetrics:
        return StageMetrics(
            wall_s=self.wall_s, gpu_s=self.gpu_s, peak_mem_mb=self.peak_mem_mb, cost_usd=cost_usd
        )


@contextmanager
def measure(on_gpu: bool = False) -> Iterator[_Meter]:
    """Time a block. On a GPU stage the whole wall time is billed as GPU-seconds,
    because the rented GPU is held for the duration either way."""
    meter = _Meter()
    started_tracing = not tracemalloc.is_tracing()
    if started_tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()
    t0 = time.perf_counter()
    try:
        yield meter
    finally:
        meter.wall_s = time.perf_counter() - t0
        meter.peak_mem_mb = tracemalloc.get_traced_memory()[1] / 2**20
        if started_tracing:
            tracemalloc.stop()
        if on_gpu:
            meter.gpu_s = meter.wall_s
