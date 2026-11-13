"""Self-measurement for stages: wall time and peak memory, recorded on every run.

Memory is the peak resident set size (RSS) of the worker process during the stage, sampled by a
background thread. It is not traced per allocation: tracemalloc hooks every allocation and made
numpy-heavy stages such as camera fitting about three times slower. RSS includes the worker's
baseline (interpreter, loaded models), which is what sizing a machine needs. A spike shorter
than the sampling interval can be missed; memory still held when the stage ends is not.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import psutil

from proof3d_pipeline.models import StageMetrics

SAMPLE_INTERVAL_S = 0.05


@dataclass
class _Meter:
    wall_s: float = 0.0
    peak_mem_mb: float = 0.0
    gpu_s: float = 0.0

    def metrics(self, cost_usd: float | None = None) -> StageMetrics:
        return StageMetrics(
            wall_s=self.wall_s, gpu_s=self.gpu_s, peak_mem_mb=self.peak_mem_mb, cost_usd=cost_usd
        )


class _PeakRss(threading.Thread):
    def __init__(self, interval_s: float) -> None:
        super().__init__(name="peak-rss", daemon=True)
        self._process = psutil.Process()
        self._interval_s = interval_s
        self._done = threading.Event()
        self.peak = self._rss()

    def _rss(self) -> int:
        rss: int = self._process.memory_info().rss
        return rss

    def run(self) -> None:
        while not self._done.wait(self._interval_s):
            self.peak = max(self.peak, self._rss())

    def finish(self) -> int:
        self._done.set()
        self.join()
        self.peak = max(self.peak, self._rss())
        return self.peak


@contextmanager
def measure(on_gpu: bool = False, interval_s: float = SAMPLE_INTERVAL_S) -> Iterator[_Meter]:
    """Time a block and record the process's peak memory during it. On a GPU stage the whole
    wall time is billed as GPU-seconds, because the rented GPU is held for the duration."""
    meter = _Meter()
    sampler = _PeakRss(interval_s)
    sampler.start()
    t0 = time.perf_counter()
    try:
        yield meter
    finally:
        meter.wall_s = time.perf_counter() - t0
        meter.peak_mem_mb = sampler.finish() / 2**20
        if on_gpu:
            meter.gpu_s = meter.wall_s
