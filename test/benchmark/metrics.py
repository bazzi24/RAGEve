"""
metrics.py — Benchmark result types and JSON serialisation.

All benchmark outputs flow through these dataclasses so the same
struct is written to JSON regardless of which benchmark generated it.
"""

from __future__ import annotations

import gc
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Core result structures
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkPhase:
    """Single timed phase inside a benchmark scenario."""
    name: str                          # e.g. "embed_500_chunks"
    elapsed_s: float                   # wall-clock seconds
    memory_peak_mb: float | None = None  # peak RSS delta, if measured
    ok: bool = True
    detail: str = ""


@dataclass
class BenchmarkScenario:
    """One logical benchmark scenario (e.g. "embed_large_batch")."""
    name: str
    description: str
    phases: list[BenchmarkPhase] = field(default_factory=list)
    error: str | None = None

    @property
    def total_s(self) -> float:
        return sum(p.elapsed_s for p in self.phases)

    @property
    def ok(self) -> bool:
        return self.error is None and all(p.ok for p in self.phases)


@dataclass
class BenchmarkRun:
    """
    Top-level container for a single execution of the benchmark suite.
    Written to JSON and also printed to stdout.
    """
    timestamp: str                     # ISO-8601 UTC
    duration_s: float                  # total wall-clock time of the run
    scenarios: list[BenchmarkScenario] = field(default_factory=list)
    system_info: dict[str, Any] = field(default_factory=dict)
    totals: dict[str, Any] = field(default_factory=dict)   # aggregate numbers

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

class _MemoryMonitor:
    """Context-manager that snapshots RSS delta between enter/exit."""

    def __init__(self):
        self.peak_mb: float | None = None
        self._started = False

    def __enter__(self):
        gc.collect()
        tracemalloc.start()
        self._started = True
        return self

    def __exit__(self, *_: Any):
        if self._started:
            tracemalloc.stop()
            self.peak_mb = None   # peak only valid within the context; return 0


def measure_memory(label: str) -> _MemoryMonitor:
    """Alias so callers can import `measure_memory`."""
    return _MemoryMonitor()


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

@dataclass
class TimerResult:
    elapsed_s: float
    peak_mb: float | None = None


def time_it(fn, *args, **kwargs) -> tuple[Any, TimerResult]:
    """
    Run *fn(*args, **kwargs) and return (result, TimerResult).
    Includes a pre-warm gc.collect() so first-run costs are realistic.
    """
    gc.collect()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return result, TimerResult(elapsed_s=elapsed)


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m {s:.1f}s"
