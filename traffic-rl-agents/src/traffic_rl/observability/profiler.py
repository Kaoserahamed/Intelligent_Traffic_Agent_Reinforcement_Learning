"""Lightweight performance instrumentation (no external dependencies).

Used by the trainer to answer the questions that actually matter in production:
where does wall time go (SUMO stepping vs. inference vs. optimisation), and how
many environment steps per second are we getting on this instance type.
"""

from __future__ import annotations

import functools
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Iterator, Mapping

F = TypeVar("F", bound="Callable[..., Any]")


@dataclass
class PhaseStats:
    """Accumulated timings for one named phase."""

    name: str
    count: int = 0
    total_s: float = 0.0
    samples: list[float] = field(default_factory=list)

    @property
    def mean_s(self) -> float:
        return self.total_s / self.count if self.count else 0.0

    @property
    def mean_ms(self) -> float:
        return self.mean_s * 1000.0

    @property
    def p95_s(self) -> float:
        if not self.samples:
            return 0.0
        ordered = sorted(self.samples)
        index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
        return ordered[index]

    @property
    def std_s(self) -> float:
        return statistics.pstdev(self.samples) if len(self.samples) > 1 else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "count": float(self.count),
            "total_s": round(self.total_s, 6),
            "mean_ms": round(self.mean_ms, 4),
            "p95_ms": round(self.p95_s * 1000.0, 4),
            "std_ms": round(self.std_s * 1000.0, 4),
        }


class PhaseTimer:
    """Collect per-phase timings across an entire run.

    >>> timer = PhaseTimer()
    >>> with timer.phase("sumo.step"):
    ...     pass
    >>> timer.report()["sumo.step"]["count"]
    1.0
    """

    def __init__(self, keep_samples: bool = True) -> None:
        self.keep_samples = keep_samples
        self._phases: dict[str, PhaseStats] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[PhaseStats]:
        """Time the enclosed block under ``name``."""
        stats = self._phases.setdefault(name, PhaseStats(name))
        start = time.perf_counter()
        try:
            yield stats
        finally:
            elapsed = time.perf_counter() - start
            stats.count += 1
            stats.total_s += elapsed
            if self.keep_samples:
                stats.samples.append(elapsed)

    def timed(self, name: str) -> Callable[[F], F]:
        """Decorator form of :meth:`phase`."""

        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.phase(name):
                    return func(*args, **kwargs)

            return wrapper  # type: ignore[return-value]

        return decorator

    @property
    def phases(self) -> Mapping[str, PhaseStats]:
        return dict(self._phases)

    def report(self) -> dict[str, dict[str, float]]:
        """Return ``{phase: stats}`` sorted by total time (descending)."""
        return {
            name: stats.as_dict()
            for name, stats in sorted(
                self._phases.items(), key=lambda item: item[1].total_s, reverse=True
            )
        }

    def reset(self) -> None:
        self._phases.clear()

    def summary_line(self, steps: int | None = None) -> str:
        """One-line human summary suitable for a log record."""
        total = sum(s.total_s for s in self._phases.values())
        parts = [f"{name}={s.total_s:.1f}s" for name, s in self._phases.items()]
        extra = f" | {steps / total:.1f} steps/s" if steps and total > 0 else ""
        return f"total={total:.1f}s | " + " ".join(parts) + extra


class StepRateMeter:
    """Track throughput (environment steps per second) with a sliding window."""

    def __init__(self, window: int = 100) -> None:
        self.window = window
        self._times: list[float] = []
        self.steps = 0
        self.started = time.perf_counter()

    def tick(self, n: int = 1) -> None:
        self.steps += n
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self.window:
            del self._times[: len(self._times) - self.window]

    @property
    def instantaneous(self) -> float:
        """Steps/second over the sliding window (0.0 until enough samples)."""
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0

    @property
    def average(self) -> float:
        span = time.perf_counter() - self.started
        return self.steps / span if span > 0 else 0.0

    def snapshot(self) -> dict[str, float]:
        return {
            "global_step": float(self.steps),
            "steps_per_second": round(self.instantaneous, 3),
            "steps_per_second_avg": round(self.average, 3),
            "wall_time_s": round(time.perf_counter() - self.started, 3),
        }


def process_rss_mb() -> float:
    """Return the resident set size of this process in MiB (best effort).

    Uses the standard library only (``resource`` on POSIX, the Windows API via
    ``ctypes``) and returns ``0.0`` when unavailable, so callers can log it
    unconditionally without platform checks.
    """
    try:
        import resource  # type: ignore[import-not-found]

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return usage / 1024.0  # Linux reports KiB
    except Exception:
        pass
    try:  # pragma: no cover - Windows only
        import ctypes
        from ctypes import wintypes

        class _ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [  # noqa: RUF012 - ctypes requires this exact class attribute
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        ):
            return counters.WorkingSetSize / (1024.0 * 1024.0)
    except Exception:
        pass
    return 0.0
