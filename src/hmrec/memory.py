"""Measure process memory and its peak (AGENTS.md §4, proposal §1.3).

The memory budget is 8 GB shared by CPU and GPU, and one task should stay
below about 6 GB (`memory.limit_gb` in configs/base.yaml). Long tasks log memory
at start and end with MemoryTracker:

    with MemoryTracker("r4_itemcf D1", logger, limit_gb=6.0, sample_interval_seconds=0.5):
        run_itemcf(...)

All numbers are RSS (resident set size) of the current process, from psutil.
On Apple Silicon, MPS tensors live in unified memory; part of that memory may
not be counted in RSS, so treat GPU-heavy readings as a lower bound.
"""

import logging
import threading
import time
from types import TracebackType

import psutil

BYTES_PER_GB = 1024**3


def bytes_to_gb(num_bytes: int) -> float:
    """Convert bytes to GB, where 1 GB = 1024**3 bytes."""
    return num_bytes / BYTES_PER_GB


def get_current_rss_bytes() -> int:
    """Return the current resident memory (RSS) of this process, in bytes."""
    rss: int = psutil.Process().memory_info().rss
    return rss


def log_memory(logger: logging.Logger, label: str) -> int:
    """Log the current RSS of this process and return it.

    Args:
        logger: Where to log.
        label: Short description of the current step, e.g. "after loading transactions".

    Returns:
        Current RSS in bytes.
    """
    rss_bytes = get_current_rss_bytes()
    logger.info("memory | %s | rss=%.2f GB", label, bytes_to_gb(rss_bytes))
    return rss_bytes


class MemoryTracker:
    """Context manager that records RSS at start, at end, and the peak in between.

    A background thread samples RSS every `sample_interval_seconds`. The peak
    is the largest of all samples plus the start and end readings. Spikes
    shorter than the sample interval can be missed, so the peak is a lower bound.

    On exit it logs start / end / peak and the elapsed time, and logs a WARNING
    if the peak is above `limit_gb`. It never raises because of memory use.

    Attributes:
        start_rss_bytes: RSS when the block started.
        end_rss_bytes: RSS when the block ended.
        peak_rss_bytes: Largest RSS seen during the block.
        elapsed_seconds: Wall-clock duration of the block.
    """

    def __init__(
        self,
        label: str,
        logger: logging.Logger,
        limit_gb: float,
        sample_interval_seconds: float,
    ) -> None:
        """Create a tracker. Nothing is measured until the `with` block starts.

        Args:
            label: Name of the task, used in log lines.
            logger: Where to log.
            limit_gb: Peak budget in GB; a warning is logged above it.
            sample_interval_seconds: Time between two RSS samples.
        """
        if limit_gb <= 0:
            raise ValueError(f"limit_gb must be > 0, got {limit_gb}")
        if sample_interval_seconds <= 0:
            raise ValueError(f"sample_interval_seconds must be > 0, got {sample_interval_seconds}")
        self.label = label
        self.logger = logger
        self.limit_gb = limit_gb
        self.sample_interval_seconds = sample_interval_seconds

        self.start_rss_bytes = 0
        self.end_rss_bytes = 0
        self.peak_rss_bytes = 0
        self.elapsed_seconds = 0.0

        self._start_time = 0.0
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._sampler: threading.Thread | None = None

    def __enter__(self) -> "MemoryTracker":
        """Take the first reading and start the sampling thread."""
        self._start_time = time.perf_counter()
        self.start_rss_bytes = get_current_rss_bytes()
        self.peak_rss_bytes = self.start_rss_bytes
        self.logger.info(
            "memory | %s | start rss=%.2f GB", self.label, bytes_to_gb(self.start_rss_bytes)
        )
        self._stop_event.clear()
        # daemon=True: if the main program crashes, this thread must not keep it alive.
        self._sampler = threading.Thread(target=self._sample_loop, daemon=True)
        self._sampler.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop sampling, take the final reading and log the summary.

        Exceptions from the `with` block are not suppressed.
        """
        # Read the end value BEFORE stopping, while the block's objects are still alive.
        self.end_rss_bytes = get_current_rss_bytes()
        self._update_peak(self.end_rss_bytes)
        self._stop_event.set()
        if self._sampler is not None:
            self._sampler.join()
        self.elapsed_seconds = time.perf_counter() - self._start_time
        self._log_summary()

    def _sample_loop(self) -> None:
        """Sample RSS until the stop event is set."""
        # Event.wait returns False on timeout (keep sampling) and True once stopped.
        while not self._stop_event.wait(self.sample_interval_seconds):
            self._update_peak(get_current_rss_bytes())

    def _update_peak(self, rss_bytes: int) -> None:
        """Raise the stored peak if `rss_bytes` is larger."""
        with self._lock:
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss_bytes)

    def _log_summary(self) -> None:
        """Log start / end / peak, and warn if the peak is above the limit."""
        peak_gb = bytes_to_gb(self.peak_rss_bytes)
        self.logger.info(
            "memory | %s | end rss=%.2f GB | peak rss=%.2f GB | elapsed=%.1f s",
            self.label,
            bytes_to_gb(self.end_rss_bytes),
            peak_gb,
            self.elapsed_seconds,
        )
        if peak_gb > self.limit_gb:
            self.logger.warning(
                "memory | %s | peak rss=%.2f GB is above the limit of %.2f GB",
                self.label,
                peak_gb,
                self.limit_gb,
            )
