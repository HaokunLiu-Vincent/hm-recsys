"""Measure process memory and its peak (AGENTS.md §4, proposal §1.3).

The memory budget is 8 GB shared by CPU and GPU, and one task should stay
below about 6 GB (`memory.limit_gb` in configs/base.yaml). Long tasks log memory
at start and end with MemoryTracker:

    with MemoryTracker("r4_itemcf D1", logger, limit_gb=6.0, sample_interval_seconds=0.5):
        run_itemcf(...)

All numbers are RSS (resident set size). There are TWO kinds of peak, and they
answer different questions:

    sampled peak (MemoryTracker.peak_rss_bytes, psutil)
        The largest RSS a background thread saw while a `with` block ran. It can
        be measured per STAGE (one tracker per stage), so it shows WHERE memory
        is used. But a spike shorter than the sampling interval is missed, so it
        is only a LOWER BOUND of the stage's real peak.

    process max RSS (read_process_peak_memory, resource.getrusage -> ru_maxrss)
        The largest RSS the operating system recorded for this process since it
        STARTED. It is the TRUE peak (no spike is missed; `/usr/bin/time -l`
        prints the same counter), but there is only one number for the whole
        process: it cannot be reset, so it cannot say which stage caused it.

Use the process max RSS to decide whether a run stayed below the limit, and the
sampled peaks to find the stage that is responsible. Every MemoryTracker logs
both when its block ends, and a script calls log_process_peak_memory as its last
step so the run log (and the run's result record) holds the true peak of the run.

Child processes are not part of this process's RSS. Their peak is read
separately (RUSAGE_CHILDREN): the largest RSS of any single child that has
already finished and been waited for (it is a maximum, not a sum; 0 = none yet).

On Apple Silicon, MPS tensors live in unified memory; part of that memory may
not be counted in RSS, so treat GPU-heavy readings as a lower bound.
"""

import logging
import resource
import sys
import threading
import time
from dataclasses import dataclass
from types import TracebackType

import psutil

BYTES_PER_GB = 1024**3
BYTES_PER_KB = 1024


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


# ---------------------------------------------------------------- true peak (ru_maxrss)


@dataclass(frozen=True)
class ProcessPeakMemory:
    """The true peak RSS recorded by the operating system (see the module docstring).

    Attributes:
        process_max_rss_bytes: Largest RSS of THIS process since it started
            (RUSAGE_SELF). Never decreases.
        children_max_rss_bytes: Largest RSS of any single child process that
            has finished and been waited for (RUSAGE_CHILDREN); 0 if there is
            none yet. A maximum over the children, not their sum.
    """

    process_max_rss_bytes: int
    children_max_rss_bytes: int

    @property
    def process_max_rss_gb(self) -> float:
        """process_max_rss_bytes in GB (1 GB = 1024**3 bytes)."""
        return bytes_to_gb(self.process_max_rss_bytes)

    @property
    def children_max_rss_gb(self) -> float:
        """children_max_rss_bytes in GB (1 GB = 1024**3 bytes)."""
        return bytes_to_gb(self.children_max_rss_bytes)


def ru_maxrss_to_bytes(ru_maxrss: int, platform: str) -> int:
    """Convert a `ru_maxrss` value of resource.getrusage to bytes.

    The unit depends on the operating system: macOS reports BYTES, Linux (and
    the other Unix systems Python supports) reports KILOBYTES of 1024 bytes.
    Mixing them up would make a reading wrong by a factor of 1024.

    Args:
        ru_maxrss: The raw value from the operating system.
        platform: sys.platform of the system that produced it ("darwin" = macOS).
    """
    if ru_maxrss < 0:
        raise ValueError(f"ru_maxrss must be >= 0, got {ru_maxrss}")
    if platform == "darwin":
        return ru_maxrss
    return ru_maxrss * BYTES_PER_KB


def read_process_peak_memory() -> ProcessPeakMemory:
    """Read the true peak RSS of this process and of its finished children."""
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return ProcessPeakMemory(
        process_max_rss_bytes=ru_maxrss_to_bytes(own, sys.platform),
        children_max_rss_bytes=ru_maxrss_to_bytes(children, sys.platform),
    )


def log_process_peak_memory(
    logger: logging.Logger, label: str, limit_gb: float
) -> ProcessPeakMemory:
    """Log the true peak RSS of the run so far; call it as the LAST step of a script.

    Logs a WARNING if the process max RSS is above `limit_gb`. This is the real
    check against the memory budget: it cannot miss a short spike. Children are
    logged only if a child process has finished.

    Args:
        logger: Where to log.
        label: Name of the run, e.g. "m5 lock check r4_itemcf".
        limit_gb: Peak budget in GB (memory.limit_gb in configs/base.yaml).

    Returns:
        The reading, so the caller can store it in the run's result record.
    """
    peak = read_process_peak_memory()
    logger.info(
        "memory | %s | process max rss=%.2f GB (true peak since process start, ru_maxrss)",
        label,
        peak.process_max_rss_gb,
    )
    if peak.children_max_rss_bytes > 0:
        logger.info(
            "memory | %s | children max rss=%.2f GB (largest finished child process)",
            label,
            peak.children_max_rss_gb,
        )
    if peak.process_max_rss_gb > limit_gb:
        logger.warning(
            "memory | %s | process max rss=%.2f GB is above the limit of %.2f GB",
            label,
            peak.process_max_rss_gb,
            limit_gb,
        )
    return peak


# ---------------------------------------------------------------- sampled peak per block


class MemoryTracker:
    """Context manager that records RSS at start, at end, and the peak in between.

    A background thread samples RSS every `sample_interval_seconds`. The peak
    is the largest of all samples plus the start and end readings. Spikes
    shorter than the sample interval can be missed, so the peak is a lower bound.

    On exit it logs start / end / sampled peak and the elapsed time, and logs a
    WARNING if the sampled peak is above `limit_gb`. It never raises because of
    memory use. The same log line also shows the process max RSS (the true peak
    since the process started, see the module docstring): if that number rises
    between two stages' lines, the spike happened in the later stage, even when
    its sampled peak missed it.

    Attributes:
        start_rss_bytes: RSS when the block started.
        end_rss_bytes: RSS when the block ended.
        peak_rss_bytes: Largest SAMPLED RSS during the block (a lower bound).
        process_max_rss_bytes: True peak RSS of the whole process from its
            start until the block ended (not only this block).
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
        self.process_max_rss_bytes = 0
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
        self.process_max_rss_bytes = read_process_peak_memory().process_max_rss_bytes
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
        """Log end / sampled peak / process max, and warn if the sampled peak is above the limit."""
        peak_gb = bytes_to_gb(self.peak_rss_bytes)
        # "peak rss" is the sampled peak of THIS block (the wording of earlier run
        # logs is kept); "process max rss" is the true peak of the whole process so far.
        self.logger.info(
            "memory | %s | end rss=%.2f GB | peak rss=%.2f GB (sampled) | "
            "process max rss=%.2f GB | elapsed=%.1f s",
            self.label,
            bytes_to_gb(self.end_rss_bytes),
            peak_gb,
            bytes_to_gb(self.process_max_rss_bytes),
            self.elapsed_seconds,
        )
        if peak_gb > self.limit_gb:
            self.logger.warning(
                "memory | %s | peak rss=%.2f GB is above the limit of %.2f GB",
                self.label,
                peak_gb,
                self.limit_gb,
            )
