import logging
import resource
import subprocess
import sys

import numpy as np
import pytest

from hmrec.memory import (
    BYTES_PER_GB,
    MemoryTracker,
    ProcessPeakMemory,
    bytes_to_gb,
    get_current_rss_bytes,
    log_memory,
    log_process_peak_memory,
    read_process_peak_memory,
    ru_maxrss_to_bytes,
)

# Outside the "hmrec" namespace on purpose: setup_logging() turns off propagation
# for "hmrec", and pytest's caplog only sees records that reach the root logger.
LOGGER = logging.getLogger("test_memory")
# 64 MB of float32 values. Big enough to show up clearly in RSS, small enough to be fast.
ALLOCATION_ELEMENTS = 16 * 1024 * 1024
ALLOCATION_BYTES = ALLOCATION_ELEMENTS * 4


def test_bytes_to_gb() -> None:
    assert bytes_to_gb(BYTES_PER_GB) == 1.0
    assert bytes_to_gb(BYTES_PER_GB // 2) == 0.5


def test_current_rss_is_positive() -> None:
    assert get_current_rss_bytes() > 0


def test_log_memory_logs_and_returns_rss(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=LOGGER.name):
        rss = log_memory(LOGGER, "checkpoint")
    assert rss > 0
    assert "memory | checkpoint | rss=" in caplog.text


def test_tracker_sees_allocation_inside_block(caplog: pytest.LogCaptureFixture) -> None:
    with (
        caplog.at_level(logging.INFO, logger=LOGGER.name),
        MemoryTracker("alloc", LOGGER, limit_gb=100.0, sample_interval_seconds=0.01) as tracker,
    ):
        # np.ones writes every element, so the pages are really resident.
        block = np.ones(ALLOCATION_ELEMENTS, dtype=np.float32)
        assert block.nbytes == ALLOCATION_BYTES

    # The end reading is taken while `block` is still alive, so the peak must
    # include most of the 64 MB. We allow slack for allocator behaviour.
    assert tracker.peak_rss_bytes >= tracker.start_rss_bytes + ALLOCATION_BYTES // 2
    assert tracker.peak_rss_bytes >= tracker.end_rss_bytes
    assert tracker.elapsed_seconds > 0
    assert "memory | alloc | start rss=" in caplog.text
    assert "peak rss=" in caplog.text
    # The true peak of the whole process can never be below what the sampler saw
    # (a few pages of slack: the two numbers are read by different system calls).
    assert tracker.process_max_rss_bytes >= tracker.peak_rss_bytes - 1024 * 1024
    # Both peaks are in the summary line, under different names.
    summary = next(r.getMessage() for r in caplog.records if "end rss=" in r.getMessage())
    assert "peak rss=" in summary and "(sampled)" in summary and "process max rss=" in summary
    del block


def test_tracker_warns_above_limit(caplog: pytest.LogCaptureFixture) -> None:
    tiny_limit_gb = 1e-6  # any real process is above ~1 KB
    with (
        caplog.at_level(logging.INFO, logger=LOGGER.name),
        MemoryTracker("tiny", LOGGER, limit_gb=tiny_limit_gb, sample_interval_seconds=0.01),
    ):
        pass
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "above the limit" in warnings[0].getMessage()


def test_tracker_does_not_swallow_exceptions() -> None:
    with (
        pytest.raises(RuntimeError, match="boom"),
        MemoryTracker("err", LOGGER, limit_gb=100.0, sample_interval_seconds=0.01),
    ):
        raise RuntimeError("boom")


@pytest.mark.parametrize(("limit_gb", "interval"), [(0.0, 0.1), (1.0, 0.0)])
def test_tracker_rejects_bad_settings(limit_gb: float, interval: float) -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        MemoryTracker("bad", LOGGER, limit_gb=limit_gb, sample_interval_seconds=interval)


# ---------------------------------------------------------------- true peak (ru_maxrss)


def test_ru_maxrss_unit_is_bytes_on_macos_and_kilobytes_elsewhere() -> None:
    # macOS reports bytes: 3 GB stays 3 GB.
    assert ru_maxrss_to_bytes(3 * BYTES_PER_GB, "darwin") == 3 * BYTES_PER_GB
    # Linux reports kilobytes of 1024 bytes: 3 GB is 3 * 1024 * 1024 KB.
    assert ru_maxrss_to_bytes(3 * 1024 * 1024, "linux") == 3 * BYTES_PER_GB
    assert ru_maxrss_to_bytes(0, "linux") == 0
    with pytest.raises(ValueError, match="must be >= 0"):
        ru_maxrss_to_bytes(-1, "darwin")


def test_process_peak_uses_the_unit_of_this_platform() -> None:
    # The same raw counter, converted with THIS platform's unit, must be at least the
    # current RSS: a factor 1024 too small (KB read as bytes) would fail here, and a
    # factor 1024 too large (bytes read as KB) would give hundreds of GB.
    peak = read_process_peak_memory()
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert peak.process_max_rss_bytes == ru_maxrss_to_bytes(raw, sys.platform)
    assert peak.process_max_rss_bytes >= get_current_rss_bytes() // 2
    assert peak.process_max_rss_gb < 64.0  # this machine cannot hold more


def test_process_peak_never_decreases_and_sees_a_released_allocation() -> None:
    before = read_process_peak_memory().process_max_rss_bytes
    # 4x the allocation of the tracker test, so it clearly exceeds anything this
    # test process has held so far; it is released before the next reading.
    block = np.ones(4 * ALLOCATION_ELEMENTS, dtype=np.float32)
    held = get_current_rss_bytes()
    del block
    after = read_process_peak_memory().process_max_rss_bytes
    assert after >= before  # the counter cannot go down
    # The memory is free again, but the true peak still remembers it. This is what
    # the sampled peak cannot promise for a short spike.
    assert after >= held - 1024 * 1024


def test_children_peak_is_separate_from_this_process() -> None:
    # A child that holds about 200 MB and exits. Its memory is never part of this
    # process's RSS, so it must show up in the children number instead.
    child_bytes = 200 * 1024 * 1024
    code = f"block = bytearray({child_bytes}); block[::4096] = bytes(len(block[::4096]))"
    subprocess.run([sys.executable, "-c", code], check=True)
    peak = read_process_peak_memory()
    assert peak.children_max_rss_bytes >= child_bytes
    assert peak.children_max_rss_gb == bytes_to_gb(peak.children_max_rss_bytes)


def test_log_process_peak_names_the_true_peak_and_the_children(
    caplog: pytest.LogCaptureFixture,
) -> None:
    subprocess.run([sys.executable, "-c", "pass"], check=True)  # make sure a child has finished
    with caplog.at_level(logging.INFO, logger=LOGGER.name):
        peak = log_process_peak_memory(LOGGER, "run", limit_gb=100.0)
    assert isinstance(peak, ProcessPeakMemory)
    messages = [record.getMessage() for record in caplog.records]
    assert any("memory | run | process max rss=" in message for message in messages)
    assert any("memory | run | children max rss=" in message for message in messages)
    assert not [record for record in caplog.records if record.levelno == logging.WARNING]


def test_log_process_peak_warns_above_the_limit(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=LOGGER.name):
        log_process_peak_memory(LOGGER, "run", limit_gb=1e-6)
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "process max rss=" in warnings[0].getMessage()
    assert "above the limit" in warnings[0].getMessage()
