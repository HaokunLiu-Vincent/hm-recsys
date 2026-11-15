import logging

import numpy as np
import pytest

from hmrec.memory import (
    BYTES_PER_GB,
    MemoryTracker,
    bytes_to_gb,
    get_current_rss_bytes,
    log_memory,
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
