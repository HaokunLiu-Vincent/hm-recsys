"""Tests for hmrec.run_records: the row of one M6 process and the memory-limit stop."""

from pathlib import Path

import pytest

from hmrec.csv_records import read_rows
from hmrec.memory import BYTES_PER_GB, ProcessPeakMemory
from hmrec.run_records import (
    GENERATED,
    RUN_COLUMNS,
    RunStamp,
    StepResult,
    append_run_row,
    check_memory_limit,
    run_row,
)

STAMP = RunStamp("abc123", False)
RESULT = StepResult("F1", "r4_itemcf", GENERATED, 7_000_000, 25.5, 3.5, 0, "")


def _peak(process_gb: float, children_gb: float = 0.0) -> ProcessPeakMemory:
    return ProcessPeakMemory(int(process_gb * BYTES_PER_GB), int(children_gb * BYTES_PER_GB))


def test_row_keeps_sampled_and_true_peak_apart(tmp_path: Path) -> None:
    row = run_row(RESULT, total_seconds=30.25, peak=_peak(4.0, 0.5), stamp=STAMP)
    assert tuple(row) == RUN_COLUMNS
    # 3.5 GB was sampled during generate; the whole process peaked at 4 GB.
    assert (row["sampled_peak_rss_gb"], row["process_max_rss_gb"]) == ("3.5", "4.0")
    assert row["children_max_rss_gb"] == "0.5"
    assert (row["generate_seconds"], row["total_seconds"]) == ("25.5", "30.25")
    assert (row["fold"], row["step"], row["status"], row["n_rows"]) == (
        "F1",
        "r4_itemcf",
        "generated",
        "7000000",
    )
    path = tmp_path / "m6_runs.csv"
    append_run_row(path, row)
    append_run_row(path, row)
    assert len(read_rows(path, RUN_COLUMNS)) == 2


def test_unknown_status_is_refused() -> None:
    with pytest.raises(ValueError, match="status must be one of"):
        StepResult("F1", "r1_popularity", "done", 1, 0.0, 0.0, 0, "")


def test_memory_limit_uses_the_true_peak() -> None:
    check_memory_limit(_peak(6.0), limit_gb=6.0, label="run")  # exactly at the limit: allowed
    with pytest.raises(MemoryError, match=r"process max RSS 6\.50 GB is above the limit of 6\.00"):
        check_memory_limit(_peak(6.5), limit_gb=6.0, label="run")
