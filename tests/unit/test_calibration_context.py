"""Tests for hmrec.calibration.context: how a calibration process starts and ends."""

import csv
from dataclasses import replace
from pathlib import Path

import pytest

import hmrec.calibration.context as context_module
from hmrec.calibration.context import (
    CalibrationContext,
    ItemOutcome,
    finish_item,
    finish_repeat,
    platform_name,
    require_committed_code,
    require_known_run,
    timed,
)
from hmrec.calibration.records import Measurement, RepeatCheck
from hmrec.config import load_base_config, load_calibration_config
from hmrec.memory import ProcessPeakMemory
from hmrec.paths import build_paths, calibration_output_paths, get_paths
from hmrec.platform_fields import PlatformFields
from hmrec.run_records import RunStamp

REPO_PATHS = get_paths()
BASE = load_base_config(REPO_PATHS.base_config_file)
CONFIG = load_calibration_config(REPO_PATHS.m10_calibration_config_file)
COMMIT = "0123456789abcdef0123456789abcdef01234567"
PLATFORM = PlatformFields(
    "linux",
    "aarch64",
    "/usr/lib/libgomp.so.1",
    "libgomp1 14.2.0-19",
    "not loaded",
    COMMIT,
    "b" * 64,
)
GB = 1024**3


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CalibrationContext:
    """A context whose artifacts and results lie under tmp_path; the platform is a fixed record."""
    paths = build_paths(BASE, tmp_path)
    monkeypatch.setattr(context_module, "read_platform_fields", lambda repo_root: PLATFORM)
    return CalibrationContext(
        paths=paths,
        main_paths=calibration_output_paths(paths, "linux-arm64", "run1", "main"),
        recorded_model_paths=calibration_output_paths(
            paths, "linux-arm64", "run1", "recorded_model"
        ),
        base=BASE,
        config=CONFIG,
        run="run1",
        platform_name="linux-arm64",
        stamp=RunStamp(COMMIT, False),
        started=0.0,
    )


def test_platform_name() -> None:
    assert platform_name("linux", "aarch64") == "linux-arm64"
    assert platform_name("linux", "x86_64") == "linux-amd64"
    assert platform_name("darwin", "arm64") == "darwin-arm64"
    assert platform_name("linux", "riscv64") == "linux-riscv64"  # an unknown name is kept


def test_only_the_configured_runs_are_allowed() -> None:
    require_known_run("run1", CONFIG)
    require_known_run("run2", CONFIG)
    with pytest.raises(ValueError, match="--run must be one of"):
        require_known_run("run3", CONFIG)


def test_a_calibration_refuses_uncommitted_code() -> None:
    require_committed_code(RunStamp(COMMIT, False))
    with pytest.raises(RuntimeError, match="runs only from committed code"):
        require_committed_code(RunStamp(COMMIT, True))


def test_the_output_trees_lie_in_the_calibration_folders(context: CalibrationContext) -> None:
    root = context.paths.artifacts_root / "m10_calibration" / "linux-arm64" / "run1"
    assert context.main_paths.artifacts_root == root / "main"
    assert context.recorded_model_paths.artifacts_root == root / "recorded_model"
    # A stage-1 function that is given the output paths writes below the calibration tree.
    assert context.main_paths.candidates_file("V4", "r5_wals") == (
        root / "main" / "candidates" / "V4" / "r5_wals.parquet"
    )
    assert context.main_paths.results_root == context.paths.results_root / "m10_calibration"
    # Inputs keep their stage-1 locations.
    assert context.main_paths.data_root == context.paths.data_root


def test_timed_returns_the_value_the_seconds_and_a_sampled_peak(
    context: CalibrationContext,
) -> None:
    value, seconds, sampled_peak_gb = timed(context, "test block", lambda: 41 + 1)
    assert value == 42
    assert seconds >= 0.0 and sampled_peak_gb > 0.0


def test_finish_item_writes_the_comparison_rows_and_then_the_run_row(
    context: CalibrationContext,
) -> None:
    outcome = ItemOutcome(
        item="B",
        scope="r5_wals",
        item_seconds=12.5,
        sampled_peak_rss_gb=1.5,
        item_peak=ProcessPeakMemory(2 * GB, 0),
        measurements=[Measurement("B", "r5_wals", "recall", 0.75, 0.5)],
        detail="test",
    )
    finish_item(context, outcome)
    (comparison,) = _read(context.paths.calibration_comparisons_file)
    assert (comparison["item"], comparison["quantity"], comparison["run"]) == (
        "B",
        "recall",
        "run1",
    )
    assert comparison["openmp_version"] == "libgomp1 14.2.0-19"
    assert comparison["code_tree_sha256"] == "b" * 64
    (run,) = _read(context.paths.calibration_runs_file)
    assert (run["item_seconds"], run["item_max_rss_gb"], run["num_threads"]) == ("12.5", "2.0", "6")
    # The peak at the end of the process is at least the item's own peak... of THIS test
    # process, so only its presence and sign are checked.
    assert float(run["process_max_rss_gb"]) > 0.0
    # Nothing was written outside results/m10_calibration/.
    written = sorted(path.name for path in context.paths.results_root.rglob("*") if path.is_file())
    assert written == ["comparisons.csv", "runs.csv"]


def test_finish_item_fails_after_the_rows_when_the_peak_is_above_the_limit(
    context: CalibrationContext,
) -> None:
    tiny_limit = replace(BASE, memory=replace(BASE.memory, limit_gb=0.001))
    outcome = ItemOutcome("A", "r1_popularity", 1.0, 0.1, ProcessPeakMemory(GB, 0), [], "test")
    with pytest.raises(MemoryError, match="above the limit"):
        finish_item(replace(context, base=tiny_limit), outcome)
    # The run is on record although the check failed.
    assert len(_read(context.paths.calibration_runs_file)) == 1


def test_finish_repeat_writes_one_row_per_check(context: CalibrationContext) -> None:
    checks = [
        RepeatCheck("D", "model (text of every tree)", True, "500 trees"),
        RepeatCheck("D", "Top-12", False, "columns with different values: ['score']"),
    ]
    finish_repeat(context, checks, "run1", "run2")
    rows = _read(context.paths.calibration_repeatability_file)
    assert [row["bit_identical"] for row in rows] == ["True", "False"]
    assert {row["first_run"] for row in rows} == {"run1"}
