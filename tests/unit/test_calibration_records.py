"""Tests for hmrec.calibration.records: the result tables of the M10 calibration."""

import csv
from pathlib import Path

import pytest

from hmrec.calibration.records import (
    COMPARISON_COLUMNS,
    REPEAT_COLUMNS,
    RUN_COLUMNS,
    Measurement,
    ProcessRecord,
    RepeatCheck,
    RunIdentity,
    append_comparisons,
    append_repeat_checks,
    append_run,
    comparison_row,
    difference_texts,
    value_text,
    within_text,
)
from hmrec.csv_records import read_rows
from hmrec.paths import get_paths
from hmrec.platform_fields import PLATFORM_COLUMNS, PlatformFields

COMMIT = "0123456789abcdef0123456789abcdef01234567"
PLATFORM = PlatformFields(
    os="linux",
    architecture="aarch64",
    openmp_library="/usr/lib/aarch64-linux-gnu/libgomp.so.1.0.0",
    openmp_version="libgomp1 14.2.0-19",
    blas_libraries="/venv/numpy.libs/libscipy_openblas64_.so",
    image_commit=COMMIT,
    code_tree_sha256="a" * 64,
)
IDENTITY = RunIdentity(run="run1", git_commit=COMMIT, git_dirty=False, platform=PLATFORM)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_value_text() -> None:
    assert value_text(0.1) == "0.1"  # repr(): reads back as the same float64
    assert value_text(7) == "7"
    assert value_text(True) == "True"
    assert value_text("none") == "none"
    assert value_text(None) == ""


def test_differences_are_computed_only_for_two_numbers() -> None:
    # 0.75 - 0.5 = 0.25; 0.25 / 0.5 = 0.5 (all exact in binary).
    assert difference_texts(0.75, 0.5) == ("0.25", "0.5")
    # Integers: 7 - 5 = 2; 2 / 5 = 0.4.
    assert difference_texts(7, 5) == ("2", "0.4")
    # A stage-1 value of 0 has no relative difference.
    assert difference_texts(3, 0) == ("3", "")
    # Flags, texts and missing records give no difference.
    assert difference_texts(True, True) == ("", "")
    assert difference_texts("none", "none") == ("", "")
    assert difference_texts(0.5, None) == ("", "")


def test_within_text() -> None:
    assert (within_text(True), within_text(False), within_text(None)) == ("inside", "outside", "")


def test_a_comparison_row_carries_the_run_the_commit_and_the_platform() -> None:
    measurement = Measurement(
        "B", "r5_wals", "recall", 0.75, 0.5, "relative difference <= 0.01", False, "results/x.csv"
    )
    row = comparison_row(measurement, IDENTITY, "2026-11-22T10:00:00")
    assert tuple(row) == COMPARISON_COLUMNS
    assert row["measured"] == "0.75" and row["stage1"] == "0.5"
    assert (row["difference"], row["relative_difference"]) == ("0.25", "0.5")
    assert row["provisional_tolerance"] == "relative difference <= 0.01"
    assert row["within_tolerance"] == "outside"
    assert (row["run"], row["git_commit"], row["git_dirty"]) == ("run1", COMMIT, "False")
    for name in PLATFORM_COLUMNS:
        assert row[name] == getattr(PLATFORM, name)


def test_tables_are_appended_with_a_fixed_header(tmp_path: Path) -> None:
    comparisons = tmp_path / "m10_calibration" / "comparisons.csv"
    first = Measurement("A", "r1_popularity", "rank_rows_identical", True, True, "identical", True)
    second = Measurement("A", "r1_popularity", "differing_rank_rows", 0, 0)
    append_comparisons(comparisons, [first], IDENTITY)
    append_comparisons(comparisons, [second], IDENTITY)
    rows = _read(comparisons)
    assert [row["quantity"] for row in rows] == ["rank_rows_identical", "differing_rank_rows"]
    assert rows[0]["measured"] == "True" and rows[0]["within_tolerance"] == "inside"
    assert rows[1]["within_tolerance"] == ""  # no tolerance for this quantity

    runs = tmp_path / "m10_calibration" / "runs.csv"
    record = ProcessRecord("B", "r5_wals", 1.5, 2.5, 3.0, 3.5, 4.0, 6, 6.0, 12.0, "6072000 rows")
    append_run(runs, record, IDENTITY)
    (run_row,) = _read(runs)
    assert tuple(run_row) == RUN_COLUMNS
    assert (run_row["item_seconds"], run_row["item_max_rss_gb"]) == ("1.5", "3.5")
    assert run_row["code_tree_sha256"] == "a" * 64

    repeats = tmp_path / "m10_calibration" / "repeatability.csv"
    checks = [RepeatCheck("D", "model (text of every tree)", True, "500 trees")]
    append_repeat_checks(repeats, checks, "run1", "run2", IDENTITY)
    (repeat_row,) = _read(repeats)
    assert tuple(repeat_row) == REPEAT_COLUMNS
    assert (repeat_row["first_run"], repeat_row["second_run"]) == ("run1", "run2")
    assert repeat_row["bit_identical"] == "True"


def test_a_table_with_another_header_is_not_appended_to(tmp_path: Path) -> None:
    path = tmp_path / "comparisons.csv"
    path.write_text("item,scope\nA,r1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected"):
        append_comparisons(path, [Measurement("A", "r1", "rows", 1, 1)], IDENTITY)
    assert path.read_text(encoding="utf-8") == "item,scope\nA,r1\n"  # nothing was written


def test_the_committed_calibration_tables_have_the_fixed_headers_and_say_what_they_ran_on() -> None:
    # results/m10_calibration/ holds the calibration of week V4 in the arm64 container
    # (2026-11-22). read_rows fails on another header. Every row must name its run, a full
    # commit, a verified code tree and the OpenMP runtime: a container result is only
    # meaningful together with what it was computed on.
    paths = get_paths()
    tables = (
        (paths.calibration_comparisons_file, COMPARISON_COLUMNS),
        (paths.calibration_runs_file, RUN_COLUMNS),
        (paths.calibration_repeatability_file, REPEAT_COLUMNS),
    )
    for path, columns in tables:
        rows = read_rows(path, columns)
        assert rows, f"{path} is empty or missing"
        for row in rows:
            assert row["run"] in ("run1", "run2")
            assert len(row["git_commit"]) == 40 and row["git_dirty"] == "False"
            assert row["image_commit"] == row["git_commit"]
            assert len(row["code_tree_sha256"]) == 64
            assert (row["os"], row["architecture"]) == ("linux", "aarch64")
            # A process that computed something loaded an OpenMP runtime and names it. The
            # comparison of two Ranker runs only reads files, so it may have loaded none.
            if path != paths.calibration_repeatability_file:
                assert row["openmp_library"] != "not loaded"
    # The Ranker rows ran on the system OpenMP runtime the Dockerfile pins.
    ranker_rows = [
        row for row in read_rows(paths.calibration_runs_file, RUN_COLUMNS) if row["item"] == "D"
    ]
    assert ranker_rows
    assert {row["openmp_version"] for row in ranker_rows} == {"libgomp1 14.2.0-19"}
