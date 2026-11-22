"""Tests for the record configs/platforms/linux-amd64.yaml (stage 3, M13; docs/mlops.md §8.3).

The record was written after the calibration on Cloud Run. It must say the same
as the arm64 record wherever the two platforms are the same (acceptance, library
versions, settings, the Ranker lock), name its own platform, and rest on the
calibration tables of M13 as they were committed.
"""

import hashlib
import re
import subprocess
from pathlib import Path

from hmrec.config import load_yaml
from hmrec.csv_records import read_rows
from hmrec.paths import get_paths
from hmrec.platform_fields import ARCHITECTURE_NAMES
from hmrec.platform_records import load_platform_record
from hmrec.weekly.replay_compare import ACCEPTANCE, COMPARISON_COLUMNS, STRUCTURAL

REPO = Path(__file__).resolve().parents[2]
PATHS = get_paths()
RECORD_FILE = PATHS.platform_record_file("linux-amd64")
ARM64_FILE = PATHS.platform_record_file("linux-arm64")


def test_the_record_names_the_platform_of_cloud_run() -> None:
    record = load_platform_record(RECORD_FILE)
    assert (record.platform, record.os, record.architecture) == ("linux-amd64", "linux", "x86_64")
    # The name the tolerance-accepted entry looks for in an x86-64 Linux process
    # (hmrec.ranker.platform_entry.platform_record_name).
    assert f"linux-{ARCHITECTURE_NAMES['x86_64']}" == record.platform


def test_fields_and_thresholds_are_those_of_the_arm64_record() -> None:
    record, arm64 = load_platform_record(RECORD_FILE), load_platform_record(ARM64_FILE)
    assert record.acceptance == arm64.acceptance
    assert record.acceptance.max_relative_difference == 0.01
    same = (
        "base_image",
        "lightgbm_version",
        "openmp_version",
        "num_threads",
        "torch_version",
        "r6_fine_tune_setting",
        "ranker_lock_path",
        "ranker_lock_sha256",
    )
    for field in same:
        assert getattr(record, field) == getattr(arm64, field), field
    raw, raw_arm64 = load_yaml(RECORD_FILE), load_yaml(ARM64_FILE)
    assert set(raw) == set(raw_arm64)
    for section in ("ranker_runtime", "recall_runtime", "settings", "acceptance", "ranker_lock"):
        assert raw[section] == raw_arm64[section], section


def test_the_decision_entry_exists() -> None:
    record = load_platform_record(RECORD_FILE)
    prefix = "docs/decisions.md "
    assert record.decision_record.startswith(prefix)
    title = "## " + record.decision_record.removeprefix(prefix)
    assert title in (REPO / "docs" / "decisions.md").read_text(encoding="utf-8").splitlines()


def test_the_calibration_basis_is_the_committed_tables_of_m13() -> None:
    calibration = load_yaml(RECORD_FILE)["calibration"]
    assert calibration["week"] == "V1"
    (image,) = calibration["images"]
    assert re.fullmatch(r"[0-9a-f]{40}", image["commit"])
    assert re.fullmatch(r"[0-9a-f]{64}", image["code_tree_sha256"])
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", image["digest"])
    # The image is the one the calibration ticks recorded in their step rows.
    steps = (PATHS.cloud_results_dir / "calibration" / "steps.csv").read_text(encoding="utf-8")
    assert image["commit"] in steps
    commit = calibration["result_tables_commit"]
    assert re.fullmatch(r"[0-9a-f]{40}", commit)
    assert len(calibration["result_tables"]) == 6
    if (REPO / ".git").exists():
        # A checkout with history: the recorded sha256 is the one of each table as it
        # was at that commit.
        for path, recorded_sha256 in calibration["result_tables"].items():
            content = subprocess.run(
                ["git", "show", f"{commit}:{path}"], cwd=REPO, capture_output=True, check=True
            ).stdout
            assert hashlib.sha256(content).hexdigest() == recorded_sha256, path
    else:
        # Inside an image there is no git history: the tables exist and the values are
        # sha256 values.
        for path, recorded_sha256 in calibration["result_tables"].items():
            assert (REPO / path).is_file(), path
            assert re.fullmatch(r"[0-9a-f]{64}", recorded_sha256)


def test_the_calibration_was_inside_the_acceptance_the_record_states() -> None:
    record = load_platform_record(RECORD_FILE)
    calibration = load_yaml(RECORD_FILE)["calibration"]
    table = PATHS.cloud_results_dir / "calibration" / "comparison.csv"
    rows = [row for row in read_rows(table, COMPARISON_COLUMNS) if row["week"] == "V1"]
    accepted = [row for row in rows if row["kind"] == ACCEPTANCE]
    structural = [row for row in rows if row["kind"] == STRUCTURAL]
    # Six channels, the union and MAP@12; two structural checks.
    assert len(accepted) == 8 and len(structural) == 2
    assert all(row["inside"] == "True" for row in [*accepted, *structural])
    largest = max(float(row["relative_difference"]) for row in accepted)
    assert largest <= record.acceptance.max_relative_difference
    # The record quotes the largest difference, rounded up.
    assert largest <= calibration["largest_relative_difference"] < largest * 1.01
    assert all(float(row["value"]) == 0.0 for row in structural)
