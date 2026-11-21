"""Tests for hmrec.calibration.references: the stage-1 records and the input check."""

import hashlib
from pathlib import Path

import pytest

from hmrec.calibration.references import (
    single_row,
    stage1_candidates_row,
    stage1_dataset_row,
    stage1_ranker_metrics,
    stage1_ranker_trial,
    stage1_run_row,
    verify_files,
)
from hmrec.config import load_calibration_config
from hmrec.paths import get_paths

PATHS = get_paths()
CONFIG = load_calibration_config(PATHS.m10_calibration_config_file)
LOCKED_CONFIG_ID = "ca1a3cdfce49"


def test_single_row_needs_exactly_one_match(tmp_path: Path) -> None:
    table = tmp_path / "runs.csv"
    table.write_text(
        "fold,step,seconds\nV4,r1,1.5\nV4,r2,0.5\nV3,r1,1.0\nV3,r1,9.0\n", encoding="utf-8"
    )
    assert single_row(table, {"fold": "V4", "step": "r2"}) == {
        "fold": "V4",
        "step": "r2",
        "seconds": "0.5",
    }
    with pytest.raises(ValueError, match="found 0"):
        single_row(table, {"fold": "V4", "step": "r9"})
    # Two rows for one step: which one is the record? Neither is used.
    with pytest.raises(ValueError, match="found 2"):
        single_row(table, {"fold": "V3", "step": "r1"})
    with pytest.raises(FileNotFoundError, match="stage-1 result table not found"):
        single_row(tmp_path / "missing.csv", {"fold": "V4"})


def test_the_committed_tables_hold_the_stage_1_records_of_the_calibration_week() -> None:
    # The numbers of docs/m10_calibration_plan.md §4, read from the committed tables.
    assert CONFIG.fold == "V4" and CONFIG.fine_tune_fold == "V3"
    r5 = stage1_candidates_row(PATHS, "V4", "r5_wals")
    assert (r5["n_rows"], r5["recall"]) == ("6072000", "0.09497516644781423")
    union = stage1_candidates_row(PATHS, "V4", "union")
    assert (union["n_rows"], union["recall"]) == ("22293247", "0.2761626403149555")
    assert stage1_run_row(PATHS, "V4", "r4_itemcf")["n_rows"] == "6513892"
    assert stage1_run_row(PATHS, "V3", "r6_twotower")["status"] == "generated"


def test_the_ranker_reference_is_the_m8_rehearsal_of_the_locked_configuration() -> None:
    trial = stage1_ranker_trial(PATHS, CONFIG, LOCKED_CONFIG_ID)
    assert (trial["num_trees"], trial["train_folds"]) == ("500", "F4 V1 V2 V3")
    metrics = stage1_ranker_metrics(PATHS, CONFIG, LOCKED_CONFIG_ID)
    assert sorted(metrics) == [100, 200, 300, 400, 500]
    # MAP@12 of V4 at the locked 500 trees, as recorded by M7 and reproduced by the rehearsal.
    assert metrics[500]["map@12"] == 0.034849701577104486
    assert set(metrics[500]) == set(CONFIG.metrics)
    assert stage1_dataset_row(PATHS, CONFIG)["n_rows"] == "10453387"
    with pytest.raises(ValueError, match="has no row"):
        stage1_ranker_metrics(PATHS, CONFIG, "000000000000")


def test_verify_files(tmp_path: Path) -> None:
    (tmp_path / "queries").mkdir()
    file = tmp_path / "queries" / "V4.parquet"
    file.write_bytes(b"content")
    expected = {"queries/V4.parquet": hashlib.sha256(b"content").hexdigest()}
    verify_files(tmp_path, expected, ["queries/V4.parquet"])  # the recorded file: no error
    file.write_bytes(b"other content")
    with pytest.raises(ValueError, match="sha256"):
        verify_files(tmp_path, expected, ["queries/V4.parquet"])
    with pytest.raises(KeyError, match="records no sha256 for 'labels/V4.parquet'"):
        verify_files(tmp_path, expected, ["labels/V4.parquet"])
    file.unlink()
    with pytest.raises(FileNotFoundError):
        verify_files(tmp_path, expected, ["queries/V4.parquet"])
