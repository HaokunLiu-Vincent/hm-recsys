"""Tests for hmrec.final.rehearsal: the rehearsal must equal M7 exactly; any difference is found."""

import json
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest
from final_example import LOCKED_TREES, NOW, InProcessWorker, copy_workspace

from hmrec.csv_records import read_rows
from hmrec.eval.monitor import MONITOR_COLUMNS
from hmrec.final.context import FinalContext
from hmrec.final.gate import PASSED, REHEARSAL_CHECK_COLUMNS, FinalRunRefused
from hmrec.final.rehearsal import (
    code_check,
    compare_metric_rows,
    metric_checks,
    model_check,
    model_experiments,
    monitor_check,
    record_rehearsal_check,
    rehearsal_checks,
    top_k_check,
    trained_sample_sha256,
)
from hmrec.final.run import run_rehearsal
from hmrec.final.trial import experiment_config_id, sample_file
from hmrec.hpo.v_records import MetricRow
from hmrec.locks import file_sha256

METRICS = {"map@12": 0.5, "recall@12": 0.25}


def _row(num_trees: int, map_value: float = 0.5, n_queries: int = 40) -> MetricRow:
    scores = {**METRICS, "map@12": map_value}
    return MetricRow("full", "abc", "V4", num_trees, n_queries, scores, None)


def _rehearsed(tmp_path: Path) -> FinalContext:
    return copy_workspace(tmp_path, rehearsed=True)


# ---------------------------------------------------------------- compare_metric_rows


def test_equal_rows_pass_one_check_per_number_of_trees() -> None:
    rows = [_row(10), _row(20, 0.6)]
    checks = compare_metric_rows(rows, list(rows), "V4", (10, 20), "full")
    assert [check.name for check in checks] == [
        "full metrics at 10 trees",
        "full metrics at 20 trees",
    ]
    assert all(check.passed and check.detail == "" for check in checks)


def test_the_smallest_difference_of_a_float_fails() -> None:
    next_float = 0.5000000000000001  # the float right above 0.5
    assert next_float != 0.5
    checks = compare_metric_rows([_row(10, next_float)], [_row(10)], "V4", (10,), "full")
    assert not checks[0].passed
    assert checks[0].detail == "map@12 0.5000000000000001 != 0.5"


def test_another_number_of_queries_or_a_missing_row_fails() -> None:
    checks = compare_metric_rows([_row(10, n_queries=39)], [_row(10)], "V4", (10,), "full")
    assert not checks[0].passed and checks[0].detail == "n_queries 39 != 40"
    missing_ours = compare_metric_rows([], [_row(10)], "V4", (10,), "full")
    assert (
        not missing_ours[0].passed and "the rehearsal has no single row" in missing_ours[0].detail
    )
    missing_theirs = compare_metric_rows([_row(10)], [], "V4", (10,), "full")
    assert "the M7 record has no single row" in missing_theirs[0].detail
    twice = compare_metric_rows([_row(10), _row(10)], [_row(10)], "V4", (10,), "full")
    assert not twice[0].passed  # recorded twice is not "one row"


# ---------------------------------------------------------------- on the tiny workspace


def test_the_rehearsal_reproduces_m7_exactly(tmp_path: Path) -> None:
    context = _rehearsed(tmp_path)
    assert model_experiments(context) == ("full", "minus_r6")
    checks = rehearsal_checks(context)
    assert [check.name for check in checks] == [
        "full metrics at 10 trees",
        "minus_r6 metrics at 10 trees",
        "rrf_baseline metrics at 0 trees",
        "full trial ran with the code at HEAD",
        "full model: first 10 trees",
        "full Top-12 at 10 trees",
        "minus_r6 trial ran with the code at HEAD",
        "minus_r6 model: first 10 trees",
        "minus_r6 Top-12 at 10 trees",
        "recall monitoring rows",
    ]
    assert [check for check in checks if not check.passed] == []
    row = read_rows(context.paths.m8_rehearsal_check_file, REHEARSAL_CHECK_COLUMNS)[-1]
    assert (row["status"], row["n_checks"], row["failed_checks"]) == (PASSED, "10", "")
    assert (row["target_fold"], row["num_trees"]) == ("V4", str(LOCKED_TREES))
    assert row["config_id"] == context.locked.config_id


def test_the_check_row_records_the_samples_the_rehearsal_trained_on(tmp_path: Path) -> None:
    context = _rehearsed(tmp_path)
    recorded = trained_sample_sha256(context)
    assert sorted(recorded) == ["full", "minus_r6"]
    for experiment, by_fold in recorded.items():
        assert sorted(by_fold) == ["F4", "V1", "V2", "V3"]
        for fold, sha256 in by_fold.items():
            assert sha256 == file_sha256(sample_file(context, experiment, fold))
    row = read_rows(context.paths.m8_rehearsal_check_file, REHEARSAL_CHECK_COLUMNS)[-1]
    assert json.loads(row["sample_sha256_json"]) == recorded


def test_m7_trained_more_trees_and_only_the_first_ones_are_compared(tmp_path: Path) -> None:
    context = _rehearsed(tmp_path)
    # M7's full model has 20 trees (as the real one has 1000); the lock keeps 10.
    config_id = context.locked.config_id
    m7_model = context.paths.ranker_model_file("v_phase", "full", config_id, "V4")
    ours = context.paths.ranker_model_file("m8_rehearsal", "full", config_id, "V4")
    assert file_sha256(m7_model) != file_sha256(ours)  # different files (20 against 10 trees)
    assert model_check(context, "full").passed  # but the same first 10 trees


def test_a_changed_recorded_metric_fails_the_rehearsal(tmp_path: Path) -> None:
    context = _rehearsed(tmp_path)
    metrics_file = context.paths.v_phase_metrics_file
    text = metrics_file.read_text(encoding="utf-8")
    lines = text.splitlines()
    # The M7 row of the locked configuration at the locked number of trees: change its MAP.
    index = next(
        i
        for i, line in enumerate(lines)
        if line.startswith(f"full,{context.locked.config_id},V4,10,")
    )
    cells = lines[index].split(",")
    cells[5] = repr(float(cells[5]) / 2)
    lines[index] = ",".join(cells)
    metrics_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    failed = [check.name for check in metric_checks(context) if not check.passed]
    assert failed == ["full metrics at 10 trees"]
    row = record_rehearsal_check(context, NOW)
    assert row["status"] == "failed" and row["failed_checks"] == "full metrics at 10 trees"
    with pytest.raises(FinalRunRefused, match="did NOT reproduce M7 on V4"):
        run_rehearsal(context, InProcessWorker(context), tmp_path, NOW)


def test_another_model_or_another_top_12_fails(tmp_path: Path) -> None:
    context = _rehearsed(tmp_path)
    config_id = context.locked.config_id
    # M7's saved Top-12 with one score changed.
    m7_predictions = context.paths.ranker_predictions_file("v_phase", "full", config_id, "V4")
    table = pl.read_parquet(m7_predictions)
    changed = table.with_columns(
        pl.when(pl.int_range(pl.len()) == 0).then(pl.col("score") + 1e-9).otherwise(pl.col("score"))
    )
    changed.write_parquet(m7_predictions)
    assert not top_k_check(context, "full").passed
    assert top_k_check(context, "minus_r6").passed
    # M7's model file replaced by another file: it is no longer the recorded one.
    m7_model = context.paths.ranker_model_file("v_phase", "full", config_id, "V4")
    m7_model.write_text(m7_model.read_text() + "\n", encoding="utf-8")
    check = model_check(context, "full")
    assert not check.passed and "is not the recorded model file" in check.detail


def test_other_monitoring_rows_fail(tmp_path: Path) -> None:
    context = _rehearsed(tmp_path)
    assert monitor_check(context).passed
    path = context.paths.m6_candidates_results_file
    rows = read_rows(path, MONITOR_COLUMNS)
    recall_column = MONITOR_COLUMNS.index("recall")
    lines = path.read_text(encoding="utf-8").splitlines()
    cells = lines[1].split(",")
    cells[recall_column] = repr(float(rows[0]["recall"]) + 0.125)
    lines[1] = ",".join(cells)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    check = monitor_check(context)
    assert not check.passed and check.detail.startswith("r1_popularity.recall:")


def test_a_rehearsal_trial_of_other_code_does_not_count(tmp_path: Path) -> None:
    context = _rehearsed(tmp_path)
    assert code_check(context, "full").passed
    # The same rows seen from a process at another commit: the trial ran with other code.
    other = replace(context, code_changed=lambda commit: True)
    check = code_check(other, "full")
    assert not check.passed and "it ran at aaaaaaaaaaaa" in check.detail
    assert experiment_config_id(other, "full") == context.locked.config_id
