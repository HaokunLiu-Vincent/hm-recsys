"""Tests for hmrec.final.process: one step in one process, recorded; the gate comes first."""

import time
from pathlib import Path

import pytest
from final_example import COMMIT, NOW, copy_workspace, load_context

from hmrec.csv_records import read_rows
from hmrec.final.context import MODE_FORMAL, MODE_REHEARSAL
from hmrec.final.gate import FinalRunRefused, begin_formal_run
from hmrec.final.process import run_worker, step_label
from hmrec.final.run import (
    STEP_BASELINE,
    STEP_DATASET,
    STEP_MONITOR,
    STEP_SAMPLE,
    STEP_TRIAL,
    StepRequest,
)
from hmrec.hpo.v_records import M7_RUN_COLUMNS, read_metric_rows, read_trial_rows
from hmrec.ranker.runtime import read_lightgbm_runtime

METRICS = ("map@12", "recall@12", "ndcg@12")


def test_step_label() -> None:
    assert step_label(StepRequest(STEP_DATASET, MODE_FORMAL, "T1", "minus_r6")) == (
        "formal_dataset_minus_r6"
    )


def test_a_formal_step_without_a_rehearsal_reads_nothing(tmp_path: Path) -> None:
    context = copy_workspace(tmp_path)
    for fold in ("T1", "T2", "T3", "T4"):
        context.paths.labels_file(fold).unlink()  # a read would be FileNotFoundError
    for step in (STEP_MONITOR, STEP_BASELINE, STEP_SAMPLE, STEP_DATASET, STEP_TRIAL):
        with pytest.raises(FinalRunRefused, match="no rehearsal is recorded"):
            run_worker(context, StepRequest(step, MODE_FORMAL, "T1"), time.perf_counter())
    assert not context.paths.m8_runs_file.exists()
    assert not context.paths.t_phase_recall_file.exists()


def test_a_formal_step_needs_the_recorded_start_of_the_formal_run(tmp_path: Path) -> None:
    context = copy_workspace(tmp_path, rehearsed=True)
    # The rehearsal passed, but the driver has not recorded the start: a worker started by
    # hand must not be the first reader of a T label.
    with pytest.raises(FinalRunRefused, match="the start of the formal run is not recorded"):
        run_worker(context, StepRequest(STEP_MONITOR, MODE_FORMAL, "T1"), time.perf_counter())
    assert not context.paths.t_phase_recall_file.exists()


def test_a_dirty_tree_stops_every_step(tmp_path: Path) -> None:
    context = copy_workspace(tmp_path)
    dirty = load_context(context.paths, COMMIT, dirty=True)
    for mode, fold in ((MODE_REHEARSAL, "V4"), (MODE_FORMAL, "T1")):
        with pytest.raises(FinalRunRefused, match="needs a clean working tree"):
            run_worker(dirty, StepRequest(STEP_MONITOR, mode, fold), time.perf_counter())


def test_a_rehearsal_step_cannot_reach_a_t_fold(tmp_path: Path) -> None:
    context = copy_workspace(tmp_path)
    context.paths.labels_file("T1").unlink()
    for step in (STEP_MONITOR, STEP_BASELINE, STEP_DATASET, STEP_TRIAL):
        with pytest.raises(ValueError, match="only the configured rehearsal fold"):
            run_worker(context, StepRequest(step, MODE_REHEARSAL, "T1"), time.perf_counter())
    with pytest.raises(ValueError, match="samples are drawn only in the formal run"):
        run_worker(context, StepRequest(STEP_SAMPLE, MODE_REHEARSAL, "T1"), time.perf_counter())


def test_helper_steps_write_a_run_row_with_time_memory_and_runtime(tmp_path: Path) -> None:
    context = copy_workspace(tmp_path, rehearsed=True)
    begin_formal_run(context, NOW)
    run_worker(context, StepRequest(STEP_SAMPLE, MODE_FORMAL, "T1"), time.perf_counter())
    rows = read_rows(context.paths.m8_runs_file, M7_RUN_COLUMNS)
    row = rows[-1]
    assert (row["fold"], row["step"], row["status"]) == ("T1", "formal_sample_full", "generated")
    assert row["n_rows"] == "120" and row["detail"] == "40 queries, 40 positives"
    assert float(row["process_max_rss_gb"]) > 0 and float(row["total_seconds"]) > 0
    runtime = read_lightgbm_runtime()
    assert row["lightgbm_version"] == runtime.lightgbm_version
    assert row["libomp_version"] == runtime.libomp_version
    assert (row["git_commit"], row["git_dirty"]) == (COMMIT, "False")
    # The rehearsal wrote rows for its helper steps too (monitor, baseline, two Datasets).
    assert [r["step"] for r in rows[:4]] == [
        "rehearsal_monitor_full",
        "rehearsal_baseline_full",
        "rehearsal_dataset_full",
        "rehearsal_dataset_minus_r6",
    ]


def test_a_trial_writes_its_metric_rows_and_then_its_trial_row(tmp_path: Path) -> None:
    context = copy_workspace(tmp_path, rehearsed=True)
    paths = context.paths
    trial = read_trial_rows(paths.m8_rehearsal_results_file)[0]
    assert (trial["phase"], trial["experiment"], trial["step"]) == ("V", "full", "rehearsal")
    assert (trial["fold"], trial["train_folds"]) == ("V4", "F4 V1 V2 V3")
    assert trial["config_id"] == context.locked.config_id and trial["num_trees"] == "10"
    assert trial["evaluated"] == "True" and trial["git_commit"] == COMMIT
    assert len(trial["model_sha256"]) == 64 and float(trial["process_max_rss_gb"]) > 0
    assert trial["lightgbm_version"] == read_lightgbm_runtime().lightgbm_version
    metric_rows = read_metric_rows(paths.m8_rehearsal_metrics_file, METRICS)
    assert [(row.experiment, row.num_trees) for row in metric_rows] == [
        ("rrf_baseline", 0),
        ("full", 10),
        ("minus_r6", 10),
    ]
