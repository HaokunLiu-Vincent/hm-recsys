"""Integration test of the final evaluation (M8) on tiny data: rehearsal, then the formal run.

The workspace (tests/unit/final_example.py) holds 9 tiny folds, a tiny Ranker lock
and what M7 / M6 recorded for the rehearsal fold, made with the M7 / M6 code. The
real driver (hmrec.final.run) runs every step with the real worker code
(hmrec.final.process.run_worker), in this process instead of one process per step.

What is checked end to end:
    - the rehearsal reproduces M7 exactly and reads no T label;
    - the formal run starts only after it, records the first read of T labels first,
      samples only V4 and the first three T folds, reuses M7's V1-V3 sample files,
      and trains one model per T fold on the right four weeks;
    - the result tables, the verdict on G1-G3 and the saved Top-12;
    - a second formal run starts nothing and changes nothing.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import polars as pl
import pytest

from hmrec.csv_records import read_rows
from hmrec.eval.monitor import MONITOR_COLUMNS
from hmrec.final.gate import (
    LABEL_ACCESS_COLUMNS,
    REHEARSAL_CHECK_COLUMNS,
    FinalRunRefused,
)
from hmrec.final.goals import GOAL_COLUMNS
from hmrec.final.report import final_summaries, final_verdicts, summary_lines
from hmrec.final.run import STEP_SAMPLE, run_formal, run_rehearsal
from hmrec.final.trial import experiment_config_id, sample_file
from hmrec.hpo.v_records import M7_RUN_COLUMNS, read_metric_rows, read_trial_rows
from hmrec.locks import file_sha256
from hmrec.ranker.train import load_model

UNIT_TESTS = Path(__file__).resolve().parents[1] / "unit"
METRICS = ("map@12", "recall@12", "ndcg@12")
T_FOLDS = ("T1", "T2", "T3", "T4")
WINDOWS = {
    "T1": "V1 V2 V3 V4",
    "T2": "V2 V3 V4 T1",
    "T3": "V3 V4 T1 T2",
    "T4": "V4 T1 T2 T3",
}


def _load(name: str) -> ModuleType:
    """Import a helper module of tests/unit by file (it is not a package)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, UNIT_TESTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# final_example imports its two helper modules by name: load them first.
ranker_example = _load("ranker_example")
v_phase_example = _load("v_phase_example")
final_example = _load("final_example")
NOW = final_example.NOW
LOCKED_TREES = final_example.LOCKED_TREES


def test_rehearsal_then_formal_run(tmp_path: Path) -> None:
    context = final_example.build_final_workspace(tmp_path)
    paths = context.paths
    full_id = experiment_config_id(context, "full")
    ablation_id = experiment_config_id(context, "minus_r6")
    reused = {
        (experiment, fold): file_sha256(sample_file(context, experiment, fold))
        for experiment in ("full", "minus_r6")
        for fold in ("V1", "V2", "V3")
    }

    # ---- the formal run cannot start before the rehearsal
    with pytest.raises(FinalRunRefused, match="no rehearsal is recorded"):
        run_formal(context, final_example.InProcessWorker(context), tmp_path, NOW)

    # ---- rehearsal: V4 with the M8 code path equals M7, and no T file is produced
    rehearsal_worker = final_example.InProcessWorker(context)
    check = run_rehearsal(context, rehearsal_worker, tmp_path, NOW)
    assert (check["status"], check["n_checks"], check["failed_checks"]) == ("passed", "10", "")
    assert {request.fold for request in rehearsal_worker.calls} == {"V4"}
    assert not paths.t_phase_results_file.exists()
    assert not paths.t_phase_label_access_file.exists()
    assert not sample_file(context, "full", "T1").exists()
    rehearsal_metrics = read_metric_rows(paths.m8_rehearsal_metrics_file, METRICS)
    m7_metrics = read_metric_rows(paths.v_phase_metrics_file, METRICS)
    for row in rehearsal_metrics:
        same = [
            other
            for other in m7_metrics
            if (other.experiment, other.config_id, other.fold, other.num_trees)
            == (row.experiment, row.config_id, row.fold, row.num_trees)
        ]
        assert len(same) == 1 and same[0].scores == row.scores  # float64 ==

    # ---- the formal run
    worker = final_example.InProcessWorker(context)
    verdicts = run_formal(context, worker, tmp_path, NOW)
    assert len(worker.calls) == 32  # 4 monitor + 4 baseline + 2 x (4 samples + 4 x 2)

    # The first read of T labels was recorded, once, with the commit.
    access = read_rows(paths.t_phase_label_access_file, LABEL_ACCESS_COLUMNS)
    assert len(access) == 1 and access[0]["event"] == "first_t_label_read"
    assert (access[0]["started_at"], access[0]["git_commit"]) == (NOW, final_example.COMMIT)

    # Samples: V4 and the first three T folds are new; V1-V3 are M7's files, untouched;
    # T4 is never sampled.
    sampled = [(r.experiment, r.fold) for r in worker.calls if r.step == STEP_SAMPLE]
    assert sampled == [
        (experiment, fold)
        for experiment in ("full", "minus_r6")
        for fold in ("V4", "T1", "T2", "T3")
    ]
    for (experiment, fold), sha256 in reused.items():
        assert file_sha256(sample_file(context, experiment, fold)) == sha256
    assert not sample_file(context, "full", "T4").exists()

    # Trials: one model per (experiment, T fold), trained on the four weeks before it,
    # with the locked number of trees.
    trials = read_trial_rows(paths.t_phase_results_file)
    assert [(t["experiment"], t["fold"]) for t in trials] == [
        (experiment, fold) for experiment in ("full", "minus_r6") for fold in T_FOLDS
    ]
    for trial in trials:
        assert trial["train_folds"] == WINDOWS[trial["fold"]]
        assert (trial["phase"], trial["step"]) == ("T", "formal")
        assert trial["num_trees"] == str(LOCKED_TREES) and trial["git_dirty"] == "False"
        expected_id = full_id if trial["experiment"] == "full" else ablation_id
        assert trial["config_id"] == expected_id
        model_file = paths.ranker_model_file(
            "t_phase", trial["experiment"], expected_id, trial["fold"]
        )
        assert file_sha256(model_file) == trial["model_sha256"]
        assert load_model(model_file).num_trees() == LOCKED_TREES

    # Metrics: T folds are evaluated only at the locked number of trees.
    metric_rows = read_metric_rows(paths.t_phase_metrics_file, METRICS)
    assert sorted({(row.experiment, row.num_trees) for row in metric_rows}) == [
        ("full", LOCKED_TREES),
        ("minus_r6", LOCKED_TREES),
        ("rrf_baseline", 0),
    ]
    assert len(metric_rows) == 12 and all(row.n_queries == 40 for row in metric_rows)

    # The saved Top-12 of every T fold.
    for fold in T_FOLDS:
        top = pl.read_parquet(paths.ranker_predictions_file("t_phase", "full", full_id, fold))
        assert top.height == 40 * 12 and top["num_trees"].unique().to_list() == [LOCKED_TREES]
        assert top.group_by("query_id").len()["len"].unique().to_list() == [12]

    # Recall monitoring: 6 channels and the union for each T fold, with labels.
    recall_rows = read_rows(paths.t_phase_recall_file, MONITOR_COLUMNS)
    assert len(recall_rows) == 28
    assert [row["fold"] for row in recall_rows if row["scope"] == "union"] == list(T_FOLDS)
    assert all(row["recall"] != "" and row["labels_read"] == "True" for row in recall_rows)

    # The verdict: the tiny folds are easy, so all three goals are met; what matters is
    # that the numbers in the verdict are the numbers of the tables.
    summaries = final_summaries(context)
    full, baseline = summaries["full"], summaries["rrf_baseline"]
    assert [verdict.goal for verdict in verdicts] == ["G1", "G2", "G3"]
    assert verdicts[0].value == repr(full.pooled["map@12"]) and verdicts[0].met
    assert verdicts[1].value == "1.0" and verdicts[1].met  # every label is a candidate
    above = sum(full.by_fold["map@12"][f] > baseline.by_fold["map@12"][f] for f in T_FOLDS)
    assert verdicts[2].value == str(above) == "4" and verdicts[2].met
    # 40 queries in every fold: pooled and equal-fold MAP are the same number here.
    assert full.pooled["map@12"] == pytest.approx(full.equal_fold["map@12"], rel=1e-12)
    # Without R6 the tiny model loses the tt_dot feature only; it still beats the baseline.
    assert summaries["minus_r6"].pooled["map@12"] > baseline.pooled["map@12"]
    goals = read_rows(paths.t_phase_goals_file, GOAL_COLUMNS)
    assert [(g["goal"], g["met"]) for g in goals] == [
        ("G1", "True"),
        ("G2", "True"),
        ("G3", "True"),
    ]
    assert [g["threshold"] for g in goals] == ["0.032", "0.24", "3"]
    lines = summary_lines(context, verdicts)
    assert sum(line.startswith("G") for line in lines) == 3 and len(lines) == 12

    # Helper processes are recorded with time, memory and runtime.
    runs = read_rows(paths.m8_runs_file, M7_RUN_COLUMNS)
    assert len(runs) == 4 + 24  # rehearsal: 4 helper steps; formal: 8 + 2 x (4 + 4)
    assert all(float(run["process_max_rss_gb"]) > 0 and run["lightgbm_version"] for run in runs)

    # ---- running the formal stage again starts nothing and changes nothing
    tables = {
        path: path.read_bytes()
        for path in (
            paths.t_phase_results_file,
            paths.t_phase_metrics_file,
            paths.t_phase_recall_file,
            paths.t_phase_goals_file,
            paths.t_phase_label_access_file,
            paths.m8_runs_file,
        )
    }
    again = final_example.InProcessWorker(context)
    assert run_formal(context, again, tmp_path, "2026-11-19T09:00:00") == verdicts
    assert again.calls == []
    assert all(path.read_bytes() == content for path, content in tables.items())
    assert final_verdicts(context) == verdicts
    assert len(read_rows(paths.m8_rehearsal_check_file, REHEARSAL_CHECK_COLUMNS)) == 1
