"""Tests for hmrec.tracking.records: which runs and metrics the MLflow view holds."""

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest
from run_manifest_example import example_manifest
from tracking_example import (
    CHANNEL_RECALL,
    COMMIT,
    EXPERIMENT,
    REPO,
    fold_v1,
    ranker_lock,
    weekly_run,
    write_weekly_tables,
)

from hmrec.config import load_versioning_config
from hmrec.data.folds import load_fold_manifest
from hmrec.paths import get_paths
from hmrec.run_manifest import MetricRecord, write_run_manifest
from hmrec.tracking.records import (
    IMPORTED_RECORD,
    PIPELINE_RUN,
    RECORD_ID_TAG,
    RECORD_KIND_TAG,
    MetricSource,
    as_number,
    calibration_runs,
    check_unique_keys,
    metric_key,
    pipeline_run,
    select_row,
    source_from_text,
    stage1_weekly_run,
)
from hmrec.tracking.view import view_runs


def test_metric_key_uses_only_what_mlflow_allows() -> None:
    assert metric_key("map@12") == "map_at_12"
    assert metric_key("recall@100") == "recall_at_100"
    assert metric_key("union_recall") == "union_recall"
    assert metric_key("share (new)") == "share _new_"


def test_as_number_accepts_only_finite_numbers() -> None:
    assert as_number("0.25") == 0.25
    assert as_number("7203500") == 7203500.0
    for text in ("", "True", "deterministic", "nan", "inf"):
        assert as_number(text) is None


def test_select_row_needs_exactly_one_row() -> None:
    rows = [
        {"fold": "V1", "scope": "a"},
        {"fold": "V1", "scope": "b"},
        {"fold": "V2", "scope": "a"},
    ]
    assert select_row(rows, (("fold", "V1"), ("scope", "b")), "t") == rows[1]
    with pytest.raises(ValueError, match="found 2"):
        select_row(rows, (("fold", "V1"),), "t")
    with pytest.raises(ValueError, match="found 0"):
        select_row(rows, (("fold", "V3"),), "t")


def test_source_text_round_trip() -> None:
    source = MetricSource("results/t.csv", "", (("fold", "V1"), ("scope", "union")), "recall")
    assert source_from_text(source.to_text()) == source
    with pytest.raises(ValueError, match="not a metric source"):
        source_from_text(json.dumps({"file": "x"}))


def test_stage1_weekly_run_takes_the_locked_rows(tmp_path: Path) -> None:
    run = weekly_run(tmp_path)
    values = {metric.key: metric.value for metric in run.metrics}
    # Six channels, the union, the Ranker and the baseline: nine metrics.
    expected = {f"recall.{channel}": value for channel, value in CHANNEL_RECALL.items()}
    expected |= {"union_recall": 0.25, "map_at_12.ranker": 0.03, "map_at_12.rrf_baseline": 0.018}
    assert values == expected
    assert (run.record_id, run.record_kind, run.experiment) == (
        "stage1/V1",
        IMPORTED_RECORD,
        EXPERIMENT,
    )
    assert "(imported)" in run.run_name
    lock = ranker_lock()
    assert run.params["anchor"] == "2020-07-29"
    assert run.params["ranker_config_id"] == lock.selection.config_id
    assert run.params["n_queries"] == "1000"
    tags = run.all_tags()
    assert tags[RECORD_KIND_TAG] == IMPORTED_RECORD
    assert tags[RECORD_ID_TAG] == "stage1/V1"
    assert tags["hmrec.ranker_git_commit"] == COMMIT
    # Every metric says where it came from.
    ranker = next(metric for metric in run.metrics if metric.key == "map_at_12.ranker")
    assert ranker.source == MetricSource(
        "results/metrics.csv",
        "",
        (
            ("experiment", "full"),
            ("config_id", lock.selection.config_id),
            ("fold", "V1"),
            ("num_trees", str(lock.model.num_trees)),
        ),
        "map@12",
    )
    assert source_from_text(tags["source.map_at_12.ranker"]) == ranker.source
    assert len([key for key in tags if key.startswith("source.")]) == 9


def test_stage1_weekly_run_fails_on_a_missing_or_repeated_row(tmp_path: Path) -> None:
    tables = write_weekly_tables(tmp_path)
    recall = tables.recall_file.read_text(encoding="utf-8")
    tables.recall_file.write_text(recall + "V1,union,0.26\r\n", encoding="utf-8")
    with pytest.raises(ValueError, match="found 2"):
        stage1_weekly_run(fold_v1(), tables, ranker_lock(), EXPERIMENT, tmp_path)
    tables.recall_file.write_text(recall.replace("V1,union,0.25\n", ""), encoding="utf-8")
    with pytest.raises(ValueError, match="found 0"):
        stage1_weekly_run(fold_v1(), tables, ranker_lock(), EXPERIMENT, tmp_path)
    tables.recall_file.write_text(recall.replace("0.25", "n/a"), encoding="utf-8")
    with pytest.raises(ValueError, match="not a number"):
        stage1_weekly_run(fold_v1(), tables, ranker_lock(), EXPERIMENT, tmp_path)


COMPARISON_HEADER = (
    "item,scope,quantity,measured,stage1,difference,relative_difference,run,git_commit,"
    "os,architecture,openmp_version,image_commit,code_tree_sha256,finished_at"
)
PROCESS_HEADER = "item,scope,item_seconds,total_seconds,item_max_rss_gb,process_max_rss_gb,run"
STAMP = f"{COMMIT},linux,aarch64,unknown,{COMMIT},{'d' * 64},2026-11-21T16:16:38"


def _calibration_tables(root: Path) -> tuple[Path, Path]:
    folder = root / "results" / "m10_calibration"
    folder.mkdir(parents=True)
    comparisons = [
        COMPARISON_HEADER,
        f"B,r5_wals,recall,0.0999,0.1,-0.0001,-0.001,run1,{STAMP}",
        f"B,r5_wals,file_sha256_equal,False,True,,,run1,{STAMP}",
        f"B,r5_wals,rows,7203500,7203500,0,0.0,run1,{STAMP}",
        f"D,500 trees,map@12,0.0348,0.0348,0.0,0.0,run1,{STAMP}",
        f"B,r5_wals,recall,0.1,0.1,0.0,0.0,run2,{STAMP}",
    ]
    processes = [PROCESS_HEADER, "B,r5_wals,116.5,120.25,5.5,6.0,run1"]
    (folder / "comparisons.csv").write_text("\r\n".join(comparisons) + "\r\n", encoding="utf-8")
    (folder / "runs.csv").write_text("\r\n".join(processes) + "\r\n", encoding="utf-8")
    return folder / "comparisons.csv", folder / "runs.csv"


def test_calibration_runs_one_per_run_item_and_scope(tmp_path: Path) -> None:
    comparisons, processes = _calibration_tables(tmp_path)
    runs = calibration_runs(comparisons, processes, "m10 (test)", tmp_path)
    assert [run.record_id for run in runs] == [
        "m10/run1/B/r5_wals",
        "m10/run1/D/500 trees",
        "m10/run2/B/r5_wals",
    ]
    assert all(run.record_kind == IMPORTED_RECORD for run in runs)
    first = {metric.key: metric.value for metric in runs[0].metrics}
    # Numbers become metrics (measured, the stage-1 record, the relative difference); the
    # process row adds time and memory. "False" is not a number: it becomes a parameter.
    assert first == {
        "recall": 0.0999,
        "recall.stage1": 0.1,
        "recall.relative_difference": -0.001,
        "rows": 7203500.0,
        "rows.stage1": 7203500.0,
        "rows.relative_difference": 0.0,
        "process.item_seconds": 116.5,
        "process.total_seconds": 120.25,
        "process.item_max_rss_gb": 5.5,
        "process.process_max_rss_gb": 6.0,
    }
    assert runs[0].params == {
        "run": "run1",
        "item": "B",
        "scope": "r5_wals",
        "measured.file_sha256_equal": "False",
    }
    assert runs[0].tags["hmrec.platform"] == "linux-aarch64"
    # A group without a process row has only its comparison metrics; "@" is replaced.
    assert {metric.key for metric in runs[1].metrics} == {
        "map_at_12",
        "map_at_12.stage1",
        "map_at_12.relative_difference",
    }
    source = runs[1].metrics[0].source
    assert source.file == "results/m10_calibration/comparisons.csv"
    assert dict(source.where) == {
        "run": "run1",
        "item": "D",
        "scope": "500 trees",
        "quantity": "map@12",
    }
    assert source.column == "measured"


def test_pipeline_run_from_a_manifest(tmp_path: Path) -> None:
    manifest = example_manifest()
    manifest_file = tmp_path / "artifacts" / "pipeline" / "runs" / manifest.run_id / "manifest.json"
    write_run_manifest(manifest_file, manifest)
    run = pipeline_run(manifest, manifest_file, "weekly (test)", tmp_path)
    assert (run.record_id, run.record_kind) == ("pipeline/2020-08-05-eval-0123456", PIPELINE_RUN)
    values = {metric.key: metric.value for metric in run.metrics}
    # Labelled metrics describe the week the tick closed; the others the week it opened.
    assert values["closed_week/map_at_12"] == 0.030112271332103004
    assert values["closed_week/union_recall"] == 0.2543785197141435
    assert values["opened_week/candidates_per_query"] == 306.5
    assert values["step.O3_train_ranker.seconds"] == 278.0
    assert values["step.O3_train_ranker.process_max_rss_gb"] == 6.5
    assert values["max_process_rss_gb"] == 6.5
    assert values["total_seconds"] == 800.0
    # 3 metrics, 2 per step for 3 steps, 3 of the whole tick.
    assert len(values) == 12
    assert run.params["mode"] == "eval"
    assert run.tags["hmrec.image_digest"].startswith("sha256:")
    assert run.tags["hmrec.platform"] == "linux-arm64"
    labelled = next(metric for metric in run.metrics if metric.key == "closed_week/map_at_12")
    assert labelled.source == MetricSource(
        f"artifacts/pipeline/runs/{manifest.run_id}/manifest.json",
        "metrics",
        (("name", "map@12"), ("week", "2020-07-29")),
        "value",
    )


def test_pipeline_run_refuses_a_metric_of_another_week(tmp_path: Path) -> None:
    other_week = (MetricRecord("map@12", date(2020, 7, 22), 0.03),)
    manifest = replace(example_manifest(), metrics=other_week)
    with pytest.raises(ValueError, match="neither the tick's anchor"):
        pipeline_run(manifest, tmp_path / "manifest.json", "weekly (test)", tmp_path)


def test_two_metrics_with_one_key_are_refused(tmp_path: Path) -> None:
    run = weekly_run(tmp_path)
    with pytest.raises(ValueError, match="repeat"):
        check_unique_keys(replace(run, metrics=(run.metrics[0], run.metrics[0])))


def test_the_view_of_this_repository() -> None:
    paths = get_paths()
    config = load_versioning_config(paths.versioning_config_file).mlflow
    manifest = load_fold_manifest(paths.fold_manifest_file)
    runs = view_runs(paths, config, manifest, ranker_lock())
    weekly = [run for run in runs if run.experiment == config.experiment_stage1]
    # The eight weeks of the replay, each with nine metrics.
    assert [run.record_id for run in weekly] == [
        f"stage1/{fold}" for fold in ("V1", "V2", "V3", "V4", "T1", "T2", "T3", "T4")
    ]
    assert all(len(run.metrics) == 9 for run in weekly)
    v1 = {metric.key: metric.value for metric in weekly[0].metrics}
    # results/v_phase_metrics.csv, row (full, ca1a3cdfce49, V1, 500), column map@12;
    # results/m6_candidates.csv, row (V1, union), column recall.
    assert v1["map_at_12.ranker"] == 0.030112271332103004
    assert v1["map_at_12.rrf_baseline"] == 0.018510960720404597
    assert v1["union_recall"] == 0.2543785197141435
    t1 = {metric.key: metric.value for metric in weekly[4].metrics}
    # results/t_phase_metrics.csv and results/t_phase_recall.csv: recorded numbers, no label.
    assert t1["map_at_12.ranker"] == 0.03495997634509058
    assert t1["recall.r6_twotower"] == 0.17326230234379905
    calibration = [run for run in runs if run.experiment == config.experiment_calibration]
    # The calibration tables are append-only: there are at least the 37 groups of M10.
    assert len(calibration) >= 37
    assert all(run.record_kind == IMPORTED_RECORD for run in weekly + calibration)
    assert all("(imported)" in run.experiment for run in weekly + calibration)
    assert paths.project_root == REPO
