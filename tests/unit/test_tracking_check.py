"""Tests for hmrec.tracking.check: every logged metric against its source."""

from dataclasses import replace
from pathlib import Path

from run_manifest_example import example_manifest
from tracking_example import as_logged, weekly_run

from hmrec.run_manifest import write_run_manifest
from hmrec.tracking.check import check_logged_runs, source_value
from hmrec.tracking.records import MetricSource, pipeline_run


def test_a_correct_view_passes(tmp_path: Path) -> None:
    run = weekly_run(tmp_path)
    result = check_logged_runs([as_logged(run)], [run], tmp_path)
    assert (result.n_runs, result.n_metrics, result.problems) == (1, 9, ())
    assert result.passed


def test_a_metric_that_differs_from_its_source_is_reported(tmp_path: Path) -> None:
    run = weekly_run(tmp_path)
    logged = as_logged(run)
    # The smallest possible difference: the next float64 after 0.03.
    logged.metrics["map_at_12.ranker"] = 0.030000000000000002
    result = check_logged_runs([logged], [run], tmp_path)
    assert not result.passed
    assert result.problems == (
        "stage1 (test) / stage1/V1 / map_at_12.ranker: MLflow has 0.030000000000000002, "
        "the source has 0.03",
    )


def test_the_source_is_read_again_from_the_file(tmp_path: Path) -> None:
    run = weekly_run(tmp_path)
    # The table changes after the run was built: the logged value no longer equals it.
    table = tmp_path / "results" / "recall.csv"
    table.write_text(
        table.read_text(encoding="utf-8").replace("V1,union,0.25", "V1,union,0.26"),
        encoding="utf-8",
    )
    result = check_logged_runs([as_logged(run)], [run], tmp_path)
    assert result.problems == (
        "stage1 (test) / stage1/V1 / union_recall: MLflow has 0.25, the source has 0.26",
    )


def test_a_metric_without_a_source_or_with_a_lost_source_is_reported(tmp_path: Path) -> None:
    run = weekly_run(tmp_path)
    logged = as_logged(run)
    del logged.tags["source.union_recall"]
    result = check_logged_runs([logged], [run], tmp_path)
    assert result.problems == (
        "stage1 (test) / stage1/V1 / union_recall: the metric has no source tag",
    )
    (tmp_path / "results" / "metrics.csv").unlink()
    result = check_logged_runs([as_logged(run)], [run], tmp_path)
    assert len(result.problems) == 2
    assert all("source file not found: results/metrics.csv" in line for line in result.problems)


def test_missing_unexpected_and_incomplete_runs_are_reported(tmp_path: Path) -> None:
    run = weekly_run(tmp_path)
    assert check_logged_runs([], [run], tmp_path).problems == (
        "stage1 (test) / stage1/V1: expected, but not in the store",
    )
    assert check_logged_runs([as_logged(run)], [], tmp_path).problems == (
        "stage1 (test) / stage1/V1: in the store, but not expected",
    )
    incomplete = as_logged(run)
    del incomplete.metrics["union_recall"]
    assert check_logged_runs([incomplete], [run], tmp_path).problems == (
        "stage1 (test) / stage1/V1: metric keys differ: ['union_recall']",
    )
    twice = check_logged_runs([as_logged(run), as_logged(run)], [run], tmp_path)
    assert "two runs with the same record id" in twice.problems[0]


def test_metrics_of_a_run_manifest_are_checked_against_the_file(tmp_path: Path) -> None:
    manifest = example_manifest()
    manifest_file = tmp_path / "artifacts" / "pipeline" / "runs" / manifest.run_id / "manifest.json"
    write_run_manifest(manifest_file, manifest)
    run = pipeline_run(manifest, manifest_file, "weekly (test)", tmp_path)
    result = check_logged_runs([as_logged(run)], [run], tmp_path)
    assert (result.n_metrics, result.problems) == (12, ())
    # A list inside the manifest, and its top level.
    file = f"artifacts/pipeline/runs/{manifest.run_id}/manifest.json"
    step = MetricSource(file, "steps", (("name", "O3_train_ranker"),), "seconds")
    assert source_value(tmp_path, step) == 278.0
    assert source_value(tmp_path, MetricSource(file, "", (), "total_seconds")) == 800.0
    # The view built from another manifest than the file holds: reported, metric by metric.
    other = replace(manifest, total_seconds=801.0)
    wrong = pipeline_run(other, manifest_file, "weekly (test)", tmp_path)
    result = check_logged_runs([as_logged(wrong)], [wrong], tmp_path)
    assert result.problems == (
        f"weekly (test) / pipeline/{manifest.run_id} / total_seconds: MLflow has 801.0, "
        "the source has 800.0",
    )
