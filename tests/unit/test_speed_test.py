"""Tests for hmrec.hpo.speed_test: the histogram decision, the summary and the run order."""

from pathlib import Path

import pytest
from ranker_example import METRICS, REPO, tiny_inputs
from v_phase_example import RULE, TINY_GRID_RAW, FakeLauncher

from hmrec.config import RankerTunedParams, load_ranker_config
from hmrec.hpo.speed_test import (
    EVALUATED_STEP,
    TIMING_STEPS,
    histogram_timing,
    report_lines,
    run_speed_test,
    summarize,
    training_data_lines,
)
from hmrec.hpo.v_phase import (
    BASELINE_SCRIPT,
    DATASET_SCRIPT,
    SAMPLE_SCRIPT,
    TRIAL_SCRIPT,
    GridContext,
    parse_v_phase_grid,
)
from hmrec.hpo.v_records import MetricRow, read_trial_rows
from hmrec.ranker.sampling import SampleRecord

UNDECIDED = {"mode": "undecided", "comparison_trees": 6, "warmup_trees": 2,
             "min_row_wise_speedup": 0.15}  # fmt: skip


def _timing_row(seconds_per_tree: float, peak_gb: float) -> dict[str, str]:
    return {"seconds_per_tree": repr(seconds_per_tree), "process_max_rss_gb": repr(peak_gb)}


def test_histogram_timing_applies_the_rule() -> None:
    faster = histogram_timing(_timing_row(1.0, 2.0), _timing_row(0.7, 3.5), 8.0, 0.15)
    assert faster.chosen_mode == "row_wise"
    assert (faster.col_wise_peak_gb, faster.row_wise_peak_gb) == (2.0, 3.5)
    close = histogram_timing(_timing_row(1.0, 2.0), _timing_row(0.9, 3.5), 8.0, 0.15)
    assert close.chosen_mode == "col_wise"
    too_big = histogram_timing(_timing_row(1.0, 2.0), _timing_row(0.5, 8.2), 8.0, 0.15)
    assert too_big.chosen_mode == "col_wise"


def _metric(config: str, k: int, value: float, train: float | None) -> MetricRow:
    experiment = "rrf_baseline" if config == "rrf_baseline" else "full"
    return MetricRow(experiment, config, "V1", k, 100, dict.fromkeys(METRICS, value), train)


TRIAL = {
    "experiment": "full", "config_id": "c1", "fold": "V1", "histogram_mode": "col_wise",
    "num_trees": "300", "n_train_rows": "10714226", "seconds_per_tree": "0.5",
    "train_seconds": "150.0", "score_seconds": "60.0", "n_scored_rows": "25226147",
    "process_max_rss_gb": "4.5", "lightgbm_version": "4.7.0", "libomp_version": "22.1.4",
    "libomp_path": "/opt/homebrew/Cellar/libomp/22.1.4/lib/libomp.dylib",
}  # fmt: skip
TIMING = histogram_timing(_timing_row(1.0, 2.0), _timing_row(0.9, 3.5), 8.0, 0.15)


def test_summary_above_the_baseline() -> None:
    rows = [
        _metric("c1", 100, 0.024, 0.60),
        _metric("c1", 200, 0.027, 0.63),
        _metric("c1", 300, 0.029, 0.65),
        _metric("rrf_baseline", 0, 0.020, None),
        _metric("other", 300, 0.001, 0.1),
    ]
    summary = summarize(TIMING, TRIAL, rows, "V1", "map@12")
    assert [row.num_trees for row in summary.checkpoints] == [100, 200, 300]
    assert summary.baseline.scores["map@12"] == 0.020
    assert summary.below_baseline is False and summary.train_metric_rises is True
    lines = report_lines(summary, [], METRICS)
    assert not any(line.startswith("STOP") for line in lines)
    assert any("300 trees: map@12 0.029000" in line for line in lines)
    assert any("RRF baseline: map@12 0.020000" in line for line in lines)
    assert any("LightGBM 4.7.0; OpenMP runtime /opt/homebrew/Cellar/libomp/22.1.4" in line
               for line in lines)  # fmt: skip


def test_summary_below_the_baseline_says_stop() -> None:
    rows = [
        _metric("c1", 100, 0.010, 0.60),
        _metric("c1", 200, 0.015, 0.58),  # the training metric went DOWN
        _metric("c1", 300, 0.019, 0.61),
        _metric("rrf_baseline", 0, 0.020, None),
    ]
    summary = summarize(TIMING, TRIAL, rows, "V1", "map@12")
    # Only the LAST checkpoint decides: 0.019 < 0.020.
    assert summary.below_baseline is True and summary.train_metric_rises is False
    lines = report_lines(summary, [], METRICS)
    assert lines[-1].startswith("STOP") and "do not extrapolate" in lines[-1]


def test_summary_needs_the_baseline_and_the_checkpoints() -> None:
    with pytest.raises(ValueError, match="needs its checkpoints and one baseline row"):
        summarize(TIMING, TRIAL, [_metric("c1", 100, 0.02, 0.5)], "V1", "map@12")
    with pytest.raises(ValueError, match="needs its checkpoints and one baseline row"):
        summarize(TIMING, TRIAL, [_metric("rrf_baseline", 0, 0.02, None)], "V1", "map@12")


def test_training_data_lines() -> None:
    record = SampleRecord(
        fold="F1", experiment="full", ratio=40, seed=42, n_rows=2951046, n_queries=41420,
        n_positives=72385, rows_per_query_mean=71.2, rows_per_query_p50=41.0,
        rows_per_query_p90=123.0, columns=(), sha256="", features_record_sha256="", git_commit="",
    )  # fmt: skip
    (line,) = training_data_lines([record])
    assert "F1 r=40: 2,951,046 rows, 41,420 queries" in line
    assert "positives 72,385 (2.4529%)" in line
    assert "mean 71.2, P50 41, P90 123" in line


def _context(tmp_path: Path, launcher_setup: dict[str, float]) -> tuple[GridContext, FakeLauncher]:
    inputs = tiny_inputs(tmp_path, histogram=UNDECIDED)
    launcher = FakeLauncher(inputs, lambda tuned, fold, k: 0.02 + k / 10000, baseline=0.015)
    launcher.seconds_per_tree.update(launcher_setup)
    grid = parse_v_phase_grid(TINY_GRID_RAW)
    context = GridContext(
        inputs, grid, RULE, Path("scripts"), launcher, "abc", check_speed_test_model=False
    )
    return context, launcher


def test_speed_test_runs_every_part_once_and_picks_row_wise(tmp_path: Path) -> None:
    context, launcher = _context(tmp_path, {"col_wise": 1.0, "row_wise": 0.7})
    summary = run_speed_test(context)
    # histogram.mode in the config is still undecided: the speed test chooses itself.
    assert context.inputs.ranker.histogram.mode == "undecided"
    assert summary.timing.chosen_mode == "row_wise"
    assert summary.trial["histogram_mode"] == "row_wise" and summary.trial["step"] == EVALUATED_STEP
    assert [row.num_trees for row in summary.checkpoints] == [10, 20]
    assert summary.below_baseline is False

    sampled = [args["fold"][0] for name, args in launcher.calls if name == SAMPLE_SCRIPT]
    assert sampled == ["F1", "F2", "F3", "F4"]  # the training folds of V1, nothing else
    assert launcher.count(DATASET_SCRIPT) == 1
    trials = [args for name, args in launcher.calls if name == TRIAL_SCRIPT]
    assert [args["step"][0] for args in trials] == [*TIMING_STEPS.values(), EVALUATED_STEP]
    # The two timing runs train 6 trees without evaluation; all three are the default config.
    assert ["no-eval" in args for args in trials] == [True, True, False]
    assert [args["num-trees"][0] for args in trials] == ["6", "6", "20"]
    assert all(args["area"] == ["speed_test"] and args["target"] == ["V1"] for args in trials)
    defaults = context.inputs.ranker.tuned_defaults
    assert defaults == RankerTunedParams(4, 5, 2)
    assert all(args["num-leaves"] == ["4"] and args["ratio"] == ["2"] for args in trials)
    assert launcher.count(BASELINE_SCRIPT) == 1
    # Nothing was written to the grid's tables.
    paths = context.inputs.paths
    assert not paths.v_phase_results_file.exists() and not paths.v_phase_metrics_file.exists()
    assert len(read_trial_rows(paths.m7_speed_test_file)) == 3

    calls_before = len(launcher.calls)
    assert run_speed_test(context).timing.chosen_mode == "row_wise"
    assert len(launcher.calls) == calls_before  # every part is recorded: nothing runs again


def test_speed_test_keeps_col_wise_when_row_wise_is_not_clearly_faster(tmp_path: Path) -> None:
    context, launcher = _context(tmp_path, {"col_wise": 1.0, "row_wise": 0.9})
    summary = run_speed_test(context)
    assert summary.timing.chosen_mode == "col_wise"
    assert summary.trial["histogram_mode"] == "col_wise"


def test_real_speed_test_settings() -> None:
    ranker = load_ranker_config(REPO / "configs" / "ranker.yaml")
    assert (ranker.speed_test.target_fold, ranker.speed_test.num_trees) == ("V1", 300)
    assert (ranker.histogram.comparison_trees, ranker.histogram.warmup_trees) == (50, 10)
    assert ranker.histogram.min_row_wise_speedup == 0.15
