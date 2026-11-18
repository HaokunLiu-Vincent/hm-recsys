"""Tests for hmrec.hpo.v_records: the V-phase result tables and pooled scores.

Pooled score by hand (query-weighted, proposal §4.1): two folds with 100 and 300
queries and MAP 0.02 and 0.04:
    pooled     = (100 * 0.02 + 300 * 0.04) / 400 = 0.035
    equal fold = (0.02 + 0.04) / 2               = 0.03
"""

from pathlib import Path

import pytest
from ranker_example import METRICS, tiny_inputs

from hmrec.csv_records import read_rows
from hmrec.hpo.v_records import (
    AREA_SPEED_TEST,
    AREA_V_PHASE,
    M7_RUN_COLUMNS,
    TRIAL_COLUMNS,
    MetricRow,
    RankerTrialRecord,
    append_m7_run_row,
    append_metric_row,
    append_trial,
    complete_tree_counts,
    config_rows,
    equal_fold_score,
    find_trial,
    metric_columns,
    pooled_curve,
    pooled_score,
    read_metric_rows,
    read_trial_rows,
    results_files,
    rows_at,
    scores_at,
)
from hmrec.ranker.runtime import RUNTIME_COLUMNS, LightGBMRuntime
from hmrec.run_records import RUN_COLUMNS


def _row(fold: str, num_trees: int, n_queries: int, value: float, config: str = "c1") -> MetricRow:
    scores = {"map@12": value, "recall@12": value * 2, "ndcg@12": value * 3}
    return MetricRow("full", config, fold, num_trees, n_queries, scores, None)


def _record(**changes: object) -> RankerTrialRecord:
    values: dict[str, object] = {
        "phase": "V",
        "experiment": "full",
        "step": "leaves_x_min_data",
        "config_id": "0123456789ab",
        "fold": "V1",
        "train_folds": ("F1", "F2", "F3", "F4"),
        "params": {"num_leaves": 31, "min_data_in_leaf": 500, "negatives_per_positive": 40},
        "config": {"seed": 42},
        "histogram_mode": "col_wise",
        "num_trees": 2000,
        "evaluated": True,
        "n_train_rows": 10,
        "n_train_queries": 3,
        "n_scored_rows": 50,
        "n_queries": 5,
        "load_seconds": 0.1,
        "train_seconds": 1 / 3,
        "seconds_per_tree": 0.5,
        "score_seconds": 2.0,
        "total_seconds": 3.0,
        "sampled_peak_rss_gb": 1.25,
        "process_max_rss_gb": 1.5,
        "model_sha256": "f" * 64,
        "runtime": LightGBMRuntime(
            "4.7.0", "/opt/homebrew/Cellar/libomp/22.1.4/lib/libomp.dylib", "22.1.4"
        ),
        "git_commit": "abc",
        "git_dirty": False,
        "finished_at": "2026-11-18T10:00:00",
    }
    values.update(changes)
    return RankerTrialRecord(**values)  # pyright: ignore[reportArgumentType]  test helper


def test_trial_rows_keep_both_memory_peaks(tmp_path: Path) -> None:
    path = tmp_path / "v_phase.csv"
    append_trial(path, _record())
    append_trial(path, _record(fold="V2"))
    rows = read_trial_rows(path)
    assert list(rows[0]) == list(TRIAL_COLUMNS)
    # The sampled peak and the true peak (ru_maxrss) are separate columns.
    assert rows[0]["sampled_peak_rss_gb"] == "1.25" and rows[0]["process_max_rss_gb"] == "1.5"
    assert float(rows[0]["train_seconds"]) == 1 / 3  # repr() round-trips exactly
    assert rows[0]["train_folds"] == "F1 F2 F3 F4"
    assert [row["fold"] for row in rows] == ["V1", "V2"]
    assert read_trial_rows(tmp_path / "missing.csv") == []
    # LightGBM and the OpenMP runtime the trial's process loaded (libomp is not in uv.lock).
    assert rows[0]["lightgbm_version"] == "4.7.0" and rows[0]["libomp_version"] == "22.1.4"
    assert rows[0]["libomp_path"] == "/opt/homebrew/Cellar/libomp/22.1.4/lib/libomp.dylib"


def test_m7_run_rows_carry_the_runtime(tmp_path: Path) -> None:
    assert (*RUN_COLUMNS, *RUNTIME_COLUMNS) == M7_RUN_COLUMNS
    assert RUNTIME_COLUMNS == ("lightgbm_version", "libomp_path", "libomp_version")
    path = tmp_path / "m7_runs.csv"
    run_row = dict.fromkeys(RUN_COLUMNS, "x")
    append_m7_run_row(path, run_row, LightGBMRuntime("4.7.0", "/lib/libomp.dylib", "unknown"))
    (row,) = read_rows(path, M7_RUN_COLUMNS)
    assert row["fold"] == "x" and row["lightgbm_version"] == "4.7.0"
    assert (row["libomp_path"], row["libomp_version"]) == ("/lib/libomp.dylib", "unknown")


def test_find_trial_with_and_without_step(tmp_path: Path) -> None:
    path = tmp_path / "trials.csv"
    append_trial(path, _record(step="histogram_col_wise", evaluated=False))
    append_trial(path, _record(step="speed_test"))
    rows = read_trial_rows(path)
    found = find_trial(rows, "full", "0123456789ab", "V1", "speed_test")
    assert found is not None and found["evaluated"] == "True"
    assert find_trial(rows, "full", "0123456789ab", "V2") is None
    assert find_trial(rows, "minus_r6", "0123456789ab", "V1", "speed_test") is None
    # Without a step both rows match: in the grid a configuration runs once per fold.
    with pytest.raises(ValueError, match="recorded twice"):
        find_trial(rows, "full", "0123456789ab", "V1")


def test_metric_rows_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    first = MetricRow("full", "c1", "V1", 100, 82802, dict.fromkeys(METRICS, 1 / 3), 0.123)
    baseline_scores: dict[str, float] = dict.fromkeys(METRICS, 0.01)
    baseline = MetricRow("rrf_baseline", "rrf_baseline", "V1", 0, 82802, baseline_scores, None)
    append_metric_row(path, first, METRICS, "abc", "t")
    append_metric_row(path, baseline, METRICS, "abc", "t")
    assert read_metric_rows(path, METRICS) == [first, baseline]
    assert metric_columns(METRICS)[5:9] == (*METRICS, "train_ndcg@12")
    with pytest.raises(ValueError, match="expected"):
        append_metric_row(path, first, ("map@12",), "abc", "t")


def test_pooled_and_equal_fold_scores_by_hand() -> None:
    fold_rows = {"V1": _row("V1", 100, 100, 0.02), "V2": _row("V2", 100, 300, 0.04)}
    assert pooled_score(fold_rows, "map@12") == pytest.approx(0.035)
    assert equal_fold_score(fold_rows, "map@12") == pytest.approx(0.03)
    assert pooled_score(fold_rows, "recall@12") == pytest.approx(0.07)


def test_curve_uses_only_tree_counts_that_every_fold_has() -> None:
    rows = [
        _row("V1", 100, 100, 0.02),
        _row("V2", 100, 300, 0.04),
        _row("V1", 200, 100, 0.03),
        _row("V2", 200, 300, 0.05),
        _row("V1", 300, 100, 0.09),  # V2 has no row at 300 yet
        _row("V1", 100, 100, 0.5, config="other"),
    ]
    own = config_rows(rows, "full", "c1")
    assert len(own) == 5
    assert complete_tree_counts(own, ["V1", "V2"]) == [100, 200]
    curve = pooled_curve(own, ["V1", "V2"], "map@12")
    assert curve == {100: pytest.approx(0.035), 200: pytest.approx(0.045)}
    scores = scores_at(own, ["V1", "V2"], "map@12", 200, "c1")
    assert scores.trial_id == "c1" and scores.fold_scores == {"V1": 0.03, "V2": 0.05}
    assert scores.pooled_score == pytest.approx(0.045)
    with pytest.raises(ValueError, match="no metric row at 300 trees for folds \\['V2'\\]"):
        rows_at(own, ["V1", "V2"], 300)
    with pytest.raises(ValueError, match="recorded twice"):
        rows_at([*own, _row("V1", 100, 100, 0.02)], ["V1", "V2"], 100)


def test_areas_have_separate_files(tmp_path: Path) -> None:
    paths = tiny_inputs(tmp_path).paths
    grid_files = results_files(paths, AREA_V_PHASE)
    speed_files = results_files(paths, AREA_SPEED_TEST)
    assert [file.name for file in grid_files] == ["v_phase.csv", "v_phase_metrics.csv"]
    assert [file.name for file in speed_files] == ["m7_speed_test.csv", "m7_speed_test_metrics.csv"]
    with pytest.raises(ValueError, match="area must be one of"):
        results_files(paths, "d_phase")
    # The two areas of the final evaluation (M8) have their own tables.
    assert results_files(paths, "t_phase") == (
        paths.t_phase_results_file,
        paths.t_phase_metrics_file,
    )
    assert results_files(paths, "m8_rehearsal") == (
        paths.m8_rehearsal_results_file,
        paths.m8_rehearsal_metrics_file,
    )
    assert len({*grid_files, *speed_files, *results_files(paths, "t_phase")}) == 6
