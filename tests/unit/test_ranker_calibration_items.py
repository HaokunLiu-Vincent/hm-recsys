"""Tests for hmrec.calibration.ranker_items (LightGBM side; the file name puts it there).

The item itself (training the locked model on four weeks, scoring 22 million
rows) runs only on the full data. Here: the rows it writes are checked on small
hand-made inputs, and the two platform branches of its first step are checked:
on the laptop the tolerance-accepted entry refuses, in a container it accepts
the platform and then stops at the first missing input.
"""

import shutil
import sys
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from hmrec.calibration.context import CalibrationContext
from hmrec.calibration.ranker_items import (
    END_TO_END_AREA,
    load_ranker_setup,
    metric_measurements,
    model_measurements,
    require_new_outputs,
    run_end_to_end_ranker,
    top_k_measurements,
)
from hmrec.calibration.records import Measurement
from hmrec.config import load_base_config, load_calibration_config
from hmrec.hpo.v_records import MetricRow
from hmrec.hpo.v_trial import TrialOutcome
from hmrec.paths import build_paths, calibration_output_paths, get_paths
from hmrec.run_records import RunStamp

REPO = Path(__file__).resolve().parents[2]
REPO_PATHS = get_paths()
CONFIG = load_calibration_config(REPO_PATHS.m10_calibration_config_file)
COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _context(root: Path) -> CalibrationContext:
    """A context on a copy of configs/ under `root` (no data, no artifacts)."""
    shutil.copytree(REPO / "configs", root / "configs")
    base = load_base_config(root / "configs" / "base.yaml")
    paths = build_paths(base, root)
    return CalibrationContext(
        paths=paths,
        main_paths=calibration_output_paths(paths, "linux-arm64", "run1", "main"),
        recorded_model_paths=calibration_output_paths(
            paths, "linux-arm64", "run1", "recorded_model"
        ),
        base=base,
        config=CONFIG,
        run="run1",
        platform_name="linux-arm64",
        stamp=RunStamp(COMMIT, False),
        started=0.0,
    )


def _by_key(measurements: list[Measurement]) -> dict[tuple[str, str], Measurement]:
    return {(m.scope, m.quantity): m for m in measurements}


def test_the_first_step_on_this_platform(tmp_path: Path) -> None:
    context = _context(tmp_path)
    if sys.platform == "darwin":
        # The laptop: the tolerance-accepted entry refuses. The strict entry is the only
        # one here, and a calibration of the Ranker runs in a container.
        with pytest.raises(RuntimeError, match="runs on Linux only"):
            load_ranker_setup(context)
    else:
        # A container: the platform is accepted in calibration mode and the lock checks
        # pass; the temporary project has no sample files, so the input check stops it.
        with pytest.raises(FileNotFoundError):
            load_ranker_setup(context)


def test_a_t_fold_is_refused_before_any_input_is_read(tmp_path: Path) -> None:
    context = _context(tmp_path)
    t_fold = replace(context, config=replace(CONFIG, fold="T1"))
    if sys.platform == "darwin":
        # The laptop: the entry refuses the platform before the fold is even looked at.
        with pytest.raises(RuntimeError, match="runs on Linux only"):
            load_ranker_setup(t_fold)
    else:
        # A container: the M7 window refuses a T fold, before a sample or label is read.
        with pytest.raises(ValueError, match="M7 predicts and evaluates only V folds"):
            load_ranker_setup(t_fold)


def _outcome(scores_by_trees: dict[int, dict[str, float]]) -> TrialOutcome:
    rows = tuple(
        MetricRow("full", "ca1a3cdfce49", "V4", num_trees, 72035, scores, None)
        for num_trees, scores in scores_by_trees.items()
    )
    return TrialOutcome(
        "ca1a3cdfce49", {}, ("F4", "V1", "V2", "V3"), 1, 1, 1, 72035, 0.0, 0.0, 0.0, 0.0, "", rows
    )


def test_metric_rows_and_the_tolerance_on_map_at_the_locked_trees(tmp_path: Path) -> None:
    context = _context(tmp_path)
    recorded = {
        100: {"map@12": 0.030, "recall@12": 0.070, "ndcg@12": 0.050},
        500: {"map@12": 0.040, "recall@12": 0.080, "ndcg@12": 0.060},
    }
    # At 500 trees MAP@12 is 0.5% above the record (inside 1%); Recall@12 is 5% below it,
    # but docs/mlops.md 8.3 names no tolerance for Recall@12, so it is not judged.
    measured = {
        100: {"map@12": 0.020, "recall@12": 0.070, "ndcg@12": 0.050},
        500: {"map@12": 0.0402, "recall@12": 0.076, "ndcg@12": 0.060},
    }
    rows = _by_key(metric_measurements(context, _outcome(measured).metric_rows, recorded))
    assert len(rows) == 6
    judged = rows[("500 trees", "map@12")]
    assert (judged.measured, judged.stage1) == (0.0402, 0.040)
    assert judged.tolerance == "relative difference <= 0.01" and judged.within_tolerance is True
    # Earlier checkpoints and the other metrics are reported without a verdict.
    assert rows[("100 trees", "map@12")].within_tolerance is None
    assert rows[("500 trees", "recall@12")].within_tolerance is None
    # 2% above the record at the locked trees: outside.
    outside = {500: {"map@12": 0.0408, "recall@12": 0.080, "ndcg@12": 0.060}}
    rows = _by_key(metric_measurements(context, _outcome(outside).metric_rows, recorded))
    assert rows[("500 trees", "map@12")].within_tolerance is False
    with pytest.raises(ValueError, match="stage 1 recorded no metrics at 300 trees"):
        metric_measurements(context, _outcome({300: measured[100]}).metric_rows, recorded)


def _model_text(leaf_values: list[str]) -> str:
    lines = ["tree", "version=v4", ""]
    for index, value in enumerate(leaf_values):
        lines.extend([f"Tree={index}", f"leaf_value={value}", ""])
    lines.extend(["end of trees", "", "feature_importances:"])
    return "\n".join(lines)


def test_model_measurements_name_the_first_tree_that_differs(tmp_path: Path) -> None:
    recorded_file, measured_file = tmp_path / "recorded.txt", tmp_path / "measured.txt"
    recorded_file.write_text(_model_text(["0.1", "0.2", "0.3", "0.4"]), encoding="utf-8")
    # Trees 0 and 1 are the same; tree 2 is the first that differs.
    measured_file.write_text(_model_text(["0.1", "0.2", "0.31", "0.41"]), encoding="utf-8")
    rows = _by_key(model_measurements(measured_file, recorded_file, "aaa", "bbb"))
    assert (rows[("model", "trees")].measured, rows[("model", "trees")].stage1) == (4, 4)
    assert rows[("model", "leading_identical_trees")].measured == 2
    assert rows[("model", "first_differing_tree")].measured == 2
    assert rows[("model", "model_sha256_equal")].measured is False
    same = _by_key(model_measurements(recorded_file, recorded_file, "aaa", "aaa"))
    assert same[("model", "leading_identical_trees")].measured == 4
    assert same[("model", "first_differing_tree")].measured == "none"
    assert same[("model", "model_sha256_equal")].measured is True


def _predictions(path: Path, lists: dict[int, dict[int, list[int]]]) -> None:
    """Write num_trees -> query_id -> ranked articles as a predictions file."""
    rows = [
        (num_trees, query_id, rank, article, 1.0 / rank)
        for num_trees, queries in lists.items()
        for query_id, articles in queries.items()
        for rank, article in enumerate(articles, start=1)
    ]
    schema = {
        "num_trees": pl.Int32,
        "query_id": pl.Int32,
        "rank": pl.Int32,
        "article_idx": pl.Int32,
        "score": pl.Float64,
    }
    pl.DataFrame(rows, schema=schema, orient="row").write_parquet(path)


def test_top_k_measurements_use_only_the_locked_number_of_trees(tmp_path: Path) -> None:
    # At 500 trees: query 1 has the same list, query 2 has one of two articles replaced.
    # At 100 trees everything differs; that checkpoint must not be read.
    _predictions(
        tmp_path / "recorded.parquet", {100: {1: [9, 8], 2: [7, 6]}, 500: {1: [1, 2], 2: [3, 4]}}
    )
    _predictions(
        tmp_path / "measured.parquet", {100: {1: [5, 5], 2: [5, 5]}, 500: {1: [1, 2], 2: [3, 5]}}
    )
    rows = _by_key(
        top_k_measurements(tmp_path / "measured.parquet", tmp_path / "recorded.parquet", 500, "x")
    )
    assert rows[("500 trees", "queries")].measured == 2
    assert rows[("500 trees", "identical_top12_list_query_share")].measured == 0.5
    assert rows[("500 trees", "identical_top12_set_query_share")].measured == 0.5
    # (2/2 + 1/2) / 2 = 0.75
    assert rows[("500 trees", "mean_top12_overlap")].measured == 0.75


def test_require_new_outputs(tmp_path: Path) -> None:
    require_new_outputs((tmp_path / "a.parquet", tmp_path / "model.txt"))
    (tmp_path / "model.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Nothing is reused"):
        require_new_outputs((tmp_path / "a.parquet", tmp_path / "model.txt"))


# ---------------------------------------------------------------- item E


def test_item_e_rows_carry_their_own_item_and_the_same_tolerance_on_map(tmp_path: Path) -> None:
    context = _context(tmp_path)
    recorded = {500: {"map@12": 0.040, "recall@12": 0.080, "ndcg@12": 0.060}}
    # End to end, MAP@12 is 0.25% below the record: inside the provisional 1%.
    measured = {500: {"map@12": 0.0399, "recall@12": 0.0799, "ndcg@12": 0.0599}}
    rows = _by_key(metric_measurements(context, _outcome(measured).metric_rows, recorded, "E"))
    assert {row.item for row in rows.values()} == {"E"}
    assert rows[("500 trees", "map@12")].within_tolerance is True
    assert rows[("500 trees", "recall@12")].within_tolerance is None
    _predictions(tmp_path / "recorded.parquet", {500: {1: [1, 2], 2: [3, 4]}})
    _predictions(tmp_path / "measured.parquet", {500: {1: [2, 1], 2: [3, 4]}})
    top = top_k_measurements(
        tmp_path / "measured.parquet", tmp_path / "recorded.parquet", 500, "x", "E"
    )
    assert {row.item for row in top} == {"E"}
    by_quantity = {row.quantity: row.measured for row in top}
    # Query 1 has the same two articles in another order; query 2 the same list.
    assert by_quantity["identical_top12_list_query_share"] == 0.5
    assert by_quantity["identical_top12_set_query_share"] == 1.0
    assert by_quantity["mean_top12_overlap"] == 1.0


def test_item_e_writes_into_its_own_area_and_starts_like_item_d(tmp_path: Path) -> None:
    # Its Top-12 must not collide with item D's files of the same run.
    assert END_TO_END_AREA == "m10_end_to_end"
    context = _context(tmp_path)
    if sys.platform == "darwin":
        # The laptop: the tolerance-accepted entry refuses, as for item D.
        with pytest.raises(RuntimeError, match="runs on Linux only"):
            run_end_to_end_ranker(context)
    else:
        # A container: the platform is accepted; the temporary project has no sample files.
        with pytest.raises(FileNotFoundError):
            run_end_to_end_ranker(context)
