"""Tests for weekly.replay_compare: the comparison of a replayed week with its stage-1 record.

Hand examples for the arithmetic; the real result tables of the repository
(results/, also in the container image) for reading a week's record.
"""

import math
import shutil
from datetime import date
from pathlib import Path

import polars as pl
import pytest
from weekly_example import (
    STAMP,
    V1,
    V2,
    FakeSteps,
    build_workspace,
    patch_driver_environment,
)

from hmrec.csv_records import read_rows
from hmrec.data.folds import load_fold_manifest
from hmrec.locks import ALL_CHANNELS, load_ranker_lock
from hmrec.paths import StateLayout, get_paths
from hmrec.run_manifest import EVAL_MODE
from hmrec.weekly.driver import run_tick
from hmrec.weekly.outputs import WeekEvaluation
from hmrec.weekly.replay_compare import (
    ACCEPTANCE,
    COMPARISON_COLUMNS,
    REPORTED,
    STRUCTURAL,
    Comparison,
    append_comparisons,
    compare_replay,
    compare_week,
    compared_weeks,
    copy_chain_records,
    is_inside,
    metric_comparisons,
    outside,
    pair_overlap,
    pooled,
    read_week_record,
    record_tables,
    serve_overlap,
    structural_comparisons,
)

PATHS = get_paths()
MANIFEST = load_fold_manifest(PATHS.fold_manifest_file)
LOCK = load_ranker_lock(PATHS.ranker_lock_file)
THRESHOLD = 0.01


def test_relative_difference_and_the_one_percent_criterion() -> None:
    close = Comparison(ACCEPTANCE, "recall", "r5_wals", 0.0995, 0.10, "table")
    assert close.relative_difference == pytest.approx(0.005)
    assert is_inside(close, THRESHOLD) is True
    far = Comparison(ACCEPTANCE, "map@12", "", 0.0398, 0.04, "table")
    assert far.relative_difference == pytest.approx(0.005)
    assert is_inside(Comparison(ACCEPTANCE, "map@12", "", 0.0395, 0.04, "t"), THRESHOLD) is False
    # A reported quantity is never judged, however far it is.
    assert (
        is_inside(Comparison(REPORTED, "candidate_rows", "r6", 1.0, 100.0, "t"), THRESHOLD) is None
    )
    # A structural check is an equality, not a tolerance.
    assert is_inside(Comparison(STRUCTURAL, "duplicate_keys", "", 0.0, 0.0, "t"), THRESHOLD) is True
    assert (
        is_inside(Comparison(STRUCTURAL, "duplicate_keys", "", 1.0, 0.0, "t"), THRESHOLD) is False
    )
    assert Comparison(STRUCTURAL, "x", "", 0.0, 0.0, "t").relative_difference == 0.0
    assert math.isinf(Comparison(STRUCTURAL, "x", "", 2.0, 0.0, "t").relative_difference)


def test_pooled_score_is_weighted_by_queries() -> None:
    # (100 x 0.02 + 300 x 0.04) / 400 = 0.035, not the plain mean 0.03.
    assert pooled([(100, 0.02), (300, 0.04)]) == pytest.approx(0.035)
    with pytest.raises(ValueError, match="no queries"):
        pooled([])


def test_pair_overlap_counts_recorded_pairs_and_identical_scores() -> None:
    schema = {"query_id": pl.Int32, "article_idx": pl.Int32, "score": pl.Float32}
    recorded = pl.DataFrame(
        [(1, 10, 0.5), (1, 11, 0.4), (2, 10, 0.3), (2, 12, 0.2)], schema=schema, orient="row"
    )
    ours = pl.DataFrame(
        [(1, 10, 0.5), (1, 11, 0.41), (2, 10, 0.3), (2, 13, 0.9), (3, 1, 0.1)],
        schema=schema,
        orient="row",
    )
    share, identical, n_ours = pair_overlap(ours, recorded)
    # 3 of the 4 recorded pairs are ours; 2 of those 3 have the same float32 score.
    assert (share, identical, n_ours) == (0.75, pytest.approx(2 / 3), 5)


def _evaluation(**changes: object) -> WeekEvaluation:
    values = {
        "week": "V1",
        "anchor": "2020-07-29",
        "n_queries": 100,
        "n_label_pairs": 150,
        "channel_recall_metric": dict.fromkeys(ALL_CHANNELS, "recall@100"),
        "channel_recall": dict.fromkeys(ALL_CHANNELS, 0.10),
        "union_recall": 0.25,
        "zero_hit_query_share": 0.5,
        "ranker": {"map@12": 0.03},
        "rrf_baseline": {"map@12": 0.02},
        "hit_value": 1.0,
    }
    return WeekEvaluation(**{**values, **changes})  # type: ignore[arg-type]  # test values


def test_a_weeks_acceptance_rows_are_the_six_channels_the_union_and_map() -> None:
    record = read_week_record(PATHS, MANIFEST.get("V1"), LOCK)
    made_up = _evaluation(
        channel_recall=dict.fromkeys(ALL_CHANNELS, 0.9), union_recall=0.9, ranker={"map@12": 0.9}
    )
    rows = metric_comparisons(made_up, record, feature_rows=1000)
    acceptance = [(row.quantity, row.subject) for row in rows if row.kind == ACCEPTANCE]
    assert acceptance == [
        *(("recall", channel) for channel in ALL_CHANNELS),
        ("recall", "union"),
        ("map@12", ""),
    ]
    reported = {row.quantity: row for row in rows if row.kind == REPORTED}
    assert set(reported) == {"rrf_baseline_map@12", "deduplicated_candidates", "queries"}
    assert reported["deduplicated_candidates"].record == record.deduplicated_candidates
    # Every acceptance row of the made-up week is far from the real record: all outside.
    assert len(outside(rows, THRESHOLD)) == 8


def test_structural_checks_read_the_feature_steps_own_numbers() -> None:
    good = {
        "structure/feature_rows": 900.0,
        "structure/deduplicated_candidates": 900.0,
        "structure/duplicate_keys": 0.0,
    }
    assert outside(structural_comparisons(good), THRESHOLD) == []
    bad = {**good, "structure/feature_rows": 899.0, "structure/duplicate_keys": 2.0}
    failed = outside(structural_comparisons(bad), THRESHOLD)
    assert [(row.quantity, row.value) for row in failed] == [
        ("feature_rows_minus_deduplicated_candidates", -1.0),
        ("duplicate_keys", 2.0),
    ]


def test_the_record_of_a_week_is_the_locked_configurations_row() -> None:
    v1 = read_week_record(PATHS, MANIFEST.get("V1"), LOCK)
    # results/v_phase_metrics.csv: the RRF baseline of V1 (the first row of the table).
    assert (v1.rrf_map_at_12, v1.n_queries) == (0.018510960720404597, 82802)
    assert v1.map_at_12 > v1.rrf_map_at_12 and set(v1.channel_recall) == set(ALL_CHANNELS)
    assert (v1.recall_source, v1.metrics_source) == (
        "results/m6_candidates.csv",
        "results/v_phase_metrics.csv",
    )
    # T weeks: the M8 tables. Recorded numbers are read; no label file is opened.
    t4 = read_week_record(PATHS, MANIFEST.get("T4"), LOCK)
    assert (t4.map_at_12, t4.n_queries) == (0.04243513587830168, 68984)
    assert record_tables(PATHS, MANIFEST.get("T1")) == (
        PATHS.t_phase_recall_file,
        PATHS.t_phase_metrics_file,
    )
    with pytest.raises(ValueError, match="no stage-1 record to compare with"):
        record_tables(PATHS, MANIFEST.get("F4"))


def test_a_week_is_compared_once(tmp_path: Path) -> None:
    table = tmp_path / "comparison.csv"
    rows = [Comparison(ACCEPTANCE, "map@12", "", 0.0301, 0.03, "results/v_phase_metrics.csv")]
    append_comparisons(table, "V1", "2020-07-29", rows, THRESHOLD, "abc123")
    assert compared_weeks(table) == {"V1"}
    written = read_rows(table, COMPARISON_COLUMNS)[0]
    assert (written["inside"], written["threshold"], written["kind"]) == (
        "True",
        "0.01",
        ACCEPTANCE,
    )
    assert float(written["relative_difference"]) == pytest.approx(0.1 / 30)
    with pytest.raises(ValueError, match="a week is compared once"):
        append_comparisons(table, "V1", "2020-07-29", rows, THRESHOLD, "abc123")


def test_serve_overlap_compares_the_lists_of_the_same_customers() -> None:
    def top(lists: dict[int, list[int]]) -> pl.DataFrame:
        rows = [
            (q, rank + 1, article)
            for q, articles in lists.items()
            for rank, article in enumerate(articles)
        ]
        return pl.DataFrame(
            rows,
            schema={"query_id": pl.Int32, "rank": pl.Int32, "article_idx": pl.Int32},
            orient="row",
        )

    ours = top({1: [10, 11, 12], 2: [10, 11, 12], 3: [1, 2, 3], 4: [7, 8, 9]})
    reference = top({1: [10, 11, 12], 2: [12, 11, 10], 3: [1, 5, 6], 9: [1, 2, 3]})
    values = serve_overlap(ours, reference)
    # Customers 1, 2, 3 are in both. 1: identical; 2: the same set in another order;
    # 3: 1 of 3 articles in common, the same first article.
    assert (values["n_customers"], values["n_customers_in_reference"]) == (4.0, 3.0)
    assert values["identical_list_share"] == pytest.approx(1 / 3)
    assert values["same_articles_share"] == pytest.approx(2 / 3)
    assert values["mean_overlap"] == pytest.approx((3 + 3 + 1) / 9)
    assert values["first_article_identical_share"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="no query in common"):
        serve_overlap(top({4: [1, 2, 3]}), reference)


def test_chain_records_are_copied_once_per_week(tmp_path: Path) -> None:
    layout = StateLayout(tmp_path / "pipeline")
    anchors = [date(2020, 7, 29), date(2020, 8, 5), date(2020, 8, 12)]
    for anchor, fold in zip(anchors[:2], ("V1", "V2"), strict=True):
        folder = layout.week_dir(anchor) / "twotower"
        folder.mkdir(parents=True)
        (folder / "chain_record.csv").write_text(
            f"fold,model,sha256\n{fold},{fold}.pt,{'a' * 64}\n"
        )
    table = tmp_path / "results" / "twotower_chain.csv"
    # The third week is not closed yet: it has no record file.
    assert copy_chain_records(layout, anchors, table) == 2
    assert copy_chain_records(layout, anchors, table) == 0
    assert [row["fold"] for row in read_rows(table, ("fold", "model", "sha256"))] == ["V1", "V2"]


def test_a_state_of_small_records_is_compared_without_its_candidate_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A replay in the cloud is compared from a copy of its small records (M13). The state
    # below is what two ticks leave, written by the fake steps; no candidate file of
    # stage 1 exists in the workspace, and none of the pipeline is read.
    patch_driver_environment(monkeypatch)
    paths = build_workspace(tmp_path)
    for table in ("m6_candidates.csv", "v_phase_metrics.csv"):
        paths.results_root.mkdir(parents=True, exist_ok=True)
        shutil.copy(PATHS.results_root / table, paths.results_root / table)
    for anchor in (V1, V2):
        assert run_tick(paths, anchor, EVAL_MODE, FakeSteps(paths), STAMP).succeeded
    weeks = [MANIFEST.get("V1"), MANIFEST.get("V2")]
    result = compare_replay(
        paths, weeks, date(2020, 9, 23), LOCK, THRESHOLD, "abc123", with_candidates=False
    )
    # V1 is closed (by the tick at V2) and compared; V2 is open; no pooled value yet.
    assert result.closed == ("V1",) and result.compared_now == ("V1",)
    assert result.serve_row is None and result.chain_records_copied == 1
    rows = read_rows(paths.replay_table_file("comparison"), COMPARISON_COLUMNS)
    kinds = {(row["kind"], row["quantity"]) for row in rows}
    assert (ACCEPTANCE, "recall") in kinds and (ACCEPTANCE, "map@12") in kinds
    assert (STRUCTURAL, "duplicate_keys") in kinds
    # Eight acceptance rows and two structural rows, as for any week; nothing of the
    # candidate files.
    assert sum(row["kind"] == ACCEPTANCE for row in rows) == 8
    assert sum(row["kind"] == STRUCTURAL for row in rows) == 2
    assert all(row["quantity"] != "recorded_pairs_present" for row in rows)
    # The made-up numbers of the fake steps are far from the records: reported as outside.
    assert {name for name, _ in result.failed} == {"V1"}
    # With candidates the same call needs the stage-1 candidate files, which are not here.
    with pytest.raises(FileNotFoundError):
        compare_week(paths, StateLayout(paths.pipeline_state_dir), weeks[0], LOCK)
