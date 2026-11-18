"""Tests for hmrec.final.report: pooled / equal-fold numbers and the recorded verdict."""

from pathlib import Path

import pytest
from final_example import NOW, load_context, write_configs

from hmrec.csv_records import read_rows
from hmrec.final.goals import GOAL_COLUMNS, GoalVerdict
from hmrec.final.report import record_verdicts, system_summary, union_recall_by_fold
from hmrec.hpo.v_records import MetricRow

FOLDS = ("T1", "T2")
METRICS = ("map@12", "recall@12")


def _row(fold: str, n_queries: int, map_value: float, recall: float, k: int = 10) -> MetricRow:
    scores = {"map@12": map_value, "recall@12": recall}
    return MetricRow("full", "abc", fold, k, n_queries, scores, None)


def test_pooled_is_query_weighted_and_equal_fold_is_the_plain_mean() -> None:
    # T1: 100 queries with MAP 0.02; T2: 300 queries with MAP 0.04.
    #   pooled     = (100 * 0.02 + 300 * 0.04) / 400 = 14 / 400 = 0.035
    #   equal fold = (0.02 + 0.04) / 2               = 0.03
    rows = [_row("T1", 100, 0.02, 0.10), _row("T2", 300, 0.04, 0.20)]
    summary = system_summary(rows, "full", "abc", FOLDS, 10, METRICS)
    assert summary.n_queries == {"T1": 100, "T2": 300}
    assert summary.by_fold["map@12"] == {"T1": 0.02, "T2": 0.04}
    assert summary.pooled["map@12"] == pytest.approx(0.035, rel=1e-12)
    assert summary.equal_fold["map@12"] == pytest.approx(0.03, rel=1e-12)
    # recall: pooled (100 * 0.10 + 300 * 0.20) / 400 = 0.175; equal fold 0.15.
    assert summary.pooled["recall@12"] == pytest.approx(0.175, rel=1e-12)
    assert summary.equal_fold["recall@12"] == pytest.approx(0.15, rel=1e-12)


def test_a_summary_needs_exactly_one_row_per_fold() -> None:
    rows = [_row("T1", 100, 0.02, 0.10)]
    with pytest.raises(ValueError, match="no metric row at 10 trees for folds"):
        system_summary(rows, "full", "abc", FOLDS, 10, METRICS)
    twice = [*rows, _row("T1", 100, 0.02, 0.10), _row("T2", 300, 0.04, 0.20)]
    with pytest.raises(ValueError, match="recorded twice"):
        system_summary(twice, "full", "abc", FOLDS, 10, METRICS)
    # Rows of another configuration or another number of trees are not this system's.
    other = [_row("T1", 100, 0.02, 0.10, k=20), _row("T2", 300, 0.04, 0.20, k=20)]
    with pytest.raises(ValueError, match="no metric row at 10 trees"):
        system_summary(other, "full", "abc", FOLDS, 10, METRICS)


def test_union_recall_is_read_from_the_union_rows_only() -> None:
    rows = [
        {"fold": "T1", "scope": "r1_popularity", "recall": "0.07"},
        {"fold": "T1", "scope": "union", "recall": "0.25"},
        {"fold": "T2", "scope": "union", "recall": "0.24"},
    ]
    assert union_recall_by_fold(rows, FOLDS) == {"T1": 0.25, "T2": 0.24}
    with pytest.raises(ValueError, match="fold T2 has 0 union recall rows"):
        union_recall_by_fold(rows[:2], FOLDS)
    # A union row without a recall (as M6 wrote for T folds) does not count.
    no_recall = [*rows[:2], {"fold": "T2", "scope": "union", "recall": ""}]
    with pytest.raises(ValueError, match="fold T2 has 0 union recall rows"):
        union_recall_by_fold(no_recall, FOLDS)


def _verdicts(g1_met: bool) -> list[GoalVerdict]:
    return [
        GoalVerdict("G1", "pooled MAP", "0.033" if g1_met else "0.031", "0.032", g1_met, "d1"),
        GoalVerdict("G2", "union recall", "0.25", "0.24", True, "d2"),
        GoalVerdict("G3", "folds above", "4", "3", True, "d3"),
    ]


def test_the_verdict_is_written_once_and_cannot_change(tmp_path: Path) -> None:
    context = load_context(write_configs(tmp_path))
    record_verdicts(context, _verdicts(True), NOW)
    rows = read_rows(context.paths.t_phase_goals_file, GOAL_COLUMNS)
    assert [(row["goal"], row["met"]) for row in rows] == [
        ("G1", "True"),
        ("G2", "True"),
        ("G3", "True"),
    ]
    assert rows[0]["threshold"] == "0.032" and rows[0]["decided_at"] == NOW
    # The same verdict again (the run is continued, or the report is printed): no new rows.
    record_verdicts(context, _verdicts(True), "2026-11-19T08:00:00")
    assert len(read_rows(context.paths.t_phase_goals_file, GOAL_COLUMNS)) == 3
    # Another verdict for the same run is an error, not a second set of rows.
    with pytest.raises(ValueError, match="records the verdict"):
        record_verdicts(context, _verdicts(False), NOW)
    assert len(read_rows(context.paths.t_phase_goals_file, GOAL_COLUMNS)) == 3
