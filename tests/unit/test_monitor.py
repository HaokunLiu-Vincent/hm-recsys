"""Tests for eval.monitor: candidate counts, label metrics for F / V only, the low-recall flag.

Hand example, fold F1 (three queries), three channels a, b, c with quota 2:
    labels:  q1 -> {10, 11, 12}, q2 -> {20}, q3 -> {30}
    a:       q1 -> 10 (rank 1), 11 (rank 2); q2 -> 21
    b:       q1 -> 11, 40;                   q2 -> 20
    c:       q1 -> 12
    q3 has no candidate in any channel.

    recall@2  a: q1 2/3, q2 0,   q3 0 -> mean 2/9
              b: q1 1/3, q2 1,   q3 0 -> mean 4/9
              c: q1 1/3, q2 0,   q3 0 -> mean 1/9
    union recall: q1 3/3, q2 1/1, q3 0 -> mean 2/3; queries without any hit: q3 -> 1/3
    candidates per query
              a: q1 2, q2 1, q3 0 -> 3 rows, mean 1, P50 1, P90 2, empty 1/3
              union: q1 {10, 11, 40, 12} = 4, q2 {21, 20} = 2, q3 0 -> 6 rows, mean 2, P50 2, P90 4

Recorded D1-D4 scores (in the fake lock files): a 0.25 each, b 0.40 each, c 0.12 each.
    a: 2/9 = 0.2222 is 11.1% below 0.25 -> flagged (more than 10% below)
    b: 4/9 = 0.4444 is above 0.40        -> not flagged
    c: 1/9 = 0.1111 is 7.4% below 0.12   -> not flagged
"""

from pathlib import Path

import polars as pl
import pytest

from conftest import make_queries
from hmrec.config import CandidateRunConfig
from hmrec.eval.monitor import (
    MONITOR_COLUMNS,
    UNION,
    FoldCandidates,
    compare_with_previous,
    flagged_channels,
    label_metrics,
    load_label_evaluator,
    monitor_rows,
    previous_fold_names,
    recall_history,
    require_label_phase,
    summarize_counts,
)
from hmrec.eval.report import FoldEvaluator
from hmrec.locks import RULE_SELECTION, LockedChannel, RecordedScore
from hmrec.run_records import RunStamp

INT32 = pl.Int32
STAMP = RunStamp("abc123", False)
CONFIG = CandidateRunConfig(previous_folds=4, max_relative_recall_drop=0.10, max_fine_tune_loss=7.0)
FOLD_NAMES = ("D1", "D2", "D3", "D4", "F1", "F2", "T1")
D_SCORES = {"a": 0.25, "b": 0.40, "c": 0.12}


def _candidates(rows: list[tuple[int, int, int]]) -> pl.DataFrame:
    """(query_id, article_idx, rank) rows."""
    return pl.DataFrame(
        {
            "query_id": [row[0] for row in rows],
            "article_idx": [row[1] for row in rows],
            "rank": [row[2] for row in rows],
        },
        schema={"query_id": INT32, "article_idx": INT32, "rank": INT32},
    )


CANDIDATES = {
    "a": _candidates([(1, 10, 1), (1, 11, 2), (2, 21, 1)]),
    "b": _candidates([(1, 11, 1), (1, 40, 2), (2, 20, 1)]),
    "c": _candidates([(1, 12, 1)]),
}
LABEL_ROWS = [(1, 10), (1, 11), (1, 12), (2, 20), (3, 30)]


def _labels(fold: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "query_id": [query for query, _ in LABEL_ROWS],
            "fold": [fold] * len(LABEL_ROWS),
            "article_idx": [article for _, article in LABEL_ROWS],
        },
        schema={"query_id": INT32, "fold": pl.String, "article_idx": INT32},
    )


def _lock(channel: str) -> LockedChannel:
    score = D_SCORES[channel]
    folds: dict[str, float] = dict.fromkeys(("D1", "D2", "D3", "D4"), score)
    record = RecordedScore("results/d_phase.csv", "recall@2", score, folds)
    return LockedChannel(channel, RULE_SELECTION, "0" * 12, "0" * 12, "test", 2, record, {})


LOCKED = {channel: _lock(channel) for channel in CANDIDATES}


def _data(fold: str, phase: str) -> FoldCandidates:
    queries = make_queries([(1, 1), (2, 2), (3, 3)], fold=fold)
    return FoldCandidates(fold, phase, queries, CANDIDATES, LOCKED, FOLD_NAMES)


def _by_scope(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["scope"]: row for row in rows}


# ---------------------------------------------------------------- F fold: counts and label metrics


def test_label_metrics_hand_computed() -> None:
    metrics = dict.fromkeys(CANDIDATES, "recall@2")
    result = label_metrics(FoldEvaluator("F1", _labels("F1")), CANDIDATES, metrics)
    assert result.channel_recall == pytest.approx({"a": 2 / 9, "b": 4 / 9, "c": 1 / 9})
    assert result.union_recall == pytest.approx(2 / 3)
    assert result.zero_hit_query_share == pytest.approx(1 / 3)


def test_rows_of_an_f_fold_hand_computed() -> None:
    evaluator = FoldEvaluator("F1", _labels("F1"))
    rows = monitor_rows(_data("F1", "F"), evaluator, [], CONFIG, query_chunk_size=2, stamp=STAMP)
    assert [row["scope"] for row in rows] == ["a", "b", "c", UNION]
    assert all(tuple(row) == MONITOR_COLUMNS for row in rows)
    by_scope = _by_scope(rows)
    a = by_scope["a"]
    assert (a["n_queries"], a["n_rows"], a["quota"], a["metric"]) == ("3", "3", "2", "recall@2")
    assert (a["candidates_per_query_mean"], a["candidates_per_query_p50"]) == ("1.0", "1.0")
    assert a["candidates_per_query_p90"] == "2.0"
    assert float(a["empty_query_share"]) == pytest.approx(1 / 3)
    assert float(a["recall"]) == pytest.approx(2 / 9)
    assert a["labels_read"] == "True"
    union = by_scope[UNION]
    assert (union["n_rows"], union["quota"], union["metric"]) == ("6", "6", "union_recall")
    assert (union["candidates_per_query_mean"], union["candidates_per_query_p90"]) == ("2.0", "4.0")
    assert float(union["recall"]) == pytest.approx(2 / 3)
    assert float(union["zero_hit_query_share"]) == pytest.approx(1 / 3)
    # The union has no "previous 4 weeks" comparison (it is per channel, proposal §6.6).
    assert (union["previous_mean_recall"], union["low_recall_flag"]) == ("", "")


def test_low_recall_is_flagged_against_the_four_weeks_before() -> None:
    evaluator = FoldEvaluator("F1", _labels("F1"))
    rows = monitor_rows(_data("F1", "F"), evaluator, [], CONFIG, query_chunk_size=10, stamp=STAMP)
    by_scope = _by_scope(rows)
    # F1 is compared with D1-D4, whose scores come from the lock files.
    assert by_scope["a"]["previous_folds"] == "D1 D2 D3 D4"
    assert float(by_scope["a"]["previous_mean_recall"]) == pytest.approx(0.25)
    assert float(by_scope["a"]["recall_vs_previous"]) == pytest.approx((2 / 9) / 0.25 - 1)
    assert by_scope["a"]["low_recall_flag"] == "True"  # 11.1% below
    assert by_scope["b"]["low_recall_flag"] == "False"  # above the mean
    assert by_scope["c"]["low_recall_flag"] == "False"  # 7.4% below: within 10%
    assert flagged_channels(rows) == ["a"]


def test_the_next_fold_uses_the_rows_already_written() -> None:
    evaluator = FoldEvaluator("F1", _labels("F1"))
    f1_rows = monitor_rows(_data("F1", "F"), evaluator, [], CONFIG, 10, STAMP)
    history = recall_history(LOCKED, f1_rows)
    assert list(history["a"]) == ["D1", "D2", "D3", "D4", "F1"]
    assert history["a"]["F1"] == pytest.approx(2 / 9)
    f2_rows = monitor_rows(
        _data("F2", "F"), FoldEvaluator("F2", _labels("F2")), f1_rows, CONFIG, 10, STAMP
    )
    a = _by_scope(f2_rows)["a"]
    # F2 is compared with D2, D3, D4 and F1: (3 x 0.25 + 2/9) / 4.
    assert a["previous_folds"] == "D2 D3 D4 F1"
    assert float(a["previous_mean_recall"]) == pytest.approx((0.75 + 2 / 9) / 4)


# ---------------------------------------------------------------- T fold: no labels at all


def test_t_fold_rows_have_counts_but_no_label_based_number() -> None:
    rows = monitor_rows(_data("T1", "T"), None, [], CONFIG, query_chunk_size=2, stamp=STAMP)
    by_scope = _by_scope(rows)
    for scope in ("a", "b", "c", UNION):
        row = by_scope[scope]
        assert row["labels_read"] == "False"
        for label_cell in ("recall", "previous_mean_recall", "recall_vs_previous"):
            assert row[label_cell] == ""
        assert (row["low_recall_flag"], row["zero_hit_query_share"]) == ("", "")
    # The numbers that need no labels are all there.
    assert (by_scope["a"]["n_rows"], by_scope[UNION]["n_rows"]) == ("3", "6")
    assert by_scope[UNION]["candidates_per_query_mean"] == "2.0"
    assert flagged_channels(rows) == []


def test_an_evaluator_for_a_t_fold_is_refused(tmp_path: Path) -> None:
    # Even with a perfectly valid label table at hand, M6 refuses to score a T fold.
    evaluator = FoldEvaluator("T1", _labels("T1"))
    with pytest.raises(ValueError, match="M6 must not read the labels of fold T1"):
        monitor_rows(_data("T1", "T"), evaluator, [], CONFIG, 10, STAMP)
    # load_label_evaluator checks the phase BEFORE opening the file: the path does not
    # exist, and the error is still the phase error, not FileNotFoundError.
    missing = tmp_path / "labels" / "T1.parquet"
    with pytest.raises(ValueError, match="T metrics are computed in M8"):
        load_label_evaluator("T1", "T", missing)
    with pytest.raises(ValueError, match="M6 must not read the labels of fold D4"):
        require_label_phase("D4", "D")
    for fold, phase in (("F1", "F"), ("V3", "V")):
        require_label_phase(fold, phase)  # no error


def test_load_label_evaluator_reads_an_f_fold(tmp_path: Path) -> None:
    labels_file = tmp_path / "F1.parquet"
    _labels("F1").write_parquet(labels_file)
    evaluator = load_label_evaluator("F1", "F", labels_file)
    assert evaluator.fold == "F1" and evaluator.n_queries == 3


# ---------------------------------------------------------------- pieces


def test_previous_fold_names() -> None:
    assert previous_fold_names("F1", FOLD_NAMES, 4) == ("D1", "D2", "D3", "D4")
    assert previous_fold_names("T1", FOLD_NAMES, 4) == ("D3", "D4", "F1", "F2")
    assert previous_fold_names("D2", FOLD_NAMES, 4) == ("D1",)  # not enough folds before
    with pytest.raises(ValueError, match="unknown fold"):
        previous_fold_names("V1", FOLD_NAMES, 4)


def test_compare_with_previous() -> None:
    history = {"D1": 0.24, "D2": 0.26, "D3": 0.25, "D4": 0.25}
    previous = ("D1", "D2", "D3", "D4")
    # Mean 0.25. 0.23 is 8% below: not flagged. 0.22 is 12% below: flagged.
    within = compare_with_previous(0.23, history, previous, 4, 0.10)
    assert within.previous_mean == pytest.approx(0.25)
    assert within.relative_change == pytest.approx(-0.08)
    assert within.is_low is False
    assert compare_with_previous(0.22, history, previous, 4, 0.10).is_low is True
    assert compare_with_previous(0.40, history, previous, 4, 0.10).is_low is False


def test_no_comparison_without_all_previous_folds() -> None:
    history = {"D1": 0.25, "D2": 0.25, "D4": 0.25}  # D3 has no recall
    missing = compare_with_previous(0.01, history, ("D1", "D2", "D3", "D4"), 4, 0.10)
    assert (missing.previous_mean, missing.relative_change, missing.is_low) == (None, None, False)
    too_few = compare_with_previous(0.01, history, ("D1", "D2"), 4, 0.10)
    assert (too_few.previous_mean, too_few.is_low) == (None, False)


def test_summarize_counts_refuses_a_fold_without_queries() -> None:
    empty = pl.DataFrame(schema={"query_id": INT32, "fold": pl.String, "value": pl.Float64})
    with pytest.raises(ValueError, match="no queries"):
        summarize_counts(empty)
