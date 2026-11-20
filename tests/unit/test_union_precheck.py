"""Tests for hpo.union_precheck: replaying the selection, candidate files, union tables, summary.

Hand example (fold D1), three channels a, b, c:
    labels:  q1 -> {10, 11, 12}, q2 -> {20}, q3 -> {30}
    a:       q1 -> 10 (rank 1), 11 (rank 2); q2 -> 21
    b:       q1 -> 11, 40;                   q2 -> 20
    c:       q1 -> 12
    q3 has no candidate in any channel.

    union recall (all):  q1 = 3/3, q2 = 1/1, q3 = 0       -> mean 2/3
    without a:           q1 = {11, 12} = 2/3, q2 = 1, q3 = 0 -> mean 5/9
    without b:           q1 = 1, q2 = 0 (21 is not a label), q3 = 0 -> mean 1/3
    without c:           q1 = {10, 11} = 2/3, q2 = 1, q3 = 0 -> mean 5/9
    distinct candidates: q1 = {10, 11, 40, 12} = 4, q2 = {21, 20} = 2, q3 = 0
    zero-hit queries:    only q3 -> 1/3
    a's recall@2:        q1 = 2/3, q2 = 0, q3 = 0         -> mean 2/9

Fold D2 (one query, q101 -> {50}; a: 50; b, c: nothing):
    all = 1, without a = 0, without b = without c = 1; 1 distinct candidate.
"""

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from conftest import make_queries, make_recall_inputs
from hmrec.config import load_base_config, load_selection_rule, load_yaml
from hmrec.data.folds import Fold
from hmrec.eval.report import FoldEvaluator, MultiFoldSummary
from hmrec.hpo.grid import parse_step
from hmrec.hpo.records import TrialRecord
from hmrec.hpo.runner import FoldContext, make_trial_id
from hmrec.hpo.union_precheck import (
    SelectedChannel,
    count_candidates_per_query,
    fold_tables,
    load_or_generate_candidates,
    precheck_scopes,
    select_channel,
    summarize_precheck,
    union_recall_tables,
    write_precheck_table,
)

REPO = Path(__file__).resolve().parents[2]
RULE = load_selection_rule(REPO / "configs" / "selection_rule.yaml")
R1_DEFAULT = load_yaml(REPO / "configs" / "recall" / "r1_popularity.yaml")
FOLDS: tuple[str, ...] = ("D1", "D2", "D3", "D4")
INT32 = pl.Int32


def _record(channel: str, config: dict, score: float, metric: str = "recall@2") -> TrialRecord:
    """A finished trial with the same score on all 4 folds."""
    fold_scores = dict.fromkeys(FOLDS, score)
    summary = MultiFoldSummary(fold_scores, dict.fromkeys(FOLDS, 10), score, score, {})
    return TrialRecord(
        phase="D",
        channel=channel,
        step="x",
        trial_id=make_trial_id(channel, config),
        config=config,
        params={},
        metric=metric,
        summary=summary,
        extra_pooled={},
        seconds=1.0,
        peak_rss_gb=0.1,
        git_commit="abc",
        git_dirty=False,
        finished_at="2026-11-17T00:00:00",
        diagnostics={},
    )


def _tau_step(stop_at_boundary: bool) -> tuple:
    raw = {
        "name": "tau_days",
        "kind": "select",
        "order": "nearest_to_current",
        "stop_at_boundary": stop_at_boundary,
        "candidates": [{"tau_days": 2.0}, {"tau_days": 4.0}],
    }
    return (parse_step(raw, "test"),)


# ---------------------------------------------------------------- selection replay


def test_select_channel_replays_the_rule_without_running_trials() -> None:
    # Default tau 3 scores 0.10; tau 2 scores 0.11 (+10%, 4/4 folds) -> switch to 2.
    tau2 = {**R1_DEFAULT, "tau_days": 2.0}
    tau4 = {**R1_DEFAULT, "tau_days": 4.0}
    records = [
        _record("r1_popularity", dict(R1_DEFAULT), 0.10),
        _record("r1_popularity", tau2, 0.11),
        _record("r1_popularity", tau4, 0.09),
    ]
    finished = {record.trial_id: record for record in records}
    choice = select_channel("r1_popularity", _tau_step(False), R1_DEFAULT, finished, RULE)
    assert choice.raw_config["tau_days"] == 2.0
    assert choice.record is records[1]


def test_select_channel_refuses_a_missing_trial() -> None:
    default = _record("r1_popularity", dict(R1_DEFAULT), 0.10)
    with pytest.raises(ValueError, match="not in results/d_phase.csv"):
        select_channel(
            "r1_popularity", _tau_step(False), R1_DEFAULT, {default.trial_id: default}, RULE
        )


def test_select_channel_refuses_an_open_stop() -> None:
    # tau 4 (top of {2, 3, 4}) wins a stop_at_boundary step -> stop: no final config.
    records = [
        _record("r1_popularity", dict(R1_DEFAULT), 0.10),
        _record("r1_popularity", {**R1_DEFAULT, "tau_days": 2.0}, 0.09),
        _record("r1_popularity", {**R1_DEFAULT, "tau_days": 4.0}, 0.12),
    ]
    finished = {record.trial_id: record for record in records}
    with pytest.raises(ValueError, match="ended in 'stop'"):
        select_channel("r1_popularity", _tau_step(True), R1_DEFAULT, finished, RULE)


# ---------------------------------------------------------------- hand example


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


def _context(fold: str, query_ids: list[int], labels: list[tuple[int, int]]) -> FoldContext:
    label_table = pl.DataFrame(
        {
            "query_id": [query for query, _ in labels],
            "fold": [fold] * len(labels),
            "article_idx": [article for _, article in labels],
        },
        schema={"query_id": INT32, "fold": pl.String, "article_idx": INT32},
    )
    queries = make_queries([(query, query) for query in query_ids], fold=fold)
    fold_info = Fold(name=fold, phase="D", anchor=date(2020, 1, 1), label_end=date(2020, 1, 7))
    inputs = make_recall_inputs([], 100)
    return FoldContext(fold_info, queries, inputs, FoldEvaluator(fold, label_table))


D1_CONTEXT = _context("D1", [1, 2, 3], [(1, 10), (1, 11), (1, 12), (2, 20), (3, 30)])
D1_CANDIDATES = {
    "a": _candidates([(1, 10, 1), (1, 11, 2), (2, 21, 1)]),
    "b": _candidates([(1, 11, 1), (1, 40, 2), (2, 20, 1)]),
    "c": _candidates([(1, 12, 1)]),
}
D2_CONTEXT = _context("D2", [101], [(101, 50)])
D2_CANDIDATES = {"a": _candidates([(101, 50, 1)]), "b": _candidates([]), "c": _candidates([])}
METRICS: dict[str, str] = {"a": "recall@2", "b": "recall@2", "c": "recall@2"}


def test_union_recall_tables_hand_computed() -> None:
    tables = union_recall_tables(D1_CANDIDATES, D1_CONTEXT)
    values = {name: table["value"].to_list() for name, table in tables.items()}
    assert values["all"] == [1.0, 1.0, 0.0]
    assert values["without_a"] == [pytest.approx(2 / 3), 1.0, 0.0]
    assert values["without_b"] == [1.0, 0.0, 0.0]
    assert values["without_c"] == [pytest.approx(2 / 3), 1.0, 0.0]


def test_count_candidates_per_query_dedups_and_keeps_empty_queries() -> None:
    counts = count_candidates_per_query(D1_CANDIDATES, D1_CONTEXT.queries)
    assert counts["query_id"].to_list() == [1, 2, 3]
    assert counts["value"].to_list() == [4.0, 2.0, 0.0]


def _selected(channel: str, pooled: float) -> SelectedChannel:
    # The record only supplies metric and pooled score here; a real channel name
    # is needed to build its trial_id, so an R1 config stands in.
    record = _record("r1_popularity", dict(R1_DEFAULT), pooled)
    return SelectedChannel(channel, dict(R1_DEFAULT), record)


def test_summarize_precheck_hand_computed() -> None:
    tables = [
        fold_tables(D1_CANDIDATES, D1_CONTEXT, METRICS),
        fold_tables(D2_CANDIDATES, D2_CONTEXT, METRICS),
    ]
    scopes = precheck_scopes(["D1", "D2"], {"D1_D2": ("D1", "D2")})
    selected = [_selected("a", 0.5), _selected("b", 0.1), _selected("c", 0.1)]
    summary = summarize_precheck(tables, scopes, selected)
    rows = {row["measure"]: row for row in summary.iter_rows(named=True)}
    # Pooled over the 4 queries of D1 + D2: all = (1 + 1 + 0 + 1) / 4.
    assert rows["union_recall_all"]["D1"] == pytest.approx(2 / 3)
    assert rows["union_recall_all"]["D2"] == 1.0
    assert rows["union_recall_all"]["pooled"] == pytest.approx(3 / 4)
    assert rows["union_recall_all"]["D1_D2"] == pytest.approx(3 / 4)
    # without a, pooled: (2/3 + 1 + 0 + 0) / 4 = 5/12, so the drop is 3/4 - 5/12 = 1/3.
    assert rows["union_recall_without_a"]["pooled"] == pytest.approx(5 / 12)
    assert rows["union_recall_drop_without_a"]["pooled"] == pytest.approx(1 / 3)
    assert rows["union_recall_drop_without_b"]["D1"] == pytest.approx(2 / 3 - 1 / 3)
    assert rows["zero_hit_query_share"]["D1"] == pytest.approx(1 / 3)
    assert rows["zero_hit_query_share"]["pooled"] == pytest.approx(1 / 4)
    # Candidate counts: D1 [4, 2, 0], D2 [1].
    assert rows["candidates_per_query_mean"]["pooled"] == pytest.approx(7 / 4)
    assert rows["candidates_per_query_p50"]["D1"] == 2.0
    assert rows["candidates_per_query_p90"]["D1"] == 4.0
    # Channel a's recall@2: D1 = 2/9 (see module docstring), D2 = 1.
    assert rows["a_recall@2"]["D1"] == pytest.approx(2 / 9)
    assert rows["a_recall@2"]["pooled"] == pytest.approx((2 / 3 + 1) / 4)
    assert rows["a_recall@2"]["recorded_pooled"] == 0.5
    assert rows["union_recall_all"]["recorded_pooled"] is None


def test_union_recall_tables_need_two_channels() -> None:
    with pytest.raises(ValueError, match="at least 2 channels"):
        union_recall_tables({"a": D1_CANDIDATES["a"]}, D1_CONTEXT)


# ---------------------------------------------------------------- files


def test_candidates_are_written_once_and_read_back(tmp_path: Path) -> None:
    # R2 on a tiny fold: customer 1 bought article 20 on day 95, anchor day 100.
    r2_default = load_yaml(REPO / "configs" / "recall" / "r2_repurchase.yaml")
    choice = SelectedChannel(
        "r2_repurchase", dict(r2_default), _record("r2_repurchase", dict(r2_default), 0.1)
    )
    context = FoldContext(
        D1_CONTEXT.fold,
        make_queries([(1, 1)]),
        make_recall_inputs([(1, 20, 95)], 100),
        D1_CONTEXT.evaluator,
    )
    base = load_base_config(REPO / "configs" / "base.yaml")
    path = tmp_path / "D1" / "r2_repurchase_x.parquet"
    first = load_or_generate_candidates(choice, context, base, path)
    assert first["article_idx"].to_list() == [20]
    written_at = path.stat().st_mtime_ns
    second = load_or_generate_candidates(choice, context, base, path)
    assert second.equals(first)
    assert path.stat().st_mtime_ns == written_at  # read back, not rewritten


def test_write_precheck_table_never_replaces(tmp_path: Path) -> None:
    path = tmp_path / "precheck.csv"
    table = pl.DataFrame({"measure": ["x"], "pooled": [0.1]})
    write_precheck_table(table, path)
    with pytest.raises(FileExistsError):
        write_precheck_table(table, path)
