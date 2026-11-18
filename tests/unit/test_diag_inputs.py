"""Tests for hmrec.diagnostics.recorded and hmrec.diagnostics.inputs on the tiny workspace."""

from pathlib import Path

import polars as pl
import pytest
from diag_example import (
    BASELINE_MAP,
    BASELINE_RECALL,
    MANIFEST,
    RANKER_MAP,
    RANKER_RECALL,
    UNION_RECALL,
    anchor_day,
    build_diag_workspace,
    monitor_row,
    query_ids,
    transaction_lines,
    write_metric_rows,
    write_transactions,
)

from hmrec.csv_records import append_row
from hmrec.diagnostics.inputs import (
    read_lines_before_anchor,
    read_queries,
    read_ranker_per_query,
    read_union_hits,
    read_wide_summary,
)
from hmrec.diagnostics.recorded import (
    area_of_phase,
    check_close,
    recall_file,
    recorded_fold,
    recorded_union_recall,
)
from hmrec.eval.diagnostic_labels import load_diagnostic_evaluator
from hmrec.eval.monitor import MONITOR_COLUMNS

V1, T1 = MANIFEST.get("V1"), MANIFEST.get("T1")


def test_recorded_fold_reads_the_locked_rows_of_both_phases(tmp_path: Path) -> None:
    context = build_diag_workspace(tmp_path)
    for fold in (V1, T1):
        recorded = recorded_fold(context, fold)
        assert (recorded.fold, recorded.phase, recorded.n_queries) == (fold.name, fold.phase, 4)
        assert recorded.union_recall == UNION_RECALL
        # The rows at the locked number of trees, not the ones at 5 trees.
        assert recorded.ranker["map@12"] == RANKER_MAP
        assert recorded.ranker["recall@12"] == RANKER_RECALL
        assert recorded.baseline["map@12"] == BASELINE_MAP
        assert recorded.baseline["recall@12"] == BASELINE_RECALL
    paths = context.paths
    assert recall_file(paths, "V") == paths.m6_candidates_results_file
    assert recall_file(paths, "T") == paths.t_phase_recall_file
    with pytest.raises(ValueError, match="no recorded Ranker results"):
        area_of_phase("D")


def test_recorded_fold_refuses_missing_and_repeated_rows(tmp_path: Path) -> None:
    context = build_diag_workspace(tmp_path)
    paths = context.paths
    # The M6 table's union row of a T fold has no recall; the T table has it exactly once.
    m6_rows = [monitor_row(T1, "union", "")]
    with pytest.raises(ValueError, match="exactly one recorded union recall"):
        recorded_union_recall(m6_rows, "T1")
    write_metric_rows(paths, V1)  # V1's metric rows a second time
    with pytest.raises(ValueError, match="recorded twice"):
        recorded_fold(context, V1)
    append_row(paths.t_phase_recall_file, MONITOR_COLUMNS, monitor_row(T1, "union", "0.5"))
    with pytest.raises(ValueError, match="exactly one recorded union recall"):
        recorded_fold(context, T1)


def test_check_close_uses_the_tolerance_of_the_project() -> None:
    check_close("x", 0.5 + 1e-13, 0.5)
    with pytest.raises(ValueError, match="differs from the recorded"):
        check_close("V1 union recall", 0.5 + 1e-11, 0.5)


def test_saved_values_are_read_at_the_locked_number_of_trees(tmp_path: Path) -> None:
    context = build_diag_workspace(tmp_path)
    values = read_ranker_per_query(context, V1)
    assert values.columns == ["query_id", "fold", "value"]
    assert values["query_id"].to_list() == query_ids(V1)
    assert values["value"].to_list() == [1.0, 0.0, 0.5, 1.0]  # not the zeros saved at 5 trees
    queries = read_queries(context, T1)
    assert queries["customer_idx"].to_list() == [1, 2, 3, 100 + 12]  # T1 is fold number 12


def test_wide_summary_gives_the_rrf_top_k_and_one_days_value_per_query(tmp_path: Path) -> None:
    context = build_diag_workspace(tmp_path)
    wide = read_wide_summary(context, V1)
    q0, q1, q2, q3 = query_ids(V1)
    top = wide.baseline_top
    # q0: articles 1 (R1 rank 1) and 2 (R2 rank 1) tie at 1 / 61 -> article_idx ascending.
    assert top.filter(pl.col("query_id") == q0)["article_idx"].to_list() == [1, 2, 5, 6]
    assert top.filter(pl.col("query_id") == q2)["article_idx"].to_list() == [5, 4, 6]
    assert top.filter(pl.col("query_id") == q0)["rank"].to_list() == [1, 2, 3, 4]
    assert wide.days["query_id"].to_list() == [q0, q1, q2, q3]
    days = wide.days["user_days_since_last_purchase"].to_list()
    assert (days[0], days[2], days[3]) == (3.0, 45.0, 30.0) and days[1] != days[1]  # NaN


def test_a_changed_part_file_is_not_read(tmp_path: Path) -> None:
    context = build_diag_workspace(tmp_path)
    part = context.paths.features_dir("V1") / "part-00000.parquet"
    pl.read_parquet(part).with_columns(pl.col("rrf_score") * 2).write_parquet(part)
    with pytest.raises(ValueError, match="not the frozen file"):
        read_wide_summary(context, V1)


def test_union_hits_are_the_candidates_that_are_labels(tmp_path: Path) -> None:
    context = build_diag_workspace(tmp_path)
    evaluator = load_diagnostic_evaluator(V1, context.paths.labels_file("V1"), None)
    hits = read_union_hits(context, V1, evaluator)
    q0, _, q2, q3 = query_ids(V1)
    # q0: article 1 from R1 and article 2 from R2; q2: article 4; q3: article 5; q1: nothing.
    assert sorted(hits.rows()) == [(q0, 1), (q0, 2), (q2, 4), (q3, 5)]
    union = evaluator.union_recall_of_hits(hits)
    assert union["value"].to_list() == [1.0, 0.0, 0.5, 1.0]


def test_lines_before_the_anchor_stop_before_the_anchor_day(tmp_path: Path) -> None:
    context = build_diag_workspace(tmp_path)
    lines = read_lines_before_anchor(context, T1)
    assert lines.columns == ["article_idx", "t_dat_day"]
    # Article 5 was sold ON T1's anchor day: that line is label-week data and is not read.
    assert sorted(lines.rows()) == [(1, 0), (1, 300), (2, anchor_day("V1") - 10), (4, 0)]
    assert max(lines["t_dat_day"].to_list()) < anchor_day("T1")
    # Poison: more lines on the anchor day and after it change nothing.
    poison = [(9, 5, anchor_day("T1")), (9, 8, anchor_day("T1") + 1), (9, 6, anchor_day("T1") + 30)]
    write_transactions(context.paths, [*transaction_lines(), *poison])
    assert sorted(read_lines_before_anchor(context, T1).rows()) == sorted(lines.rows())
    # For the next fold the same line of article 5 IS history (7 days before T2's anchor).
    t2_lines = read_lines_before_anchor(context, MANIFEST.get("T2"))
    assert (5, anchor_day("T1")) in t2_lines.rows()
