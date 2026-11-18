"""Tests for hmrec.diagnostics.run: one whole run of the diagnostics on the tiny workspace.

The expected numbers are computed by hand in tests/unit/diag_example.py.
"""

import copy
from pathlib import Path

import polars as pl
import pytest
from diag_example import (
    COMMIT,
    FORMAL_COMMIT,
    MANIFEST,
    NOW,
    SCRIPT,
    TINY_DIAGNOSTICS,
    anchor_day,
    build_diag_workspace,
    file_snapshot,
    make_context,
    transaction_lines,
    write_transactions,
)

from hmrec.csv_records import read_rows
from hmrec.diagnostics.access import ACCESS_COLUMNS, DiagnosticsRefused
from hmrec.diagnostics.bootstrap import BOOTSTRAP_COLUMNS
from hmrec.diagnostics.recorded import recorded_fold
from hmrec.diagnostics.records import PROVENANCE_COLUMNS
from hmrec.diagnostics.run import (
    check_group_counts,
    fold_per_query,
    needs_t_labels,
    recorded_pooled,
    run_diagnostics,
)
from hmrec.eval.diagnostic_labels import DiagnosticAccess

DIAG_FILES = {
    "results/t_diag_label_access.csv",
    "results/t_diag_decomposition.csv",
    "results/t_diag_concentration.csv",
    "results/t_diag_query_groups.csv",
    "results/t_diag_bootstrap.csv",
}
RESULT_FOLDS = ("V1", "V2", "V3", "V4", "T1", "T2", "T3", "T4")


def _by_fold(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["fold"]: row for row in rows}


def test_a_run_writes_only_the_five_diagnostics_files(tmp_path: Path) -> None:
    context = build_diag_workspace(tmp_path)
    before = file_snapshot(tmp_path)
    run_diagnostics(context, NOW, SCRIPT)
    after = file_snapshot(tmp_path)
    assert set(after) - set(before) == DIAG_FILES
    # Every file that was there before is byte-for-byte the same: the M8 and M7 result
    # tables, labels, candidates, features, saved predictions and the processed data.
    assert {name: after[name] for name in before} == before


def test_table_a_decomposition(tmp_path: Path) -> None:
    context = build_diag_workspace(tmp_path)
    tables = run_diagnostics(context, NOW, SCRIPT)
    assert [row["fold"] for row in tables.decomposition] == list(RESULT_FOLDS)
    row = _by_fold(tables.decomposition)["T1"]
    assert (row["phase"], row["n_queries"], row["union_recall"]) == ("T", "4", "0.625")
    # Recall@12 0.625 of a union recall of 0.625: every found label is in the Top-12.
    assert row["lambdamart_recall@12_over_union"] == "1.0"
    assert (row["lambdamart_map@12"], row["rrf_baseline_map@12"]) == ("0.625", "0.5625")


def test_table_b_concentration(tmp_path: Path) -> None:
    context = build_diag_workspace(tmp_path)
    rows = _by_fold(run_diagnostics(context, NOW, SCRIPT).concentration)
    assert list(rows) == list(MANIFEST.names)  # all 16 folds, D1 ... T4
    new, unsold = "n_articles_first_sale_within_28d", "n_articles_first_sale_in_label_week"
    for row in rows.values():
        # 6 label pairs on 5 articles; article 1 has 2 of them, the next article 1.
        assert (row["n_label_pairs"], row["n_label_articles"]) == ("6", "5")
        assert (row["top1_pair_share"], row["top2_pair_share"]) == (repr(2 / 6), repr(3 / 6))
    # D1: articles 2, 5 and 8 have no sale before the anchor.
    assert (rows["D1"][new], rows["D1"][unsold]) == ("0", "3")
    assert rows["D1"]["pair_share_first_sale_in_label_week"] == repr(3 / 6)
    # V1: article 2 was first sold 10 days before the anchor (new); 5 and 8 not yet.
    assert (rows["V1"][new], rows["V1"][unsold]) == ("1", "2")
    assert rows["V1"]["pair_share_first_sale_within_28d"] == repr(1 / 6)
    # V4: article 2 is 31 days old now (no longer new).
    assert (rows["V4"][new], rows["V4"][unsold]) == ("0", "2")
    # T1: article 5 was first sold ON the anchor day. That is the label week, so it
    # counts as "first sale in the label week", not as history.
    assert (rows["T1"][new], rows["T1"][unsold]) == ("0", "2")
    # T2: the same sale is 7 days before the anchor now: article 5 is new; only 8 is unsold.
    assert (rows["T2"][new], rows["T2"][unsold]) == ("1", "1")
    assert rows["T2"]["anchor"] == MANIFEST.get("T2").anchor.isoformat()


def test_table_b_ignores_sales_on_and_after_the_anchor(tmp_path: Path) -> None:
    context = build_diag_workspace(tmp_path)
    clean = _by_fold(run_diagnostics(context, NOW, SCRIPT).concentration)
    # Poison: article 8 (never sold before) gets sales on V1's anchor day and later.
    poison = [(9, 8, anchor_day("V1")), (9, 8, anchor_day("V1") + 3)]
    write_transactions(context.paths, [*transaction_lines(), *poison])
    poisoned = _by_fold(run_diagnostics(context, "2026-11-19T11:00:00", SCRIPT).concentration)
    for fold in ("D1", "D4", "F4", "V1"):  # anchors up to V1: the poison is not history
        assert poisoned[fold] == clean[fold]
    # From V2 on these lines ARE before the anchor, so article 8 becomes a new article.
    assert poisoned["V2"]["n_articles_first_sale_in_label_week"] == "1"


def test_table_c_query_groups(tmp_path: Path) -> None:
    context = build_diag_workspace(tmp_path)
    rows = run_diagnostics(context, NOW, SCRIPT).query_groups
    assert len(rows) == 3 * len(RESULT_FOLDS)
    t1 = {row["query_group"]: row for row in rows if row["fold"] == "T1"}
    assert list(t1) == ["no_history", "recent_purchase", "other"]
    # q1 never bought; q0 (3 days) and q3 (30 days) bought recently; q2 (45 days) is "other".
    assert [t1[group]["n_queries"] for group in t1] == ["1", "2", "1"]
    assert [t1[group]["query_share"] for group in t1] == ["0.25", "0.5", "0.25"]
    assert [t1[group]["union_recall"] for group in t1] == ["0.0", "1.0", "0.5"]
    assert [t1[group]["lambdamart_map@12"] for group in t1] == ["0.0", "1.0", "0.5"]
    assert [t1[group]["rrf_baseline_map@12"] for group in t1] == ["0.0", "1.0", "0.25"]


def test_table_d_bootstrap(tmp_path: Path) -> None:
    context = build_diag_workspace(tmp_path)
    rows = run_diagnostics(context, NOW, SCRIPT).bootstrap
    assert [(row["phase"], row["quantity"]) for row in rows] == [
        ("T", "lambdamart"),
        ("T", "rrf_baseline"),
        ("T", "lambdamart_minus_rrf_baseline"),
        ("V", "lambdamart"),
        ("V", "rrf_baseline"),
        ("V", "lambdamart_minus_rrf_baseline"),
    ]
    ranker, baseline, difference = rows[:3]
    assert ranker["folds"] == "T1 T2 T3 T4"
    # 16 queries of 7 customers: customers 1, 2, 3 are in all four weeks (4 queries each),
    # and each week has one customer of its own.
    assert (ranker["n_customers"], ranker["n_queries"]) == ("7", "16")
    assert (ranker["estimate"], baseline["estimate"], difference["estimate"]) == (
        "0.625",
        "0.5625",
        "0.0625",
    )
    assert (ranker["n_replicates"], ranker["seed"], ranker["confidence_level"]) == (
        "200",
        "7",
        "0.95",
    )
    for row in rows:
        assert float(row["ci_low"]) <= float(row["estimate"]) <= float(row["ci_high"])
    # The Ranker is never below the baseline on any query here, so no replicate of the
    # paired difference is negative.
    assert float(difference["ci_low"]) >= 0.0


def test_rows_carry_the_commit_and_the_start_of_their_run(tmp_path: Path) -> None:
    context = build_diag_workspace(tmp_path)
    run_diagnostics(context, NOW, SCRIPT)
    run_diagnostics(context, "2026-11-20T09:00:00", SCRIPT)  # a second run appends
    paths = context.paths
    rows = read_rows(paths.t_diag_bootstrap_file, (*BOOTSTRAP_COLUMNS, *PROVENANCE_COLUMNS))
    assert [row["run_started_at"] for row in rows] == [NOW] * 6 + ["2026-11-20T09:00:00"] * 6
    assert {row["git_commit"] for row in rows} == {COMMIT}
    assert rows[:6] != rows[6:] and [r["ci_low"] for r in rows[:6]] == [
        r["ci_low"] for r in rows[6:]
    ]
    access = read_rows(paths.t_diag_label_access_file, ACCESS_COLUMNS)
    assert [row["recorded_at"] for row in access] == [NOW, "2026-11-20T09:00:00"]
    assert {row["formal_run_commit"] for row in access} == {FORMAL_COMMIT}


def test_a_refused_run_reads_and_writes_nothing(tmp_path: Path) -> None:
    paths = build_diag_workspace(tmp_path).paths
    # Without the label files a run that got past the gate would fail with another error.
    for fold in MANIFEST.names:
        paths.labels_file(fold).unlink()
    before = file_snapshot(tmp_path)
    for context in (make_context(paths, dirty=True), make_context(paths, changed=("M configs/x",))):
        with pytest.raises(DiagnosticsRefused):
            run_diagnostics(context, NOW, SCRIPT)
    assert file_snapshot(tmp_path) == before


def test_a_run_without_the_t_phase_asks_for_no_access(tmp_path: Path) -> None:
    paths = build_diag_workspace(tmp_path).paths
    raw = copy.deepcopy(TINY_DIAGNOSTICS)
    raw.update(concentration_phases=["D", "V"], query_group_phases=["V"], bootstrap_phases=["V"])
    # Even a dirty tree is no obstacle: no T label is read (table A reads result tables).
    context = make_context(paths, dirty=True, diagnostics=raw)
    assert not needs_t_labels(context) and needs_t_labels(make_context(paths))
    tables = run_diagnostics(context, NOW, SCRIPT)
    assert not paths.t_diag_label_access_file.exists()
    assert [row["fold"] for row in tables.decomposition] == list(RESULT_FOLDS)
    assert {row["phase"] for row in tables.bootstrap} == {"V"}


def test_a_recorded_number_that_is_not_reproduced_stops_the_run(tmp_path: Path) -> None:
    context = build_diag_workspace(tmp_path)
    paths = context.paths
    # T3's saved per-query values no longer give the recorded MAP@12 of 0.625.
    saved = paths.ranker_per_query_file("t_phase", "full", context.ranker.config_id, "T3")
    pl.read_parquet(saved).with_columns(pl.col("map@12") * 0.5).write_parquet(saved)
    with pytest.raises(ValueError, match="T3 Ranker map@12: recomputed"):
        run_diagnostics(context, NOW, SCRIPT)
    # Nothing was reported. The access row stays: T labels were read before the check failed.
    written = {path.name for path in paths.results_root.glob("t_diag_*.csv")}
    assert written == {"t_diag_label_access.csv"}


def test_the_group_sizes_must_add_up_to_the_recorded_queries(tmp_path: Path) -> None:
    context = build_diag_workspace(tmp_path)
    access = DiagnosticAccess(COMMIT, FORMAL_COMMIT, NOW, "descriptive diagnostics")
    t1 = MANIFEST.get("T1")
    per_query = fold_per_query(context, t1, access)
    recorded = recorded_fold(context, t1)
    check_group_counts(per_query, recorded)  # 1 + 2 + 1 = 4 recorded queries
    with pytest.raises(ValueError, match="hold 3 queries, but the result tables record 4"):
        check_group_counts(per_query.head(3), recorded)
    unknown = per_query.with_columns(pl.lit("somewhere").alias("query_group"))
    with pytest.raises(ValueError, match="unknown query groups"):
        check_group_counts(unknown, recorded)
    with pytest.raises(TypeError, match="DiagnosticAccess"):
        fold_per_query(context, t1, None)  # a T fold's labels are not read without access


def test_recorded_pooled_is_the_query_weighted_mean(tmp_path: Path) -> None:
    context = build_diag_workspace(tmp_path)
    folds = [recorded_fold(context, MANIFEST.get(name)) for name in ("V1", "V2")]
    # Two folds of 4 and 12 queries with MAP 0.625 and 0.125: (4 * 0.625 + 12 * 0.125) / 16.
    bigger = type(folds[1])(**{**folds[1].__dict__, "n_queries": 12, "ranker": {"map@12": 0.125}})
    assert recorded_pooled([folds[0], bigger], "ranker", "map@12") == 0.25
    assert recorded_pooled(folds, "baseline", "map@12") == 0.5625
