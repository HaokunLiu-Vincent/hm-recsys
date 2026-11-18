"""Tests for kaggle.rehearsal: Top-K lists compared by customer; the gate before the real run.

Example (k = 2): the fold has queries q1 (customer 5) and q2 (customer 6); M8 saved
    q1: articles 10, 11    q2: 20, 21
The rehearsal numbers its queries differently (q900 = customer 5, q901 = customer 6,
q902 = customer 7, who is not a query of the fold).
"""

from pathlib import Path

import polars as pl
import pytest

from hmrec.config import load_base_config
from hmrec.csv_records import read_rows
from hmrec.kaggle.rehearsal import (
    REHEARSAL_COLUMNS,
    RehearsalOutcome,
    TopComparison,
    compare_top_lists,
    record_rehearsal,
    require_passed_rehearsal,
)
from hmrec.paths import ProjectPaths, build_paths
from hmrec.run_records import RunStamp

REPO = Path(__file__).resolve().parents[2]
BASE = load_base_config(REPO / "configs" / "base.yaml")


def _top(rows: list[tuple[int, int, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows, schema={"query_id": pl.Int32, "rank": pl.Int32, "article_idx": pl.Int32}, orient="row"
    )


def _queries(pairs: list[tuple[int, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        pairs, schema={"query_id": pl.Int32, "customer_idx": pl.Int32}, orient="row"
    )


OLD_QUERIES = _queries([(1, 5), (2, 6)])
OLD_TOP = _top([(1, 1, 10), (1, 2, 11), (2, 1, 20), (2, 2, 21)])
NEW_QUERIES = _queries([(900, 5), (901, 6), (902, 7)])


def _compare(new_rows: list[tuple[int, int, int]]) -> TopComparison:
    return compare_top_lists(_top(new_rows), NEW_QUERIES, OLD_TOP, OLD_QUERIES)


def test_lists_are_matched_by_customer_not_by_query_id() -> None:
    same = [(900, 1, 10), (900, 2, 11), (901, 1, 20), (901, 2, 21), (902, 1, 99), (902, 2, 98)]
    result = _compare(same)
    # Customer 7 is not a query of the fold: its list is not compared.
    assert (result.n_queries, result.n_identical, result.share) == (2, 2, 1.0)


def test_another_order_or_another_article_is_not_identical() -> None:
    swapped = [(900, 1, 11), (900, 2, 10), (901, 1, 20), (901, 2, 21)]
    assert _compare(swapped).n_identical == 1  # the same two articles, other order
    replaced = [(900, 1, 10), (900, 2, 11), (901, 1, 20), (901, 2, 22)]
    assert _compare(replaced).n_identical == 1
    shorter = [(900, 1, 10), (900, 2, 11), (901, 1, 20)]
    assert _compare(shorter).n_identical == 1


def test_a_fold_query_without_a_new_list_counts_as_different() -> None:
    result = _compare([(900, 1, 10), (900, 2, 11)])
    assert (result.n_queries, result.n_identical, result.share) == (2, 1, 0.5)


def test_every_fold_query_needs_a_saved_list() -> None:
    with pytest.raises(ValueError, match="2 queries but 1 saved"):
        compare_top_lists(_top([]), NEW_QUERIES, OLD_TOP.head(2), OLD_QUERIES)


def _outcome(n_identical: int = 1000, trees: str = "t", checks: bool = True) -> RehearsalOutcome:
    return RehearsalOutcome(
        n_customers=1500,
        n_shards=2,
        new_trees_sha256=trees,
        m8_trees_sha256="t",
        comparison=TopComparison(1000, n_identical),
        min_identical_share=0.999,
        format_checks_passed=checks,
    )


def test_all_three_checks_are_needed_to_pass() -> None:
    assert _outcome().passed
    # 999 of 1000 = 0.999 is still enough; 998 is not.
    assert _outcome(n_identical=999).passed and not _outcome(n_identical=998).passed
    assert not _outcome(trees="other").passed and not _outcome(trees="other").trees_identical
    assert not _outcome(checks=False).passed


def _paths(tmp_path: Path) -> ProjectPaths:
    return build_paths(BASE, tmp_path)


def test_outcome_is_recorded_with_its_numbers(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    record_rehearsal(paths, _outcome(n_identical=998), RunStamp("c1", False))
    rows = read_rows(paths.kaggle_rehearsal_file, REHEARSAL_COLUMNS)
    assert len(rows) == 1
    row = rows[0]
    assert (row["n_fold_queries"], row["n_identical_top"]) == ("1000", "998")
    assert float(row["identical_top_share"]) == 0.998 and row["min_identical_top_share"] == "0.999"
    assert (row["trees_identical"], row["format_checks_passed"], row["passed"]) == (
        "True",
        "True",
        "False",
    )
    assert (row["git_commit"], row["n_customers"], row["n_shards"]) == ("c1", "1500", "2")


def test_real_run_needs_the_last_rehearsal_to_have_passed_with_this_code(tmp_path: Path) -> None:
    paths = _paths(tmp_path)

    def unchanged(commit: str) -> bool:
        return False

    with pytest.raises(RuntimeError, match="no rehearsal is recorded"):
        require_passed_rehearsal(paths, unchanged)
    record_rehearsal(paths, _outcome(), RunStamp("c1", False))
    assert require_passed_rehearsal(paths, unchanged) == "c1"
    # The code changed after the rehearsal: it must be run again.
    with pytest.raises(RuntimeError, match="the code changed since"):
        require_passed_rehearsal(paths, lambda commit: True)
    # A later failed rehearsal closes the gate again, whatever passed before.
    record_rehearsal(paths, _outcome(n_identical=10), RunStamp("c2", False))
    with pytest.raises(RuntimeError, match="did not pass"):
        require_passed_rehearsal(paths, unchanged)
    # A rehearsal from a dirty tree does not count either.
    record_rehearsal(paths, _outcome(), RunStamp("c3", True))
    with pytest.raises(RuntimeError, match="did not pass from a clean working tree"):
        require_passed_rehearsal(paths, unchanged)
