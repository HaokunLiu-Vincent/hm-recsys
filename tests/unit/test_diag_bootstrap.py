"""Tests for hmrec.diagnostics.bootstrap (table D): the cluster bootstrap over customers."""

import numpy as np
import polars as pl
import pytest

from hmrec.diagnostics.bootstrap import (
    BOOTSTRAP_COLUMNS,
    QUANTITIES,
    bootstrap_rows,
    cluster_bootstrap,
    customer_totals,
    draw_customer_weights,
    pooled_mean,
)


def _per_query(rows: list[tuple[int, float, float]]) -> pl.DataFrame:
    """Rows (customer_idx, ranker_ap, baseline_ap), one per query."""
    return pl.DataFrame(
        rows,
        schema={"customer_idx": pl.Int32, "ranker_ap": pl.Float64, "baseline_ap": pl.Float64},
        orient="row",
    )


# Customer 1 has three queries (three weeks), customer 2 one, customer 3 two.
PER_QUERY = _per_query(
    [(3, 0.5, 0.0), (1, 1.0, 0.5), (2, 0.0, 0.0), (1, 1.0, 0.5), (3, 0.5, 0.25), (1, 0.0, 0.5)]
)


def test_customer_totals_puts_all_queries_of_a_customer_in_one_row() -> None:
    totals = customer_totals(PER_QUERY)
    assert totals.rows() == [(1, 3, 2.0, 1.5), (2, 1, 0.0, 0.0), (3, 2, 1.0, 0.25)]


def test_pooled_mean_of_one_replicate_by_hand() -> None:
    # Customer 1 drawn twice, customer 2 never, customer 3 once:
    # AP sum = 2 * 2.0 + 0 + 1 * 1.0 = 5.0; queries = 2 * 3 + 0 + 1 * 2 = 8.
    weights = np.array([2, 0, 1])
    sums = np.array([2.0, 0.0, 1.0])
    n_queries = np.array([3, 1, 2])
    assert pooled_mean(weights, sums, n_queries) == 5.0 / 8.0


def test_draws_are_customers_with_replacement_and_reproducible() -> None:
    weights = draw_customer_weights(n_customers=5, n_replicates=50, seed=3)
    assert len(weights) == 50
    assert all(w.shape == (5,) and int(w.sum()) == 5 and (w >= 0).all() for w in weights)
    # With replacement: in some replicate a customer is drawn more than once.
    assert any(int(w.max()) > 1 for w in weights)
    again = draw_customer_weights(n_customers=5, n_replicates=50, seed=3)
    assert all((a == b).all() for a, b in zip(weights, again, strict=True))
    other_seed = draw_customer_weights(n_customers=5, n_replicates=50, seed=4)
    assert any((a != b).any() for a, b in zip(weights, other_seed, strict=True))
    with pytest.raises(ValueError, match="must be > 0"):
        draw_customer_weights(0, 10, 3)


def test_cluster_bootstrap_estimates_and_intervals() -> None:
    totals = customer_totals(PER_QUERY)
    intervals = cluster_bootstrap(totals, n_replicates=300, seed=11, confidence_level=0.95)
    assert list(intervals) == list(QUANTITIES)
    # Real data: Ranker (2.0 + 0.0 + 1.0) / 6 = 0.5; baseline (1.5 + 0 + 0.25) / 6.
    assert intervals["lambdamart"].estimate == 0.5
    assert intervals["rrf_baseline"].estimate == 1.75 / 6
    assert intervals["lambdamart_minus_rrf_baseline"].estimate == 0.5 - 1.75 / 6
    for interval in intervals.values():
        assert interval.low <= interval.estimate <= interval.high
    assert cluster_bootstrap(totals, 300, 11, 0.95) == intervals  # the same seed, the same result
    # A narrower confidence level gives an interval inside the wider one.
    narrow = cluster_bootstrap(totals, 300, 11, 0.5)["lambdamart"]
    assert intervals["lambdamart"].low <= narrow.low <= narrow.high <= intervals["lambdamart"].high


def test_the_interval_is_the_quantiles_of_the_replicates_by_hand() -> None:
    totals = customer_totals(PER_QUERY)
    n_queries = np.array([3, 1, 2])
    weights = draw_customer_weights(3, 200, seed=5)
    ranker = np.array([pooled_mean(w, np.array([2.0, 0.0, 1.0]), n_queries) for w in weights])
    baseline = np.array([pooled_mean(w, np.array([1.5, 0.0, 0.25]), n_queries) for w in weights])
    intervals = cluster_bootstrap(totals, 200, 5, 0.9)
    # Both ends in ONE numpy call, as the module does (alpha / 2 = 0.05 and 0.95).
    low, high = np.quantile(ranker, [0.05, 0.95])
    assert (intervals["lambdamart"].low, intervals["lambdamart"].high) == (float(low), float(high))
    # The difference is PAIRED: the same draws for both systems, replicate by replicate.
    low, high = np.quantile(ranker - baseline, [0.05, 0.95])
    difference = intervals["lambdamart_minus_rrf_baseline"]
    assert (difference.low, difference.high) == (float(low), float(high))


def test_identical_customers_give_an_interval_of_width_zero() -> None:
    # Whoever is drawn, the pooled mean is the same: there is no customer uncertainty.
    same = _per_query([(1, 0.5, 0.25), (2, 0.5, 0.25), (3, 0.5, 0.25)])
    intervals = cluster_bootstrap(customer_totals(same), 100, 1, 0.95)
    assert (intervals["lambdamart"].low, intervals["lambdamart"].high) == (0.5, 0.5)
    assert intervals["lambdamart_minus_rrf_baseline"].low == 0.25


def test_bootstrap_rows_have_the_table_columns() -> None:
    totals = customer_totals(PER_QUERY)
    intervals = cluster_bootstrap(totals, 100, 1, 0.95)
    rows = bootstrap_rows("T", ("T1", "T2"), "map@12", totals, intervals, 100, 1, 0.95)
    assert [row["quantity"] for row in rows] == list(QUANTITIES)
    assert all(list(row) == list(BOOTSTRAP_COLUMNS) for row in rows)
    first = rows[0]
    assert (first["phase"], first["folds"], first["metric"]) == ("T", "T1 T2", "map@12")
    assert (first["n_customers"], first["n_queries"]) == ("3", "6")
    assert (first["n_replicates"], first["seed"], first["confidence_level"]) == ("100", "1", "0.95")
    assert first["estimate"] == "0.5"
    with pytest.raises(ValueError, match="confidence_level"):
        cluster_bootstrap(totals, 100, 1, 1.0)
