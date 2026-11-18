"""Tests for hmrec.diagnostics.query_groups (table C), with hand-computed numbers."""

import polars as pl
import pytest

from hmrec.diagnostics.query_groups import (
    GROUPS,
    assign_query_groups,
    build_per_query,
    column_mean,
    query_days,
    query_group_columns,
    query_group_rows,
)

NAN = float("nan")
DAYS = "user_days_since_last_purchase"


def _days(rows: list[tuple[int, float | None]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema={"query_id": pl.Int32, DAYS: pl.Float32}, orient="row")


def _values(values: dict[int, float]) -> pl.DataFrame:
    return pl.DataFrame(
        {"query_id": list(values), "fold": ["V1"] * len(values), "value": list(values.values())},
        schema={"query_id": pl.Int32, "fold": pl.String, "value": pl.Float64},
    )


def _queries(customers: dict[int, int]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "query_id": list(customers),
            "fold": ["V1"] * len(customers),
            "customer_idx": list(customers.values()),
        },
        schema={"query_id": pl.Int32, "fold": pl.String, "customer_idx": pl.Int32},
    )


def test_query_days_keeps_one_row_per_query_also_for_nan() -> None:
    # Three candidate rows of query 1 (NaN: never bought) and two of query 2.
    per_query = query_days(_days([(2, 5.0), (1, NAN), (1, NAN), (2, 5.0), (1, NAN)]))
    assert per_query["query_id"].to_list() == [1, 2]
    assert per_query[DAYS].is_nan().to_list() == [True, False]
    assert per_query[DAYS][1] == 5.0


def test_query_days_refuses_values_that_cannot_be_right() -> None:
    with pytest.raises(ValueError, match="differs between the candidate rows"):
        query_days(_days([(1, 5.0), (1, 6.0)]))
    with pytest.raises(ValueError, match="differs between the candidate rows"):
        query_days(_days([(1, 5.0), (1, NAN)]))
    # 0 days would be a purchase ON the anchor day: that is the label week, not history.
    with pytest.raises(ValueError, match="NaN or >= 1"):
        query_days(_days([(1, 0.0)]))
    with pytest.raises(ValueError, match="nulls"):
        query_days(_days([(1, None)]))


def test_groups_at_the_boundaries() -> None:
    days = _days([(1, NAN), (2, 1.0), (3, 30.0), (4, 31.0), (5, 700.0)])
    groups = assign_query_groups(days, recent_purchase_days=30)
    assert groups.rows() == [
        (1, "no_history"),
        (2, "recent_purchase"),  # bought yesterday
        (3, "recent_purchase"),  # exactly 30 days ago still counts
        (4, "other"),
        (5, "other"),
    ]
    with pytest.raises(ValueError, match="recent_purchase_days must be > 0"):
        assign_query_groups(days, recent_purchase_days=0)


def _per_query() -> pl.DataFrame:
    return build_per_query(
        queries=_queries({1: 10, 2: 20, 3: 30, 4: 40}),
        groups=assign_query_groups(_days([(1, 3.0), (2, NAN), (3, 45.0), (4, 30.0)]), 30),
        union_recall=_values({1: 1.0, 2: 0.0, 3: 0.5, 4: 1.0}),
        ranker_ap=_values({4: 1.0, 3: 0.5, 2: 0.0, 1: 1.0}),  # another row order on purpose
        baseline_ap=_values({1: 1.0, 2: 0.0, 3: 0.25, 4: 0.5}),
    )


def test_build_per_query_joins_one_row_per_query() -> None:
    table = _per_query()
    assert table.columns == [
        "query_id",
        "fold",
        "customer_idx",
        "query_group",
        "union_recall",
        "ranker_ap",
        "baseline_ap",
    ]
    assert table.rows() == [
        (1, "V1", 10, "recent_purchase", 1.0, 1.0, 1.0),
        (2, "V1", 20, "no_history", 0.0, 0.0, 0.0),
        (3, "V1", 30, "other", 0.5, 0.5, 0.25),
        (4, "V1", 40, "recent_purchase", 1.0, 1.0, 0.5),
    ]
    assert column_mean(table, "baseline_ap") == (1.0 + 0.0 + 0.25 + 0.5) / 4


def test_build_per_query_refuses_tables_with_other_queries() -> None:
    union = _values({1: 1.0, 2: 0.0})
    groups = assign_query_groups(_days([(1, 3.0), (2, NAN)]), 30)
    queries = _queries({1: 10, 2: 20})
    good = _values({1: 1.0, 2: 0.0})
    with pytest.raises(ValueError, match="the Ranker's per-query values"):
        build_per_query(queries, groups, union, _values({1: 1.0}), good)  # a query is missing
    with pytest.raises(ValueError, match="the baseline's per-query values"):
        build_per_query(queries, groups, union, good, _values({1: 1.0, 3: 0.0}))  # another query
    with pytest.raises(ValueError, match="the query list"):
        build_per_query(_queries({1: 10, 2: 20, 3: 30}), groups, union, good, good)
    with pytest.raises(ValueError, match="the query groups"):
        build_per_query(queries, groups.head(1), union, good, good)


def test_query_group_rows_by_hand() -> None:
    rows = query_group_rows(_per_query(), "V1", "V", "map@12")
    assert [row["query_group"] for row in rows] == list(GROUPS)
    assert list(rows[0]) == list(query_group_columns("map@12"))
    no_history, recent, other = rows
    # no_history: query 2 only. recent_purchase: queries 1 and 4. other: query 3.
    assert (no_history["n_queries"], no_history["query_share"]) == ("1", "0.25")
    assert (no_history["union_recall"], no_history["lambdamart_map@12"]) == ("0.0", "0.0")
    assert (recent["n_queries"], recent["query_share"]) == ("2", "0.5")
    assert recent["union_recall"] == "1.0"
    assert recent["lambdamart_map@12"] == "1.0"
    assert recent["rrf_baseline_map@12"] == "0.75"  # (1.0 + 0.5) / 2
    assert (other["union_recall"], other["rrf_baseline_map@12"]) == ("0.5", "0.25")


def test_an_empty_group_has_no_means() -> None:
    everyone_recent = _per_query().with_columns(pl.lit("recent_purchase").alias("query_group"))
    no_history, recent, other = query_group_rows(everyone_recent, "V1", "V", "map@12")
    assert (no_history["n_queries"], no_history["query_share"]) == ("0", "0.0")
    assert (no_history["union_recall"], no_history["lambdamart_map@12"]) == ("", "")
    assert (recent["n_queries"], other["n_queries"]) == ("4", "0")
