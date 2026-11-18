"""Table C: queries grouped by purchase history, and each group's scores.

Every query of a fold is put into one of three groups by the customer's
purchases BEFORE the anchor. The group comes from the saved wide-table feature
user_days_since_last_purchase (proposal §8.5), which is the same for every
candidate row of a query:

    no_history        NaN: the customer has no purchase before the anchor
    recent_purchase   1 <= days <= recent_purchase_days (configs/m8_diagnostics.yaml: 30).
                      days = anchor - last purchase day; yesterday is 1. The anchor
                      day itself is never history.
    other             days > recent_purchase_days

The per-query table of a fold (build_per_query) has one row per query:

    query_id (int32), fold (str), customer_idx (int32), query_group (str),
    union_recall, ranker_ap, baseline_ap (float64)

and table C reports, per group: the number of queries, their share of the fold,
and the means of the three values. Means use math.fsum over the queries sorted
by query_id (hmrec.eval.metrics.macro_mean), like every metric of the project.
"""

import polars as pl

from hmrec.data.id_maps import CUSTOMER_IDX
from hmrec.diagnostics.records import float_text
from hmrec.eval.metrics import FOLD, QUERY_ID, VALUE, macro_mean

DAYS = "user_days_since_last_purchase"
QUERY_GROUP = "query_group"
NO_HISTORY = "no_history"
RECENT_PURCHASE = "recent_purchase"
OTHER = "other"
GROUPS = (NO_HISTORY, RECENT_PURCHASE, OTHER)

UNION_RECALL = "union_recall"
RANKER_AP = "ranker_ap"
BASELINE_AP = "baseline_ap"
VALUE_COLUMNS = (UNION_RECALL, RANKER_AP, BASELINE_AP)


def query_days(wide_days: pl.DataFrame) -> pl.DataFrame:
    """One row per query with its user_days_since_last_purchase.

    Args:
        wide_days: Candidate rows of the wide table (many per query). Columns:
            query_id (int32), user_days_since_last_purchase (float32, NaN = the
            customer never bought before the anchor).

    Returns:
        One row per query, the same two columns, sorted by query_id.

    Raises:
        ValueError: If the value is null, differs between the rows of a query, or
            is a number below 1 (the anchor day or later would be label-week data).
    """
    if wide_days[DAYS].null_count() > 0:
        raise ValueError(f"{DAYS} has nulls; missing values of the wide table are NaN")
    # Dedup at (query_id, value): the feature is a property of the customer, so all
    # candidate rows of a query carry the same value (polars treats NaN as equal to NaN).
    per_query = wide_days.select(QUERY_ID, DAYS).unique()
    if per_query[QUERY_ID].is_duplicated().any():
        raise ValueError(f"{DAYS} differs between the candidate rows of one query")
    too_small = per_query.filter(pl.col(DAYS).is_not_nan() & (pl.col(DAYS) < 1))
    if too_small.height > 0:
        raise ValueError(f"{DAYS} must be NaN or >= 1 (yesterday = 1), got {too_small[DAYS].min()}")
    return per_query.sort(QUERY_ID)


def assign_query_groups(days: pl.DataFrame, recent_purchase_days: int) -> pl.DataFrame:
    """Put every query into one of GROUPS (module docstring).

    Args:
        days: One row per query: query_id, user_days_since_last_purchase (query_days).
        recent_purchase_days: Largest number of days that still counts as recent.

    Returns:
        One row per query: query_id (int32), query_group (str); sorted by query_id.
    """
    if recent_purchase_days <= 0:
        raise ValueError(f"recent_purchase_days must be > 0, got {recent_purchase_days}")
    group = (
        pl.when(pl.col(DAYS).is_nan())
        .then(pl.lit(NO_HISTORY))
        .when(pl.col(DAYS) <= recent_purchase_days)
        .then(pl.lit(RECENT_PURCHASE))
        .otherwise(pl.lit(OTHER))
        .alias(QUERY_GROUP)
    )
    return days.select(QUERY_ID, group).sort(QUERY_ID)


def _check_same_queries(expected: pl.DataFrame, table: pl.DataFrame, name: str) -> None:
    """Fail unless `table` has exactly the queries of `expected`, one row each."""
    if table[QUERY_ID].sort().to_list() != expected[QUERY_ID].sort().to_list():
        raise ValueError(
            f"{name} does not have exactly the labelled queries of the fold "
            f"({table.height} rows, {expected.height} labelled queries)"
        )


def build_per_query(
    queries: pl.DataFrame,
    groups: pl.DataFrame,
    union_recall: pl.DataFrame,
    ranker_ap: pl.DataFrame,
    baseline_ap: pl.DataFrame,
) -> pl.DataFrame:
    """Join everything known about each query of one fold into one row.

    Args:
        queries: The fold's query list: query_id, fold, customer_idx.
        groups: query_id, query_group (assign_query_groups).
        union_recall: query_id, fold, value: one row per LABELLED query (the evaluator).
        ranker_ap: query_id, fold, value: the saved per-query AP of the locked Ranker.
        baseline_ap: query_id, fold, value: the per-query AP of the RRF baseline.

    Returns:
        The per-query table (module docstring), sorted by query_id.

    Raises:
        ValueError: If any table does not have exactly the labelled queries.
    """
    for name, table in (
        ("the query list", queries),
        ("the query groups", groups),
        ("the Ranker's per-query values", ranker_ap),
        ("the baseline's per-query values", baseline_ap),
    ):
        _check_same_queries(union_recall, table, name)
    table = queries.select(QUERY_ID, FOLD, CUSTOMER_IDX).join(groups, on=QUERY_ID)
    for column, values in (
        (UNION_RECALL, union_recall),
        (RANKER_AP, ranker_ap),
        (BASELINE_AP, baseline_ap),
    ):
        table = table.join(values.select(QUERY_ID, pl.col(VALUE).alias(column)), on=QUERY_ID)
    return table.sort(QUERY_ID)


def column_mean(per_query: pl.DataFrame, column: str) -> float:
    """The mean of one value column over the rows (math.fsum, sorted by query_id)."""
    return macro_mean(per_query.select(QUERY_ID, FOLD, pl.col(column).alias(VALUE)))


def query_group_columns(map_metric: str) -> tuple[str, ...]:
    """The columns of table C (without the provenance columns)."""
    return (
        "fold",
        "phase",
        QUERY_GROUP,
        "n_queries",
        "query_share",
        UNION_RECALL,
        f"lambdamart_{map_metric}",
        f"rrf_baseline_{map_metric}",
    )


def query_group_rows(
    per_query: pl.DataFrame, fold: str, phase: str, map_metric: str
) -> list[dict[str, str]]:
    """The three rows of one fold for table C, in the order of GROUPS.

    A group without any query gets n_queries 0, share 0 and empty mean cells.

    Args:
        per_query: The fold's per-query table (build_per_query).
        fold: Fold name.
        phase: The fold's phase.
        map_metric: "map@12" (names the two MAP columns).
    """
    rows: list[dict[str, str]] = []
    for group in GROUPS:
        members = per_query.filter(pl.col(QUERY_GROUP) == group)
        has_queries = members.height > 0
        means = {
            column: column_mean(members, column) if has_queries else None
            for column in VALUE_COLUMNS
        }
        rows.append(
            {
                "fold": fold,
                "phase": phase,
                QUERY_GROUP: group,
                "n_queries": str(members.height),
                "query_share": float_text(members.height / per_query.height),
                UNION_RECALL: float_text(means[UNION_RECALL]),
                f"lambdamart_{map_metric}": float_text(means[RANKER_AP]),
                f"rrf_baseline_{map_metric}": float_text(means[BASELINE_AP]),
            }
        )
    return rows
