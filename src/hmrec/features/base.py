"""Shared pieces of the feature groups (proposal §8.1).

Every feature of a fold is computed as of the fold's anchor a: only purchases
with t_dat < a are used. All purchase lines come from window_lines /
lines_before below, which call hmrec.data.asof (the only place that filters by
date) and add

    age_days = anchor_day - t_dat_day      (yesterday = 1)

so "days since ..." features are simply the smallest age_days.

Windows (configs/features.yaml), all ending before the anchor:
    1w = [a - 7, a)    prev_1w = [a - 14, a - 7)    4w = [a - 28, a)    52w = [a - 364, a)
A feature without a window uses all history t_dat < a.

Missing values: a COUNT with no data is 0. A "days since", a mean, a share or
a ratio that cannot be defined (never bought, empty window, zero denominator,
one side missing) is NaN.

Feature code never imports recall/ and never reads labels.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl

from hmrec.data.asof import TRANSACTION_DAY, history_window
from hmrec.data.day_index import date_to_day_index

QUERY_ID = "query_id"
ARTICLE_IDX = "article_idx"
CUSTOMER_IDX = "customer_idx"
KEYS = [QUERY_ID, ARTICLE_IDX]
AGE_DAYS = "age_days"
PRICE = "price"
SALES_CHANNEL = "sales_channel_id"
PRODUCT_CODE = "product_code"
AGE = "age"
AGE_BUCKET = "age_bucket"
NAN = float("nan")

# The six article category fields of proposal §8.4 and the two customer fields of §8.5.
ARTICLE_CATEGORY_COLUMNS = (
    "product_type_no",
    "product_group_name",
    "garment_group_no",
    "index_group_no",
    "section_no",
    "colour_group_code",
)
CUSTOMER_CATEGORY_COLUMNS = ("club_member_status", "fashion_news_frequency")


@dataclass(frozen=True)
class FoldData:
    """Everything the feature groups read for one fold (no labels, no recall code).

    Attributes:
        fold: Fold name, e.g. "F1".
        anchor: First day of the label week; features use t_dat < anchor only.
        day_zero: Day 0 of t_dat_day.
        transactions: Processed transactions (lazy); one row per purchase line
            with customer_idx, article_idx, t_dat_day (int16), price,
            sales_channel_id. Cut only through window_lines / lines_before.
        customers: One row per customer: customer_idx, age (null if unknown),
            age_bucket, fn, active, club_member_status, fashion_news_frequency.
        articles: One row per article of the catalog: article_idx,
            product_code and the six category fields of §8.4.
        queries: One row per query of the fold: query_id, customer_idx.
    """

    fold: str
    anchor: date
    day_zero: date
    transactions: pl.LazyFrame
    customers: pl.DataFrame
    articles: pl.DataFrame
    queries: pl.DataFrame

    @property
    def anchor_day(self) -> int:
        """The anchor as a t_dat_day index (via hmrec.data.day_index)."""
        return date_to_day_index(self.anchor, self.day_zero)

    @property
    def query_customers(self) -> np.ndarray:
        """Distinct customers of the fold's queries, int32, ascending."""
        return np.unique(self.queries[CUSTOMER_IDX].to_numpy()).astype(np.int32)


def _with_age_days(lines: pl.DataFrame, anchor_day: int) -> pl.DataFrame:
    """Add age_days = anchor_day - t_dat_day (int32) and fail if any is below 1.

    age_days < 1 would mean a purchase on the anchor day or later reached a
    feature: a time leak.
    """
    aged = lines.with_columns(
        (pl.lit(anchor_day, dtype=pl.Int32) - pl.col(TRANSACTION_DAY).cast(pl.Int32)).alias(
            AGE_DAYS
        )
    )
    if aged.height > 0:
        youngest = aged[AGE_DAYS].min()
        if not isinstance(youngest, int) or youngest < 1:
            raise ValueError(
                f"purchase lines have age_days={youngest} (< 1): rows on or after the anchor leaked"
            )
    return aged


def _of_customers(transactions: pl.LazyFrame, customers: np.ndarray | None) -> pl.LazyFrame:
    """Keep only the given customers' lines (None = everyone). Not a date filter."""
    if customers is None:
        return transactions
    wanted = pl.Series(CUSTOMER_IDX, customers, dtype=pl.Int32).implode()
    return transactions.filter(pl.col(CUSTOMER_IDX).is_in(wanted))


def window_lines(
    data: FoldData,
    lookback_days: int,
    columns: Sequence[str],
    days_before_anchor: int = 0,
    customers: np.ndarray | None = None,
) -> pl.DataFrame:
    """Purchase lines of a window that ends at the anchor or a number of days before it.

    The window is [a - days_before_anchor - lookback_days, a - days_before_anchor).
    With days_before_anchor = 0 this is the usual window ending at the anchor,
    e.g. 4w = window_lines(data, 28, ...). prev_1w = [a - 14, a - 7) is
    window_lines(data, 7, ..., days_before_anchor=7): a window that ends 7 days
    BEFORE the anchor, so it can never reach the anchor either.

    Args:
        data: The fold.
        lookback_days: Window length in days (> 0).
        columns: Columns to read (t_dat_day is always added).
        days_before_anchor: How many days before the anchor the window ends (>= 0).
        customers: If given, only these customers' lines.

    Returns:
        One row per purchase line with `columns`, t_dat_day and age_days
        (int32, counted from the fold's anchor, so always >= 1).
    """
    if days_before_anchor < 0:
        raise ValueError(f"days_before_anchor must be >= 0, got {days_before_anchor}")
    window_end = data.anchor - timedelta(days=days_before_anchor)
    lines = history_window(
        _of_customers(data.transactions, customers),
        window_end,
        data.day_zero,
        lookback_days,
        columns=list(columns),
    )
    return _with_age_days(lines, data.anchor_day)


def lines_before(
    data: FoldData, columns: Sequence[str], customers: np.ndarray | None = None
) -> pl.DataFrame:
    """Every purchase line before the anchor: t_dat_day in [0, anchor_day) (all history).

    Returns:
        As window_lines.
    """
    if data.anchor_day <= 0:
        raise ValueError(f"anchor {data.anchor} must be after day_zero {data.day_zero}")
    return window_lines(data, data.anchor_day, columns, customers=customers)


def as_feature(name: str, count: bool = False) -> pl.Expr:
    """A table column as a float32 feature: a missing COUNT is 0, any other missing value NaN."""
    column = pl.col(name).cast(pl.Float32)
    return column.fill_null(0.0 if count else NAN).alias(name)


@dataclass(frozen=True)
class HistoryArrays:
    """The query customers' last purchases before the anchor (hmrec.data.histories).

    The same history R6 predicts with. Used by img_sim_hist, txt_sim_hist,
    tt_sim_max_hist, tt_sim_top5_hist and ui_attr_match_cnt.

    Attributes:
        customer_idx: int32 [Q], strictly increasing; row q of the two arrays
            below belongs to this customer.
        history: int32 [Q, L]; slot 0 is the most recent purchase, empty slots
            hold the padding index n_articles.
        history_mask: bool [Q, L], True for filled slots.
    """

    customer_idx: np.ndarray
    history: np.ndarray
    history_mask: np.ndarray


def history_rows(arrays: HistoryArrays) -> pl.DataFrame:
    """The customers' last purchases as a table: one row per filled history slot.

    Returns:
        customer_idx (int32), article_idx (int32). An article bought on two
        different days appears twice; a customer without purchases has no row.
    """
    filled = arrays.history_mask.ravel()
    customers = np.repeat(arrays.customer_idx, arrays.history.shape[1])[filled]
    articles = arrays.history.ravel()[filled]
    return pl.DataFrame(
        {CUSTOMER_IDX: customers, ARTICLE_IDX: articles},
        schema={CUSTOMER_IDX: pl.Int32, ARTICLE_IDX: pl.Int32},
    )
