"""The ONLY place in the project that filters transactions by date (AGENTS.md §3.1).

Two windows exist, both half-open and both measured in integer days
(`t_dat_day`, see hmrec.data.day_index):

    history:    [anchor_day - lookback_days, anchor_day)   -> history_window
    label week: [anchor_day, label_end_day + 1)            -> label_week_transactions

The anchor day belongs to the label week, never to the history. Both functions
turn calendar dates into day indexes with hmrec.data.day_index (the only place
that knows the formula), filter, collect, and then CHECK the result: if any row
falls outside the window, they raise instead of returning leaked data.

`label_week_transactions` reads the answer of a fold. It may be called only by
hmrec.data.queries; tests/unit/test_asof.py scans the code base and fails if any
other file uses it.

Input tables come from data/processed/transactions.parquet, read with
hmrec.data.metadata.scan_processed_parquet.
"""

from collections.abc import Sequence
from datetime import date

import polars as pl

from hmrec.data.day_index import DAY_INDEX_DTYPE, date_to_day_index
from hmrec.data.folds import Fold

TRANSACTION_DAY = "t_dat_day"


def _check_day_column(transactions: pl.LazyFrame) -> None:
    """Fail unless the table has an int16 `t_dat_day` column."""
    schema = transactions.collect_schema()
    if TRANSACTION_DAY not in schema:
        raise ValueError(
            f"transactions has no {TRANSACTION_DAY!r} column; columns are {schema.names()}"
        )
    if schema[TRANSACTION_DAY] != DAY_INDEX_DTYPE:
        raise ValueError(
            f"transactions.{TRANSACTION_DAY} has dtype {schema[TRANSACTION_DAY]}, "
            f"expected {DAY_INDEX_DTYPE}"
        )


def _select_columns(columns: Sequence[str] | None) -> list[str] | None:
    """Return the columns to read, always including t_dat_day (needed for the check)."""
    if columns is None:
        return None
    selected = list(columns)
    if TRANSACTION_DAY not in selected:
        selected.append(TRANSACTION_DAY)
    return selected


def _filter_window(
    transactions: pl.LazyFrame,
    start_day: int,
    end_day_exclusive: int,
    columns: Sequence[str] | None,
) -> pl.DataFrame:
    """Keep rows with start_day <= t_dat_day < end_day_exclusive, then check the result.

    Raises:
        ValueError: If any returned row is outside the window. This should be
            impossible after the filter; the check guards against future edits
            of this function (for example a wrong comparison operator).
    """
    _check_day_column(transactions)
    selected = _select_columns(columns)
    if selected is not None:
        transactions = transactions.select(selected)
    day = pl.col(TRANSACTION_DAY)
    window = transactions.filter((day >= start_day) & (day < end_day_exclusive)).collect()
    _assert_inside_window(window, start_day, end_day_exclusive)
    return window


def _assert_inside_window(window: pl.DataFrame, start_day: int, end_day_exclusive: int) -> None:
    """Fail if any t_dat_day in `window` is outside [start_day, end_day_exclusive).

    Uses `raise`, not the `assert` statement, so the check also runs under
    `python -O` (which removes assert statements).
    """
    if window.height == 0:
        return
    min_day = window[TRANSACTION_DAY].min()
    max_day = window[TRANSACTION_DAY].max()
    if not isinstance(min_day, int) or not isinstance(max_day, int):
        raise ValueError(f"t_dat_day min/max are not integers: {min_day!r}, {max_day!r}")
    if min_day < start_day:
        raise ValueError(f"window contains t_dat_day={min_day}, which is before {start_day}")
    if max_day >= end_day_exclusive:
        raise ValueError(
            f"window contains t_dat_day={max_day}, which is not before {end_day_exclusive}"
        )


def history_window(
    transactions: pl.LazyFrame,
    anchor: date,
    day_zero: date,
    lookback_days: int,
    columns: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Return the purchase lines in the history window [anchor - lookback_days, anchor).

    This is the input side of a fold. The anchor day itself is NOT included,
    because it is the first day of the label week. Example: anchor day 42 and
    lookback_days 7 keep t_dat_day 35..41.

    Duplicate purchase lines are kept (they are repeat purchases).

    Args:
        transactions: One row per purchase line, with an int16 `t_dat_day`
            column (data/processed/transactions.parquet).
        anchor: First day of the label week.
        day_zero: Day 0 of t_dat_day (PreprocessConfig.day_zero).
        lookback_days: Window length in days; must be > 0.
        columns: Columns to return. `t_dat_day` is always added. None = all columns.

    Returns:
        One row per purchase line inside the window, in the input row order.

    Raises:
        ValueError: If lookback_days <= 0, `t_dat_day` is missing or not int16,
            or a returned row is outside the window.
    """
    if lookback_days <= 0:
        raise ValueError(f"lookback_days must be > 0, got {lookback_days}")
    anchor_day = date_to_day_index(anchor, day_zero)
    start_day = anchor_day - lookback_days
    return _filter_window(transactions, start_day, anchor_day, columns)


def label_week_transactions(
    transactions: pl.LazyFrame,
    fold: Fold,
    day_zero: date,
    columns: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Return the purchase lines of a fold's label week [anchor, label_end].

    ONLY hmrec.data.queries may call this function (it reads the answers).
    tests/unit/test_asof.py enforces this.

    In day indexes the window is [anchor_day, label_end_day + 1), which is
    the same as [anchor, anchor + 7d) for a 7-day fold.

    Args:
        transactions: One row per purchase line, with an int16 `t_dat_day` column.
        fold: The fold; uses fold.anchor and fold.label_end (inclusive).
        day_zero: Day 0 of t_dat_day.
        columns: Columns to return. `t_dat_day` is always added. None = all columns.

    Returns:
        One row per purchase line inside the label week (duplicates kept).

    Raises:
        ValueError: If `t_dat_day` is missing or not int16, or a returned row is
            outside the label week.
    """
    anchor_day = date_to_day_index(fold.anchor, day_zero)
    end_day_exclusive = date_to_day_index(fold.label_end_exclusive, day_zero)
    return _filter_window(transactions, anchor_day, end_day_exclusive, columns)
