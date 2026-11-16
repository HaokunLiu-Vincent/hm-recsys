"""The ONE place that converts calendar dates to integer day indexes and back.

Processed transactions store the purchase date as `t_dat_day` (int16):

    t_dat_day = (t_dat - day_zero).days

with day_zero = 2018-09-20 from configs/preprocess.yaml, so the first day of
the data is 0 and 2020-09-22 is 733.

Why int16: 31.8M rows x 2 bytes instead of 4 (Date) or 8 (Datetime), and the
largest value (733) is far below the int16 limit of 32,767.

The formula lives only in `date_to_day_index_expr` and `day_index_to_date_expr`.
The scalar helpers evaluate those same expressions on a single value, so there is
no second copy of the formula anywhere in the project. Every other module must call
these functions instead of writing the subtraction itself.
"""

from datetime import date

import polars as pl

# int16 is the storage dtype of t_dat_day (see module docstring).
DAY_INDEX_DTYPE = pl.Int16


def date_to_day_index_expr(dates: pl.Expr, day_zero: date) -> pl.Expr:
    """Turn a Date expression into the int16 day index.

    Args:
        dates: Expression of dtype Date.
        day_zero: The date that becomes day 0.

    Returns:
        Int16 expression: (dates - day_zero).days. Dates before day_zero give
        negative values. The cast is strict, so a value outside the int16 range
        raises an error instead of silently wrapping around.
    """
    days_since_zero = (dates - pl.lit(day_zero)).dt.total_days()
    return days_since_zero.cast(DAY_INDEX_DTYPE, strict=True)


def day_index_to_date_expr(day_indexes: pl.Expr, day_zero: date) -> pl.Expr:
    """Turn an integer day index expression back into a Date expression.

    Args:
        day_indexes: Integer expression, e.g. pl.col("t_dat_day").
        day_zero: The date that is day 0.

    Returns:
        Date expression: day_zero + day_indexes days.
    """
    return (pl.lit(day_zero) + pl.duration(days=day_indexes)).cast(pl.Date)


def date_to_day_index(day: date, day_zero: date) -> int:
    """Scalar version of date_to_day_index_expr, e.g. to turn an anchor date into a day index.

    Args:
        day: Any calendar date.
        day_zero: The date that becomes day 0.

    Returns:
        (day - day_zero).days as a Python int.
    """
    result = pl.select(date_to_day_index_expr(pl.lit(day, dtype=pl.Date), day_zero)).item()
    return int(result)


def day_index_to_date(day_index: int, day_zero: date) -> date:
    """Scalar version of day_index_to_date_expr.

    Args:
        day_index: Integer day index.
        day_zero: The date that is day 0.

    Returns:
        The calendar date day_zero + day_index days.
    """
    result = pl.select(day_index_to_date_expr(pl.lit(day_index), day_zero)).item()
    if not isinstance(result, date):
        raise TypeError(f"expected a date, got {result!r}")
    return result
