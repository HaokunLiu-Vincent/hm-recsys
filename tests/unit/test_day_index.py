from datetime import date

import polars as pl
import pytest

from hmrec.data.day_index import (
    date_to_day_index,
    date_to_day_index_expr,
    day_index_to_date,
    day_index_to_date_expr,
)

DAY_ZERO = date(2018, 9, 20)


def test_first_and_last_day_of_the_data() -> None:
    assert date_to_day_index(date(2018, 9, 20), DAY_ZERO) == 0
    # 2018-09-20 -> 2019-09-20 is 365 days, 2019-09-20 -> 2020-09-20 is 366 (leap year),
    # plus 2 more days to 2020-09-22: 365 + 366 + 2 = 733.
    assert date_to_day_index(date(2020, 9, 22), DAY_ZERO) == 733


def test_date_before_day_zero_is_negative() -> None:
    assert date_to_day_index(date(2018, 9, 19), DAY_ZERO) == -1


@pytest.mark.parametrize("day_index", [0, 1, 365, 733])
def test_round_trip(day_index: int) -> None:
    assert date_to_day_index(day_index_to_date(day_index, DAY_ZERO), DAY_ZERO) == day_index


def test_expressions_on_a_column() -> None:
    frame = pl.DataFrame({"d": [date(2018, 9, 20), date(2018, 9, 27)]})
    days = frame.select(date_to_day_index_expr(pl.col("d"), DAY_ZERO).alias("day"))
    assert days.schema["day"] == pl.Int16
    assert days["day"].to_list() == [0, 7]
    back = days.select(day_index_to_date_expr(pl.col("day"), DAY_ZERO).alias("d"))
    assert back["d"].to_list() == frame["d"].to_list()


def test_overflow_raises_instead_of_wrapping() -> None:
    # 2200-01-01 is 66,212 days after day zero, beyond the int16 limit of 32,767.
    with pytest.raises(pl.exceptions.InvalidOperationError):
        date_to_day_index(date(2200, 1, 1), DAY_ZERO)
