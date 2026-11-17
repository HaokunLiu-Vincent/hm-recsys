"""Tests for features.user (group D) on the fold of feature_example.py.

Hand-computed customer table (anchor day 400):
    customer 0: lines T1 (395, 0.50, store), T2 (399, 0.25, online), T6 (390, 0.30, store)
        4w 3, 52w 3; last purchase 400 - 399 = 1; price (0.50 + 0.25 + 0.30) / 3 = 0.35;
        online 1/3; distinct articles {0, 1} -> 2; purchase days {390, 395, 399} -> 3;
        age 30; codes: ACTIVE 1, NONE 2; fn 1, active 1
    customer 1: lines T3 (380, 0.10, online), T4 (399, 0.50, online), T5 (day 30: outside 52w)
        4w 2, 52w 2; last purchase 1; price 0.30; online 2/2 = 1; distinct articles {1, 0} -> 2;
        purchase days {380, 399} -> 2 (day 30 is outside 52w);
        age unknown -> NaN; club missing -> 0; Regularly 3; fn 0, active 0
    customer 2: no purchase: counts 0; last purchase, price, online share NaN;
        age 50; PRE-CREATE 2; Monthly 1
"""

import dataclasses
import math

import polars as pl
import pytest
from feature_example import CONFIG, LINES, POISON, make_fold, random_lines, row_of, shuffled

from hmrec.features.categories import build_category_codes
from hmrec.features.user import USER_FEATURES, build_user_table


def _table(lines: list = LINES):
    fold = make_fold(lines)
    return build_user_table(fold, CONFIG, build_category_codes(fold.articles, fold.customers))


def test_user_table_hand_computed() -> None:
    table = _table()
    assert table.columns == ["customer_idx", *USER_FEATURES]
    assert table["customer_idx"].to_list() == [0, 1, 2]  # one row per query customer
    first = row_of(table, "customer_idx", 0)
    assert (first["age"], first["club_member_status"], first["fashion_news_frequency"]) == (
        30,
        1,
        2,
    )
    assert (first["fn"], first["active"]) == (1.0, 1.0)
    assert (first["user_cnt_4w"], first["user_cnt_52w"], first["user_uniq_items_52w"]) == (3, 3, 2)
    assert first["user_days_since_last_purchase"] == 1
    assert first["user_purchase_days_52w"] == 3
    assert first["user_price_mean_52w"] == pytest.approx(0.35, rel=1e-6)
    assert first["user_online_share_52w"] == pytest.approx(1 / 3, rel=1e-6)
    second = row_of(table, "customer_idx", 1)
    assert math.isnan(second["age"])
    assert (second["club_member_status"], second["fashion_news_frequency"]) == (0, 3)
    # T5 (day 30) is before the 52w window: 2 lines, not 3.
    assert (second["user_cnt_4w"], second["user_cnt_52w"], second["user_uniq_items_52w"]) == (
        2,
        2,
        2,
    )
    assert second["user_price_mean_52w"] == pytest.approx(0.30, rel=1e-6)
    assert second["user_purchase_days_52w"] == 2
    assert second["user_online_share_52w"] == 1.0


def test_customer_without_purchases_has_zero_counts_and_nan_elsewhere() -> None:
    row = row_of(_table(), "customer_idx", 2)
    assert (row["user_cnt_4w"], row["user_cnt_52w"], row["user_uniq_items_52w"]) == (0, 0, 0)
    assert row["user_purchase_days_52w"] == 0
    for name in ("user_days_since_last_purchase", "user_price_mean_52w", "user_online_share_52w"):
        assert math.isnan(row[name]), name
    assert (row["age"], row["club_member_status"], row["fashion_news_frequency"]) == (50, 2, 1)


def test_dtypes_and_no_nulls() -> None:
    table = _table()
    for name in USER_FEATURES:
        is_category = name in ("club_member_status", "fashion_news_frequency")
        assert str(table.schema[name]) == ("Int16" if is_category else "Float32"), name
    assert all(table[name].null_count() == 0 for name in table.columns)


def test_poison_on_and_after_the_anchor_changes_nothing() -> None:
    # The poison gives customer 2 purchases on day 400 and 401: they must stay "never bought".
    assert _table(LINES + POISON).equals(_table())


def test_a_query_customer_missing_from_the_customers_table_is_an_error() -> None:
    fold = make_fold()
    codes = build_category_codes(fold.articles, fold.customers)
    without_two = dataclasses.replace(
        fold, customers=fold.customers.filter(pl.col("customer_idx") != 2)
    )
    with pytest.raises(ValueError, match="1 query customers are not in the customers table"):
        build_user_table(without_two, CONFIG, codes)


def test_two_lines_on_one_day_are_one_purchase_day() -> None:
    # Customer 0 buys two more lines on day 399 (already a purchase day) and one on day 100.
    table = _table([*LINES, (0, 2, 399, 0.1, 1), (0, 3, 399, 0.1, 1), (0, 3, 100, 0.1, 1)])
    row = row_of(table, "customer_idx", 0)
    assert (row["user_cnt_52w"], row["user_purchase_days_52w"]) == (6, 4)  # days 100, 390, 395, 399


def test_price_mean_does_not_depend_on_the_row_order() -> None:
    lines = random_lines(seed=2)
    reference = _table(lines)["user_price_mean_52w"].to_list()
    for seed in range(5):
        assert _table(shuffled(lines, seed))["user_price_mean_52w"].to_list() == reference, seed
