"""Tests for features.item (group C) on the fold of feature_example.py.

Hand-computed article table (anchor day 400; see feature_example for the lines):
    article 0: lines T1 (day 395), T2 (399), T4 (399)
        1w 3, prev_1w 0, 4w 3; buyers {0, 1} -> 2; first sale 400 - 395 = 5, last 400 - 399 = 1
        price (0.50 + 0.25 + 0.50) / 3 = 0.416667; online T2, T4 of 3 -> 2/3
        buyer ages: customer 0 is 30, customer 1 unknown -> mean over known = 30
        52w: customer 0 bought on days 395 and 399 (2 days), customer 1 on 1 day -> 1/2
        discount ratio: week mean 0.416667 / year mean 0.416667 (the same three lines) = 1
    article 1: lines T3 (380), T6 (390)
        1w 0, prev_1w 1 (T6), 4w 2; buyers 2; first sale 20, last 10
        price (0.10 + 0.30) / 2 = 0.20; online T3 of 2 -> 1/2; buyer age 30; repurchase 0/2 = 0
        no sale in the last week -> discount ratio NaN
    article 2: line T5 (day 30) only: all counts 0; first = last = 370 days;
        nothing in 4w or 52w -> price, online share, buyer age, repurchase ratio NaN
    article 3: never sold: counts 0, everything else NaN
"""

import math

import pytest
from feature_example import CONFIG, LINES, POISON, make_fold, random_lines, row_of, shuffled

from hmrec.features.categories import build_category_codes
from hmrec.features.item import build_item_table

NAN = float("nan")


def _table(lines: list = LINES):
    fold = make_fold(lines)
    return build_item_table(fold, CONFIG, build_category_codes(fold.articles, fold.customers))


def _check(row: dict, expected: dict) -> None:
    for name, value in expected.items():
        if isinstance(value, float) and math.isnan(value):
            assert math.isnan(row[name]), name
        else:
            assert row[name] == pytest.approx(value, rel=1e-6), name


def test_item_table_hand_computed() -> None:
    table = _table()
    assert table["article_idx"].to_list() == [0, 1, 2, 3]  # every catalog article, sorted
    _check(
        row_of(table, "article_idx", 0),
        {
            "item_cnt_1w": 3,
            "item_cnt_prev_1w": 0,
            "item_cnt_4w": 3,
            "item_uniq_buyers_4w": 2,
            "item_days_since_first_sale": 5,
            "item_days_since_last_sale": 1,
            "item_price_mean_4w": 1.25 / 3,
            "item_online_share_4w": 2 / 3,
            "item_buyer_age_mean_4w": 30,
            "item_repurchase_ratio_52w": 0.5,
            "item_discount_ratio": 1.0,
        },
    )
    _check(
        row_of(table, "article_idx", 1),
        {
            "item_cnt_1w": 0,
            "item_cnt_prev_1w": 1,
            "item_cnt_4w": 2,
            "item_uniq_buyers_4w": 2,
            "item_days_since_first_sale": 20,
            "item_days_since_last_sale": 10,
            "item_price_mean_4w": 0.2,
            "item_online_share_4w": 0.5,
            "item_buyer_age_mean_4w": 30,
            "item_repurchase_ratio_52w": 0.0,
            "item_discount_ratio": NAN,
        },
    )


def test_counts_are_zero_and_other_features_nan_without_data() -> None:
    table = _table()
    counts = {"item_cnt_1w": 0, "item_cnt_prev_1w": 0, "item_cnt_4w": 0, "item_uniq_buyers_4w": 0}
    undefined = {
        "item_price_mean_4w": NAN,
        "item_online_share_4w": NAN,
        "item_buyer_age_mean_4w": NAN,
        "item_repurchase_ratio_52w": NAN,
        "item_discount_ratio": NAN,
    }
    # Article 2 was sold once, 370 days ago: outside every window, inside "all history".
    days = {"item_days_since_first_sale": 370, "item_days_since_last_sale": 370}
    _check(row_of(table, "article_idx", 2), {**counts, **undefined, **days})
    # Article 3 was never sold.
    never = {"item_days_since_first_sale": NAN, "item_days_since_last_sale": NAN}
    _check(row_of(table, "article_idx", 3), {**counts, **undefined, **never})


def test_dtypes_and_category_codes() -> None:
    table = _table()
    categories = [
        "product_type_no",
        "product_group_name",
        "garment_group_no",
        "index_group_no",
        "section_no",
        "colour_group_code",
    ]
    for name in table.columns:
        expected = (
            "Int32" if name == "article_idx" else ("Int16" if name in categories else "Float32")
        )
        assert str(table.schema[name]) == expected, name
    # Codes: sorted distinct values from 1 (see feature_example).
    assert table.select(categories).rows() == [
        (1, 1, 1, 1, 1, 2),
        (1, 1, 2, 1, 1, 1),
        (2, 2, 2, 2, 2, 2),
        (2, 2, 2, 2, 2, 2),
    ]
    assert all(table[name].null_count() == 0 for name in table.columns)


def test_poison_on_and_after_the_anchor_changes_nothing() -> None:
    # The poison adds lines of every article on the anchor day and later, also of
    # article 3, which must stay "never sold".
    assert _table(LINES + POISON).equals(_table())


def test_discount_ratio_is_last_weeks_price_over_the_years_price() -> None:
    # One more line of article 0, 200 days before the anchor at price 1.00:
    #   week mean (0.50 + 0.25 + 0.50) / 3 = 0.416667
    #   year mean (0.50 + 0.25 + 0.50 + 1.00) / 4 = 0.5625  ->  ratio 0.740741
    table = _table([*LINES, (1, 0, 200, 1.0, 1)])
    assert row_of(table, "article_idx", 0)["item_discount_ratio"] == pytest.approx(
        (1.25 / 3) / 0.5625, rel=1e-6
    )
    # Article 1 was sold in the year but not in the last week: NaN, not 0.
    assert math.isnan(row_of(table, "article_idx", 1)["item_discount_ratio"])


def test_price_means_do_not_depend_on_the_row_order() -> None:
    # 4,000 lines with random float32 prices: each mean adds up hundreds of values, and
    # in another row order the float64 sum differs in its last bits. The stored
    # float32 features must still be IDENTICAL (prices are cast to float64 first).
    lines = random_lines(seed=1)
    reference = _table(lines)
    assert reference["item_price_mean_4w"].is_nan().sum() == 0
    for seed in range(5):
        other = _table(shuffled(lines, seed))
        for name in ("item_price_mean_4w", "item_discount_ratio"):
            assert other[name].to_list() == reference[name].to_list(), (name, seed)
