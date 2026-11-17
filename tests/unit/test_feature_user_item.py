"""Tests for features.user_item (group E) on the fold of feature_example.py.

w(a) = exp(-a / 30) is the decay weight of a line that is a days old. Lines by customer:
    customer 0: T1 art 0 (age 5), T2 art 0 (age 1), T6 art 1 (age 10)        all inside 52w
    customer 1: T3 art 1 (age 20), T4 art 0 (age 1), T5 art 2 (age 370, outside 52w)
    customer 2: nothing
Article fields (type, group, garment, index, section, colour):
    art 0: 5, A, 1, 1, 3, 9      art 1: 5, A, 2, 1, 3, 8      art 2: 7, B, 2, 2, 4, 9
Last purchases (hmrec.data.histories): customer 0 [0, 0, 1], customer 1 [0, 1, 2].
All customers' 52w lines: T1, T2, T3, T4, T6 = 5 lines, all of product type 5
    -> site-wide share of type 5 is 1, of type 7 is 0.

(q1 = customer 0, article 0)
    cnt_item 2, days 1; product_code 10 (arts 0, 1): 3 lines, days 1
    52w shares (3 lines): type 5 3/3; garment 1 2/3; index 1 3/3; colour 9 2/3;
        section 3 3/3; group A 3/3
    price ratio 0.416667 / 0.35 = 1.190476; age gap |30 - 30| = 0
    decay: type 5 w5 + w1 + w10; garment 1 w5 + w1; section 3 w5 + w1 + w10; colour 9 w5 + w1
    days since type 5: 1; since section 3: 1
    lift 1 / 1 = 1; channel gap |1/3 - 2/3| = 1/3
    attr match: history values type {5}, group {A}, garment {1, 2}, index {1}, section {3},
        colour {9, 8} -> article 0 matches all 6
(q1 = customer 0, article 2)
    cnt 0, days NaN; product_code 20: 0, NaN
    shares: type 7 0; garment 2 1/3 (T6); index 2 0; colour 9 2/3; section 4 0; group B 0
    price ratio NaN, age gap NaN (article 2 has no 4w lines)
    decay: type 7 0; garment 2 w10; section 4 0; colour 9 w5 + w1; days since type / section NaN
    lift NaN (site-wide share of type 7 is 0); channel gap NaN (no 4w online share)
    attr match: garment 2 and colour 9 -> 2
(q2 = customer 1, article 2)
    cnt 1, days 370 (T5 counts: all history); product_code 20: 1, 370
    52w shares (2 lines: T3, T4): type 7 0; garment 2 1/2; index 2 0; colour 9 1/2;
        section 4 0; group B 0
    price ratio NaN, age gap NaN
    decay: type 7 w370; garment 2 w20 + w370; section 4 w370; colour 9 w1 + w370
    days since type 7: 370; since section 4: 370
    lift NaN; channel gap NaN; attr match: history [0, 1, 2] holds every value of art 2 -> 6
(q3 = customer 2, article 0)
    cnt 0, NaN, 0, NaN; all six shares NaN (no 52w line); price ratio NaN; age gap |50 - 30| = 20
    decays 0; days since NaN; lift NaN; channel gap NaN (no user online share);
    attr match NaN (no purchase at all)
"""

import dataclasses
import math

import polars as pl
import pytest
from feature_example import (
    CONFIG,
    LINES,
    POISON,
    QUERIES,
    make_fold,
    make_history,
    random_lines,
    shuffled,
)

from hmrec.features.base import history_rows
from hmrec.features.categories import build_category_codes
from hmrec.features.item import build_item_table
from hmrec.features.user import build_user_table
from hmrec.features.user_item import (
    USER_ITEM_FEATURES,
    add_user_item_features,
    build_user_item_tables,
)

PAIRS = [(1, 0), (1, 2), (2, 2), (3, 0)]  # (query_id, article_idx)
NAN = float("nan")


def w(age_days: int) -> float:
    return math.exp(-age_days / 30)


def _features(lines: list = LINES, pairs: list = PAIRS) -> pl.DataFrame:
    fold = make_fold(lines)
    codes = build_category_codes(fold.articles, fold.customers)
    chunk = pl.DataFrame(
        {"query_id": [q for q, _ in pairs], "article_idx": [a for _, a in pairs]},
        schema={"query_id": pl.Int32, "article_idx": pl.Int32},
    ).join(QUERIES, on="query_id", maintain_order="left")
    chunk = chunk.join(
        build_item_table(fold, CONFIG, codes), on="article_idx", maintain_order="left"
    )
    chunk = chunk.join(
        build_user_table(fold, CONFIG, codes), on="customer_idx", maintain_order="left"
    )
    tables = build_user_item_tables(fold, CONFIG, history_rows(make_history(lines)))
    return add_user_item_features(chunk, tables)


def _check(table: pl.DataFrame, position: int, expected: dict[str, float]) -> None:
    assert set(expected) == set(USER_ITEM_FEATURES)
    row = table.row(position, named=True)
    for name, want in expected.items():
        if math.isnan(want):
            assert math.isnan(row[name]), name
        else:
            assert row[name] == pytest.approx(want, rel=1e-6, abs=1e-12), name


def test_customer_who_bought_the_article_hand_computed() -> None:
    table = _features()
    assert table.select("query_id", "article_idx").rows() == PAIRS  # chunk order kept
    assert len(USER_ITEM_FEATURES) == 21
    _check(
        table,
        0,
        {
            "ui_cnt_item": 2,
            "ui_days_since_item": 1,
            "ui_cnt_pcode": 3,
            "ui_days_since_pcode": 1,
            "ui_share_ptype_52w": 1.0,
            "ui_share_garment_52w": 2 / 3,
            "ui_share_index_52w": 1.0,
            "ui_share_colour_52w": 2 / 3,
            "ui_price_ratio": (1.25 / 3) / 0.35,
            "ui_age_gap": 0.0,
            "ui_decay_ptype": w(5) + w(1) + w(10),
            "ui_decay_garment": w(5) + w(1),
            "ui_decay_section": w(5) + w(1) + w(10),
            "ui_decay_colour": w(5) + w(1),
            "ui_days_since_ptype": 1,
            "ui_days_since_section": 1,
            "ui_share_section_52w": 1.0,
            "ui_share_pgroup_52w": 1.0,
            "ui_lift_ptype_52w": 1.0,
            "ui_channel_gap": 1 / 3,
            "ui_attr_match_cnt": 6,
        },
    )


def test_customer_who_never_bought_the_article_hand_computed() -> None:
    _check(
        _features(),
        1,
        {
            "ui_cnt_item": 0,
            "ui_days_since_item": NAN,
            "ui_cnt_pcode": 0,
            "ui_days_since_pcode": NAN,
            "ui_share_ptype_52w": 0.0,
            "ui_share_garment_52w": 1 / 3,
            "ui_share_index_52w": 0.0,
            "ui_share_colour_52w": 2 / 3,
            "ui_price_ratio": NAN,
            "ui_age_gap": NAN,
            "ui_decay_ptype": 0.0,
            "ui_decay_garment": w(10),
            "ui_decay_section": 0.0,
            "ui_decay_colour": w(5) + w(1),
            "ui_days_since_ptype": NAN,
            "ui_days_since_section": NAN,
            "ui_share_section_52w": 0.0,
            "ui_share_pgroup_52w": 0.0,
            "ui_lift_ptype_52w": NAN,
            "ui_channel_gap": NAN,
            "ui_attr_match_cnt": 2,
        },
    )


def test_old_purchase_counts_for_history_features_but_not_for_52w() -> None:
    # Customer 1 bought article 2 once, 370 days ago: it is in "all history" (counts,
    # days since, decay) and in the last-20 history, but outside every 52w share.
    _check(
        _features(),
        2,
        {
            "ui_cnt_item": 1,
            "ui_days_since_item": 370,
            "ui_cnt_pcode": 1,
            "ui_days_since_pcode": 370,
            "ui_share_ptype_52w": 0.0,
            "ui_share_garment_52w": 0.5,
            "ui_share_index_52w": 0.0,
            "ui_share_colour_52w": 0.5,
            "ui_price_ratio": NAN,
            "ui_age_gap": NAN,
            "ui_decay_ptype": w(370),
            "ui_decay_garment": w(20) + w(370),
            "ui_decay_section": w(370),
            "ui_decay_colour": w(1) + w(370),
            "ui_days_since_ptype": 370,
            "ui_days_since_section": 370,
            "ui_share_section_52w": 0.0,
            "ui_share_pgroup_52w": 0.0,
            "ui_lift_ptype_52w": NAN,
            "ui_channel_gap": NAN,
            "ui_attr_match_cnt": 6,
        },
    )


def test_customer_without_purchases_has_zero_counts_and_nan_elsewhere() -> None:
    zero = ("ui_cnt_item", "ui_cnt_pcode", "ui_decay_ptype", "ui_decay_garment")
    zero += ("ui_decay_section", "ui_decay_colour")
    expected = {name: (0.0 if name in zero else NAN) for name in USER_ITEM_FEATURES}
    expected["ui_age_gap"] = 20.0  # |50 - 30|: defined without any purchase of the customer
    _check(_features(), 3, expected)


def test_lift_is_the_users_share_over_the_site_wide_share() -> None:
    # One more 52w line: customer 1 buys article 2 (product type 7) on day 390.
    #   site-wide 52w: 6 lines, type 5 on 5 of them (5/6), type 7 on 1 (1/6)
    #   customer 0 (3 lines, all type 5): article 0 -> 1 / (5/6) = 1.2; article 2 -> 0 / (1/6) = 0
    #   customer 1 (3 lines: T3, T4 type 5; the new one type 7): article 2 -> (1/3) / (1/6) = 2
    #   customer 2: no 52w line -> NaN
    table = _features([*LINES, (1, 2, 390, 0.2, 1)])
    lift = table["ui_lift_ptype_52w"].to_list()
    assert lift[0] == pytest.approx(1.2, rel=1e-6)
    assert lift[1] == 0.0
    assert lift[2] == pytest.approx(2.0, rel=1e-6)
    assert math.isnan(lift[3])


def test_dtypes_no_nulls_and_no_temporary_columns() -> None:
    table = _features()
    for name in USER_ITEM_FEATURES:
        assert str(table.schema[name]) == "Float32", name
        assert table[name].null_count() == 0, name
    assert not [name for name in table.columns if name.startswith("__")]


def test_poison_on_and_after_the_anchor_changes_nothing() -> None:
    # The poison adds lines of every customer and article on the anchor day and later,
    # also for customer 2, who must stay "without any purchase" in all 21 features.
    assert _features(LINES + POISON).equals(_features())


def test_price_ratio_does_not_depend_on_the_row_order() -> None:
    lines = random_lines(seed=3)
    pairs = [(query, article) for query in (1, 2, 3) for article in range(4)]
    reference = _features(lines, pairs)["ui_price_ratio"].to_list()
    assert not any(math.isnan(value) for value in reference)
    for seed in range(5):
        assert _features(shuffled(lines, seed), pairs)["ui_price_ratio"].to_list() == reference


def test_tables_hold_only_query_customers() -> None:
    fold = make_fold()
    only_q1 = pl.DataFrame(
        {"query_id": [1], "customer_idx": [0]},
        schema={"query_id": pl.Int32, "customer_idx": pl.Int32},
    )
    recent = history_rows(make_history()).filter(pl.col("customer_idx") == 0)
    tables = build_user_item_tables(dataclasses.replace(fold, queries=only_q1), CONFIG, recent)
    assert tables.by_article["customer_idx"].unique().to_list() == [0]
    assert tables.year_lines.rows() == [(0, 3)]
    assert tables.with_history["customer_idx"].to_list() == [0]
    # The site-wide share is over ALL customers' lines, also when only one is queried.
    assert tables.global_type_share.rows() == [(5, 1.0)]
