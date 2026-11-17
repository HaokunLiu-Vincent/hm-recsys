"""One tiny fold shared by the feature tests (not a test module itself).

Day zero is FIXTURE_DAY_ZERO; the anchor is day 400. Windows:
    1w = [393, 400)   prev_1w = [386, 393)   4w = [372, 400)   52w = [36, 400)

Purchase lines (customer, article, day, price, sales channel; channel 2 = online):
    T1  0  0  395  0.50  1
    T2  0  0  399  0.25  2
    T3  1  1  380  0.10  2
    T4  1  0  399  0.50  2
    T5  1  2   30  0.20  1     370 days before the anchor: outside 52w, inside "all history"
    T6  0  1  390  0.30  1     in prev_1w
POISON: lines on the anchor day (400) and later. No feature may change because of them.

Articles (product_code, product_type_no, product_group_name, garment_group_no,
index_group_no, section_no, colour_group_code):
    0: 10, 5, "A", 1, 1, 3, 9        1: 10, 5, "A", 2, 1, 3, 8
    2: 20, 7, "B", 2, 2, 4, 9        3: 30, 7, "B", 2, 2, 4, 9   (never sold)
Customers (age, age_bucket, fn, active, club_member_status, fashion_news_frequency):
    0: 30, AGE_25_34, 1, 1, "ACTIVE", "NONE"
    1: unknown, AGE_UNKNOWN, 0, 0, missing, "Regularly"
    2: 50, AGE_45_54, 0, 0, "PRE-CREATE", "Monthly"            (never bought anything)
Queries: q1 -> customer 0, q2 -> customer 1, q3 -> customer 2.

Category codes (0 = missing, then the sorted distinct values):
    product_type_no 5 -> 1, 7 -> 2      product_group_name A -> 1, B -> 2
    garment_group_no 1 -> 1, 2 -> 2     index_group_no 1 -> 1, 2 -> 2
    section_no 3 -> 1, 4 -> 2           colour_group_code 8 -> 1, 9 -> 2
    club_member_status ACTIVE -> 1, PRE-CREATE -> 2
    fashion_news_frequency Monthly -> 1, NONE -> 2, Regularly -> 3
"""

from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
from conftest import FIXTURE_DAY_ZERO

from hmrec.config import load_features_config
from hmrec.data.histories import recent_purchase_histories
from hmrec.features.base import FoldData, HistoryArrays

REPO = Path(__file__).resolve().parents[2]
CONFIG = load_features_config(REPO / "configs" / "features.yaml")
ANCHOR_DAY = 400
ANCHOR = FIXTURE_DAY_ZERO + timedelta(days=ANCHOR_DAY)
Line = tuple[int, int, int, float, int]
LINES: list[Line] = [
    (0, 0, 395, 0.50, 1),
    (0, 0, 399, 0.25, 2),
    (1, 1, 380, 0.10, 2),
    (1, 0, 399, 0.50, 2),
    (1, 2, 30, 0.20, 1),
    (0, 1, 390, 0.30, 1),
]
# On the anchor day and after it: more lines of every article and customer,
# including the customer and the article that have no history at all.
POISON: list[Line] = [
    (0, 1, 400, 9.0, 1),
    (1, 2, 400, 9.0, 2),
    (2, 0, 400, 9.0, 2),
    (2, 3, 401, 9.0, 1),
    (0, 3, 500, 9.0, 2),
    (1, 0, 733, 9.0, 1),
]


def make_transactions(lines: list[Line]) -> pl.DataFrame:
    """Processed-format transactions from the example's lines."""
    return pl.DataFrame(
        {
            "customer_idx": [line[0] for line in lines],
            "article_idx": [line[1] for line in lines],
            "t_dat_day": [line[2] for line in lines],
            "price": [line[3] for line in lines],
            "sales_channel_id": [line[4] for line in lines],
        },
        schema={
            "customer_idx": pl.Int32,
            "article_idx": pl.Int32,
            "t_dat_day": pl.Int16,
            "price": pl.Float32,
            "sales_channel_id": pl.Int8,
        },
    )


CUSTOMERS = pl.DataFrame(
    {
        "customer_idx": [0, 1, 2],
        "age": [30, None, 50],
        "age_bucket": ["AGE_25_34", "AGE_UNKNOWN", "AGE_45_54"],
        "fn": [1, 0, 0],
        "active": [1, 0, 0],
        "club_member_status": ["ACTIVE", None, "PRE-CREATE"],
        "fashion_news_frequency": ["NONE", "Regularly", "Monthly"],
    },
    schema={
        "customer_idx": pl.Int32,
        "age": pl.Int8,
        "age_bucket": pl.String,
        "fn": pl.Int8,
        "active": pl.Int8,
        "club_member_status": pl.String,
        "fashion_news_frequency": pl.String,
    },
)
ARTICLES = pl.DataFrame(
    {
        "article_idx": [0, 1, 2, 3],
        "product_code": [10, 10, 20, 30],
        "product_type_no": [5, 5, 7, 7],
        "product_group_name": ["A", "A", "B", "B"],
        "garment_group_no": [1, 2, 2, 2],
        "index_group_no": [1, 1, 2, 2],
        "section_no": [3, 3, 4, 4],
        "colour_group_code": [9, 8, 9, 9],
    },
    schema={
        "article_idx": pl.Int32,
        "product_code": pl.Int32,
        "product_type_no": pl.Int16,
        "product_group_name": pl.String,
        "garment_group_no": pl.Int16,
        "index_group_no": pl.Int8,
        "section_no": pl.Int8,
        "colour_group_code": pl.Int8,
    },
)
QUERIES = pl.DataFrame(
    {"query_id": [1, 2, 3], "customer_idx": [0, 1, 2]},
    schema={"query_id": pl.Int32, "customer_idx": pl.Int32},
)


def make_fold(lines: list[Line] = LINES) -> FoldData:
    """The example fold; pass LINES + POISON for the time cut-off tests."""
    return FoldData(
        fold="F1",
        anchor=ANCHOR,
        day_zero=FIXTURE_DAY_ZERO,
        transactions=make_transactions(lines).lazy(),
        customers=CUSTOMERS,
        articles=ARTICLES,
        queries=QUERIES,
    )


def row_of(table: pl.DataFrame, key: str, value: int) -> dict:
    """The single row of `table` whose `key` column equals `value`, as a dict."""
    rows = table.filter(pl.col(key) == value).to_dicts()
    assert len(rows) == 1
    return rows[0]


def make_history(lines: list[Line] = LINES) -> HistoryArrays:
    """The three customers' last purchases before the anchor (the function R6 uses).

    With LINES: customer 0 -> articles [0, 0, 1] (days 399, 395, 390),
    customer 1 -> [0, 1, 2] (days 399, 380, 30), customer 2 -> nothing.
    """
    customers = np.array([0, 1, 2], dtype=np.int32)
    history, mask = recent_purchase_histories(
        make_transactions(lines).lazy(),
        customers,
        ANCHOR,
        FIXTURE_DAY_ZERO,
        CONFIG.history_length,
        padding_idx=ARTICLES.height,
    )
    return HistoryArrays(customers, history, mask)


def random_lines(seed: int, n_lines: int = 4000) -> list[Line]:
    """Many random purchase lines inside the 52w window, with random float32 prices.

    For the row-order tests: thousands of prices per mean, so a sum in another
    order really does differ in its last float64 bits.
    """
    rng = np.random.default_rng(seed)
    prices = rng.uniform(0.005, 0.6, size=n_lines).astype(np.float32)
    return [
        (
            int(rng.integers(0, 3)),
            int(rng.integers(0, 4)),
            int(rng.integers(36, ANCHOR_DAY)),
            float(prices[i]),
            int(rng.integers(1, 3)),
        )
        for i in range(n_lines)
    ]


def shuffled(lines: list[Line], seed: int) -> list[Line]:
    """The same lines in another row order."""
    order = np.random.default_rng(seed).permutation(len(lines))
    return [lines[int(position)] for position in order]
