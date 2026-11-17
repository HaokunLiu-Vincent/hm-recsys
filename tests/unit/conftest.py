"""Shared fixtures for the asof / catalog / queries tests.

See tests/fixtures/asof/README.md for the rows and the hand-computed expectations.
"""

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from hmrec.data.day_index import day_index_to_date
from hmrec.data.folds import Fold, FoldManifest
from hmrec.recall.base import RecallInputs

ASOF_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "asof"

# Day 0 of the fixture (NOT the real 2018-09-20); keeps the day numbers small.
FIXTURE_DAY_ZERO = date(2020, 1, 1)
D1_ANCHOR = date(2020, 2, 12)  # day 42
D1 = Fold(name="D1", phase="D", anchor=D1_ANCHOR, label_end=date(2020, 2, 18))
D2 = Fold(name="D2", phase="D", anchor=date(2020, 2, 19), label_end=date(2020, 2, 25))

PROCESSED_TRANSACTIONS_SCHEMA = pl.Schema(
    {
        "customer_idx": pl.Int32,
        "article_idx": pl.Int32,
        "t_dat_day": pl.Int16,
        "price": pl.Float32,
        "sales_channel_id": pl.Int8,
    }
)


def make_transactions(rows: list[tuple[int, int, int]]) -> pl.DataFrame:
    """Build processed-format transactions from (customer_idx, article_idx, t_dat_day) rows."""
    return pl.DataFrame(
        {
            "customer_idx": [row[0] for row in rows],
            "article_idx": [row[1] for row in rows],
            "t_dat_day": [row[2] for row in rows],
            "price": [0.5] * len(rows),
            "sales_channel_id": [1] * len(rows),
        },
        schema=PROCESSED_TRANSACTIONS_SCHEMA,
    )


@pytest.fixture
def fixture_transactions() -> pl.DataFrame:
    """The 17 rows of tests/fixtures/asof/transactions.csv with processed dtypes."""
    return pl.read_csv(ASOF_FIXTURES / "transactions.csv", schema=PROCESSED_TRANSACTIONS_SCHEMA)


@pytest.fixture
def fixture_manifest() -> FoldManifest:
    """Two adjacent test folds: D1 (ordinal 0) and D2 (ordinal 1)."""
    return FoldManifest(folds=(D1, D2))


# ---------------------------------------------------------------- recall helpers


def make_queries(pairs: list[tuple[int, int]], fold: str = "D1") -> pl.DataFrame:
    """Build a query list from (query_id, customer_idx) pairs."""
    return pl.DataFrame(
        {
            "query_id": [pair[0] for pair in pairs],
            "customer_idx": [pair[1] for pair in pairs],
            "fold": [fold] * len(pairs),
        },
        schema={"query_id": pl.Int32, "customer_idx": pl.Int32, "fold": pl.String},
    )


def make_recall_inputs(
    rows: list[tuple[int, int, int]],
    anchor_day: int,
    customers: dict[int, str] | None = None,
    product_codes: dict[int, int] | None = None,
    active_pool: list[int] | None = None,
) -> RecallInputs:
    """Build RecallInputs on FIXTURE_DAY_ZERO with the anchor at `anchor_day`.

    Args:
        rows: (customer_idx, article_idx, t_dat_day) purchase lines.
        anchor_day: Anchor as a day index (so age_days = anchor_day - t_dat_day).
        customers: customer_idx -> age_bucket.
        product_codes: article_idx -> product_code (the catalog).
        active_pool: article_idx list of A(a).
    """
    customers = customers or {}
    product_codes = product_codes or {}
    return RecallInputs(
        anchor=day_index_to_date(anchor_day, FIXTURE_DAY_ZERO),
        day_zero=FIXTURE_DAY_ZERO,
        transactions=make_transactions(rows).lazy(),
        customers=pl.DataFrame(
            {"customer_idx": list(customers), "age_bucket": list(customers.values())},
            schema={"customer_idx": pl.Int32, "age_bucket": pl.String},
        ),
        articles=pl.DataFrame(
            {"article_idx": list(product_codes), "product_code": list(product_codes.values())},
            schema={"article_idx": pl.Int32, "product_code": pl.Int32},
        ),
        active_pool=pl.DataFrame(
            {"article_idx": active_pool or []}, schema={"article_idx": pl.Int32}
        ),
    )


# ---------------------------------------------------------------- two-tower helpers

TINY_N_ARTICLES = 6


def make_tiny_articles() -> pl.DataFrame:
    """Six articles for the two-tower tests.

    With one_hot_max_cardinality = 3 (TINY_MODEL), the category codes are:
    product_type_no 7 codes -> embedding, product_group_name 3 -> one-hot,
    garment_group_no 2 -> one-hot, index_group_no 3 -> one-hot,
    section_no 7 -> embedding, colour_group_code 7 -> embedding (codes include 0).
    """
    return pl.DataFrame(
        {
            "article_idx": list(range(TINY_N_ARTICLES)),
            "product_code": [1, 1, 2, 2, 3, 3],
            "product_type_no": [1, 2, 3, 4, 5, 6],
            "product_group_name": ["a", "a", "b", "b", "a", "b"],
            "garment_group_no": [1] * TINY_N_ARTICLES,
            "index_group_no": [1, 2, 1, 2, 1, 2],
            "section_no": [1, 2, 3, 4, 5, 6],
            "colour_group_code": [1, 2, 3, 4, 5, 6],
        },
        schema_overrides={"article_idx": pl.Int32},
    )


def make_tiny_customers(n_customers: int = 4) -> pl.DataFrame:
    """Customers 0..n-1 with small profile vocabularies (one null each)."""
    statuses = ["ACTIVE", None, "ACTIVE", "PRE-CREATE"]
    news = ["NONE", "Regularly", None, "NONE"]
    return pl.DataFrame(
        {
            "customer_idx": list(range(n_customers)),
            "age_bucket": ["AGE_18_24", "AGE_25_34", "AGE_UNKNOWN", "AGE_18_24"][:n_customers],
            "club_member_status": statuses[:n_customers],
            "fashion_news_frequency": news[:n_customers],
        },
        schema_overrides={"customer_idx": pl.Int32},
    )


def make_tiny_embeddings(seed: int = 0) -> dict:
    """Image (width 4) and text (width 3) vectors; article 2 has no image."""
    rng = np.random.default_rng(seed)
    image = rng.standard_normal((TINY_N_ARTICLES, 4)).astype(np.float32)
    image[2] = 0.0
    return {
        "image": image,
        "image_mask": np.array([True, True, False, True, True, True]),
        "text": rng.standard_normal((TINY_N_ARTICLES, 3)).astype(np.float32),
        "text_mask": np.ones(TINY_N_ARTICLES, dtype=np.bool_),
    }
