"""Shared fixtures for the asof / catalog / queries tests.

See tests/fixtures/asof/README.md for the rows and the hand-computed expectations.
"""

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from hmrec.data.folds import Fold, FoldManifest

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
