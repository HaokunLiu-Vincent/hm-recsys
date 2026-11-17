"""Tests for recall.inputs: the fold inputs shared by the D-phase trials and M6.

Active pool rule (configs/catalog.yaml): >= 3 purchase lines in [anchor - 28d, anchor).
Anchor day 100 -> window days 72..99. Lines (customer, article, day):
    article 1: days 99, 80, 72      -> 3 lines in the window  -> active
    article 2: days 99, 98          -> 2 lines                -> not active
    article 3: days 99, 98, 71      -> day 71 is before the window: 2 lines -> not active
    article 4: days 100, 100, 101   -> on / after the anchor: 0 lines -> not active (poison)
"""

from datetime import timedelta
from pathlib import Path

import polars as pl
from conftest import FIXTURE_DAY_ZERO, make_queries, make_transactions

from hmrec.config import load_base_config, load_catalog_config, load_preprocess_config
from hmrec.data.day_index import day_index_to_date
from hmrec.data.folds import Fold
from hmrec.paths import build_paths
from hmrec.recall.inputs import SharedTables, build_recall_inputs, read_queries

REPO = Path(__file__).resolve().parents[2]
BASE = load_base_config(REPO / "configs" / "base.yaml")
CATALOG = load_catalog_config(REPO / "configs" / "catalog.yaml")
# The real preprocess config with the fixture's day zero, so day numbers stay small.
PREPROCESS = load_preprocess_config(REPO / "configs" / "preprocess.yaml")
LINES = [
    (0, 1, 99),
    (1, 1, 80),
    (2, 1, 72),
    (0, 2, 99),
    (1, 2, 98),
    (0, 3, 99),
    (1, 3, 98),
    (2, 3, 71),
]
POISON = [(0, 4, 100), (1, 4, 100), (2, 4, 101)]


def _tables(lines: list[tuple[int, int, int]]) -> SharedTables:
    customers = pl.DataFrame(
        {"customer_idx": [0, 1, 2], "age_bucket": ["AGE_18_24"] * 3},
        schema={"customer_idx": pl.Int32, "age_bucket": pl.String},
    )
    articles = pl.DataFrame(
        {"article_idx": [1, 2, 3, 4], "product_code": [10, 10, 20, 20]},
        schema={"article_idx": pl.Int32, "product_code": pl.Int32},
    )
    return SharedTables(make_transactions(lines).lazy(), customers, articles)


def _fold(day_zero_offset: int) -> Fold:
    anchor = day_index_to_date(day_zero_offset, PREPROCESS.day_zero)
    return Fold(name="F1", phase="F", anchor=anchor, label_end=anchor + timedelta(days=6))


def test_inputs_carry_the_anchor_and_the_active_pool_before_it() -> None:
    fold = _fold(100)
    inputs = build_recall_inputs(_tables(LINES), fold, PREPROCESS, CATALOG)
    assert (inputs.anchor, inputs.day_zero, inputs.anchor_day) == (
        fold.anchor,
        PREPROCESS.day_zero,
        100,
    )
    assert inputs.active_pool["article_idx"].to_list() == [1]
    assert inputs.customers.columns == ["customer_idx", "age_bucket"]
    assert inputs.articles.columns == ["article_idx", "product_code"]


def test_purchases_on_and_after_the_anchor_do_not_change_the_pool() -> None:
    clean = build_recall_inputs(_tables(LINES), _fold(100), PREPROCESS, CATALOG)
    poisoned = build_recall_inputs(_tables(LINES + POISON), _fold(100), PREPROCESS, CATALOG)
    # Article 4 has 3 lines, all on / after the anchor: it must not become active.
    assert poisoned.active_pool.equals(clean.active_pool)


def test_read_queries_returns_the_folds_query_file(tmp_path: Path) -> None:
    paths = build_paths(BASE, tmp_path)
    queries = make_queries([(40_000_000, 7), (40_000_001, 9)], fold="F1")
    paths.queries_file("F1").parent.mkdir(parents=True)
    queries.write_parquet(paths.queries_file("F1"))
    assert read_queries(paths, "F1").equals(queries)
    assert FIXTURE_DAY_ZERO is not None  # (the fixture day zero is not used here on purpose)
