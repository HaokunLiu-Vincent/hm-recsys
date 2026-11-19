"""Tests for hmrec.data.catalog. Expected values: tests/fixtures/asof/README.md."""

from pathlib import Path

import polars as pl
import pytest

from conftest import D1_ANCHOR, FIXTURE_DAY_ZERO, make_transactions
from hmrec.config import ActivePoolConfig, load_catalog_config
from hmrec.data.catalog import build_active_pool, build_catalog, count_purchase_lines

REPO_CATALOG_YAML = Path(__file__).resolve().parents[2] / "configs" / "catalog.yaml"
POOL_CONFIG = ActivePoolConfig(window_days=28, min_purchase_lines=3)


def test_repo_catalog_config_matches_proposal() -> None:
    # proposal §2.4: >= 3 purchase lines in [a - 28d, a). Frozen, not a hyperparameter.
    config = load_catalog_config(REPO_CATALOG_YAML)
    assert config.active_pool == POOL_CONFIG


def test_build_catalog_returns_sorted_article_idx() -> None:
    articles = pl.DataFrame({"article_idx": [2, 0, 1], "product_code": [7, 8, 9]})
    articles = articles.with_columns(pl.col("article_idx").cast(pl.Int32))
    catalog = build_catalog(articles.lazy())
    assert catalog["article_idx"].to_list() == [0, 1, 2]
    assert catalog.columns == ["article_idx"]


def test_build_catalog_rejects_duplicates() -> None:
    articles = pl.DataFrame({"article_idx": [0, 0]}, schema={"article_idx": pl.Int32})
    with pytest.raises(ValueError, match="duplicate"):
        build_catalog(articles.lazy())


def test_count_purchase_lines_counts_duplicates() -> None:
    history = make_transactions([(0, 5, 1), (0, 5, 1), (1, 5, 2), (1, 6, 2)])
    counts = count_purchase_lines(history)
    assert counts.rows() == [(5, 3), (6, 1)]


def test_active_pool_hand_computed(fixture_transactions: pl.DataFrame) -> None:
    pool = build_active_pool(fixture_transactions.lazy(), D1_ANCHOR, FIXTURE_DAY_ZERO, POOL_CONFIG)
    # Window [14, 42):
    #   article 0: days 14, 30, 41 -> 3 lines (both window ends included)  -> in
    #   article 1: days 30, 41 (day 13 is outside)                          -> 2, out
    #   article 2: days 20, 20 (duplicate), 35                              -> 3, in
    #   article 3: days 40, 41 (day 42 is the anchor, not history)         -> 2, out
    assert pool.rows() == [(0, 3), (2, 3)]
    assert pool.schema == pl.Schema({"article_idx": pl.Int32, "purchase_line_count": pl.Int32})


def test_active_pool_ignores_poison_on_and_after_anchor(
    fixture_transactions: pl.DataFrame,
) -> None:
    clean = build_active_pool(fixture_transactions.lazy(), D1_ANCHOR, FIXTURE_DAY_ZERO, POOL_CONFIG)
    # Article 1 and 3 each miss the threshold by one line. Many lines on or after
    # the anchor (day 42) must not push them into the pool, nor add article 5.
    poison = make_transactions([(9, 1, 42), (9, 1, 42), (9, 3, 42), (9, 3, 50), (9, 5, 43)] * 3)
    poisoned_input = pl.concat([fixture_transactions, poison])
    poisoned = build_active_pool(poisoned_input.lazy(), D1_ANCHOR, FIXTURE_DAY_ZERO, POOL_CONFIG)
    assert poisoned.equals(clean)
