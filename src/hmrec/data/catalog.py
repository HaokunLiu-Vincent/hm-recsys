"""Article catalog C and active pool A(a) (proposal §2.4).

- Catalog C: every article in articles.csv. One row per article_idx.
- Active pool A(a): articles with at least `min_purchase_lines` purchase lines
  in [a - window_days, a). With configs/catalog.yaml this is ">= 3 lines in the
  28 days before the anchor". It approximates "currently on sale"; it is not
  real stock data.

R1, R4, R5 and R6 limit their output to A(a); R2 and R3 do not (proposal §2.4).
The two numbers come from configs/catalog.yaml and are frozen (not tuned).
"""

from datetime import date

import polars as pl

from hmrec.config import ActivePoolConfig
from hmrec.data.asof import history_window
from hmrec.data.id_maps import ARTICLE_IDX, IDX_DTYPE

PURCHASE_LINE_COUNT = "purchase_line_count"


def build_catalog(articles: pl.LazyFrame) -> pl.DataFrame:
    """Return the article catalog C.

    Args:
        articles: data/processed/articles.parquet, one row per article,
            with an int32 `article_idx` column.

    Returns:
        One row per article. Column: article_idx (int32), sorted ascending.

    Raises:
        ValueError: If article_idx is missing, has the wrong dtype, or repeats.
    """
    catalog = articles.select(ARTICLE_IDX).sort(ARTICLE_IDX).collect()
    if catalog[ARTICLE_IDX].dtype != IDX_DTYPE:
        raise ValueError(f"articles.{ARTICLE_IDX} has dtype {catalog[ARTICLE_IDX].dtype}")
    if catalog[ARTICLE_IDX].is_duplicated().any():
        raise ValueError(f"articles.{ARTICLE_IDX} has duplicate values")
    return catalog


def count_purchase_lines(history: pl.DataFrame) -> pl.DataFrame:
    """Count purchase lines per article.

    Duplicate lines count separately: two identical rows mean the item was
    bought twice, which is what "purchase records" means in proposal §2.4.

    Args:
        history: One row per purchase line, with column article_idx (int32).

    Returns:
        One row per article that appears in `history`.
        Columns: article_idx (int32), purchase_line_count (int32), sorted by article_idx.
    """
    return (
        history.group_by(ARTICLE_IDX)
        .agg(pl.len().cast(pl.Int32).alias(PURCHASE_LINE_COUNT))
        .sort(ARTICLE_IDX)
    )


def build_active_pool(
    transactions: pl.LazyFrame,
    anchor: date,
    day_zero: date,
    pool_config: ActivePoolConfig,
) -> pl.DataFrame:
    """Return the active pool A(a) for one anchor.

    Uses only the history [anchor - window_days, anchor) from hmrec.data.asof,
    so rows on or after the anchor can never change the pool.

    Args:
        transactions: One row per purchase line (processed transactions, int16 t_dat_day).
        anchor: First day of the label week.
        day_zero: Day 0 of t_dat_day.
        pool_config: window_days and min_purchase_lines from configs/catalog.yaml.

    Returns:
        One row per active article.
        Columns: article_idx (int32), purchase_line_count (int32; lines in the
        window, >= min_purchase_lines), sorted by article_idx.
    """
    history = history_window(
        transactions, anchor, day_zero, pool_config.window_days, columns=[ARTICLE_IDX]
    )
    counts = count_purchase_lines(history)
    return counts.filter(pl.col(PURCHASE_LINE_COUNT) >= pool_config.min_purchase_lines)
