"""Load what a recall channel reads for one fold (the same for D-phase trials and M6).

    tables = load_shared_tables(paths, preprocess)          # once per process
    inputs = build_recall_inputs(tables, fold, preprocess, catalog)
    queries = read_queries(paths, fold.name)

Processed tables are read only through hmrec.data.metadata.scan_processed_parquet.
The transactions stay lazy: every channel cuts them itself with hmrec.data.asof,
so nothing here filters by date except the active pool (which uses asof too).

No labels are read here. The query list (artifacts/queries/{fold}.parquet) only
says WHO is predicted for, not what they bought.
"""

from dataclasses import dataclass

import polars as pl

from hmrec.config import CatalogConfig, PreprocessConfig
from hmrec.data.catalog import build_active_pool
from hmrec.data.folds import Fold
from hmrec.data.metadata import scan_processed_parquet
from hmrec.paths import ProjectPaths
from hmrec.recall.base import RecallInputs


@dataclass(frozen=True)
class SharedTables:
    """The processed tables every fold reads.

    Attributes:
        transactions: Lazy scan, one row per purchase line (int16 t_dat_day).
        customers: One row per customer; columns customer_idx (int32), age_bucket (str).
        articles: One row per article; columns article_idx (int32), product_code (int32).
    """

    transactions: pl.LazyFrame
    customers: pl.DataFrame
    articles: pl.DataFrame


def load_shared_tables(paths: ProjectPaths, preprocess: PreprocessConfig) -> SharedTables:
    """Scan the transactions and read the two small tables (after the metadata check)."""
    metadata = paths.processed_metadata_file
    transactions = scan_processed_parquet(paths.transactions_file, metadata, preprocess)
    customers = (
        scan_processed_parquet(paths.customers_file, metadata, preprocess)
        .select("customer_idx", "age_bucket")
        .collect()
    )
    articles = (
        scan_processed_parquet(paths.articles_file, metadata, preprocess)
        .select("article_idx", "product_code")
        .collect()
    )
    return SharedTables(transactions, customers, articles)


def build_recall_inputs(
    tables: SharedTables, fold: Fold, preprocess: PreprocessConfig, catalog: CatalogConfig
) -> RecallInputs:
    """The channel inputs as of the fold's anchor, with the active pool A(a).

    A(a) uses the history [anchor - 28d, anchor) (configs/catalog.yaml), read
    through hmrec.data.asof; the anchor day itself is never included.
    """
    active_pool = build_active_pool(
        tables.transactions, fold.anchor, preprocess.day_zero, catalog.active_pool
    )
    return RecallInputs(
        fold.anchor,
        preprocess.day_zero,
        tables.transactions,
        tables.customers,
        tables.articles,
        active_pool,
    )


def read_queries(paths: ProjectPaths, fold_name: str) -> pl.DataFrame:
    """The fold's query list: one row per query; columns query_id, fold, customer_idx, anchor."""
    return pl.read_parquet(paths.queries_file(fold_name))
