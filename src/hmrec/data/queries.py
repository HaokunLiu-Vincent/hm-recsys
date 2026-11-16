"""Weekly query list and label sets G_q (proposal §2.3, §14.2).

A query is (customer, fold). It exists only if the customer bought at least one
catalog article in the fold's label week [anchor, label_end]. Its label set G_q
is the DE-DUPLICATED set of catalog articles the customer bought in that week.

query_id is unique across ALL folds (configs/queries.yaml):

    query_id = fold_ordinal * query_id_fold_offset + position

    fold_ordinal: index of the fold in configs/fold_manifest.yaml (D1 = 0 ... T4 = 15)
    position:     0-based rank of customer_idx (ascending) among the fold's queries

Example with offset 10,000,000: the 3rd query (position 2) of D2 (ordinal 1)
has query_id 10,000,002. `decode_query_id` reverses this.

Outputs (written by scripts/m0_build_queries.py):
    artifacts/queries/{fold}.parquet: one row per query.
        Columns: query_id (int32), fold (str), customer_idx (int32), anchor (date).
    artifacts/labels/{fold}.parquet: one row per (query_id, article_idx).
        Columns: query_id (int32), fold (str), article_idx (int32).

Labels may be read only by hmrec.eval and hmrec.ranker.sampling (AGENTS.md §3.1).
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from hmrec.config import INT32_MAX
from hmrec.data.asof import label_week_transactions
from hmrec.data.folds import Fold, FoldManifest
from hmrec.data.id_maps import ARTICLE_IDX, CUSTOMER_IDX, IDX_DTYPE

QUERY_ID = "query_id"
FOLD = "fold"
ANCHOR = "anchor"
QUERY_ID_DTYPE = pl.Int32
QUERY_POSITION = "query_position"

QUERIES_SCHEMA = pl.Schema(
    {QUERY_ID: QUERY_ID_DTYPE, FOLD: pl.String, CUSTOMER_IDX: IDX_DTYPE, ANCHOR: pl.Date}
)
LABELS_SCHEMA = pl.Schema({QUERY_ID: QUERY_ID_DTYPE, FOLD: pl.String, ARTICLE_IDX: IDX_DTYPE})


@dataclass(frozen=True)
class FoldQueries:
    """Queries and labels of one fold.

    Attributes:
        queries: One row per query; columns as QUERIES_SCHEMA, sorted by query_id.
        labels: One row per (query_id, article_idx); columns as LABELS_SCHEMA,
            sorted by (query_id, article_idx).
    """

    queries: pl.DataFrame
    labels: pl.DataFrame


# ---------------------------------------------------------------- query_id


def fold_ordinal(manifest: FoldManifest, fold_name: str) -> int:
    """Return the 0-based position of a fold in the manifest (D1 = 0 ... T4 = 15)."""
    if fold_name not in manifest.names:
        raise KeyError(f"unknown fold {fold_name!r}; known folds: {list(manifest.names)}")
    return manifest.names.index(fold_name)


def check_query_id_capacity(manifest: FoldManifest, query_id_fold_offset: int) -> None:
    """Fail if the largest possible query_id does not fit in int32.

    The largest query_id is (n_folds - 1) * offset + (offset - 1) = n_folds * offset - 1.
    """
    largest_query_id = len(manifest.folds) * query_id_fold_offset - 1
    if largest_query_id > INT32_MAX:
        raise ValueError(
            f"{len(manifest.folds)} folds x query_id_fold_offset={query_id_fold_offset} "
            f"gives query_id up to {largest_query_id}, which does not fit in int32"
        )


def query_id_base(manifest: FoldManifest, fold_name: str, query_id_fold_offset: int) -> int:
    """Return the query_id of position 0 in a fold: fold_ordinal * offset."""
    check_query_id_capacity(manifest, query_id_fold_offset)
    return fold_ordinal(manifest, fold_name) * query_id_fold_offset


def decode_query_id(
    query_id: int, manifest: FoldManifest, query_id_fold_offset: int
) -> tuple[str, int]:
    """Turn a query_id back into (fold name, position inside the fold).

    Example with offset 10,000,000: 10,000,002 -> ("D2", 2).

    Args:
        query_id: A query_id produced by this module.
        manifest: The fold manifest (gives the fold order).
        query_id_fold_offset: From configs/queries.yaml.

    Returns:
        (fold_name, position). position is the 0-based rank of the customer
        among the fold's queries.

    Raises:
        ValueError: If query_id is negative or its fold ordinal is not in the manifest.
    """
    if query_id < 0:
        raise ValueError(f"query_id must be >= 0, got {query_id}")
    ordinal, position = divmod(query_id, query_id_fold_offset)
    if ordinal >= len(manifest.folds):
        raise ValueError(
            f"query_id {query_id} has fold ordinal {ordinal}, but the manifest has only "
            f"{len(manifest.folds)} folds"
        )
    return manifest.folds[ordinal].name, position


# ---------------------------------------------------------------- building


def build_fold_queries_from_label_week(
    label_week: pl.DataFrame,
    catalog: pl.DataFrame,
    fold: Fold,
    manifest: FoldManifest,
    query_id_fold_offset: int,
) -> FoldQueries:
    """Build queries and labels from the purchase lines of one label week.

    Steps:
        1. Keep only lines whose article is in the catalog C (proposal §2.3:
           a query needs at least one CATALOG article).
        2. De-duplicate (customer_idx, article_idx): G_q is a set, so buying
           the same article twice in the label week counts once.
        3. Every remaining customer becomes a query. Customers are sorted by
           customer_idx and numbered 0..n-1 (position); query_id = base + position.

    Args:
        label_week: Purchase lines of the label week, from
            asof.label_week_transactions. Columns: customer_idx, article_idx (int32).
        catalog: Catalog C; column article_idx (int32).
        fold: The fold being built.
        manifest: The fold manifest (for the fold ordinal).
        query_id_fold_offset: From configs/queries.yaml.

    Returns:
        FoldQueries with queries (QUERIES_SCHEMA) and labels (LABELS_SCHEMA).

    Raises:
        ValueError: If the fold has offset or more queries, or the query_id range
            does not fit in int32.
    """
    base = query_id_base(manifest, fold.name, query_id_fold_offset)
    in_catalog = label_week.join(catalog.select(ARTICLE_IDX), on=ARTICLE_IDX, how="semi")
    label_pairs = in_catalog.select(CUSTOMER_IDX, ARTICLE_IDX).unique()

    customers = label_pairs.select(CUSTOMER_IDX).unique().sort(CUSTOMER_IDX)
    if customers.height >= query_id_fold_offset:
        raise ValueError(
            f"fold {fold.name} has {customers.height} queries, which is not below "
            f"query_id_fold_offset={query_id_fold_offset}"
        )
    queries = customers.with_columns(
        (pl.lit(base, dtype=pl.Int64) + pl.int_range(pl.len(), dtype=pl.Int64))
        .cast(QUERY_ID_DTYPE, strict=True)
        .alias(QUERY_ID),
        pl.lit(fold.name).alias(FOLD),
        pl.lit(fold.anchor, dtype=pl.Date).alias(ANCHOR),
    ).select(QUERIES_SCHEMA.names())

    labels = (
        label_pairs.join(queries.select(QUERY_ID, FOLD, CUSTOMER_IDX), on=CUSTOMER_IDX)
        .select(LABELS_SCHEMA.names())
        .sort(QUERY_ID, ARTICLE_IDX)
    )
    result = FoldQueries(queries=queries, labels=labels)
    check_fold_queries(result)
    return result


def build_fold_queries(
    transactions: pl.LazyFrame,
    catalog: pl.DataFrame,
    fold: Fold,
    manifest: FoldManifest,
    day_zero: date,
    query_id_fold_offset: int,
) -> FoldQueries:
    """Build queries and labels for one fold from the processed transactions.

    Reads only the label week [anchor, label_end] through
    asof.label_week_transactions; rows outside that week cannot change the result.

    Args:
        transactions: Processed transactions (one row per purchase line, int16
            t_dat_day), read with metadata.scan_processed_parquet.
        catalog: Catalog C from catalog.build_catalog.
        fold: The fold to build.
        manifest: The fold manifest.
        day_zero: Day 0 of t_dat_day.
        query_id_fold_offset: From configs/queries.yaml.

    Returns:
        FoldQueries for this fold.
    """
    label_week = label_week_transactions(
        transactions, fold, day_zero, columns=[CUSTOMER_IDX, ARTICLE_IDX]
    )
    return build_fold_queries_from_label_week(
        label_week, catalog, fold, manifest, query_id_fold_offset
    )


# ---------------------------------------------------------------- checks


def _check_schema(table: pl.DataFrame, expected: pl.Schema, name: str) -> None:
    """Fail if `table` does not have exactly the expected columns and dtypes."""
    if table.schema != expected:
        raise ValueError(f"{name} schema is {dict(table.schema)}, expected {dict(expected)}")


def check_fold_queries(fold_queries: FoldQueries) -> None:
    """Check the invariants of one fold's queries and labels.

    - schemas match QUERIES_SCHEMA / LABELS_SCHEMA;
    - query_id and customer_idx are unique in queries;
    - (query_id, article_idx) is unique in labels;
    - every label belongs to a query, and every query has at least one label;
    - queries and labels contain a single fold.

    Raises:
        ValueError: On the first broken invariant.
    """
    queries, labels = fold_queries.queries, fold_queries.labels
    _check_schema(queries, QUERIES_SCHEMA, "queries")
    _check_schema(labels, LABELS_SCHEMA, "labels")
    if queries[QUERY_ID].is_duplicated().any() or queries[CUSTOMER_IDX].is_duplicated().any():
        raise ValueError("queries: query_id and customer_idx must be unique")
    if labels.select(QUERY_ID, ARTICLE_IDX).is_duplicated().any():
        raise ValueError("labels: (query_id, article_idx) must be unique")
    query_ids = set(queries[QUERY_ID].to_list())
    label_query_ids = set(labels[QUERY_ID].unique().to_list())
    if label_query_ids != query_ids:
        raise ValueError(
            f"labels cover {len(label_query_ids)} query_ids but queries has {len(query_ids)}; "
            "every query needs >= 1 label and every label needs a query"
        )
    fold_names = set(queries[FOLD].to_list()) | set(labels[FOLD].to_list())
    if len(fold_names) > 1:
        raise ValueError(f"queries/labels mix several folds: {sorted(fold_names)}")


# ---------------------------------------------------------------- IO


def write_fold_queries(
    fold_queries: FoldQueries, queries_path: Path, labels_path: Path, overwrite: bool
) -> None:
    """Write queries and labels of one fold to Parquet.

    Existing files are NOT replaced unless overwrite=True: queries and labels
    are the frozen definition of a fold, and later artifacts (candidates,
    features) refer to their query_ids.

    Raises:
        FileExistsError: If a target file exists and overwrite is False.
    """
    for path in (queries_path, labels_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it")
    check_fold_queries(fold_queries)
    queries_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    fold_queries.queries.write_parquet(queries_path)
    fold_queries.labels.write_parquet(labels_path)
