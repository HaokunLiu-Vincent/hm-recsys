"""The wide feature table in `serve` mode: shared tables once, then one shard per process.

The features are the 71 locked ones (configs/features.yaml, checked against
configs/locks/features.lock.yaml), computed as of the week's anchor with the
functions M6 used (hmrec.features).

Two kinds of tables (hmrec.features.build.FoldTables):

    over ALL customers, computed ONCE per week (compute_shared_tables) and read
    by every shard:
        the article table (C group: counts, prices, buyers, sale days ... of every
        article) and the global product-type shares (for ui_type_lift)
    per customer, computed by each shard for its own customers:
        the customer table (D group), the customer x article aggregates (E group),
        the customers' last purchases (B group)

A shard is then built like a week in `eval` mode: in chunks of `query_chunk_size`
whole queries, each chunk checked (no repeated pair; tt_dot = the R6 score and
wals_dot = the R5 score, bit for bit) and written as one part file;
features.json is written LAST. The number of rows must equal the number of
deduplicated candidates of the shard.

A shard also writes, next to its features:
    query_stats.parquet    per query, the number of candidates and whether the
                           customer has any purchase before the anchor
    feature_stats.csv      per feature, rows, missing rows, min, max and mean

A shard's features are removed after the shard was scored (docs/mlops.md §8.4);
the scoring step copies these two small files first.

Brought into main from branch kaggle-late-submission (hmrec.kaggle.features)
without its Kaggle-specific parts. No label is read here; recall code is not
imported (its output is read as files). Neither torch nor lightgbm is imported.
"""

import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from hmrec.config import FeaturesConfig
from hmrec.csv_records import append_row
from hmrec.feature_files import (
    FeatureFiles,
    PartFile,
    describe_part,
    write_feature_files,
)
from hmrec.features.base import (
    CUSTOMER_IDX,
    KEYS,
    QUERY_ID,
    FoldData,
    HistoryArrays,
    history_rows,
)
from hmrec.features.build import (
    TT_DOT,
    TWOTOWER_CHANNEL,
    WALS_CHANNEL,
    WALS_DOT,
    FoldTables,
    build_chunk,
    check_dot_equals_score,
    check_no_repeated_pairs,
)
from hmrec.features.categories import CategoryCodes, build_category_codes
from hmrec.features.dense import DenseSources
from hmrec.features.item import build_item_table
from hmrec.features.stats import STATS_COLUMNS, FeatureStats
from hmrec.features.user import build_user_table
from hmrec.features.user_item import (
    article_field_table,
    build_user_item_tables,
    global_type_shares,
)
from hmrec.logging_utils import get_logger
from hmrec.run_records import RunStamp

LOGGER = get_logger("weekly.shard_features")

N_CANDIDATES = "n_candidates"
HAS_HISTORY = "has_history"
QUERY_STATS_FILE = "query_stats.parquet"
FEATURE_STATS_FILE = "feature_stats.csv"
# A customer without any purchase before the anchor has NaN here (hmrec.features.user).
LAST_PURCHASE_FEATURE = "user_days_since_last_purchase"
QUERY_STATS_SCHEMA = pl.Schema(
    {
        QUERY_ID: pl.Int32,
        CUSTOMER_IDX: pl.Int32,
        N_CANDIDATES: pl.Int32,
        HAS_HISTORY: pl.Boolean,
    }
)


# ---------------------------------------------------------------- once per week


@dataclass(frozen=True)
class SharedTables:
    """The tables over all customers, computed once per week.

    Attributes:
        item: One row per article: article_idx and the C-group features.
        global_type_share: One row per product type sold in the last 52 weeks by
            anyone: its share of all purchase lines.
    """

    item: pl.DataFrame
    global_type_share: pl.DataFrame


def compute_shared_tables(
    data: FoldData, config: FeaturesConfig, codes: CategoryCodes
) -> SharedTables:
    """Compute the two tables that use the purchases of ALL customers before the anchor.

    Neither depends on data.queries: the article table has one row per article
    of the catalog, the type shares one row per product type.
    """
    return SharedTables(
        item=build_item_table(data, config, codes),
        global_type_share=global_type_shares(data, config, article_field_table(data.articles)),
    )


def write_shared_tables(tables: SharedTables, item_file: Path, type_share_file: Path) -> None:
    """Write the two shared tables of a week (never replaced).

    Raises:
        FileExistsError: If one of the files exists.
    """
    for table, path in ((tables.item, item_file), (tables.global_type_share, type_share_file)):
        if path.exists():
            raise FileExistsError(f"{path} already exists; it is never replaced")
        path.parent.mkdir(parents=True, exist_ok=True)
        table.write_parquet(path)


def read_shared_tables(item_file: Path, type_share_file: Path) -> SharedTables:
    """Read the shared tables written by write_shared_tables.

    Raises:
        FileNotFoundError: If one is missing (the shared step did not finish).
    """
    missing = [str(file) for file in (item_file, type_share_file) if not file.is_file()]
    if missing:
        raise FileNotFoundError(f"shared feature tables not found: {missing}")
    return SharedTables(pl.read_parquet(item_file), pl.read_parquet(type_share_file))


# ---------------------------------------------------------------- one shard


def shard_tables(
    data: FoldData, config: FeaturesConfig, shared: SharedTables, recent_purchases: pl.DataFrame
) -> FoldTables:
    """The tables joined onto a shard's chunks: the shared ones plus the shard's own.

    Args:
        data: The week's tables with the SHARD's queries.
        config: configs/features.yaml.
        shared: The tables over all customers (computed once per week).
        recent_purchases: The shard customers' last purchases, one row per
            history slot: customer_idx, article_idx.
    """
    codes = build_category_codes(data.articles, data.customers)
    user_item = build_user_item_tables(data, config, recent_purchases)
    return FoldTables(
        item=shared.item,
        user=build_user_table(data, config, codes),
        # The shares over all customers come from the table computed once per week.
        user_item=replace(user_item, global_type_share=shared.global_type_share),
        codes=codes,
    )


def count_distinct_candidates(candidates: dict[str, pl.DataFrame]) -> int:
    """Number of distinct (query_id, article_idx) pairs over all channels."""
    keys = pl.concat([table.select(KEYS) for table in candidates.values()])
    return keys.unique().height


def query_stats(
    queries: pl.DataFrame, rows_per_query: pl.DataFrame, user: pl.DataFrame
) -> pl.DataFrame:
    """Per query: its number of candidates and whether its customer has a purchase history.

    Args:
        queries: The shard's queries: query_id, customer_idx.
        rows_per_query: query_id and n_candidates of every query with a candidate.
        user: The shard's customer table (D group), one row per customer.

    Returns:
        One row per query, sorted by query_id; columns as QUERY_STATS_SCHEMA.
        A query without any candidate has n_candidates 0.
    """
    history = user.select(
        CUSTOMER_IDX, pl.col(LAST_PURCHASE_FEATURE).is_not_nan().alias(HAS_HISTORY)
    )
    stats = (
        queries.select(QUERY_ID, CUSTOMER_IDX)
        .join(rows_per_query, on=QUERY_ID, how="left")
        .join(history, on=CUSTOMER_IDX, how="left")
        .with_columns(pl.col(N_CANDIDATES).fill_null(0).cast(pl.Int32))
        .sort(QUERY_ID)
    )
    if stats[HAS_HISTORY].null_count() > 0:
        raise ValueError("some queries have no row in the customer table")
    return stats.select(QUERY_STATS_SCHEMA.names())


@dataclass(frozen=True)
class ShardInputs:
    """Everything read for one shard.

    Attributes:
        data: The week's tables with the SHARD's queries (query_id, customer_idx).
        history: The shard customers' last purchases before the anchor.
        candidates: Channel -> its candidates of the shard.
        dense: The B-group sources (the shard's vectors, the week's bucket table).
    """

    data: FoldData
    history: HistoryArrays
    candidates: dict[str, pl.DataFrame]
    dense: DenseSources


@dataclass(frozen=True)
class ShardResult:
    """What building the features of one shard did.

    Attributes:
        n_rows: Rows of the shard's wide table.
        expected_rows: Distinct (query_id, article_idx) pairs of the shard's candidates.
        n_parts: Part files.
        n_queries: Queries of the shard.
        tt_dot_rows_checked: R6 rows whose tt_dot was compared with the R6 score.
        wals_dot_rows_checked: R5 rows whose wals_dot was compared with the R5 score.
    """

    n_rows: int
    expected_rows: int
    n_parts: int
    n_queries: int
    tt_dot_rows_checked: int
    wals_dot_rows_checked: int


def write_stats_rows(path: Path, rows: list[dict[str, str]]) -> None:
    """Write the per-feature statistics of a week or a shard (never replaced)."""
    if path.exists():
        raise FileExistsError(f"{path} already exists; it is never replaced")
    for row in rows:
        append_row(path, STATS_COLUMNS, row)


def write_shard_features(
    inputs: ShardInputs,
    shared: SharedTables,
    config: FeaturesConfig,
    query_chunk_size: int,
    features_dir: Path,
    stamp: RunStamp,
    details: dict[str, Any],
) -> ShardResult:
    """Build, check and write the wide table of one shard (see the module docstring).

    Args:
        inputs: Everything read for the shard.
        shared: The tables over all customers (computed once per week).
        config: configs/features.yaml.
        query_chunk_size: Queries per part file (chunking.query_chunk_size).
        features_dir: Where the part files, features.json and the two statistics
            files go.
        stamp: The code state of the run.
        details: Extra facts for features.json (week, shard).

    Raises:
        ValueError: If a check fails; features.json is then not written.
    """
    if query_chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be > 0, got {query_chunk_size}")
    started = time.perf_counter()
    data, candidates = inputs.data, inputs.candidates
    tables = shard_tables(data, config, shared, history_rows(inputs.history))
    query_ids = data.queries[QUERY_ID].unique().sort()
    stats = FeatureStats(config.features)
    parts: list[PartFile] = []
    counts: list[pl.DataFrame] = []
    timings: dict[str, float] = {}
    tt_checked, wals_checked = 0, 0
    for start in range(0, query_ids.len(), query_chunk_size):
        chunk_ids = query_ids.slice(start, query_chunk_size)
        # query_ids are sorted and unique, so a chunk is one range of ids and a
        # query is never split over two part files.
        in_chunk = pl.col(QUERY_ID).is_between(chunk_ids[0], chunk_ids[-1])
        chunk_candidates = {name: table.filter(in_chunk) for name, table in candidates.items()}
        wide = build_chunk(chunk_candidates, data, tables, inputs.dense, config, timings)
        check_no_repeated_pairs(wide)
        tt_checked += check_dot_equals_score(
            wide, chunk_candidates[TWOTOWER_CHANNEL], TWOTOWER_CHANNEL, TT_DOT
        )
        wals_checked += check_dot_equals_score(
            wide, chunk_candidates[WALS_CHANNEL], WALS_CHANNEL, WALS_DOT
        )
        part_file = features_dir / f"part-{len(parts):05d}.parquet"
        part_file.parent.mkdir(parents=True, exist_ok=True)
        wide.write_parquet(part_file)
        stats.update(wide)
        counts.append(wide.group_by(QUERY_ID).agg(pl.len().cast(pl.Int32).alias(N_CANDIDATES)))
        # Read back from disk: the recorded sha256 and row count are those of the file.
        parts.append(describe_part(part_file))
        LOGGER.info("%s, %d queries, %d rows", part_file, chunk_ids.len(), wide.height)
    n_rows = sum(part.n_rows for part in parts)
    expected_rows = count_distinct_candidates(candidates)
    if n_rows != expected_rows:
        raise ValueError(
            f"the wide table has {n_rows} rows, but the six channels hold {expected_rows} "
            "distinct (query_id, article_idx) pairs"
        )
    query_stats(data.queries, pl.concat(counts), tables.user).write_parquet(
        features_dir / QUERY_STATS_FILE
    )
    write_stats_rows(features_dir / FEATURE_STATS_FILE, stats.rows(data.fold, stamp))
    record = FeatureFiles(
        fold=data.fold,
        columns=(*KEYS, *config.names),
        categorical_features=config.categorical_names,
        parts=tuple(parts),
        n_rows=n_rows,
        n_queries=data.queries.height,
        details={
            **details,
            "tt_dot_rows_checked": tt_checked,
            "wals_dot_rows_checked": wals_checked,
            "seconds": time.perf_counter() - started,
            "git_commit": stamp.git_commit,
            "git_dirty": stamp.git_dirty,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    # Written LAST: it marks the shard's features as complete.
    write_feature_files(features_dir, record)
    return ShardResult(
        n_rows, expected_rows, len(parts), data.queries.height, tt_checked, wals_checked
    )
