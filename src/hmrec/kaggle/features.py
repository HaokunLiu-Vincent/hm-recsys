"""The wide feature table of a Kaggle run: shared tables once, then one shard per process.

The features are the 71 locked ones (configs/features.yaml, checked against
configs/locks/features.lock.yaml), computed as of the predicted week's anchor
with the functions M6 used (hmrec.features).

Two kinds of tables (hmrec.features.build.FoldTables):

    over ALL customers, computed ONCE per run (prepare_shared_tables) and read
    by every shard:
        the article table (C group: counts, prices, buyers, sale days ... of every
        article) and the global product-type shares (for ui_type_lift)
    per customer, computed by each shard for its own customers:
        the customer table (D group), the customer x article aggregates (E group),
        the customers' last purchases (B group)

A shard is then built like a fold in M6: in chunks of `query_chunk_size` whole
queries, each chunk checked (no repeated pair; tt_dot = the R6 score and
wals_dot = the R5 score, bit for bit) and written as one part file;
features.json is written LAST and marks the shard as finished. The number of
rows must equal the number of deduplicated candidates of the shard.

A shard also writes query_stats.parquet: per query, the number of candidates and
whether the customer has any purchase before the anchor (for the run's report).

No label is read here; recall code is not imported (its output is read as files).
"""

import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from hmrec.config import FeaturesConfig, load_features_config
from hmrec.data.metadata import scan_processed_parquet
from hmrec.feature_files import (
    METADATA_FILE,
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
    RecallSideFiles,
    build_chunk,
    check_dot_equals_score,
    check_no_repeated_pairs,
    load_dense_sources,
    load_history,
)
from hmrec.features.categories import (
    CategoryCodes,
    build_category_codes,
    save_or_check_category_codes,
)
from hmrec.features.dense import DenseSources
from hmrec.features.item import build_item_table
from hmrec.features.user import build_user_table
from hmrec.features.user_item import (
    article_field_table,
    build_user_item_tables,
    global_type_shares,
)
from hmrec.fusion.union import CHANNEL_PREFIXES
from hmrec.kaggle.context import KaggleContext
from hmrec.kaggle.plan import read_queries, shard_queries
from hmrec.locks import check_features_match_lock, load_features_lock, load_twotower_lock
from hmrec.logging_utils import get_logger

LOGGER = get_logger("kaggle.features")

N_CANDIDATES = "n_candidates"
HAS_HISTORY = "has_history"
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


def load_locked_features_config(context: KaggleContext) -> FeaturesConfig:
    """configs/features.yaml, after checking it against the features lock and the R6 lock.

    Raises:
        ValueError: If the config differs from configs/locks/features.lock.yaml, or
            its history length is not the one the locked R6 uses (proposal §8.3).
    """
    paths = context.paths
    config = load_features_config(paths.features_config_file)
    check_features_match_lock(config, load_features_lock(paths.features_lock_file))
    locked_history = load_twotower_lock(paths.twotower_lock_file).channel.config["history_length"]
    if config.history_length != locked_history:
        raise ValueError(
            f"features.history_length is {config.history_length}, but the locked R6 uses "
            f"{locked_history}"
        )
    return config


def load_data(context: KaggleContext, queries: pl.DataFrame) -> FoldData:
    """The processed tables and the given queries as the feature code's FoldData."""
    paths, preprocess = context.paths, context.preprocess
    metadata = paths.processed_metadata_file
    return FoldData(
        fold=context.target.name,
        anchor=context.target.anchor,
        day_zero=preprocess.day_zero,
        transactions=scan_processed_parquet(paths.transactions_file, metadata, preprocess),
        customers=scan_processed_parquet(paths.customers_file, metadata, preprocess).collect(),
        articles=scan_processed_parquet(paths.articles_file, metadata, preprocess).collect(),
        queries=queries.select(QUERY_ID, CUSTOMER_IDX),
    )


# ---------------------------------------------------------------- once per run


@dataclass(frozen=True)
class SharedTables:
    """The tables over all customers, computed once per run.

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


def _write_once(table: pl.DataFrame, path: Path) -> None:
    """Write a shared table; an existing file must hold exactly the same table."""
    if path.exists():
        if not pl.read_parquet(path).equals(table):
            raise ValueError(f"{path} exists and holds another table; it is never replaced")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(path)


def prepare_shared_tables(context: KaggleContext, config: FeaturesConfig) -> SharedTables:
    """Compute the shared tables of a run and write them (never replaced).

    Also checks the category code table against artifacts/features/category_codes.json,
    the one the ranker's training data was encoded with.
    """
    paths, run = context.paths, context.run
    data = load_data(context, read_queries(paths.kaggle_queries_file(run)))
    codes = build_category_codes(data.articles, data.customers)
    save_or_check_category_codes(paths.category_codes_file, codes)
    tables = compute_shared_tables(data, config, codes)
    _write_once(tables.item, paths.kaggle_item_table_file(run))
    _write_once(tables.global_type_share, paths.kaggle_type_share_file(run))
    LOGGER.info(
        "%s: article table %d rows, global type shares %d rows",
        run,
        tables.item.height,
        tables.global_type_share.height,
    )
    return tables


def read_shared_tables(context: KaggleContext) -> SharedTables:
    """Read the shared tables written by prepare_shared_tables."""
    paths, run = context.paths, context.run
    files = (paths.kaggle_item_table_file(run), paths.kaggle_type_share_file(run))
    missing = [str(file) for file in files if not file.is_file()]
    if missing:
        raise FileNotFoundError(f"shared feature tables not found: {missing}; run the prepare step")
    return SharedTables(pl.read_parquet(files[0]), pl.read_parquet(files[1]))


# ---------------------------------------------------------------- one shard


def shard_tables(
    data: FoldData, config: FeaturesConfig, shared: SharedTables, recent_purchases: pl.DataFrame
) -> FoldTables:
    """The tables joined onto a shard's chunks: the shared ones plus the shard's own.

    Args:
        data: The run's tables with the SHARD's queries.
        config: configs/features.yaml.
        shared: The tables over all customers (computed once per run).
        recent_purchases: The shard customers' last purchases, one row per
            history slot: customer_idx, article_idx.
    """
    codes = build_category_codes(data.articles, data.customers)
    user_item = build_user_item_tables(data, config, recent_purchases)
    return FoldTables(
        item=shared.item,
        user=build_user_table(data, config, codes),
        # The shares over all customers come from the table computed once per run.
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
        data: The run's tables with the SHARD's queries (query_id, customer_idx).
        history: The shard customers' last purchases before the anchor.
        candidates: Channel -> its candidates of the shard.
        dense: The B-group sources (the shard's vectors, the run's bucket table).
    """

    data: FoldData
    history: HistoryArrays
    candidates: dict[str, pl.DataFrame]
    dense: DenseSources


@dataclass(frozen=True)
class ShardFiles:
    """Where a shard's wide table goes.

    Attributes:
        features_dir: Folder of the part files and features.json.
        stats_file: The shard's query_stats.parquet.
    """

    features_dir: Path
    stats_file: Path

    def part_file(self, part: int) -> Path:
        """The file of one chunk of whole queries."""
        return self.features_dir / f"part-{part:05d}.parquet"


@dataclass(frozen=True)
class ShardResult:
    """What building the features of one shard did.

    Attributes:
        skipped: True if the shard was already finished and nothing was written.
        n_rows: Rows of the shard's wide table.
        n_parts: Part files.
        n_queries: Queries of the shard.
        tt_dot_rows_checked: R6 rows whose tt_dot was compared with the R6 score.
        wals_dot_rows_checked: R5 rows whose wals_dot was compared with the R5 score.
    """

    skipped: bool
    n_rows: int
    n_parts: int
    n_queries: int
    tt_dot_rows_checked: int
    wals_dot_rows_checked: int


def shard_is_finished(features_dir: Path) -> bool:
    """True if the shard is finished; part files without features.json are an error.

    Raises:
        FileExistsError: On a half-finished shard. Nothing is deleted automatically.
    """
    if (features_dir / METADATA_FILE).exists():
        return True
    leftovers = sorted(features_dir.glob("part-*.parquet")) if features_dir.exists() else []
    if leftovers:
        raise FileExistsError(
            f"{features_dir} holds {len(leftovers)} part files but no {METADATA_FILE}: a "
            "half-finished shard. Check the run log before removing anything."
        )
    return False


def write_shard_features(
    inputs: ShardInputs,
    shared: SharedTables,
    config: FeaturesConfig,
    query_chunk_size: int,
    files: ShardFiles,
    details: dict[str, Any],
) -> ShardResult:
    """Build, check and write the wide table of one shard (see the module docstring).

    Args:
        inputs: Everything read for the shard.
        shared: The tables over all customers (computed once per run).
        config: configs/features.yaml.
        query_chunk_size: Queries per part file (chunking.query_chunk_size).
        files: Where the shard's files go.
        details: Extra facts for features.json (run, shard, git commit).

    Raises:
        ValueError: If a check fails; the part files written so far stay on disk
            WITHOUT features.json, so the shard counts as not finished.
    """
    if query_chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be > 0, got {query_chunk_size}")
    started = time.perf_counter()
    data, candidates = inputs.data, inputs.candidates
    tables = shard_tables(data, config, shared, history_rows(inputs.history))
    query_ids = data.queries[QUERY_ID].unique().sort()
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
        part_file = files.part_file(len(parts))
        part_file.parent.mkdir(parents=True, exist_ok=True)
        wide.write_parquet(part_file)
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
    query_stats(data.queries, pl.concat(counts), tables.user).write_parquet(files.stats_file)
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
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    # Written LAST: it marks the shard as finished.
    write_feature_files(files.features_dir, record)
    return ShardResult(False, n_rows, len(parts), data.queries.height, tt_checked, wals_checked)


def load_shard_inputs(context: KaggleContext, config: FeaturesConfig, shard: int) -> ShardInputs:
    """Read one shard's queries, candidates and side files (written by hmrec.kaggle.recall)."""
    paths, run = context.paths, context.run
    all_queries = read_queries(paths.kaggle_queries_file(run))
    data = load_data(context, shard_queries(all_queries, shard, context.config.shard_customers))
    history = load_history(data, config)
    candidates = {
        channel: pl.read_parquet(paths.kaggle_candidates_file(run, shard, channel))
        for channel in CHANNEL_PREFIXES
    }
    side_files = RecallSideFiles(
        twotower_vectors_dir=paths.kaggle_vectors_dir(run, shard, "twotower"),
        wals_vectors_dir=paths.kaggle_vectors_dir(run, shard, "wals"),
        bucket_scores_file=paths.kaggle_r1_bucket_scores_file(run),
    )
    dense = load_dense_sources(
        paths, context.preprocess, data, history, config, context.base, side_files
    )
    return ShardInputs(data, history, candidates, dense)


def build_shard_features(context: KaggleContext, config: FeaturesConfig, shard: int) -> ShardResult:
    """Build the wide table of one shard of a run; a finished shard is skipped.

    Raises:
        FileExistsError: On a half-finished shard.
        ValueError: If a check fails.
    """
    paths, run = context.paths, context.run
    files = ShardFiles(
        paths.kaggle_features_dir(run, shard), paths.kaggle_shard_stats_file(run, shard)
    )
    if shard_is_finished(files.features_dir):
        LOGGER.warning("%s shard %d: features already exist; SKIPPED", run, shard)
        return ShardResult(True, 0, 0, 0, 0, 0)
    details = {
        "run": run,
        "shard": shard,
        "git_commit": context.stamp.git_commit,
        "git_dirty": context.stamp.git_dirty,
    }
    return write_shard_features(
        load_shard_inputs(context, config, shard),
        read_shared_tables(context),
        config,
        context.base.chunking.query_chunk_size,
        files,
        details,
    )
