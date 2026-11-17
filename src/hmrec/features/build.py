"""Build the Ranker's wide feature table of one fold (proposal §7.3, §8).

Per fold (one process, scripts/m6_build_features.py):

  1. Read the fold's inputs. Everything recall wrote is read as FILES: the six
     candidate tables, R1's bucket score table, the R5 and R6 vectors. Nothing
     is imported from recall/ and no label is read.
  2. Once per fold: the query customers' last purchases (hmrec.data.histories),
     the article table (C), the query-customer table (D), the customer x article
     aggregates (E), the history-mean embedding vectors (B).
  3. Per chunk of `query_chunk_size` queries: fuse the six channels into one
     row per (query_id, article_idx) (A), add B, join C and D, add E, order the
     columns as in configs/features.yaml, sort by (query_id, article_idx) and
     write artifacts/features/{fold}/part-{i}.parquet. A query is never split
     over two files, and its rows are consecutive.
  4. Checks on every chunk: no repeated (query_id, article_idx); for the rows R6
     returned, tt_dot equals the R6 score bit for bit; the same for wals_dot
     and R5. At the end: the row count equals the number of deduplicated
     candidates recorded by the M6 monitor (results/m6_candidates.csv).
  5. artifacts/features/{fold}/features.json is written LAST and marks the fold
     as finished. It records every part file's sha256 and row count and the
     column list (hmrec.feature_files); a reader verifies them before use.
     Features are frozen once written: a finished fold is skipped, and part
     files without features.json are a half-finished fold (an error; nothing is
     deleted or reused automatically).

The table has no label column. Labels are joined in ranker/sampling.py (M7).
"""

import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from hmrec.config import BaseConfig, FeaturesConfig, FeatureSpec, PreprocessConfig
from hmrec.data.folds import Fold
from hmrec.data.histories import recent_purchase_histories
from hmrec.data.metadata import check_processed_metadata, scan_processed_parquet
from hmrec.feature_files import FeatureFiles, PartFile, describe_part, write_feature_files
from hmrec.features.base import (
    ARTICLE_IDX,
    CUSTOMER_IDX,
    KEYS,
    QUERY_ID,
    FoldData,
    HistoryArrays,
    history_rows,
)
from hmrec.features.categories import (
    CategoryCodes,
    build_category_codes,
    save_or_check_category_codes,
)
from hmrec.features.dense import (
    TT_DOT,
    WALS_DOT,
    DenseSources,
    add_dense_features,
    add_history_similarity_features,
    similarity_vectors,
)
from hmrec.features.item import build_item_table
from hmrec.features.source import add_source_features
from hmrec.features.stats import FeatureStats
from hmrec.features.user import build_user_table
from hmrec.features.user_item import UserItemTables, add_user_item_features, build_user_item_tables
from hmrec.fusion.union import CHANNEL_PREFIXES, fuse_candidates, rank_column
from hmrec.logging_utils import get_logger
from hmrec.memory import log_memory
from hmrec.paths import ProjectPaths
from hmrec.run_records import RunStamp
from hmrec.vectors import load_vectors

LOGGER = get_logger("features.build")

TWOTOWER_CHANNEL = "r6_twotower"
# R6 article vectors are L2-normalised by the model; the tt_sim_* features rely on it.
UNIT_NORM_TOLERANCE = 1e-3
HISTORY_SIMILARITY = "history_similarity"
WALS_CHANNEL = "r5_wals"
POLARS_DTYPES: dict[str, type[pl.DataType]] = {"float32": pl.Float32, "int16": pl.Int16}


# ---------------------------------------------------------------- inputs (IO)


def load_fold_data(paths: ProjectPaths, preprocess: PreprocessConfig, fold: Fold) -> FoldData:
    """Read the tables every feature group needs (through the processed-metadata check)."""
    metadata = paths.processed_metadata_file
    return FoldData(
        fold=fold.name,
        anchor=fold.anchor,
        day_zero=preprocess.day_zero,
        transactions=scan_processed_parquet(paths.transactions_file, metadata, preprocess),
        customers=scan_processed_parquet(paths.customers_file, metadata, preprocess).collect(),
        articles=scan_processed_parquet(paths.articles_file, metadata, preprocess).collect(),
        queries=pl.read_parquet(paths.queries_file(fold.name), columns=[QUERY_ID, CUSTOMER_IDX]),
    )


def load_candidates(paths: ProjectPaths, fold: str) -> dict[str, pl.DataFrame]:
    """The six channels' candidate files of the fold, in the order R1 ... R6."""
    return {
        channel: pl.read_parquet(paths.candidates_file(fold, channel))
        for channel in CHANNEL_PREFIXES
    }


def _load_embedding(
    vectors_file: Path, mask_file: Path, n_articles: int
) -> tuple[np.ndarray, np.ndarray]:
    """One embedding matrix float32 [n_articles, dim] and its mask bool [n_articles]."""
    vectors = np.load(vectors_file, allow_pickle=False)
    mask = np.load(mask_file, allow_pickle=False)
    if vectors.dtype != np.float32 or vectors.shape[0] != n_articles or mask.shape != (n_articles,):
        raise ValueError(
            f"{vectors_file}: expected float32 [{n_articles}, dim] with a mask [{n_articles}], "
            f"got {vectors.dtype} {vectors.shape} and mask {mask.shape}"
        )
    return vectors, mask.astype(np.bool_)


def load_history(data: FoldData, config: FeaturesConfig) -> HistoryArrays:
    """The query customers' last `config.history_length` purchases before the anchor.

    From hmrec.data.histories.recent_purchase_histories, the function R6
    predicts with: the features and the model see the same history.
    """
    customers = data.query_customers
    history, history_mask = recent_purchase_histories(
        data.transactions,
        customers,
        data.anchor,
        data.day_zero,
        config.history_length,
        padding_idx=data.articles.height,
    )
    return HistoryArrays(customers, history, history_mask)


def check_unit_norms(vectors: np.ndarray, label: str) -> None:
    """Fail unless every row has L2 norm 1 (within UNIT_NORM_TOLERANCE)."""
    norms = np.linalg.norm(vectors, axis=1)
    if vectors.shape[0] > 0 and float(np.abs(norms - 1.0).max()) > UNIT_NORM_TOLERANCE:
        raise ValueError(
            f"{label}: vectors are not L2-normalised (norms {norms.min():.4f} .. {norms.max():.4f})"
        )


def load_dense_sources(
    paths: ProjectPaths,
    preprocess: PreprocessConfig,
    data: FoldData,
    history: HistoryArrays,
    config: FeaturesConfig,
    base: BaseConfig,
) -> DenseSources:
    """Read what the B group needs and build the history-mean vectors of the query customers."""
    check_processed_metadata(paths.processed_metadata_file, preprocess)
    n_articles = data.articles.height
    users_per_chunk = max(1, config.embedding_pair_chunk_size // config.history_length)
    arrays = (history.customer_idx, history.history, history.history_mask)
    image, image_mask = _load_embedding(
        paths.image_embeddings_by_idx_file, paths.image_embedding_mask_file, n_articles
    )
    image_vectors = similarity_vectors(*arrays, image, image_mask, users_per_chunk)
    del image
    text, text_mask = _load_embedding(
        paths.text_embeddings_by_idx_file, paths.text_embedding_mask_file, n_articles
    )
    text_vectors = similarity_vectors(*arrays, text, text_mask, users_per_chunk)
    del text
    twotower = load_vectors(paths.twotower_vectors_dir(data.fold), "twotower", data.fold)
    # tt_sim_max_hist / tt_sim_top5_hist read the dot product of two article vectors
    # as a cosine, which is only right if the vectors have length 1.
    check_unit_norms(twotower.item_vectors, f"{data.fold} R6 article vectors")
    return DenseSources(
        twotower=twotower,
        wals=load_vectors(paths.wals_vectors_dir(data.fold), "wals", data.fold),
        bucket_scores=pl.read_parquet(paths.r1_bucket_scores_file(data.fold)),
        image=image_vectors,
        text=text_vectors,
        history=history,
        history_similarity_top_k=config.history_similarity_top_k,
        pair_chunk_size=base.chunking.pair_chunk_size,
        embedding_pair_chunk_size=config.embedding_pair_chunk_size,
    )


# ---------------------------------------------------------------- once per fold


@dataclass(frozen=True)
class FoldTables:
    """The per-fold tables that are joined onto every chunk.

    Attributes:
        item: One row per article: article_idx and the 16 C-group features.
        user: One row per query customer: customer_idx and the 11 D-group features.
        user_item: The E-group aggregates.
        codes: The category code table (shared by all folds).
    """

    item: pl.DataFrame
    user: pl.DataFrame
    user_item: UserItemTables
    codes: CategoryCodes


def build_fold_tables(data: FoldData, config: FeaturesConfig, history: HistoryArrays) -> FoldTables:
    """Compute the article table, the customer table and the customer x article aggregates."""
    codes = build_category_codes(data.articles, data.customers)
    return FoldTables(
        item=build_item_table(data, config, codes),
        user=build_user_table(data, config, codes),
        user_item=build_user_item_tables(data, config, history_rows(history)),
        codes=codes,
    )


# ---------------------------------------------------------------- one chunk


def order_columns(wide: pl.DataFrame, features: tuple[FeatureSpec, ...]) -> pl.DataFrame:
    """Keep query_id, article_idx and the features in config order, with the declared dtypes.

    Raises:
        ValueError: If a feature column is missing, or a value is null (missing
            numbers must be NaN and missing categories code 0, never null).
    """
    missing = [feature.name for feature in features if feature.name not in wide.columns]
    if missing:
        raise ValueError(f"the wide table lacks the feature columns {missing}")
    columns = [pl.col(feature.name).cast(POLARS_DTYPES[feature.dtype]) for feature in features]
    ordered = wide.select(
        pl.col(QUERY_ID).cast(pl.Int32), pl.col(ARTICLE_IDX).cast(pl.Int32), *columns
    )
    with_null = [name for name in ordered.columns if ordered[name].null_count() > 0]
    if with_null:
        raise ValueError(f"null values in the wide table columns {with_null}")
    return ordered.sort(KEYS)


def build_chunk(
    candidates_by_channel: Mapping[str, pl.DataFrame],
    data: FoldData,
    tables: FoldTables,
    dense: DenseSources,
    config: FeaturesConfig,
    timings: dict[str, float],
) -> pl.DataFrame:
    """The wide table of the queries in the given candidates.

    Args:
        candidates_by_channel: The six channels' candidates of ONE set of queries.
        data: The fold.
        tables: The per-fold tables.
        dense: The B-group sources.
        config: configs/features.yaml.
        timings: Running seconds per step; this call ADDS the time of the
            tt_sim_* step under "history_similarity" (the caller reports the total).

    Returns:
        One row per distinct (query_id, article_idx), sorted by those two
        columns: query_id, article_idx (int32) and the features in config order.
    """
    fused = add_source_features(fuse_candidates(candidates_by_channel), config.rrf_k)
    rows = fused.join(data.queries, on=QUERY_ID, how="left", maintain_order="left")
    if rows[CUSTOMER_IDX].null_count() > 0:
        raise ValueError("some candidates belong to a query_id that is not in the fold's queries")
    rows = add_dense_features(rows, dense, data.customers)
    started = time.perf_counter()
    rows = add_history_similarity_features(rows, dense)
    timings[HISTORY_SIMILARITY] = timings.get(HISTORY_SIMILARITY, 0.0) + (
        time.perf_counter() - started
    )
    log_memory(LOGGER, f"{data.fold}: after tt_sim_* of {rows.height} rows")
    rows = rows.join(tables.item, on=ARTICLE_IDX, how="left", maintain_order="left")
    rows = rows.join(tables.user, on=CUSTOMER_IDX, how="left", maintain_order="left")
    rows = add_user_item_features(rows, tables.user_item)
    return order_columns(rows, config.features)


def check_no_repeated_pairs(wide: pl.DataFrame) -> None:
    """Fail if a (query_id, article_idx) pair appears twice."""
    if wide.select(KEYS).is_duplicated().any():
        raise ValueError("the wide table repeats a (query_id, article_idx) pair")


def check_dot_equals_score(
    wide: pl.DataFrame, candidates: pl.DataFrame, channel: str, dot_column: str
) -> int:
    """For the rows a vector channel returned: its dot feature must equal the channel's score.

    tt_dot (wals_dot) is computed for every candidate from the saved vectors;
    for the candidates R6 (R5) itself returned, it must be the very number the
    channel stored as its score, bit for bit.

    Args:
        wide: A chunk of the wide table.
        candidates: The channel's candidates of the same queries (with score).
        channel: "r6_twotower" or "r5_wals".
        dot_column: "tt_dot" or "wals_dot".

    Returns:
        The number of rows compared (= the channel's candidates in the chunk).

    Raises:
        ValueError: If a returned row is missing from the wide table, or a dot
            differs from the score.
    """
    returned = wide.filter(pl.col(rank_column(channel)).is_not_nan()).select(*KEYS, dot_column)
    joined = candidates.select(*KEYS, "score").join(returned, on=KEYS, how="inner")
    if joined.height != candidates.height or returned.height != candidates.height:
        raise ValueError(
            f"{channel}: {candidates.height} candidates, but {returned.height} wide rows carry "
            f"its rank and {joined.height} of them match"
        )
    different = joined.filter(pl.col(dot_column) != pl.col("score")).height
    if different > 0:
        raise ValueError(
            f"{channel}: {dot_column} differs from the channel's score on {different} of "
            f"{joined.height} rows"
        )
    return joined.height


# ---------------------------------------------------------------- the fold


@dataclass(frozen=True)
class FoldInputs:
    """Everything read from disk for one fold.

    Attributes:
        data: Tables, queries and the anchor.
        history: The query customers' last purchases before the anchor.
        candidates: Channel -> its candidates of the whole fold.
        dense: The B-group sources.
        expected_rows: The fold's number of deduplicated candidates as recorded
            in results/m6_candidates.csv; the wide table must have exactly this many rows.
    """

    data: FoldData
    history: HistoryArrays
    candidates: dict[str, pl.DataFrame]
    dense: DenseSources
    expected_rows: int


@dataclass(frozen=True)
class BuildResult:
    """What building one fold did.

    Attributes:
        fold: Fold name.
        skipped: True if the fold was already finished and nothing was written.
        n_rows: Rows of the wide table.
        n_parts: Part files.
        n_queries: Queries of the fold.
        tt_dot_rows_checked: R6 rows whose tt_dot was compared with the R6 score.
        wals_dot_rows_checked: R5 rows whose wals_dot was compared with the R5 score.
        seconds: Wall time of the per-fold tables plus all chunks.
        history_similarity_seconds: The part of `seconds` spent on tt_sim_max_hist
            and tt_sim_top5_hist.
        stats_rows: Rows for results/m6_features.csv (empty when skipped).
    """

    fold: str
    skipped: bool
    n_rows: int
    n_parts: int
    n_queries: int
    tt_dot_rows_checked: int
    wals_dot_rows_checked: int
    seconds: float
    history_similarity_seconds: float
    stats_rows: list[dict[str, str]]


def check_fold_not_started(paths: ProjectPaths, fold: str) -> bool:
    """True if the fold is finished (skip it); False if nothing exists yet.

    Raises:
        FileExistsError: If part files exist without features.json: a
            half-finished fold. Nothing is deleted or reused automatically.
    """
    if paths.features_metadata_file(fold).exists():
        return True
    directory = paths.features_dir(fold)
    leftovers = sorted(directory.glob("part-*.parquet")) if directory.exists() else []
    if leftovers:
        raise FileExistsError(
            f"{directory} holds {len(leftovers)} part files but no features.json: a "
            "half-finished fold. Check the run log before removing anything."
        )
    return False


def _chunk_candidates(
    candidates: Mapping[str, pl.DataFrame], first_query: int, last_query: int
) -> dict[str, pl.DataFrame]:
    """Each channel's candidates of the queries first_query ... last_query (both included)."""
    in_chunk = pl.col(QUERY_ID).is_between(first_query, last_query)
    return {channel: table.filter(in_chunk) for channel, table in candidates.items()}


def build_fold_features(
    paths: ProjectPaths,
    inputs: FoldInputs,
    config: FeaturesConfig,
    query_chunk_size: int,
    stamp: RunStamp,
) -> BuildResult:
    """Build, check and write the wide table of one fold (see the module docstring).

    Args:
        paths: Project paths (where the part files go).
        inputs: Everything read for the fold.
        config: configs/features.yaml.
        query_chunk_size: Queries per part file (chunking.query_chunk_size).
        stamp: The code state of the run.

    Raises:
        FileExistsError: On a half-finished fold.
        ValueError: If a check fails. Part files written so far stay on disk
            WITHOUT features.json, so the fold counts as not finished.
    """
    if query_chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be > 0, got {query_chunk_size}")
    data, fold = inputs.data, inputs.data.fold
    n_queries = data.queries.height
    if check_fold_not_started(paths, fold):
        LOGGER.warning("%s: features already exist; SKIPPED (they are never replaced)", fold)
        return BuildResult(fold, True, 0, 0, n_queries, 0, 0, 0.0, 0.0, [])
    started = time.perf_counter()
    tables = build_fold_tables(data, config, inputs.history)
    save_or_check_category_codes(paths.category_codes_file, tables.codes)
    LOGGER.info("%s: per-fold tables built in %.1f s", fold, time.perf_counter() - started)
    query_ids = data.queries[QUERY_ID].unique().sort()
    stats = FeatureStats(config.features)
    parts: list[PartFile] = []
    timings: dict[str, float] = {}
    tt_checked, wals_checked = 0, 0
    for start in range(0, query_ids.len(), query_chunk_size):
        chunk_ids = query_ids.slice(start, query_chunk_size)
        # query_ids are sorted and unique, so a chunk is one range of ids and a
        # query is never split over two part files.
        candidates = _chunk_candidates(inputs.candidates, chunk_ids[0], chunk_ids[-1])
        wide = build_chunk(candidates, data, tables, inputs.dense, config, timings)
        check_no_repeated_pairs(wide)
        tt_checked += check_dot_equals_score(
            wide, candidates[TWOTOWER_CHANNEL], TWOTOWER_CHANNEL, TT_DOT
        )
        wals_checked += check_dot_equals_score(
            wide, candidates[WALS_CHANNEL], WALS_CHANNEL, WALS_DOT
        )
        part_file = paths.features_part_file(fold, len(parts))
        part_file.parent.mkdir(parents=True, exist_ok=True)
        wide.write_parquet(part_file)
        stats.update(wide)
        # Read back from disk: the recorded sha256 and row count are those of the file.
        parts.append(describe_part(part_file))
        LOGGER.info(
            "%s: %s, %d queries, %d rows", fold, part_file.name, chunk_ids.len(), wide.height
        )
    n_rows = sum(part.n_rows for part in parts)
    if n_rows != inputs.expected_rows:
        raise ValueError(
            f"{fold}: the wide table has {n_rows} rows, but results/m6_candidates.csv records "
            f"{inputs.expected_rows} deduplicated candidates"
        )
    result = BuildResult(
        fold=fold,
        skipped=False,
        n_rows=n_rows,
        n_parts=len(parts),
        n_queries=n_queries,
        tt_dot_rows_checked=tt_checked,
        wals_dot_rows_checked=wals_checked,
        seconds=time.perf_counter() - started,
        history_similarity_seconds=timings.get(HISTORY_SIMILARITY, 0.0),
        stats_rows=stats.rows(fold, stamp),
    )
    write_feature_files(
        paths.features_dir(fold), feature_files_record(result, parts, config, stamp)
    )
    return result


def feature_files_record(
    result: BuildResult, parts: list[PartFile], config: FeaturesConfig, stamp: RunStamp
) -> FeatureFiles:
    """The content of features.json for a finished fold (hmrec.feature_files)."""
    return FeatureFiles(
        fold=result.fold,
        columns=(*KEYS, *config.names),
        categorical_features=config.categorical_names,
        parts=tuple(parts),
        n_rows=result.n_rows,
        n_queries=result.n_queries,
        details={
            "tt_dot_rows_checked": result.tt_dot_rows_checked,
            "wals_dot_rows_checked": result.wals_dot_rows_checked,
            "seconds": result.seconds,
            "history_similarity_seconds": result.history_similarity_seconds,
            "git_commit": stamp.git_commit,
            "git_dirty": stamp.git_dirty,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
