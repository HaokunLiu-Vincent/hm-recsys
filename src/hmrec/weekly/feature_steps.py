"""The feature steps of a tick: O2 for a week (`eval`) or shard by shard (`serve`).

NO new computation happens here. The wide table is built by the locked feature
code, hmrec.features.build, as scripts/m6_build_features.py calls it; it gets the
tick's paths (hmrec.paths.PipelinePaths), so it reads the tick's ingested
transactions and the candidates of the SAME tick.

The two structural checks of docs/mlops.md §8.3 are made here for every tick:

    5  the feature rows equal the deduplicated candidates of the same run. The
       expected number is counted here from the candidate files, independently of
       the feature code, and compared with the rows read back from the files.
    6  no (query_id, article_idx) key repeats in the features (checked chunk by
       chunk while building, and once more on the files that were written).

A failed check stops the step: features.json is not there, the driver sets the
folder aside and the tick stops.

No label is read. This module imports neither torch nor lightgbm.
"""

import polars as pl

from hmrec.config import FeaturesConfig, load_features_config
from hmrec.data.metadata import scan_processed_parquet
from hmrec.features.base import CUSTOMER_IDX, KEYS, QUERY_ID, FoldData
from hmrec.features.build import (
    FoldInputs,
    build_fold_features,
    load_candidates,
    load_dense_sources,
    load_fold_data,
    load_history,
)
from hmrec.features.categories import build_category_codes, save_or_check_category_codes
from hmrec.locks import check_features_match_lock, load_features_lock, load_twotower_lock
from hmrec.logging_utils import get_logger
from hmrec.paths import PipelinePaths, ProjectPaths
from hmrec.run_manifest import MetricRecord
from hmrec.weekly.context import OutputFile, TickContext
from hmrec.weekly.shard_features import (
    FEATURE_STATS_FILE,
    ShardInputs,
    compute_shared_tables,
    count_distinct_candidates,
    read_shared_tables,
    write_shard_features,
    write_shared_tables,
    write_stats_rows,
)
from hmrec.weekly.shards import shard_queries
from hmrec.weekly.steps import Step

LOGGER = get_logger("weekly.feature_steps")

FEATURES_RECORD_FILE = "features.json"

StepOutcome = tuple[str, list[MetricRecord], list[OutputFile]]


def load_locked_features_config(paths: ProjectPaths) -> FeaturesConfig:
    """configs/features.yaml, after checking it against the features lock and the R6 lock.

    Raises:
        ValueError: If the config differs from configs/locks/features.lock.yaml, or
            its history length is not the one the locked R6 uses (proposal §8.3).
    """
    config = load_features_config(paths.features_config_file)
    check_features_match_lock(config, load_features_lock(paths.features_lock_file))
    locked_history = load_twotower_lock(paths.twotower_lock_file).channel.config["history_length"]
    if config.history_length != locked_history:
        raise ValueError(
            f"features.history_length is {config.history_length}, but the locked R6 uses "
            f"{locked_history}"
        )
    return config


def written_keys(paths: PipelinePaths, week: str) -> tuple[int, int]:
    """(rows, distinct (query_id, article_idx) pairs) of the feature files that were written.

    Read back from disk after the build, part by part, so the answer is about
    the files and not about tables in memory.
    """
    parts = sorted(paths.features_dir(week).glob("part-*.parquet"))
    keys = pl.concat([pl.read_parquet(part, columns=KEYS) for part in parts])
    return keys.height, keys.unique().height


def require_structure(week: str, written: int, distinct: int, expected: int) -> None:
    """The two structural checks of docs/mlops.md §8.3 (5 and 6).

    Raises:
        ValueError: If the feature rows are not the deduplicated candidates of
            the run, or a key repeats.
    """
    if written != expected:
        raise ValueError(
            f"{week}: structural check failed: {written} feature rows, but the run's own "
            f"candidates hold {expected} distinct (query_id, article_idx) pairs"
        )
    if distinct != written:
        raise ValueError(
            f"{week}: structural check failed: {written - distinct} repeated "
            "(query_id, article_idx) keys in the feature files"
        )


def structure_metrics(
    context: TickContext, written: int, distinct: int, expected: int, n_queries: int
) -> list[MetricRecord]:
    """The numbers of the structural checks and candidates per query, as manifest metrics."""
    anchor = context.tick.anchor
    return [
        MetricRecord("structure/feature_rows", anchor, float(written)),
        MetricRecord("structure/deduplicated_candidates", anchor, float(expected)),
        MetricRecord("structure/duplicate_keys", anchor, float(written - distinct)),
        MetricRecord("output/candidates_per_query", anchor, written / n_queries),
    ]


# ---------------------------------------------------------------- eval: one week


def run_features(context: TickContext) -> StepOutcome:
    """O2 in `eval` mode: the 71 features of the week from this tick's candidates.

    Raises:
        ValueError: If configs/features.yaml is not the locked feature list, a
            structural check fails, or tt_dot / wals_dot differ from the scores.
    """
    view, week = context.view, context.tick
    config = load_locked_features_config(context.paths)
    data = load_fold_data(view, context.preprocess, week.fold)
    history = load_history(data, config)
    candidates = load_candidates(view, week.name)
    expected = count_distinct_candidates(candidates)
    inputs = FoldInputs(
        data=data,
        history=history,
        candidates=candidates,
        dense=load_dense_sources(view, context.preprocess, data, history, config, context.base),
        expected_rows=expected,
    )
    result = build_fold_features(
        view, inputs, config, context.base.chunking.query_chunk_size, context.stamp
    )
    write_stats_rows(view.feature_stats_file(week.name), result.stats_rows)
    written, distinct = written_keys(view, week.name)
    require_structure(week.name, written, distinct, expected)
    detail = (
        f"{written} rows in {result.n_parts} parts for {result.n_queries} queries = the "
        f"deduplicated candidates of this run; 0 repeated keys; tt_dot checked on "
        f"{result.tt_dot_rows_checked} rows, wals_dot on {result.wals_dot_rows_checked}"
    )
    LOGGER.info("O2: %s", detail)
    files = [
        OutputFile("features_record", FEATURES_RECORD_FILE),
        OutputFile("feature_stats", FEATURE_STATS_FILE),
    ]
    metrics = structure_metrics(context, written, distinct, expected, result.n_queries)
    return detail, metrics, files


# ---------------------------------------------------------------- serve: shared tables, shards


def load_week_data(context: TickContext, queries: pl.DataFrame) -> FoldData:
    """The processed tables and the given queries as the feature code's FoldData."""
    view, preprocess = context.view, context.preprocess
    metadata = view.processed_metadata_file
    return FoldData(
        fold=context.tick.name,
        anchor=context.tick.anchor,
        day_zero=preprocess.day_zero,
        transactions=scan_processed_parquet(view.transactions_file, metadata, preprocess),
        customers=scan_processed_parquet(view.customers_file, metadata, preprocess).collect(),
        articles=scan_processed_parquet(view.articles_file, metadata, preprocess).collect(),
        queries=queries.select(QUERY_ID, CUSTOMER_IDX),
    )


def run_features_shared(context: TickContext) -> StepOutcome:
    """O2 in `serve` mode, once per week: the tables over all customers.

    Also checks the category code table against the one the Ranker's training
    data was encoded with (artifacts/features/category_codes.json).
    """
    view, week = context.view, context.tick.name
    config = load_locked_features_config(context.paths)
    data = load_week_data(context, pl.read_parquet(view.queries_file(week)))
    codes = build_category_codes(data.articles, data.customers)
    save_or_check_category_codes(view.category_codes_file, codes)
    tables = compute_shared_tables(data, config, codes)
    write_shared_tables(tables, view.item_table_file(week), view.type_share_file(week))
    detail = (
        f"article table {tables.item.height} rows, global type shares "
        f"{tables.global_type_share.height} rows"
    )
    files = [
        OutputFile("features_shared/item_table", view.item_table_file(week).name),
        OutputFile("features_shared/global_type_share", view.type_share_file(week).name),
    ]
    return detail, [], files


def run_shard_features(context: TickContext, step: Step) -> StepOutcome:
    """O2 in `serve` mode: the features of one shard of customers.

    Raises:
        ValueError: If the step has no shard, or a check fails.
    """
    if step.shard is None:
        raise ValueError(f"step {step.name} has no shard")
    week = context.tick.name
    shard_view = context.view.for_shard(step.shard)
    config = load_locked_features_config(context.paths)
    all_queries = pl.read_parquet(context.view.queries_file(week))
    queries = shard_queries(all_queries, step.shard, context.pipeline.shard_customers)
    data = load_week_data(context, queries)
    history = load_history(data, config)
    inputs = ShardInputs(
        data=data,
        history=history,
        candidates=load_candidates(shard_view, week),
        dense=load_dense_sources(
            shard_view, context.preprocess, data, history, config, context.base
        ),
    )
    shared = read_shared_tables(
        context.view.item_table_file(week), context.view.type_share_file(week)
    )
    result = write_shard_features(
        inputs,
        shared,
        config,
        context.base.chunking.query_chunk_size,
        shard_view.features_dir(week),
        context.stamp,
        {"week": week, "shard": step.shard},
    )
    written, distinct = written_keys(shard_view, week)
    require_structure(f"{week} shard {step.shard}", written, distinct, result.expected_rows)
    detail = (
        f"shard {step.shard}: {written} rows in {result.n_parts} parts for {result.n_queries} "
        "queries = the deduplicated candidates of the shard; 0 repeated keys"
    )
    LOGGER.info("O2: %s", detail)
    # No file is recorded in the run manifest: a shard's features are removed after
    # the shard was scored (docs/mlops.md §8.4). The scoring step keeps the two small
    # statistics files (QUERY_STATS_FILE, FEATURE_STATS_FILE).
    return detail, [], []
