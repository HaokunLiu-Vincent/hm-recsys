"""Recall in `serve` mode: fit every channel ONCE, then write candidates shard by shard.

A `serve` tick predicts one week for every customer (1.37 million), cut into
shards of customers (hmrec.weekly.shards). The channels come only from the lock
files (hmrec.recall.locked). One process handles one channel:

    fitted = fit_channel(job, channel, load_chain)   # once, with all history before the anchor
    for every shard:  fitted.generate(queries of the shard, shard)

What is fitted once, and what a shard does:

    R1  the bucket lists (Top-quota of S(g, i) plus the global fill-in) and the
        full S(g, i) table; a shard joins its customers' age bucket to the lists.
    R2  nothing: a customer's candidates depend only on that customer's own
        purchases. A shard calls the channel's generate() unchanged.
    R3  nothing is kept: a shard calls generate() unchanged. Its market count
        M(j) is an integer count over ALL customers, so every shard computes
        exactly the same numbers.
    R4  the item graph (C_w, n_w) and the neighbour tables of every seed of the
        week's customers; a shard scores its own seeds against them.
    R5  the ALS fit (user and article vectors); a shard takes its Top-quota.
    R6  the vectors of every customer and of every article, from the chain model
        the caller loaded (the model after last week's fine-tune, step C4); a
        shard takes its Top-quota.

The side files the feature code reads are written and checked as in M6: R1's
S(g, i) table (one per week), the R5 and R6 vectors (one folder per shard).

Brought into main from branch kaggle-late-submission (hmrec.kaggle.recall)
without its Kaggle-specific parts: the R6 fine-tune is step C4 of the tick, and
a step's folder is new, so nothing is skipped or resumed inside a channel.

Files are never replaced. No label is read here. This module is on the torch side.
"""

from collections.abc import Callable
from dataclasses import dataclass

import polars as pl

from hmrec.config import BaseConfig, PopularityConfig, PreprocessConfig
from hmrec.data.asof import history_window
from hmrec.data.folds import Fold
from hmrec.locks import ALL_CHANNELS, R6_CHANNEL
from hmrec.logging_utils import get_logger
from hmrec.paths import PipelinePaths
from hmrec.recall.base import (
    ARTICLE_IDX,
    CUSTOMER_IDX,
    QUERY_ID,
    SCORE,
    OutputSpec,
    RecallInputs,
    add_age_days,
    rank_candidates,
    validate_candidates,
    write_candidates,
)
from hmrec.recall.generate import (
    R1_CHANNEL,
    R5_CHANNEL,
    TWOTOWER_SOURCE,
    WALS_SOURCE,
    check_scores_match_bucket_table,
    check_scores_match_vectors,
    write_bucket_scores,
)
from hmrec.recall.locked import LockedRecall, build_locked_channel
from hmrec.recall.r1_popularity import (
    AGE_BUCKET,
    IS_FALLBACK,
    POP_SPEC,
    PopularityChannel,
    bucket_decayed_scores,
    build_bucket_lists,
    global_decayed_scores,
)
from hmrec.recall.r4_itemcf import ItemCFChannel, ItemCFNeighbors
from hmrec.recall.r5_wals import WalsChannel
from hmrec.recall.twotower.chain import TwoTowerChainChannel
from hmrec.recall.twotower.infer import (
    TWOTOWER_SPEC,
    add_other_catalog_articles,
    candidates_from_query_vectors,
    query_vectors,
)
from hmrec.vectors import load_vectors, save_vectors, select_users
from hmrec.weekly.shards import number_of_shards, shard_queries

LOGGER = get_logger("weekly.shard_recall")

R2_CHANNEL = "r2_repurchase"
R3_CHANNEL = "r3_sibling"
R4_CHANNEL = "r4_itemcf"

# (queries of one shard, shard number) -> (candidates, rows covered by the consistency check).
ShardGenerator = Callable[[pl.DataFrame, int], tuple[pl.DataFrame, int]]


@dataclass(frozen=True)
class RecallJob:
    """Everything the sharded recall of one week reads.

    Attributes:
        paths: The tick's paths (hmrec.paths.PipelinePaths, no shard set).
        base: configs/base.yaml (chunk sizes).
        preprocess: Current preprocess config.
        locked: Both recall lock files.
        week: The predicted week.
        shard_customers: Customers per shard.
        queries: The week's query list: one row per customer, sorted by query_id.
        inputs: The channel inputs as of the week's anchor (with A(a)).
    """

    paths: PipelinePaths
    base: BaseConfig
    preprocess: PreprocessConfig
    locked: LockedRecall
    week: Fold
    shard_customers: int
    queries: pl.DataFrame
    inputs: RecallInputs


@dataclass(frozen=True)
class FittedChannel:
    """A channel after its once-per-week work, ready to generate shards.

    Attributes:
        channel: Channel name, e.g. "r4_itemcf".
        spec: The channel's output declaration.
        quota: Max candidates per query.
        generate: Produces one shard's candidates (and writes the shard's side files).
    """

    channel: str
    spec: OutputSpec
    quota: int
    generate: ShardGenerator


@dataclass(frozen=True)
class ChannelResult:
    """What generating one channel for a week did.

    Attributes:
        channel: Channel name.
        n_shards: Shards of the week.
        n_rows: Candidate rows of all shards of the channel.
        checked_rows: Rows covered by the consistency checks.
    """

    channel: str
    n_shards: int
    n_rows: int
    checked_rows: int


# ---------------------------------------------------------------- R1


def popularity_lists(
    config: PopularityConfig,
    history: pl.DataFrame,
    customers: pl.DataFrame,
    active_pool: pl.DataFrame,
    buckets: list[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """R1 (variant A1), fitted once: the list of every age bucket and the full S(g, i) table.

    The same steps as PopularityChannel.generate_with_bucket_scores, but for all
    given buckets at once and without joining any query: S(g, i) is computed ONCE
    and used for both outputs, so a non-fallback candidate's score equals its
    table entry exactly.

    Args:
        config: The locked R1 config (variant "A1").
        history: Purchase lines of [anchor - window_days, anchor) with age_days;
            columns customer_idx, article_idx, age_days.
        customers: customer_idx -> age_bucket.
        active_pool: A(a); column article_idx.
        buckets: The age buckets that need a list.

    Returns:
        (lists, table). lists: one row per (age_bucket, article_idx) of a
        bucket's list; columns age_bucket, article_idx, score (float32),
        pop_is_fallback (int8). table: one row per (age_bucket, article_idx)
        bought by the bucket in the window, NOT limited to A(a) or to the quota;
        columns age_bucket (str), article_idx (int32), score (float32), sorted
        by both keys.

    Raises:
        ValueError: If the variant is not "A1".
    """
    if config.variant != "A1":
        raise ValueError(f"R1 variant {config.variant!r} has no bucket lists (A1 only)")
    pool = active_pool.select(ARTICLE_IDX)
    all_bucket_scores = bucket_decayed_scores(history, customers, config.tau_days)
    # Only articles of the active pool A(a) can be candidates (proposal §2.4).
    bucket_scores = all_bucket_scores.join(pool, on=ARTICLE_IDX, how="semi")
    global_scores = global_decayed_scores(history, config.tau_days).join(
        pool, on=ARTICLE_IDX, how="semi"
    )
    lists = build_bucket_lists(bucket_scores, global_scores, buckets, config.quota)
    table = all_bucket_scores.select(
        pl.col(AGE_BUCKET).cast(pl.String),
        pl.col(ARTICLE_IDX).cast(pl.Int32),
        # The same float64 -> float32 cast as the candidates' score.
        pl.col(SCORE).cast(pl.Float32),
    ).sort(AGE_BUCKET, ARTICLE_IDX)
    return lists, table


def query_age_buckets(queries: pl.DataFrame, customers: pl.DataFrame) -> pl.DataFrame:
    """query_id, customer_idx and the customer's age_bucket; fails if a bucket is missing."""
    with_bucket = queries.select(QUERY_ID, CUSTOMER_IDX).join(
        customers.select(CUSTOMER_IDX, AGE_BUCKET), on=CUSTOMER_IDX, how="left"
    )
    if with_bucket[AGE_BUCKET].null_count() > 0:
        raise ValueError("some query customers have no age_bucket in the customers table")
    return with_bucket


def popularity_candidates(
    queries: pl.DataFrame, customers: pl.DataFrame, lists: pl.DataFrame, quota: int
) -> pl.DataFrame:
    """R1 candidates of some queries from the fitted bucket lists.

    Every query gets the list of its customer's age bucket, ordered by
    (pop_is_fallback ASC, score DESC, article_idx ASC).

    Returns:
        Candidates valid for POP_SPEC.
    """
    scored = query_age_buckets(queries, customers).join(lists, on=AGE_BUCKET)
    return rank_candidates(
        scored.select(QUERY_ID, ARTICLE_IDX, SCORE, IS_FALLBACK), POP_SPEC, quota
    )


def fit_popularity(job: RecallJob) -> FittedChannel:
    """R1: fit the bucket lists once and save the S(g, i) table of the week."""
    inputs = job.inputs
    channel = build_locked_channel(R1_CHANNEL, job.locked, job.base, job.paths, job.preprocess)
    if not isinstance(channel, PopularityChannel):
        raise TypeError(f"{R1_CHANNEL} is not a PopularityChannel")
    config = channel.config
    lines = history_window(
        inputs.transactions,
        inputs.anchor,
        inputs.day_zero,
        config.window_days,
        columns=[CUSTOMER_IDX, ARTICLE_IDX],
    )
    history = add_age_days(lines, inputs.anchor_day)
    buckets = query_age_buckets(job.queries, inputs.customers)[AGE_BUCKET].unique().sort()
    lists, table = popularity_lists(
        config, history, inputs.customers, inputs.active_pool, buckets.to_list()
    )
    table_file = job.paths.r1_bucket_scores_file(job.week.name)
    write_bucket_scores(table, table_file)
    # Read back from disk: the feature code reads this file.
    saved = pl.read_parquet(table_file)
    LOGGER.info(
        "%s: %d list rows, table %d rows -> %s", R1_CHANNEL, lists.height, saved.height, table_file
    )

    def generate(queries: pl.DataFrame, shard: int) -> tuple[pl.DataFrame, int]:
        candidates = popularity_candidates(queries, inputs.customers, lists, config.quota)
        label = f"{job.week.name} shard {shard} {R1_CHANNEL}"
        checked = check_scores_match_bucket_table(
            candidates, queries, inputs.customers, saved, label
        )
        return candidates, checked

    return FittedChannel(R1_CHANNEL, POP_SPEC, config.quota, generate)


# ---------------------------------------------------------------- R2, R3


def fit_per_customer_channel(job: RecallJob, channel_name: str) -> FittedChannel:
    """R2 / R3: nothing to fit; a shard calls the locked channel's generate() unchanged."""
    channel = build_locked_channel(channel_name, job.locked, job.base, job.paths, job.preprocess)

    def generate(queries: pl.DataFrame, shard: int) -> tuple[pl.DataFrame, int]:
        return channel.generate(queries, job.inputs), 0

    quota = job.locked.channel(channel_name).quota
    return FittedChannel(channel_name, channel.output_spec, quota, generate)


# ---------------------------------------------------------------- R4


def fit_itemcf(job: RecallJob) -> FittedChannel:
    """R4: build the item graph and the neighbour tables once, for all seeds of the week."""
    channel = build_locked_channel(R4_CHANNEL, job.locked, job.base, job.paths, job.preprocess)
    if not isinstance(channel, ItemCFChannel):
        raise TypeError(f"{R4_CHANNEL} is not an ItemCFChannel")
    alpha = channel.config.alpha
    # The neighbour table of a seed does not depend on which other seeds are
    # processed with it (proposal §5 R4), so one pass over all customers' seeds
    # gives every shard the tables it would have built alone.
    neighbors = channel.build_neighbors(job.queries, job.inputs, [alpha])
    LOGGER.info(
        "%s: %d seed articles, %d (query, seed) rows",
        R4_CHANNEL,
        neighbors.seed_items.size,
        neighbors.query_seeds.height,
    )

    def generate(queries: pl.DataFrame, shard: int) -> tuple[pl.DataFrame, int]:
        first, last = queries[QUERY_ID][0], queries[QUERY_ID][-1]
        shard_seeds = neighbors.query_seeds.filter(pl.col(QUERY_ID).is_between(first, last))
        shard_neighbors = ItemCFNeighbors(
            shard_seeds, neighbors.seed_items, neighbors.tables, neighbors.n_articles
        )
        return channel.candidates_from_neighbors(shard_neighbors, alpha), 0

    return FittedChannel(R4_CHANNEL, channel.output_spec, channel.config.quota, generate)


# ---------------------------------------------------------------- R5


def fit_wals(job: RecallJob) -> FittedChannel:
    """R5: fit ALS once; a shard takes its Top-quota and saves its customers' vectors."""
    inputs, week = job.inputs, job.week.name
    channel = build_locked_channel(R5_CHANNEL, job.locked, job.base, job.paths, job.preprocess)
    if not isinstance(channel, WalsChannel):
        raise TypeError(f"{R5_CHANNEL} is not a WalsChannel")
    vectors = channel.fit(inputs)

    def generate(queries: pl.DataFrame, shard: int) -> tuple[pl.DataFrame, int]:
        candidates = channel.candidates_from_vectors(queries, inputs, vectors)
        validate_candidates(candidates, channel.output_spec, channel.config.quota)
        directory = job.paths.for_shard(shard).wals_vectors_dir(week)
        # Users: only the shard's customers. Articles: every article that has a vector.
        shard_customers = queries[CUSTOMER_IDX].unique().to_numpy()
        save_vectors(vectors, directory, WALS_SOURCE, week, shard_customers, overwrite=False)
        saved = load_vectors(directory, WALS_SOURCE, week)
        label = f"{week} shard {shard} {R5_CHANNEL}"
        pair_chunk_size = job.base.chunking.pair_chunk_size
        checked = check_scores_match_vectors(candidates, queries, saved, pair_chunk_size, label)
        return candidates, checked

    return FittedChannel(R5_CHANNEL, channel.output_spec, channel.config.quota, generate)


# ---------------------------------------------------------------- R6


def fit_twotower(job: RecallJob, chain: TwoTowerChainChannel) -> FittedChannel:
    """R6: encode every customer and every article ONCE; a shard takes its Top-quota.

    The vectors are computed with the functions M6 used for a fold (query_vectors,
    add_other_catalog_articles), called once with all queries of the week. A
    shard then does what predict_twotower_fold does after that: Top-quota of
    A(a) from the vectors, save the shard's vectors, check every score against
    the saved vectors.

    Why the customers are not encoded shard by shard: the model's forward pass
    is a matrix product, and a customer encoded in a very small batch (1 to 3
    customers) can differ in the last float32 bit from the same customer in a
    larger batch. Encoding once, in the same batches whatever the shards are,
    makes the vectors (and so the candidates) independent of the sharding.

    Args:
        job: The week's recall inputs.
        chain: The chain channel with the model that predicts the week loaded
            from its checked file (hmrec.weekly.recall_steps.load_chain).
    """
    inputs, week = job.inputs, job.week.name
    config, resources, chunks = chain.config, chain.resources, chain.chunks
    if job.queries.height == 0 or inputs.active_pool.height == 0:
        raise ValueError(f"{week}: no queries or an empty active pool; nothing to predict")
    pool_vectors = query_vectors(config, resources, job.queries, inputs, chunks.encode_queries)
    # Item vectors of EVERY catalog article: tt_dot is computed for every candidate,
    # and a candidate of another channel (R2) can be outside A(a).
    vectors = add_other_catalog_articles(pool_vectors, resources, chunks.encode_queries)

    def generate(queries: pl.DataFrame, shard: int) -> tuple[pl.DataFrame, int]:
        shard_customers = queries[CUSTOMER_IDX].unique().to_numpy()
        shard_vectors = select_users(vectors, shard_customers)
        candidates = candidates_from_query_vectors(config, queries, inputs, shard_vectors, chunks)
        validate_candidates(candidates, TWOTOWER_SPEC, config.quota)
        directory = job.paths.for_shard(shard).twotower_vectors_dir(week)
        save_vectors(shard_vectors, directory, TWOTOWER_SOURCE, week, None, overwrite=False)
        saved = load_vectors(directory, TWOTOWER_SOURCE, week)
        label = f"{week} shard {shard} {R6_CHANNEL}"
        checked = check_scores_match_vectors(candidates, queries, saved, chunks.pairs, label)
        return candidates, checked

    return FittedChannel(R6_CHANNEL, TWOTOWER_SPEC, config.quota, generate)


# ---------------------------------------------------------------- one channel, all shards


def fit_channel(
    job: RecallJob, channel: str, load_chain: Callable[[], TwoTowerChainChannel]
) -> FittedChannel:
    """Do the once-per-week work of a channel.

    Args:
        job: The week's recall inputs.
        channel: One of the six channel names.
        load_chain: Loads the R6 chain model that predicts the week; called for R6 only.

    Raises:
        ValueError: If the channel name is unknown.
    """
    if channel == R1_CHANNEL:
        return fit_popularity(job)
    if channel in (R2_CHANNEL, R3_CHANNEL):
        return fit_per_customer_channel(job, channel)
    if channel == R4_CHANNEL:
        return fit_itemcf(job)
    if channel == R5_CHANNEL:
        return fit_wals(job)
    if channel == R6_CHANNEL:
        return fit_twotower(job, load_chain())
    raise ValueError(f"unknown channel {channel!r}; expected one of {ALL_CHANNELS}")


def generate_channel(
    job: RecallJob, channel: str, load_chain: Callable[[], TwoTowerChainChannel]
) -> ChannelResult:
    """Fit one channel once and write the candidates of every shard.

    Args:
        job: The week's recall inputs.
        channel: One of the six channel names.
        load_chain: Loads the R6 chain model that predicts the week (R6 only).

    Raises:
        FileExistsError: If a shard's candidates or vectors exist (never replaced).
        ValueError: If an output breaks the schema or a consistency check fails.
    """
    n_shards = number_of_shards(job.queries.height, job.shard_customers)
    fitted = fit_channel(job, channel, load_chain)
    n_rows, checked_rows = 0, 0
    for shard in range(n_shards):
        queries = shard_queries(job.queries, shard, job.shard_customers)
        candidates, checked = fitted.generate(queries, shard)
        target = job.paths.for_shard(shard).candidates_file(job.week.name, channel)
        # write_candidates validates the schema and refuses to replace a file.
        write_candidates(candidates, fitted.spec, fitted.quota, target, False)
        n_rows += candidates.height
        checked_rows += checked
        LOGGER.info(
            "%s %s shard %d/%d: %d queries, %d rows -> %s",
            job.week.name,
            channel,
            shard,
            n_shards,
            queries.height,
            candidates.height,
            target,
        )
    return ChannelResult(channel, n_shards, n_rows, checked_rows)
