"""Recall for a Kaggle run: fit every channel ONCE, then write candidates shard by shard.

A run predicts one week for many customers (the real run: 1.37 million), cut into
shards of customers (hmrec.kaggle.plan). The channels come only from the lock
files (hmrec.recall.locked). One process handles one channel:

    fitted = fit_channel(job, channel, stamp)       # once, with all history before the anchor
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
        run's customers; a shard scores its own seeds against them.
    R5  the ALS fit (user and article vectors); a shard takes its Top-quota.
    R6  the model (for the real run: the locked chain's last model, fine-tuned
        once more on the last fold's week) and the vectors of every customer of
        the run and of every article; a shard takes its Top-quota.

The side files that the feature code reads are written as in M6 and checked the
same way before the candidates: R1's S(g, i) table (one per run), the R5 and R6
vectors (one folder per shard).

Files are never replaced. A shard whose candidates file exists is skipped, so an
interrupted process can be started again. No label is read here.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from hmrec.config import PopularityConfig, load_candidate_run_config, load_catalog_config
from hmrec.data.asof import history_window
from hmrec.kaggle.context import KaggleContext
from hmrec.kaggle.plan import fine_tune_fold, number_of_shards, read_queries, shard_queries
from hmrec.locks import ALL_CHANNELS, F1_START_CHECKPOINT, R6_CHANNEL
from hmrec.logging_utils import get_logger
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
from hmrec.recall.chain_records import (
    check_model_and_record_agree,
    read_chain_records,
    start_model_for,
)
from hmrec.recall.generate import (
    R1_CHANNEL,
    R5_CHANNEL,
    WALS_SOURCE,
    check_fine_tune_loss,
    check_scores_match_bucket_table,
    check_scores_match_vectors,
    fine_tune_twotower_fold,
    require_not_started,
    write_bucket_scores,
)
from hmrec.recall.inputs import SharedTables, build_recall_inputs, load_shared_tables
from hmrec.recall.locked import (
    LockedRecall,
    ModelFile,
    build_locked_channel,
    load_locked_recall,
    load_locked_twotower_chain,
    locked_checkpoint_file,
    post_lock_folds,
)
from hmrec.recall.r1_popularity import (
    AGE_BUCKET,
    BUCKET_SCORES_SCHEMA,
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
from hmrec.run_records import RunStamp
from hmrec.vectors import METADATA_FILE, load_vectors, save_vectors, select_users

LOGGER = get_logger("kaggle.recall")

TWOTOWER_VECTORS = "twotower"

# (queries of one shard, shard number) -> (candidates, rows covered by the consistency check).
ShardGenerator = Callable[[pl.DataFrame, int], tuple[pl.DataFrame, int]]


@dataclass(frozen=True)
class RecallJob:
    """Everything the recall of one run reads.

    Attributes:
        context: The process's context (run, paths, predicted week).
        locked: Both recall lock files.
        tables: The processed tables (transactions lazy).
        queries: The run's query list: one row per customer, sorted by query_id.
        inputs: The channel inputs as of the predicted week's anchor (with A(a)).
    """

    context: KaggleContext
    locked: LockedRecall
    tables: SharedTables
    queries: pl.DataFrame
    inputs: RecallInputs


@dataclass(frozen=True)
class FittedChannel:
    """A channel after its once-per-run work, ready to generate shards.

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
    """What generating one channel for a run did.

    Attributes:
        channel: Channel name.
        n_shards: Shards of the run.
        n_generated: Shards written by this process (the others existed already).
        n_rows: Candidate rows of ALL shards of the channel.
        checked_rows: Rows covered by the consistency checks of this process.
        detail: Short free text (R6: the fine-tune).
    """

    channel: str
    n_shards: int
    n_generated: int
    n_rows: int
    checked_rows: int
    detail: str


def load_recall_job(context: KaggleContext) -> RecallJob:
    """Read the lock files, the tables and the run's queries; build the inputs as of the anchor."""
    paths = context.paths
    tables = load_shared_tables(paths, context.preprocess)
    catalog = load_catalog_config(paths.catalog_config_file)
    return RecallJob(
        context=context,
        locked=load_locked_recall(paths),
        tables=tables,
        queries=read_queries(paths.kaggle_queries_file(context.run)),
        inputs=build_recall_inputs(tables, context.target, context.preprocess, catalog),
    )


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
        columns as BUCKET_SCORES_SCHEMA, sorted by both keys.

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


def save_or_check_bucket_table(table: pl.DataFrame, path: Path) -> pl.DataFrame:
    """Write the run's S(g, i) table once; an existing file must equal the new table.

    Returns:
        The table as read back from disk (the feature code reads this file).

    Raises:
        ValueError: If the existing file holds another table (a resumed process
            must compute exactly what the first one wrote).
    """
    if not path.exists():
        write_bucket_scores(table, path)
    saved = pl.read_parquet(path)
    if saved.schema != BUCKET_SCORES_SCHEMA or not saved.equals(table):
        raise ValueError(f"{path} holds another S(g, i) table than this process computed")
    return saved


def fit_popularity(job: RecallJob) -> FittedChannel:
    """R1: fit the bucket lists once and save the S(g, i) table of the run."""
    context, inputs = job.context, job.inputs
    channel = build_locked_channel(
        R1_CHANNEL, job.locked, context.base, context.paths, context.preprocess
    )
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
    table_file = context.paths.kaggle_r1_bucket_scores_file(context.run)
    saved = save_or_check_bucket_table(table, table_file)
    LOGGER.info(
        "%s: %d list rows, table %d rows -> %s", R1_CHANNEL, lists.height, saved.height, table_file
    )

    def generate(queries: pl.DataFrame, shard: int) -> tuple[pl.DataFrame, int]:
        candidates = popularity_candidates(queries, inputs.customers, lists, config.quota)
        label = f"{context.run} shard {shard} {R1_CHANNEL}"
        checked = check_scores_match_bucket_table(
            candidates, queries, inputs.customers, saved, label
        )
        return candidates, checked

    return FittedChannel(R1_CHANNEL, POP_SPEC, config.quota, generate)


# ---------------------------------------------------------------- R2, R3


def fit_per_customer_channel(job: RecallJob, channel_name: str) -> FittedChannel:
    """R2 / R3: nothing to fit; a shard calls the locked channel's generate() unchanged."""
    context = job.context
    channel = build_locked_channel(
        channel_name, job.locked, context.base, context.paths, context.preprocess
    )

    def generate(queries: pl.DataFrame, shard: int) -> tuple[pl.DataFrame, int]:
        return channel.generate(queries, job.inputs), 0

    quota = job.locked.channel(channel_name).quota
    return FittedChannel(channel_name, channel.output_spec, quota, generate)


# ---------------------------------------------------------------- R4


def fit_itemcf(job: RecallJob) -> FittedChannel:
    """R4: build the item graph and the neighbour tables once, for all seeds of the run."""
    context = job.context
    channel_name = "r4_itemcf"
    channel = build_locked_channel(
        channel_name, job.locked, context.base, context.paths, context.preprocess
    )
    if not isinstance(channel, ItemCFChannel):
        raise TypeError(f"{channel_name} is not an ItemCFChannel")
    alpha = channel.config.alpha
    # The neighbour table of a seed does not depend on which other seeds are
    # processed with it (proposal §5 R4), so one pass over all customers' seeds
    # gives every shard the tables it would have built alone.
    neighbors = channel.build_neighbors(job.queries, job.inputs, [alpha])
    LOGGER.info(
        "%s: %d seed articles, %d (query, seed) rows",
        channel_name,
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

    return FittedChannel(channel_name, channel.output_spec, channel.config.quota, generate)


# ---------------------------------------------------------------- R5


def fit_wals(job: RecallJob) -> FittedChannel:
    """R5: fit ALS once; a shard takes its Top-quota and saves its customers' vectors."""
    context, inputs = job.context, job.inputs
    channel = build_locked_channel(
        R5_CHANNEL, job.locked, context.base, context.paths, context.preprocess
    )
    if not isinstance(channel, WalsChannel):
        raise TypeError(f"{R5_CHANNEL} is not a WalsChannel")
    vectors = channel.fit(inputs)
    week = context.target.name

    def generate(queries: pl.DataFrame, shard: int) -> tuple[pl.DataFrame, int]:
        candidates = channel.candidates_from_vectors(queries, inputs, vectors)
        validate_candidates(candidates, channel.output_spec, channel.config.quota)
        directory = context.paths.kaggle_vectors_dir(context.run, shard, WALS_SOURCE)
        # Users: only the shard's customers. Articles: every article that has a vector.
        shard_customers = queries[CUSTOMER_IDX].unique().to_numpy()
        save_vectors(vectors, directory, WALS_SOURCE, week, shard_customers, overwrite=False)
        saved = load_vectors(directory, WALS_SOURCE, week)
        label = f"{context.run} shard {shard} {R5_CHANNEL}"
        pair_chunk_size = context.base.chunking.pair_chunk_size
        checked = check_scores_match_vectors(candidates, queries, saved, pair_chunk_size, label)
        return candidates, checked

    return FittedChannel(R5_CHANNEL, channel.output_spec, channel.config.quota, generate)


# ---------------------------------------------------------------- R6


def chain_start_model(job: RecallJob) -> tuple[ModelFile, str]:
    """The locked chain's LAST model (the one that predicts the last fold) and that fold's name.

    Found and checked with hmrec.recall.chain_records.start_model_for, the
    function M6 used: the model fine-tuned on the fold before the last one
    (chain_locked/T3.pt) must exist with the sha256 of results/m6_twotower_chain.csv.
    """
    paths, lock = job.context.paths, job.locked.twotower
    fold_names = tuple(fold.name for fold in post_lock_folds(job.context.manifest))
    last_fold = fold_names[-1]
    f1_start = ModelFile(
        locked_checkpoint_file(paths, lock, F1_START_CHECKPOINT),
        lock.checkpoints[F1_START_CHECKPOINT].sha256,
    )
    records = read_chain_records(paths.m6_twotower_chain_results_file)
    start = start_model_for(
        last_fold, fold_names, f1_start, records, paths.twotower_locked_chain_checkpoint_file
    )
    return start, last_fold


def prepare_twotower_chain(job: RecallJob, stamp: RunStamp) -> tuple[TwoTowerChainChannel, str]:
    """The R6 model that predicts the run's week, loaded from a checked file.

    rehearsal:  the locked chain's last model, as it is (it predicted T4 in M6).
    submission: that model fine-tuned ONCE more, on the last fold's week, with the
                locked settings (hmrec.recall.generate.fine_tune_twotower_fold, the
                weekly fine-tune of M6). The new model is saved under
                artifacts/kaggle/submission/twotower/ and its sha256 is recorded in
                results/kaggle_twotower.csv. If it exists already (a resumed
                process), it is checked against that record and loaded; it is
                never trained twice. The fine-tune loss is checked like in M6.

    Returns:
        (the chain channel with the model loaded, a short description for the run row).
    """
    context = job.context
    paths, base, preprocess = context.paths, context.base, context.preprocess
    start, last_fold = chain_start_model(job)
    week = fine_tune_fold(context.run, context.manifest)
    if week is None:
        chain = load_locked_twotower_chain(job.locked, base, paths, preprocess, start)
        return chain, f"model {start.path.name} of the locked chain, no fine-tune"
    if week.name != last_fold:
        raise ValueError(f"the fine-tune week {week.name} is not the chain's last fold {last_fold}")

    def checkpoint_file(fold: str) -> Path:
        return paths.kaggle_twotower_checkpoint_file(context.run, fold)

    records_file = paths.kaggle_twotower_results_file
    done = check_model_and_record_agree(
        week.name, read_chain_records(records_file), checkpoint_file
    )
    if not done:
        chain = load_locked_twotower_chain(
            job.locked, base, paths, preprocess, start, checkpoint_file
        )
        catalog = load_catalog_config(paths.catalog_config_file)
        week_inputs = build_recall_inputs(job.tables, week, preprocess, catalog)
        record = fine_tune_twotower_fold(chain, week, week_inputs, start, stamp, records_file)
        max_loss = load_candidate_run_config(paths.m6_config_file).max_fine_tune_loss
        check_fine_tune_loss(record, max_loss)
    record = read_chain_records(records_file)[week.name]
    # Like every fold of M6, the week is predicted by a model LOADED from its saved
    # file (sha256 checked), not by the model object that was just trained.
    model = ModelFile(checkpoint_file(week.name), record.sha256)
    chain = load_locked_twotower_chain(job.locked, base, paths, preprocess, model, checkpoint_file)
    detail = (
        f"fine-tuned on {week.name}: {record.n_new_rows} new + {record.n_replay_rows} replay "
        f"rows, train loss {record.mean_train_loss:.4f}, alarm loss {record.alarm_loss:.4f}, "
        f"{record.fine_tune_seconds:.1f} s, sha256 {record.sha256}"
    )
    return chain, detail


def fit_twotower(job: RecallJob, chain: TwoTowerChainChannel) -> FittedChannel:
    """R6: encode every customer and every article ONCE; a shard takes its Top-quota.

    The vectors are computed with the functions M6 used for a fold (query_vectors,
    add_other_catalog_articles), called once with all queries of the run. A
    shard then does what predict_twotower_fold does after that: Top-quota of
    A(a) from the vectors, save the shard's vectors, check every score against
    the saved vectors.

    Why the customers are not encoded shard by shard: the model's forward pass
    is a matrix product, and a customer encoded in a very small batch (1 to 3
    customers) can differ in the last float32 bit from the same customer in a
    larger batch. Encoding once, in the same batches whatever the shards are,
    makes the vectors (and so the candidates) independent of the sharding.
    """
    context, inputs = job.context, job.inputs
    config, resources, chunks = chain.config, chain.resources, chain.chunks
    if job.queries.height == 0 or inputs.active_pool.height == 0:
        raise ValueError(f"{context.run}: no queries or an empty active pool; nothing to predict")
    pool_vectors = query_vectors(config, resources, job.queries, inputs, chunks.encode_queries)
    # Item vectors of EVERY catalog article: tt_dot is computed for every candidate,
    # and a candidate of another channel (R2) can be outside A(a).
    vectors = add_other_catalog_articles(pool_vectors, resources, chunks.encode_queries)
    week = context.target.name

    def generate(queries: pl.DataFrame, shard: int) -> tuple[pl.DataFrame, int]:
        shard_customers = queries[CUSTOMER_IDX].unique().to_numpy()
        shard_vectors = select_users(vectors, shard_customers)
        candidates = candidates_from_query_vectors(config, queries, inputs, shard_vectors, chunks)
        validate_candidates(candidates, TWOTOWER_SPEC, config.quota)
        directory = context.paths.kaggle_vectors_dir(context.run, shard, TWOTOWER_VECTORS)
        save_vectors(shard_vectors, directory, TWOTOWER_VECTORS, week, None, overwrite=False)
        saved = load_vectors(directory, TWOTOWER_VECTORS, week)
        label = f"{context.run} shard {shard} {R6_CHANNEL}"
        checked = check_scores_match_vectors(candidates, queries, saved, chunks.pairs, label)
        return candidates, checked

    return FittedChannel(R6_CHANNEL, TWOTOWER_SPEC, config.quota, generate)


# ---------------------------------------------------------------- one channel, all shards


def fit_channel(job: RecallJob, channel: str, stamp: RunStamp) -> tuple[FittedChannel, str]:
    """Do the once-per-run work of a channel. Returns (the fitted channel, detail text)."""
    if channel == R1_CHANNEL:
        return fit_popularity(job), ""
    if channel in ("r2_repurchase", "r3_sibling"):
        return fit_per_customer_channel(job, channel), ""
    if channel == "r4_itemcf":
        return fit_itemcf(job), ""
    if channel == R5_CHANNEL:
        return fit_wals(job), ""
    if channel == R6_CHANNEL:
        chain, detail = prepare_twotower_chain(job, stamp)
        return fit_twotower(job, chain), detail
    raise ValueError(f"unknown channel {channel!r}; expected one of {ALL_CHANNELS}")


def shard_side_files(context: KaggleContext, shard: int, channel: str) -> list[Path]:
    """The files a channel writes for a shard BEFORE its candidates file (R5, R6: the vectors)."""
    paths, run = context.paths, context.run
    if channel == R5_CHANNEL:
        return [paths.kaggle_vectors_dir(run, shard, WALS_SOURCE) / METADATA_FILE]
    if channel == R6_CHANNEL:
        return [paths.kaggle_vectors_dir(run, shard, TWOTOWER_VECTORS) / METADATA_FILE]
    return []


def stored_rows(file: Path) -> int:
    """Row count of a Parquet file, from its metadata."""
    return int(pl.scan_parquet(file).select(pl.len()).collect().item())


def generate_channel(job: RecallJob, channel: str, stamp: RunStamp) -> ChannelResult:
    """Fit one channel once and write the candidates of every shard that has none yet.

    Args:
        job: The run's recall inputs.
        channel: One of the six channel names.
        stamp: The code state of this process (recorded with the R6 fine-tune).

    Raises:
        FileExistsError: On a half-finished shard (side files without candidates).
        ValueError: If an output breaks the schema or a consistency check fails.
    """
    context = job.context
    paths, run = context.paths, context.run
    shard_size = context.config.shard_customers
    n_shards = number_of_shards(job.queries.height, shard_size)
    files = [paths.kaggle_candidates_file(run, shard, channel) for shard in range(n_shards)]
    pending = [shard for shard in range(n_shards) if not files[shard].exists()]
    checked_rows, detail = 0, ""
    if pending:
        fitted, detail = fit_channel(job, channel, stamp)
        for shard in pending:
            require_not_started(shard_side_files(context, shard, channel), files[shard])
            queries = shard_queries(job.queries, shard, shard_size)
            candidates, checked = fitted.generate(queries, shard)
            # write_candidates validates the schema and refuses to replace a file.
            write_candidates(candidates, fitted.spec, fitted.quota, files[shard], False)
            checked_rows += checked
            LOGGER.info(
                "%s %s shard %d/%d: %d queries, %d rows -> %s",
                run,
                channel,
                shard,
                n_shards,
                queries.height,
                candidates.height,
                files[shard],
            )
    return ChannelResult(
        channel=channel,
        n_shards=n_shards,
        n_generated=len(pending),
        n_rows=sum(stored_rows(file) for file in files),
        checked_rows=checked_rows,
        detail=detail,
    )
