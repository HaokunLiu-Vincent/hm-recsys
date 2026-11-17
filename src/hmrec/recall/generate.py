"""Generate ONE channel's candidates for ONE post-lock fold (M6, proposal §6.6, §7.3).

The channel is built only from configs/locks/ (hmrec.recall.locked). Outputs:

    every channel   artifacts/candidates/{fold}/{channel}.parquet
                    (validated against the channel's OutputSpec before writing)
    R1 also         artifacts/candidates/{fold}/r1_bucket_scores.parquet: the full
                    S(g, i) table, for the pop_bucket_score feature
    R5 also         artifacts/models/wals/{fold}/: user vectors of the fold's
                    query customers, item vectors of every article that has one
    R6 also         artifacts/models/twotower/vectors/{fold}/: user vectors of the
                    query customers, item vectors of EVERY catalog article,
                    computed with the model that predicts the fold;
                    then the fine-tune on the fold's week, which saves
                    artifacts/models/twotower/chain_locked/{fold}.pt and appends its
                    sha256 to results/m6_twotower_chain.csv (not for the last fold)

features/ may not import recall/, so those side files are the only way the
feature code gets R1 scores and R5 / R6 vectors. Each one is read back from
disk and checked against the candidates BEFORE the candidates are written:

    R5, R6   every candidate's score == dot product of the saved vectors
    R1       every non-fallback candidate's score == its entry of the saved table

Files are frozen once written (AGENTS.md §3.1). A step whose candidates file
exists is skipped with a warning. The candidates file is written LAST, so it
marks a finished step; side files without it are a half-finished step, which
is an error (nothing is repaired or deleted automatically).

This module never reads labels. A fold's history is cut by each channel with
hmrec.data.asof. R6's fine-tune reads the purchases of the fold's own week,
but only AFTER the fold's candidates and vectors are written.
"""

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from hmrec.config import BaseConfig, CandidateRunConfig, PreprocessConfig
from hmrec.data.folds import Fold
from hmrec.locks import F1_START_CHECKPOINT, R6_CHANNEL, file_sha256
from hmrec.logging_utils import get_logger
from hmrec.memory import MemoryTracker, bytes_to_gb
from hmrec.paths import ProjectPaths
from hmrec.recall.base import (
    ARTICLE_IDX,
    CUSTOMER_IDX,
    QUERY_ID,
    SCORE,
    RecallInputs,
    validate_candidates,
    write_candidates,
)
from hmrec.recall.chain_records import (
    ChainRecord,
    append_chain_record,
    check_model_and_record_agree,
    read_chain_records,
    start_model_for,
)
from hmrec.recall.locked import (
    LockedRecall,
    ModelFile,
    build_locked_channel,
    load_locked_twotower_chain,
    locked_checkpoint_file,
)
from hmrec.recall.r1_popularity import (
    AGE_BUCKET,
    BUCKET_SCORES_SCHEMA,
    IS_FALLBACK,
    PopularityChannel,
)
from hmrec.recall.r5_wals import WalsChannel
from hmrec.recall.twotower.chain import TwoTowerChainChannel
from hmrec.recall.twotower.infer import (
    TWOTOWER_SPEC,
    add_other_catalog_articles,
    candidates_from_query_vectors,
    query_vectors,
)
from hmrec.run_records import FINE_TUNE_ONLY, GENERATED, SKIPPED, RunStamp, StepResult
from hmrec.vectors import METADATA_FILE, UserItemVectors, dot_pairs, load_vectors, save_vectors

LOGGER = get_logger("recall.generate")

R1_CHANNEL = "r1_popularity"
R5_CHANNEL = "r5_wals"
WALS_SOURCE = "wals"
TWOTOWER_SOURCE = "twotower"


@dataclass(frozen=True)
class FoldJob:
    """Everything needed to generate channels for one fold.

    Attributes:
        paths: Project paths.
        base: configs/base.yaml (chunk sizes, threads, memory settings).
        preprocess: Current preprocess config.
        locked: Both lock files.
        fold: The fold to generate.
        fold_names: The post-lock folds in time order (F1 ... T4); the R6
            chain needs the fold before `fold` and must know the last one.
        queries: One row per query of the fold; columns query_id, fold, customer_idx.
        inputs: The channel inputs as of the fold's anchor.
    """

    paths: ProjectPaths
    base: BaseConfig
    preprocess: PreprocessConfig
    locked: LockedRecall
    fold: Fold
    fold_names: tuple[str, ...]
    queries: pl.DataFrame
    inputs: RecallInputs


# ---------------------------------------------------------------- consistency checks


def check_scores_match_vectors(
    candidates: pl.DataFrame,
    queries: pl.DataFrame,
    vectors: UserItemVectors,
    pair_chunk_size: int,
    label: str,
) -> int:
    """Every candidate's score must equal the dot product of the given vectors, bit for bit.

    Called with the vectors read back from disk: those files are what the
    wals_dot / tt_dot features will be computed from (proposal §8.3).

    Args:
        candidates: Channel output; columns query_id, article_idx, score (float32).
        queries: query_id -> customer_idx.
        vectors: The saved vectors.
        pair_chunk_size: Passed to hmrec.vectors.dot_pairs.
        label: Name for the error message, e.g. "F1 r5_wals".

    Returns:
        The number of candidate rows checked.

    Raises:
        ValueError: If a candidate's customer or article has no saved vector, or
            a score differs from its dot product.
    """
    with_customer = candidates.select(QUERY_ID, ARTICLE_IDX, SCORE).join(
        queries.select(QUERY_ID, CUSTOMER_IDX), on=QUERY_ID, how="left"
    )
    dots = dot_pairs(
        vectors,
        with_customer[CUSTOMER_IDX].to_numpy(),
        with_customer[ARTICLE_IDX].to_numpy(),
        pair_chunk_size,
    )
    scores = with_customer[SCORE].to_numpy()
    # dot_pairs returns NaN for a pair without a vector; NaN != score, so a missing
    # vector fails the check too.
    different = int(np.count_nonzero(dots != scores))
    if different > 0:
        raise ValueError(
            f"{label}: {different} of {scores.size} candidate scores differ from the dot "
            "product of the saved vectors"
        )
    return int(scores.size)


def check_scores_match_bucket_table(
    candidates: pl.DataFrame,
    queries: pl.DataFrame,
    customers: pl.DataFrame,
    bucket_scores: pl.DataFrame,
    label: str,
) -> int:
    """Every non-fallback R1 candidate's score must equal its entry of the S(g, i) table.

    Fallback rows are not checked: their score is the GLOBAL popularity S(i),
    another quantity (proposal §5 R1); the table holds the bucket scores S(g, i).

    Args:
        candidates: R1 output; columns query_id, article_idx, score, pop_is_fallback.
        queries: query_id -> customer_idx.
        customers: customer_idx -> age_bucket.
        bucket_scores: The saved table; columns age_bucket, article_idx, score (float32).
        label: Name for the error message.

    Returns:
        The number of non-fallback candidate rows checked.

    Raises:
        ValueError: If a row has no table entry or another score.
    """
    own = (
        candidates.filter(pl.col(IS_FALLBACK) == 0)
        .select(QUERY_ID, ARTICLE_IDX, SCORE)
        .join(queries.select(QUERY_ID, CUSTOMER_IDX), on=QUERY_ID, how="left")
        .join(customers.select(CUSTOMER_IDX, AGE_BUCKET), on=CUSTOMER_IDX, how="left")
    )
    table = bucket_scores.rename({SCORE: "table_score"})
    joined = own.join(table, on=[AGE_BUCKET, ARTICLE_IDX], how="left")
    # A missing entry is null, and null != score is null, so both cases are caught.
    same = (pl.col("table_score") == pl.col(SCORE)).fill_null(False)
    different = joined.filter(~same).height
    if different > 0:
        raise ValueError(
            f"{label}: {different} of {own.height} non-fallback candidates differ from "
            "(or are missing in) the saved bucket score table"
        )
    return own.height


def write_bucket_scores(bucket_scores: pl.DataFrame, path: Path) -> None:
    """Write R1's S(g, i) table; one row per (age_bucket, article_idx). Never overwrites.

    Raises:
        FileExistsError: If `path` exists.
        ValueError: If the schema is wrong, a value is null / not finite, or a
            (age_bucket, article_idx) pair repeats.
    """
    if path.exists():
        raise FileExistsError(f"{path} already exists; it is never replaced")
    if bucket_scores.schema != BUCKET_SCORES_SCHEMA:
        raise ValueError(
            f"bucket scores have schema {dict(bucket_scores.schema)}, "
            f"expected {dict(BUCKET_SCORES_SCHEMA)}"
        )
    has_null = any(bucket_scores[name].null_count() > 0 for name in bucket_scores.columns)
    if has_null or not bucket_scores[SCORE].is_finite().all():
        raise ValueError("bucket scores contain null, NaN or infinite values")
    if bucket_scores.select(AGE_BUCKET, ARTICLE_IDX).is_duplicated().any():
        raise ValueError("bucket scores repeat an (age_bucket, article_idx) pair")
    path.parent.mkdir(parents=True, exist_ok=True)
    bucket_scores.write_parquet(path)


def require_not_started(side_files: Sequence[Path], candidates_file: Path) -> None:
    """Fail if a side file exists although the candidates file (written last) does not.

    That is a half-finished step. Nothing is deleted or reused automatically:
    a human looks at what happened first.
    """
    existing = [str(path) for path in side_files if path.exists()]
    if existing:
        raise FileExistsError(
            f"{candidates_file} is missing but {existing} exist: a half-finished step. "
            "Check the run log before removing anything."
        )


# ---------------------------------------------------------------- shared pieces


def _stored_rows(candidates_file: Path) -> int:
    """Row count of an existing candidates file (from the Parquet metadata)."""
    return int(pl.scan_parquet(candidates_file).select(pl.len()).collect().item())


def _skipped(job: FoldJob, channel: str, candidates_file: Path) -> StepResult:
    """The result of a step whose candidates already exist: nothing is touched."""
    LOGGER.warning(
        "%s %s: %s already exists; SKIPPED (candidates are never replaced)",
        job.fold.name,
        channel,
        candidates_file,
    )
    return StepResult(
        job.fold.name, channel, SKIPPED, _stored_rows(candidates_file), 0.0, 0.0, 0, ""
    )


def _timed(
    job: FoldJob, label: str, compute: Callable[[], tuple[pl.DataFrame, int]]
) -> tuple[tuple[pl.DataFrame, int], float, float]:
    """Run `compute` inside a MemoryTracker.

    Returns:
        ((candidates, checked rows), wall seconds, SAMPLED peak RSS in GB).
    """
    memory = job.base.memory
    with MemoryTracker(
        f"{job.fold.name} {label}",
        LOGGER,
        limit_gb=memory.limit_gb,
        sample_interval_seconds=memory.sample_interval_seconds,
    ) as tracker:
        started = time.perf_counter()
        value = compute()
        seconds = time.perf_counter() - started
    return value, seconds, bytes_to_gb(tracker.peak_rss_bytes)


# ---------------------------------------------------------------- R1 - R5


def _popularity_outputs(job: FoldJob, channel: PopularityChannel) -> tuple[pl.DataFrame, int]:
    """R1: candidates; writes and checks the bucket score table. Returns (candidates, checked)."""
    fold = job.fold.name
    candidates, bucket_scores = channel.generate_with_bucket_scores(job.queries, job.inputs)
    validate_candidates(candidates, channel.output_spec, channel.config.quota)
    table_file = job.paths.r1_bucket_scores_file(fold)
    write_bucket_scores(bucket_scores, table_file)
    saved = pl.read_parquet(table_file)
    checked = check_scores_match_bucket_table(
        candidates, job.queries, job.inputs.customers, saved, f"{fold} {R1_CHANNEL}"
    )
    LOGGER.info(
        "%s %s: bucket score table %d rows -> %s; %d non-fallback scores match it",
        fold,
        R1_CHANNEL,
        saved.height,
        table_file,
        checked,
    )
    return candidates, checked


def _wals_outputs(job: FoldJob, channel: WalsChannel) -> tuple[pl.DataFrame, int]:
    """R5: candidates; saves and checks the vectors. Returns (candidates, checked)."""
    fold = job.fold.name
    vectors = channel.fit(job.inputs)
    candidates = channel.candidates_from_vectors(job.queries, job.inputs, vectors)
    validate_candidates(candidates, channel.output_spec, channel.config.quota)
    directory = job.paths.wals_vectors_dir(fold)
    # Users: only the fold's query customers (features are computed for queries
    # only). Articles: every article that has a vector.
    query_customers = job.queries[CUSTOMER_IDX].unique().to_numpy()
    save_vectors(vectors, directory, WALS_SOURCE, fold, query_customers, overwrite=False)
    saved = load_vectors(directory, WALS_SOURCE, fold)
    checked = check_scores_match_vectors(
        candidates, job.queries, saved, job.base.chunking.pair_chunk_size, f"{fold} {R5_CHANNEL}"
    )
    LOGGER.info(
        "%s %s: vectors of %d users, %d articles -> %s; %d scores match them",
        fold,
        R5_CHANNEL,
        saved.user_customer_idx.size,
        saved.item_article_idx.size,
        directory,
        checked,
    )
    return candidates, checked


def side_files(paths: ProjectPaths, fold: str, channel: str) -> list[Path]:
    """The files a channel writes BEFORE its candidates file (empty for R2, R3, R4)."""
    if channel == R1_CHANNEL:
        return [paths.r1_bucket_scores_file(fold)]
    if channel == R5_CHANNEL:
        return [paths.wals_vectors_dir(fold) / METADATA_FILE]
    if channel == R6_CHANNEL:
        return [paths.twotower_vectors_dir(fold) / METADATA_FILE]
    return []


def generate_statistical_channel(job: FoldJob, channel: str) -> StepResult:
    """Generate one of R1-R5 for the fold (see the module docstring).

    Raises:
        FileExistsError: On a half-finished step (side files without candidates).
        ValueError: If the output breaks the schema, or a consistency check fails.
    """
    fold = job.fold.name
    candidates_file = job.paths.candidates_file(fold, channel)
    if candidates_file.exists():
        return _skipped(job, channel, candidates_file)
    require_not_started(side_files(job.paths, fold, channel), candidates_file)
    recall_channel = build_locked_channel(channel, job.locked, job.base, job.paths, job.preprocess)
    quota = job.locked.channel(channel).quota

    def compute() -> tuple[pl.DataFrame, int]:
        if isinstance(recall_channel, PopularityChannel):
            return _popularity_outputs(job, recall_channel)
        if isinstance(recall_channel, WalsChannel):
            return _wals_outputs(job, recall_channel)
        return recall_channel.generate(job.queries, job.inputs), 0

    (candidates, checked), seconds, sampled_peak = _timed(job, f"{channel} generate", compute)
    # write_candidates validates again and refuses to replace a file.
    write_candidates(candidates, recall_channel.output_spec, quota, candidates_file, False)
    LOGGER.info("%s %s: %d rows -> %s", fold, channel, candidates.height, candidates_file)
    return StepResult(
        fold, channel, GENERATED, candidates.height, seconds, sampled_peak, checked, ""
    )


# ---------------------------------------------------------------- R6: predict, save, then fine-tune


def predict_twotower_fold(
    chain: TwoTowerChainChannel,
    fold: str,
    queries: pl.DataFrame,
    inputs: RecallInputs,
    candidates_file: Path,
    vectors_dir: Path,
) -> tuple[pl.DataFrame, int]:
    """Predict the fold with the chain's CURRENT model; save vectors, then candidates.

    The model must not have been fine-tuned on this fold yet: the caller runs
    this BEFORE chain.fine_tune_on_week, so the saved vectors and candidates
    come from a model that has only seen purchases before the fold's anchor.

    Saved vectors: user vectors of the fold's query customers; item vectors of
    every catalog article (the active pool A(a) as used for the candidates,
    plus all other articles, for the tt_dot feature of candidates outside A(a)).

    Returns:
        (candidates, number of candidate scores checked against the saved vectors).

    Raises:
        ValueError: If the fold has no queries or an empty active pool, the
            output breaks the schema, or a score differs from its dot product.
    """
    if queries.height == 0 or inputs.active_pool.height == 0:
        raise ValueError(f"{fold}: no queries or an empty active pool; nothing to predict")
    config, chunks = chain.config, chain.chunks
    pool_vectors = query_vectors(config, chain.resources, queries, inputs, chunks.encode_queries)
    vectors = add_other_catalog_articles(pool_vectors, chain.resources, chunks.encode_queries)
    candidates = candidates_from_query_vectors(config, queries, inputs, vectors, chunks)
    validate_candidates(candidates, TWOTOWER_SPEC, config.quota)
    save_vectors(vectors, vectors_dir, TWOTOWER_SOURCE, fold, None, overwrite=False)
    saved = load_vectors(vectors_dir, TWOTOWER_SOURCE, fold)
    checked = check_scores_match_vectors(
        candidates, queries, saved, chunks.pairs, f"{fold} {R6_CHANNEL}"
    )
    write_candidates(candidates, TWOTOWER_SPEC, config.quota, candidates_file, overwrite=False)
    LOGGER.info(
        "%s %s: %d rows -> %s; vectors of %d users, %d articles -> %s; %d scores match them",
        fold,
        R6_CHANNEL,
        candidates.height,
        candidates_file,
        saved.user_customer_idx.size,
        saved.item_article_idx.size,
        vectors_dir,
        checked,
    )
    return candidates, checked


def check_fine_tune_loss(record: ChainRecord, max_loss: float) -> None:
    """Fail if a weekly fine-tune ended with a loss that is not finite or above `max_loss`.

    Only a stop for a human decision (configs/m6_candidates.yaml). The loss
    never selects or changes anything.

    Raises:
        ValueError: If the training loss or the alarm-user loss is NaN,
            infinite or > max_loss.
    """
    for name, loss in (
        ("training loss", record.mean_train_loss),
        ("alarm loss", record.alarm_loss),
    ):
        if not math.isfinite(loss) or loss > max_loss:
            raise ValueError(
                f"{record.fold} {R6_CHANNEL}: fine-tune {name} is {loss!r} "
                f"(must be finite and <= {max_loss}); the chain stops for a human decision"
            )


def fine_tune_twotower_fold(
    chain: TwoTowerChainChannel,
    fold: Fold,
    inputs: RecallInputs,
    start: ModelFile,
    stamp: RunStamp,
    chain_results_file: Path,
) -> ChainRecord:
    """Fine-tune on the fold's week with the D-phase chain code, then record the saved model.

    chain.fine_tune_on_week is the same function the D-phase chain used: the
    week's purchases [anchor, anchor + 7) plus 1 : 1 replay, alarm users
    excluded, a NEW Adam, the locked E_update and fine-tune learning rate. It
    saves artifacts/models/twotower/chain_locked/{fold}.pt and never replaces a file.

    Returns:
        The appended row of results/m6_twotower_chain.csv (with the model's sha256).
    """
    result = chain.fine_tune_on_week(fold.name, inputs)
    model_file = chain.checkpoint_file(fold.name)
    n_new_rows, n_replay_rows = chain.fine_tune_row_counts[fold.name]
    record = ChainRecord(
        fold=fold.name,
        anchor=fold.anchor.isoformat(),
        start_model=start.path.name,
        start_sha256=start.sha256,
        model=model_file.name,
        sha256=file_sha256(model_file),
        n_new_rows=n_new_rows,
        n_replay_rows=n_replay_rows,
        epochs=result.epochs,
        steps=result.steps,
        n_train_samples=result.n_train_samples,
        n_alarm_samples=result.n_alarm_samples,
        mean_train_loss=result.mean_train_loss,
        alarm_loss=result.alarm_loss,
        fine_tune_seconds=result.seconds,
        sampled_peak_rss_gb=result.peak_rss_gb,
        git_commit=stamp.git_commit,
        git_dirty=stamp.git_dirty,
        finished_at=datetime.now().isoformat(timespec="seconds"),
    )
    append_chain_record(chain_results_file, record)
    LOGGER.info("%s %s: model %s sha256 %s", fold.name, R6_CHANNEL, model_file, record.sha256)
    return record


def _locked_f1_start(job: FoldJob) -> ModelFile:
    """The locked model that predicts the first post-lock fold, with its locked sha256."""
    lock = job.locked.twotower
    return ModelFile(
        locked_checkpoint_file(job.paths, lock, F1_START_CHECKPOINT),
        lock.checkpoints[F1_START_CHECKPOINT].sha256,
    )


def generate_twotower_fold(job: FoldJob, stamp: RunStamp) -> StepResult:
    """R6 for one fold: check the chain order, predict, save, then fine-tune (proposal §6.6).

    Order of the steps (each only if not done before):
      1. The model that predicts the fold: the locked f1_start for the first
         fold, else the previous fold's saved model. It must exist with the
         recorded sha256 (hmrec.recall.chain_records.start_model_for).
      2. Predict; save vectors and candidates (predict_twotower_fold).
      3. Fine-tune on the fold's week and record the model's sha256. The LAST
         fold (T4) is only predicted, never fine-tuned.
    The fine-tune loss is checked afterwards by check_recorded_fine_tune_loss,
    once the run itself is on record.

    Raises:
        ValueError / FileNotFoundError: If the chain is out of order, a sha256
            differs, or a consistency check fails.
    """
    paths, fold = job.paths, job.fold.name
    is_last_fold = fold == job.fold_names[-1]
    candidates_file = paths.candidates_file(fold, R6_CHANNEL)
    checkpoint_file = paths.twotower_locked_chain_checkpoint_file
    records = read_chain_records(paths.m6_twotower_chain_results_file)
    fine_tuned = check_model_and_record_agree(fold, records, checkpoint_file)
    if is_last_fold and fine_tuned:
        raise ValueError(f"{fold} is the last fold: it must not have a fine-tuned model")
    predicted = candidates_file.exists()
    needs_fine_tune = not is_last_fold and not fine_tuned
    if predicted and not needs_fine_tune:
        return _skipped(job, R6_CHANNEL, candidates_file)
    if not predicted:
        require_not_started(side_files(paths, fold, R6_CHANNEL), candidates_file)
    start = start_model_for(fold, job.fold_names, _locked_f1_start(job), records, checkpoint_file)
    chain = load_locked_twotower_chain(job.locked, job.base, paths, job.preprocess, start)
    LOGGER.info("%s %s: predicting with %s", fold, R6_CHANNEL, start.path)
    status, n_rows, seconds, sampled_peak, checked = FINE_TUNE_ONLY, 0, 0.0, 0.0, 0
    if predicted:
        LOGGER.warning("%s %s: candidates exist; only the fine-tune is done", fold, R6_CHANNEL)
        n_rows = _stored_rows(candidates_file)
    else:
        vectors_dir = paths.twotower_vectors_dir(fold)
        (candidates, checked), seconds, sampled_peak = _timed(
            job,
            f"{R6_CHANNEL} predict",
            lambda: predict_twotower_fold(
                chain, fold, job.queries, job.inputs, candidates_file, vectors_dir
            ),
        )
        status, n_rows = GENERATED, candidates.height
    detail = "last fold: predicted only, no fine-tune"
    if needs_fine_tune:
        # Only now may the model see the fold's own week.
        record = fine_tune_twotower_fold(
            chain, job.fold, job.inputs, start, stamp, paths.m6_twotower_chain_results_file
        )
        detail = (
            f"fine-tune {record.n_new_rows} new + {record.n_replay_rows} replay rows, "
            f"train loss {record.mean_train_loss:.4f}, alarm loss {record.alarm_loss:.4f}, "
            f"{record.fine_tune_seconds:.1f} s"
        )
    return StepResult(fold, R6_CHANNEL, status, n_rows, seconds, sampled_peak, checked, detail)


# ---------------------------------------------------------------- one entry


def generate_fold_channel(job: FoldJob, channel: str, stamp: RunStamp) -> StepResult:
    """Generate one channel for the job's fold (R6: including its fine-tune)."""
    if job.fold.name not in job.fold_names:
        raise ValueError(
            f"{job.fold.name} is not a post-lock fold; expected one of {list(job.fold_names)}"
        )
    if channel == R6_CHANNEL:
        return generate_twotower_fold(job, stamp)
    return generate_statistical_channel(job, channel)


def check_recorded_fine_tune_loss(
    job: FoldJob, result: StepResult, config: CandidateRunConfig
) -> None:
    """After an R6 run that fine-tuned: stop if the recorded loss is not finite or too high.

    Called by the script AFTER the run row is written, so a run that trips
    this check is still on record (results/m6_runs.csv and m6_twotower_chain.csv).

    Raises:
        ValueError: See check_fine_tune_loss.
    """
    if result.step != R6_CHANNEL or result.status == SKIPPED:
        return
    records = read_chain_records(job.paths.m6_twotower_chain_results_file)
    if job.fold.name in records:  # the last fold has no fine-tune and no record
        check_fine_tune_loss(records[job.fold.name], config.max_fine_tune_loss)
