"""Put a run's shards together: the submission file, its checks and the candidate counts.

    build_submission(context)   reads the Top-K of every shard, writes the
                                submission CSV and its gzip copy, reads the file
                                back and checks the format (hmrec.kaggle.submission)
    record_files(...)           one row in results/kaggle_files.csv (sizes, sha256,
                                the outcome of the format checks)
    record_candidates(...)      one row in results/kaggle_candidates.csv (rows per
                                channel, candidates per customer, customers
                                without history)
    record_submission(...)      the row of results/kaggle_submission.csv: the one
                                submission and the two scores Kaggle returned

No label is read; neither torch nor LightGBM is imported.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import polars as pl

from hmrec.csv_records import append_row, read_rows
from hmrec.data.id_maps import ARTICLE_IDX, CUSTOMER_IDX
from hmrec.data.metadata import scan_processed_parquet
from hmrec.fusion.union import CHANNEL_PREFIXES
from hmrec.kaggle.context import KaggleContext
from hmrec.kaggle.plan import number_of_shards, read_queries
from hmrec.kaggle.prepare import read_customer_ids, read_sample_customer_ids
from hmrec.kaggle.submission import (
    ARTICLE_ID,
    CUSTOMER_ID,
    QUERY_ID,
    RANK,
    CandidateStatistics,
    FileCheck,
    candidate_statistics,
    check_submission_file,
    order_like_sample,
    predictions_table,
    write_submission,
)
from hmrec.logging_utils import get_logger

LOGGER = get_logger("kaggle.report")

FILES_COLUMNS: tuple[str, ...] = (
    "run",
    "csv_file",
    "n_rows",
    "csv_bytes",
    "csv_sha256",
    "gzip_file",
    "gzip_bytes",
    "gzip_sha256",
    "n_checks",
    "failed_checks",
    "passed",
    "git_commit",
    "git_dirty",
    "written_at",
)
CANDIDATES_COLUMNS: tuple[str, ...] = (
    "run",
    "week",
    "anchor",
    "n_customers",
    "n_shards",
    "shard_customers",
    *(f"{channel}_rows" for channel in CHANNEL_PREFIXES),
    "candidate_rows",
    "candidates_per_customer_mean",
    "candidates_per_customer_p50",
    "candidates_per_customer_p90",
    "no_history_share",
    "git_commit",
    "git_dirty",
    "recorded_at",
)


def run_shards(context: KaggleContext) -> int:
    """The number of shards of the run (from its query list)."""
    queries = read_queries(context.paths.kaggle_queries_file(context.run))
    return number_of_shards(queries.height, context.config.shard_customers)


def read_all_tops(context: KaggleContext) -> pl.DataFrame:
    """The Top-K of every shard: query_id, rank, article_idx.

    Raises:
        FileNotFoundError: If a shard is not scored yet.
    """
    paths, run = context.paths, context.run
    tables = []
    for shard in range(run_shards(context)):
        file = paths.kaggle_top_file(run, shard)
        if not file.is_file():
            raise FileNotFoundError(f"{file} not found: shard {shard} is not scored yet")
        tables.append(pl.read_parquet(file, columns=[QUERY_ID, RANK, ARTICLE_IDX]))
    return pl.concat(tables)


def read_article_ids(context: KaggleContext) -> pl.DataFrame:
    """article_idx (int32) -> article_id (the original 10-character string) of every article."""
    paths = context.paths
    table = scan_processed_parquet(
        paths.article_id_map_file, paths.processed_metadata_file, context.preprocess
    )
    return table.select(ARTICLE_IDX, ARTICLE_ID).collect()


def build_submission(context: KaggleContext) -> FileCheck:
    """Write the run's submission file from the shards' Top-K and check the written file.

    The rows follow sample_submission.csv: all of its customers for the real
    run, its rows of the rehearsal's customers for a rehearsal.

    Returns:
        The outcome of the format checks (the caller records it and stops if it failed).

    Raises:
        FileExistsError: If only one of the two files exists (a half-written state).
        ValueError: If a query lacks a complete Top-K or an id is unknown.
    """
    paths, run = context.paths, context.run
    k = context.config.predictions_per_customer
    queries = read_queries(paths.kaggle_queries_file(run))
    customer_ids = read_customer_ids(context)
    article_ids = read_article_ids(context)
    run_ids = queries.select(CUSTOMER_IDX).join(customer_ids, on=CUSTOMER_IDX)[CUSTOMER_ID]
    sample_ids = read_sample_customer_ids(context)
    # The sample file's order, cut to the run's customers (all of them in the real run).
    expected_ids = sample_ids.filter(sample_ids.is_in(run_ids.implode()))
    predictions = predictions_table(read_all_tops(context), queries, customer_ids, article_ids, k)
    table = order_like_sample(predictions, expected_ids)
    csv_file = paths.kaggle_submission_file(run)
    gzip_file = paths.kaggle_submission_gzip_file(run)
    if csv_file.exists() and gzip_file.exists():
        # A resumed run: the files are never replaced, only checked again below.
        LOGGER.warning("%s: %s exists; it is checked again, not rewritten", run, csv_file)
    else:
        write_submission(table, csv_file, gzip_file)
        LOGGER.info("%s: %d rows -> %s and %s", run, table.height, csv_file, gzip_file)
    return check_submission_file(csv_file, gzip_file, expected_ids, article_ids[ARTICLE_ID], k)


def record_files(context: KaggleContext, check: FileCheck) -> None:
    """Append the written submission file and its checks to results/kaggle_files.csv."""
    paths, run, stamp = context.paths, context.run, context.stamp
    root = paths.project_root
    row = {
        "run": run,
        "csv_file": paths.kaggle_submission_file(run).relative_to(root).as_posix(),
        "n_rows": str(check.n_rows),
        "csv_bytes": str(check.csv_bytes),
        "csv_sha256": check.csv_sha256,
        "gzip_file": paths.kaggle_submission_gzip_file(run).relative_to(root).as_posix(),
        "gzip_bytes": str(check.gzip_bytes),
        "gzip_sha256": check.gzip_sha256,
        "n_checks": str(len(check.checks)),
        "failed_checks": "; ".join(check.failed),
        "passed": str(check.passed),
        "git_commit": stamp.git_commit,
        "git_dirty": str(stamp.git_dirty),
        "written_at": datetime.now().isoformat(timespec="seconds"),
    }
    append_row(paths.kaggle_files_file, FILES_COLUMNS, row)


def format_checks_passed(context: KaggleContext) -> bool:
    """Whether the run's LAST recorded submission file passed every format check.

    Raises:
        ValueError: If no file of this run and this commit is recorded.
    """
    rows = read_rows(context.paths.kaggle_files_file, FILES_COLUMNS)
    own = [
        row
        for row in rows
        if row["run"] == context.run and row["git_commit"] == context.stamp.git_commit
    ]
    if not own:
        raise ValueError(f"no submission file of the {context.run} run is recorded at this commit")
    return own[-1]["passed"] == "True"


def read_statistics(context: KaggleContext) -> CandidateStatistics:
    """The candidate statistics of the run, from the query_stats file of every shard."""
    paths, run = context.paths, context.run
    files = [paths.kaggle_shard_stats_file(run, shard) for shard in range(run_shards(context))]
    missing = [str(file) for file in files if not file.is_file()]
    if missing:
        raise FileNotFoundError(f"query statistics not found (features not built): {missing[:3]}")
    return candidate_statistics(pl.concat([pl.read_parquet(file) for file in files]))


def channel_rows(context: KaggleContext, channel: str) -> int:
    """Candidate rows of one channel over all shards (from the Parquet metadata)."""
    paths, run = context.paths, context.run
    files = [
        paths.kaggle_candidates_file(run, shard, channel) for shard in range(run_shards(context))
    ]
    return int(pl.scan_parquet(files).select(pl.len()).collect().item())


def record_candidates(context: KaggleContext) -> CandidateStatistics:
    """Append the run's candidate counts to results/kaggle_candidates.csv and return them."""
    stats = read_statistics(context)
    stamp = context.stamp
    row = {
        "run": context.run,
        "week": context.target.name,
        "anchor": context.target.anchor.isoformat(),
        "n_customers": str(stats.n_customers),
        "n_shards": str(run_shards(context)),
        "shard_customers": str(context.config.shard_customers),
        **{f"{channel}_rows": str(channel_rows(context, channel)) for channel in CHANNEL_PREFIXES},
        "candidate_rows": str(stats.n_candidate_rows),
        "candidates_per_customer_mean": repr(stats.per_customer_mean),
        "candidates_per_customer_p50": repr(stats.per_customer_p50),
        "candidates_per_customer_p90": repr(stats.per_customer_p90),
        "no_history_share": repr(stats.no_history_share),
        "git_commit": stamp.git_commit,
        "git_dirty": str(stamp.git_dirty),
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
    }
    append_row(context.paths.kaggle_candidates_results_file, CANDIDATES_COLUMNS, row)
    return stats


# ---------------------------------------------------------------- the submission's record

SUBMISSION_COLUMNS: tuple[str, ...] = (
    "recorded_at",
    "competition",
    "submission_ref",
    "submitted_at",
    "status",
    "public_score",
    "private_score",
    "file",
    "file_sha256",
    "message",
    "run_git_commit",
    "submissions_before",
    "private_position",
    "leaderboard_teams",
    "position_label",
)
POSITION_LABEL = "late submission, not an official rank"
NOT_AVAILABLE = ""


@dataclass(frozen=True)
class SubmissionResult:
    """What Kaggle returned for the one submission.

    Attributes:
        submission_ref: Kaggle's id of the submission.
        submitted_at: When Kaggle received it (as Kaggle reports it).
        status: Kaggle's status of the submission, e.g. "complete".
        public_score: The Public score (recorded next to the Private one).
        private_score: The Private score, the one that counts.
        message: The message sent with the file (it names the commit).
        submissions_before: Submissions the account had before this task.
        leaderboard_scores: The Private scores of the final leaderboard, one per
            team, if the Kaggle CLI returned them; None otherwise.
    """

    submission_ref: str
    submitted_at: str
    status: str
    public_score: float
    private_score: float
    message: str
    submissions_before: int
    leaderboard_scores: tuple[float, ...] | None


def position_among(scores: Sequence[float], score: float) -> int:
    """The place `score` would take among `scores` (higher is better): 1 + scores above it.

    A tie with an existing score shares that score's place. This is only where
    a score WOULD stand; a late submission is not ranked by Kaggle.
    """
    return 1 + sum(1 for other in scores if other > score)


def record_submission(context: KaggleContext, result: SubmissionResult) -> dict[str, str]:
    """Append the submission's row to results/kaggle_submission.csv and return it.

    The file's sha256 and the commit of the run are taken from the run's last
    passed row of results/kaggle_files.csv, so the row names exactly the file
    that was checked.

    Raises:
        ValueError: If no checked submission file of the run is recorded, or a row
            of this submission exists already (one submission, one row).
    """
    paths = context.paths
    files = [
        row
        for row in read_rows(paths.kaggle_files_file, FILES_COLUMNS)
        if row["run"] == context.run and row["passed"] == "True"
    ]
    if not files:
        raise ValueError(f"no checked submission file of the {context.run} run is recorded")
    existing = read_rows(paths.kaggle_submission_results_file, SUBMISSION_COLUMNS)
    if existing:
        raise ValueError(
            f"{paths.kaggle_submission_results_file} already has a row: one submission is made"
        )
    scores = result.leaderboard_scores
    has_position = scores is not None and len(scores) > 0
    row = {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "competition": context.config.competition,
        "submission_ref": result.submission_ref,
        "submitted_at": result.submitted_at,
        "status": result.status,
        "public_score": repr(result.public_score),
        "private_score": repr(result.private_score),
        "file": files[-1]["gzip_file"],
        "file_sha256": files[-1]["gzip_sha256"],
        "message": result.message,
        "run_git_commit": files[-1]["git_commit"],
        "submissions_before": str(result.submissions_before),
        "private_position": (
            str(position_among(scores, result.private_score)) if scores else NOT_AVAILABLE
        ),
        "leaderboard_teams": str(len(scores)) if scores else NOT_AVAILABLE,
        "position_label": POSITION_LABEL if has_position else NOT_AVAILABLE,
    }
    append_row(paths.kaggle_submission_results_file, SUBMISSION_COLUMNS, row)
    return row
