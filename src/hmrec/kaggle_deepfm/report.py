"""The submission file, the rehearsal's checks and the record of the one submission.

    build_submission(context)     reads the DeepFM Top-K of every shard, writes the
                                  submission CSV and its gzip copy, reads the file
                                  back and makes the nine format checks of the
                                  LightGBM submission (hmrec.kaggle.submission)
    record_files(...)             one row in results/kaggle_deepfm_files.csv
    rehearsal_outcome(context)    the rehearsal on T4 against the stage-2 evaluation:
                                  1. loss    the mean training loss EQUALS the one
                                             stage 2 recorded for T4 and the seed
                                             (results/m9_t_metrics.csv, float64 ==)
                                  2. Top-K   for the T4 queries, the share of
                                             customers whose list is identical to
                                             the one stage 2 saved is at least
                                             rehearsal.min_identical_top12_share
                                  3. format  the format checks passed on its file
    require_passed_rehearsal(...) the gate before the real run
    record_submission(...)        the row of results/kaggle_deepfm_submission.csv:
                                  the two scores, the place the Private score would
                                  have had, and the reading fixed before the run

Nothing here reads a label or loads a model: saved predictions are compared, not
scored. Neither torch nor LightGBM is imported.
"""

import csv
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl

from hmrec.csv_records import append_row, read_rows
from hmrec.data.id_maps import ARTICLE_IDX, CUSTOMER_IDX
from hmrec.kaggle.plan import RUN_REHEARSAL, number_of_shards, read_queries
from hmrec.kaggle.prepare import read_customer_ids, read_sample_customer_ids
from hmrec.kaggle.rehearsal import TopComparison, compare_top_lists
from hmrec.kaggle.report import (
    FILES_COLUMNS,
    POSITION_LABEL,
    SUBMISSION_COLUMNS,
    SubmissionResult,
    position_among,
    read_article_ids,
)
from hmrec.kaggle.submission import (
    ARTICLE_ID,
    CUSTOMER_ID,
    QUERY_ID,
    RANK,
    FileCheck,
    check_submission_file,
    order_like_sample,
    predictions_table,
    write_submission,
)
from hmrec.kaggle_deepfm.context import DeepFMKaggleContext
from hmrec.logging_utils import get_logger

LOGGER = get_logger("kaggle_deepfm.report")

TRAINING_FILE = "training.json"
PREDICTIONS_FILE = "predictions.parquet"
SAME = "cannot be told apart on this week"
DIFFERENT = "a difference on a single week"
REHEARSAL_COLUMNS: tuple[str, ...] = (
    "checked_at",
    "git_commit",
    "git_dirty",
    "n_customers",
    "n_shards",
    "loss_equal",
    "new_train_loss",
    "stage2_train_loss",
    "n_fold_queries",
    "n_identical_top",
    "identical_top_share",
    "min_identical_top_share",
    "format_checks_passed",
    "passed",
)
READING_COLUMNS: tuple[str, ...] = (
    "lightgbm_private_score",
    "private_relative_difference",
    "lightgbm_public_score",
    "public_relative_difference",
    "same_below_relative_difference",
    "reading",
)
DEEPFM_SUBMISSION_COLUMNS: tuple[str, ...] = (*SUBMISSION_COLUMNS, *READING_COLUMNS)
# Reads a recorded commit and says whether the code at HEAD differs from it.
CodeChanged = Callable[[str], bool]


# ---------------------------------------------------------------- the submission file


def run_shards(context: DeepFMKaggleContext) -> int:
    """The number of shards of the run (from the query list the LightGBM submission wrote)."""
    queries = read_queries(context.paths.kaggle_queries_file(context.run))
    return number_of_shards(queries.height, context.kaggle.config.shard_customers)


def read_all_tops(context: DeepFMKaggleContext) -> pl.DataFrame:
    """The DeepFM Top-K of every shard: query_id, rank, article_idx.

    Raises:
        FileNotFoundError: If a shard is not scored yet.
    """
    tables = []
    for shard in range(run_shards(context)):
        file = context.paths.kaggle_deepfm_top_file(context.run, shard)
        if not file.is_file():
            raise FileNotFoundError(f"{file} not found: shard {shard} is not scored yet")
        tables.append(pl.read_parquet(file, columns=[QUERY_ID, RANK, ARTICLE_IDX]))
    return pl.concat(tables)


def write_and_check(
    top: pl.DataFrame,
    queries: pl.DataFrame,
    customer_ids: pl.DataFrame,
    article_ids: pl.DataFrame,
    sample_ids: pl.Series,
    files: tuple[Path, Path],
    k: int,
) -> FileCheck:
    """Write a submission file from a run's Top-K and check the written file.

    The rows follow the sample file: all of its customers for the real run, its
    rows of the run's customers for a rehearsal.

    Args:
        top: The Top-k of every query: query_id, rank, article_idx.
        queries: The run's queries: query_id, customer_idx.
        customer_ids: customer_idx -> customer_id (the original string).
        article_ids: article_idx -> article_id (the original 10-character string).
        sample_ids: The customer ids of sample_submission.csv, in file order.
        files: (the CSV to write, its gzip copy). If both exist they are only
            checked again, never rewritten.
        k: Predictions per customer.

    Returns:
        The outcome of the nine format checks.

    Raises:
        FileExistsError: If only one of the two files exists (a half-written state).
        ValueError: If a query lacks a complete Top-k or an id is unknown.
    """
    csv_file, gzip_file = files
    run_ids = queries.select(CUSTOMER_IDX).join(customer_ids, on=CUSTOMER_IDX)[CUSTOMER_ID]
    # The sample file's order, cut to the run's customers (all of them in the real run).
    expected_ids = sample_ids.filter(sample_ids.is_in(run_ids.implode()))
    predictions = predictions_table(top, queries, customer_ids, article_ids, k)
    table = order_like_sample(predictions, expected_ids)
    if csv_file.exists() and gzip_file.exists():
        LOGGER.warning("%s exists; it is checked again, not rewritten", csv_file)
    else:
        write_submission(table, csv_file, gzip_file)
        LOGGER.info("%d rows -> %s and %s", table.height, csv_file, gzip_file)
    return check_submission_file(csv_file, gzip_file, expected_ids, article_ids[ARTICLE_ID], k)


def build_submission(context: DeepFMKaggleContext) -> FileCheck:
    """Write the run's DeepFM submission file from the shards' Top-K and check it."""
    paths, run, kaggle = context.paths, context.run, context.kaggle
    files = (
        paths.kaggle_deepfm_submission_file(run),
        paths.kaggle_deepfm_submission_gzip_file(run),
    )
    return write_and_check(
        read_all_tops(context),
        read_queries(paths.kaggle_queries_file(run)),
        read_customer_ids(kaggle),
        read_article_ids(kaggle),
        read_sample_customer_ids(kaggle),
        files,
        kaggle.config.predictions_per_customer,
    )


def record_files(context: DeepFMKaggleContext, check: FileCheck) -> None:
    """Append the written submission file and its checks to results/kaggle_deepfm_files.csv."""
    paths, run, stamp = context.paths, context.run, context.stamp
    root = paths.project_root
    row = {
        "run": run,
        "csv_file": paths.kaggle_deepfm_submission_file(run).relative_to(root).as_posix(),
        "n_rows": str(check.n_rows),
        "csv_bytes": str(check.csv_bytes),
        "csv_sha256": check.csv_sha256,
        "gzip_file": paths.kaggle_deepfm_submission_gzip_file(run).relative_to(root).as_posix(),
        "gzip_bytes": str(check.gzip_bytes),
        "gzip_sha256": check.gzip_sha256,
        "n_checks": str(len(check.checks)),
        "failed_checks": "; ".join(check.failed),
        "passed": str(check.passed),
        "git_commit": stamp.git_commit,
        "git_dirty": str(stamp.git_dirty),
        "written_at": datetime.now().isoformat(timespec="seconds"),
    }
    append_row(paths.kaggle_deepfm_files_file, FILES_COLUMNS, row)


def format_checks_passed(context: DeepFMKaggleContext) -> bool:
    """Whether the run's LAST recorded submission file passed every format check.

    Raises:
        ValueError: If no file of this run and this commit is recorded.
    """
    rows = read_rows(context.paths.kaggle_deepfm_files_file, FILES_COLUMNS)
    own = [
        row
        for row in rows
        if row["run"] == context.run and row["git_commit"] == context.stamp.git_commit
    ]
    if not own:
        raise ValueError(f"no submission file of the {context.run} run is recorded at this commit")
    return own[-1]["passed"] == "True"


# ---------------------------------------------------------------- the rehearsal's checks


def stage2_train_loss(context: DeepFMKaggleContext) -> float:
    """The mean training loss the stage-2 evaluation recorded for the rehearsal's fold and seed.

    Read from results/m9_t_metrics.csv: the row of the reference area, the chosen
    configuration, the seed, the fold and the configuration's epochs.

    Raises:
        ValueError: If the table has not exactly one such row.
    """
    settings, chosen = context.settings, context.chosen
    fold = context.kaggle.target.name
    wanted = (settings.reference_area, settings.config_id, str(settings.seed), fold)
    metrics_file = context.paths.m9_t_metrics_file
    if not metrics_file.is_file():
        raise ValueError(f"{metrics_file} not found: stage 2 has no record to compare with")
    with metrics_file.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if (row["area"], row["config_id"], row["seed"], row["fold"]) == wanted
            and int(row["epoch"]) == chosen.epochs
        ]
    if len(rows) != 1:
        raise ValueError(f"{metrics_file} has {len(rows)} rows for {wanted}, expected exactly one")
    return float(rows[0]["train_loss"])


def new_train_loss(context: DeepFMKaggleContext) -> float:
    """The mean training loss of the run's own model, from its training.json.

    Raises:
        FileNotFoundError: If the model is not trained yet.
    """
    file = context.paths.kaggle_deepfm_model_dir(context.run) / TRAINING_FILE
    if not file.is_file():
        raise FileNotFoundError(f"{file} not found: the model is not trained yet")
    return float(json.loads(file.read_text(encoding="utf-8"))["mean_loss"])


def stage2_top(context: DeepFMKaggleContext) -> pl.DataFrame:
    """The Top-K the stage-2 evaluation saved for the rehearsal's fold and seed.

    Returns:
        query_id, rank, article_idx; one row per (query, rank).
    """
    settings = context.settings
    directory = context.paths.deepfm_trial_dir(
        settings.reference_area, settings.config_id, settings.seed, context.kaggle.target.name
    )
    return pl.read_parquet(directory / PREDICTIONS_FILE).select(QUERY_ID, RANK, ARTICLE_IDX)


@dataclass(frozen=True)
class RehearsalOutcome:
    """The three checks of one rehearsal.

    Attributes:
        n_customers: Customers of the rehearsal.
        n_shards: Shards they were scored in.
        new_loss: Mean training loss of the rehearsal's model.
        stage2_loss: The one the stage-2 evaluation recorded.
        comparison: The Top-K comparison for the fold's queries.
        min_identical_share: The required share of identical lists.
        format_checks_passed: Whether every submission-format check passed.
    """

    n_customers: int
    n_shards: int
    new_loss: float
    stage2_loss: float
    comparison: TopComparison
    min_identical_share: float
    format_checks_passed: bool

    @property
    def loss_equal(self) -> bool:
        """Check 1: the two training losses are the same float64."""
        return self.new_loss == self.stage2_loss

    @property
    def top_identical_enough(self) -> bool:
        """Check 2: enough of the fold's queries have an identical Top-K list."""
        return self.comparison.share >= self.min_identical_share

    @property
    def passed(self) -> bool:
        """All three checks passed."""
        return self.loss_equal and self.top_identical_enough and self.format_checks_passed


def rehearsal_outcome(context: DeepFMKaggleContext) -> RehearsalOutcome:
    """Compare a finished rehearsal with the stage-2 evaluation (module docstring).

    Raises:
        ValueError: If the run is not the rehearsal, or a record to compare with is missing.
    """
    if context.run != RUN_REHEARSAL:
        raise ValueError("only a rehearsal is compared with the stage-2 evaluation")
    paths, kaggle = context.paths, context.kaggle
    new_queries = read_queries(paths.kaggle_queries_file(context.run))
    fold_queries = pl.read_parquet(
        paths.queries_file(kaggle.target.name), columns=[QUERY_ID, CUSTOMER_IDX]
    )
    comparison = compare_top_lists(
        read_all_tops(context), new_queries, stage2_top(context), fold_queries
    )
    return RehearsalOutcome(
        n_customers=new_queries.height,
        n_shards=run_shards(context),
        new_loss=new_train_loss(context),
        stage2_loss=stage2_train_loss(context),
        comparison=comparison,
        min_identical_share=kaggle.config.rehearsal.min_identical_top12_share,
        format_checks_passed=format_checks_passed(context),
    )


def record_rehearsal(context: DeepFMKaggleContext, outcome: RehearsalOutcome) -> None:
    """Append the outcome of a rehearsal to results/kaggle_deepfm_rehearsal.csv."""
    stamp = context.stamp
    row = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": stamp.git_commit,
        "git_dirty": str(stamp.git_dirty),
        "n_customers": str(outcome.n_customers),
        "n_shards": str(outcome.n_shards),
        "loss_equal": str(outcome.loss_equal),
        "new_train_loss": repr(outcome.new_loss),
        "stage2_train_loss": repr(outcome.stage2_loss),
        "n_fold_queries": str(outcome.comparison.n_queries),
        "n_identical_top": str(outcome.comparison.n_identical),
        "identical_top_share": repr(outcome.comparison.share),
        "min_identical_top_share": repr(outcome.min_identical_share),
        "format_checks_passed": str(outcome.format_checks_passed),
        "passed": str(outcome.passed),
    }
    append_row(context.paths.kaggle_deepfm_rehearsal_file, REHEARSAL_COLUMNS, row)


def require_passed_rehearsal(context: DeepFMKaggleContext, code_changed: CodeChanged) -> str:
    """Fail unless the last recorded rehearsal passed with the code that is running now.

    Args:
        context: The process's context.
        code_changed: Says whether the code at HEAD differs from a recorded commit
            (hmrec.git_state.code_changed_since: src, scripts, configs and the
            dependency files; result tables and documents do not count).

    Returns:
        The commit the passed rehearsal ran at.

    Raises:
        RuntimeError: If there is no rehearsal, the last one failed, was run from a
            dirty tree, or the code changed after it.
    """
    rows = read_rows(context.paths.kaggle_deepfm_rehearsal_file, REHEARSAL_COLUMNS)
    if not rows:
        raise RuntimeError("no rehearsal is recorded: the real run needs a passed rehearsal first")
    last = rows[-1]
    if last["passed"] != "True" or last["git_dirty"] != "False":
        raise RuntimeError(
            f"the last rehearsal ({last['checked_at']}, commit {last['git_commit'][:12]}) did "
            "not pass from a clean working tree: the real run does not start"
        )
    if code_changed(last["git_commit"]):
        raise RuntimeError(
            f"the code changed since the passed rehearsal (commit {last['git_commit'][:12]}): "
            "run the rehearsal again before the real run"
        )
    return last["git_commit"]


# ---------------------------------------------------------------- the submission's record


def relative_difference(score: float, reference: float) -> float:
    """(score - reference) / reference.

    Raises:
        ValueError: If the reference is not positive.
    """
    if reference <= 0:
        raise ValueError(f"the reference score must be > 0, got {reference}")
    return (score - reference) / reference


def reading_of(private_relative_difference: float, same_below: float) -> str:
    """The reading fixed before the run (docs/decisions.md 2026-11-20).

    |d| < same_below: "cannot be told apart on this week"; otherwise "a
    difference on a single week".
    """
    return SAME if abs(private_relative_difference) < same_below else DIFFERENT


def submission_row(
    context: DeepFMKaggleContext, result: SubmissionResult, file_row: dict[str, str]
) -> dict[str, str]:
    """The row of results/kaggle_deepfm_submission.csv of the one submission.

    Args:
        context: The process's context.
        result: What Kaggle returned.
        file_row: The row of results/kaggle_deepfm_files.csv of the submitted file.
    """
    settings = context.settings
    scores: Sequence[float] | None = result.leaderboard_scores
    private_difference = relative_difference(result.private_score, settings.lightgbm_private_score)
    public_difference = relative_difference(result.public_score, settings.lightgbm_public_score)
    return {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "competition": context.kaggle.config.competition,
        "submission_ref": result.submission_ref,
        "submitted_at": result.submitted_at,
        "status": result.status,
        "public_score": repr(result.public_score),
        "private_score": repr(result.private_score),
        "file": file_row["gzip_file"],
        "file_sha256": file_row["gzip_sha256"],
        "message": result.message,
        "run_git_commit": file_row["git_commit"],
        "submissions_before": str(result.submissions_before),
        "private_position": str(position_among(scores, result.private_score)) if scores else "",
        "leaderboard_teams": str(len(scores)) if scores else "",
        "position_label": POSITION_LABEL if scores else "",
        "lightgbm_private_score": repr(settings.lightgbm_private_score),
        "private_relative_difference": repr(private_difference),
        "lightgbm_public_score": repr(settings.lightgbm_public_score),
        "public_relative_difference": repr(public_difference),
        "same_below_relative_difference": repr(settings.same_below_relative_difference),
        "reading": reading_of(private_difference, settings.same_below_relative_difference),
    }


def record_submission(context: DeepFMKaggleContext, result: SubmissionResult) -> dict[str, str]:
    """Append the submission's row to results/kaggle_deepfm_submission.csv and return it.

    The file's sha256 and the commit of the run are taken from the run's last
    passed row of results/kaggle_deepfm_files.csv, so the row names exactly the
    file that was checked.

    Raises:
        ValueError: If no checked submission file of the run is recorded, or a row
            of this submission exists already (one submission, one row).
    """
    paths = context.paths
    files = [
        row
        for row in read_rows(paths.kaggle_deepfm_files_file, FILES_COLUMNS)
        if row["run"] == context.run and row["passed"] == "True"
    ]
    if not files:
        raise ValueError(f"no checked submission file of the {context.run} run is recorded")
    results_file = paths.kaggle_deepfm_submission_results_file
    if read_rows(results_file, DEEPFM_SUBMISSION_COLUMNS):
        raise ValueError(f"{results_file} already has a row: one submission is made")
    row = submission_row(context, result, files[-1])
    append_row(results_file, DEEPFM_SUBMISSION_COLUMNS, row)
    return row
