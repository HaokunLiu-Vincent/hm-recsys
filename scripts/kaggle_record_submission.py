"""CLI for the Kaggle late submission: record what Kaggle returned for the ONE submission.

    uv run python scripts/kaggle_record_submission.py --ref 123 --status complete
        --submitted-at "2026-11-19 12:00:00" --public-score 0.02 --private-score 0.02
        --message "..." --submissions-before 2 [--leaderboard-csv FILE --score-column Score]

Appends one row to results/kaggle_submission.csv (it refuses a second row). The
numbers are typed in from the output of the Kaggle CLI; this script uploads and
downloads nothing. With --leaderboard-csv (a final leaderboard file the CLI
returned) it also records the place the Private score would have had, labelled
"late submission, not an official rank".
"""

# Thread setup must run before polars is imported (see hmrec.threads).
from hmrec.threads import configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402

import polars as pl  # noqa: E402

from hmrec.git_state import git_state  # noqa: E402
from hmrec.kaggle.context import load_context  # noqa: E402
from hmrec.kaggle.plan import RUN_SUBMISSION  # noqa: E402
from hmrec.kaggle.report import SubmissionResult, record_submission  # noqa: E402
from hmrec.logging_utils import get_logger, setup_logging  # noqa: E402
from hmrec.paths import get_paths  # noqa: E402
from hmrec.run_records import RunStamp  # noqa: E402

LOGGER = get_logger("kaggle_record_submission")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True, help="Kaggle's id of the submission")
    parser.add_argument("--submitted-at", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--public-score", required=True, type=float)
    parser.add_argument("--private-score", required=True, type=float)
    parser.add_argument("--message", required=True)
    parser.add_argument("--submissions-before", required=True, type=int)
    parser.add_argument("--leaderboard-csv", type=Path, default=None)
    parser.add_argument("--score-column", default="Score")
    return parser.parse_args()


def main() -> None:
    """Record the submission's row."""
    args = parse_args()
    paths = get_paths()
    setup_logging(logging.INFO, log_file=paths.kaggle_log_file(RUN_SUBMISSION, "record_submission"))
    commit, dirty = git_state(paths.project_root, paths.results_root)
    context = load_context(RUN_SUBMISSION, paths, RunStamp(commit, dirty))
    scores = None
    if args.leaderboard_csv is not None:
        leaderboard = pl.read_csv(args.leaderboard_csv)
        scores = tuple(float(score) for score in leaderboard[args.score_column].to_list())
    result = SubmissionResult(
        submission_ref=args.ref,
        submitted_at=args.submitted_at,
        status=args.status,
        public_score=args.public_score,
        private_score=args.private_score,
        message=args.message,
        submissions_before=args.submissions_before,
        leaderboard_scores=scores,
    )
    row = record_submission(context, result)
    LOGGER.info("submission recorded -> %s: %s", paths.kaggle_submission_results_file, row)


if __name__ == "__main__":
    main()
