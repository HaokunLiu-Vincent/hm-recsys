"""CLI for the Kaggle late submission: write and check the submission file of a run.

    uv run python scripts/kaggle_submission.py --run rehearsal

Puts the Top-K of all shards together, writes the CSV and its gzip copy (the
real run: artifacts/kaggle/submission.csv), reads the file back and checks the
format. Records the file (results/kaggle_files.csv) and the run's candidate
counts (results/kaggle_candidates.csv), then exits with an error if a format
check failed. It uploads nothing: the submission itself is made by hand, once.
"""

# Thread setup must run before polars is imported (see hmrec.threads).
from hmrec.threads import configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402
import time  # noqa: E402

from hmrec.kaggle.context import ALL_SHARDS, DONE, finish_process, start_process  # noqa: E402
from hmrec.kaggle.plan import RUNS  # noqa: E402
from hmrec.kaggle.report import build_submission, record_candidates, record_files  # noqa: E402

STEP = "submission_file"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, choices=RUNS)
    return parser.parse_args()


def main() -> None:
    """Write and check the submission file, record it, stop on a failed check."""
    started = time.perf_counter()
    context = start_process(parse_args().run, STEP)
    check = build_submission(context)
    record_files(context, check)
    stats = record_candidates(context)
    detail = (
        f"csv sha256 {check.csv_sha256}, {len(check.checks)} format checks, failed: "
        f"{check.failed or 'none'}; {stats.n_candidate_rows} candidate rows, "
        f"{stats.per_customer_mean:.1f} per customer"
    )
    finish_process(context, STEP, ALL_SHARDS, DONE, check.n_rows, started, detail)
    if not check.passed:
        raise SystemExit(f"format checks FAILED: {check.failed}")


if __name__ == "__main__":
    main()
